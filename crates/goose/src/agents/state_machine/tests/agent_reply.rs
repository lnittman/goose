//! Covers `Agent::reply_with_state_machine`, the entry point the CLI and desktop
//! reach when the state machine is enabled.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use rmcp::model::{ElicitationAction, Tool};
use serde_json::Value;
use tokio::sync::{oneshot, Mutex};
use tokio_util::sync::CancellationToken;

use super::dummy_api::{DummyApi, ProviderFeatures};
use crate::action_required_manager::ElicitationOutcome;
use crate::agents::{Agent, AgentConfig, AgentEvent, GoosePlatform, SessionConfig};
use crate::config::permission::PermissionManager;
use crate::config::GooseMode;
use crate::conversation::message::{
    ActionRequiredData, InferenceMetadata, Message, MessageContent,
};
use crate::providers::base::{MessageStream, PermissionRouting, Provider};
use crate::session::{SessionManager, SessionType};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;

const NESTED_ELICITATION_ID: &str = "acp-provider:state-machine-test";

struct NestedElicitationProvider {
    pending: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    /// Mirrors the real provider: a claim moves the waiter out of `pending` so a
    /// second submitter cannot take it while the first is persisting.
    claimed: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl NestedElicitationProvider {
    fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(None)),
            claimed: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait]
impl Provider for NestedElicitationProvider {
    fn get_name(&self) -> &str {
        "nested-acp-test"
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let pending = self.pending.clone();
        Ok(Box::pin(async_stream::try_stream! {
            let (response_tx, response_rx) = oneshot::channel();
            *pending.lock().await = Some(response_tx);
            yield (
                Some(
                    Message::assistant()
                        .with_content(MessageContent::action_required_elicitation(
                            NESTED_ELICITATION_ID.to_string(),
                            "Choose a direction".to_string(),
                            serde_json::json!({
                                "type": "object",
                                "properties": { "direction": { "type": "string" } },
                                "required": ["direction"]
                            }),
                        ))
                        .user_only(),
                ),
                None,
            );
            response_rx.await.map_err(|error| {
                ProviderError::ExecutionError(format!("elicitation response dropped: {error}"))
            })?;
            yield (Some(Message::assistant().with_text("continued after answer")), None);
        }))
    }

    async fn has_pending_elicitation(&self, request_id: &str) -> bool {
        request_id == NESTED_ELICITATION_ID
            && (self.pending.lock().await.is_some() || self.claimed.lock().await.is_some())
    }

    async fn claim_elicitation(&self, request_id: &str) -> bool {
        if request_id != NESTED_ELICITATION_ID {
            return false;
        }
        let Some(response_tx) = self.pending.lock().await.take() else {
            return false;
        };
        *self.claimed.lock().await = Some(response_tx);
        true
    }

    async fn release_elicitation(&self, request_id: &str) {
        if request_id != NESTED_ELICITATION_ID {
            return;
        }
        if let Some(response_tx) = self.claimed.lock().await.take() {
            *self.pending.lock().await = Some(response_tx);
        }
    }

    async fn handle_elicitation_response(
        &self,
        request_id: &str,
        _user_data: &Value,
        _action: &ElicitationAction,
    ) -> bool {
        if request_id != NESTED_ELICITATION_ID {
            return false;
        }
        let claimed = self.claimed.lock().await.take();
        let response_tx = match claimed {
            Some(response_tx) => Some(response_tx),
            None => self.pending.lock().await.take(),
        };
        let Some(response_tx) = response_tx else {
            return false;
        };
        response_tx.send(()).is_ok()
    }
}

struct ActivationSensitiveProvider {
    armed: AtomicBool,
    prepared: AtomicBool,
    prepare_calls: AtomicUsize,
    queried_before_prepare: AtomicBool,
    context_queried_after_prepare: AtomicBool,
}

impl ActivationSensitiveProvider {
    fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            prepared: AtomicBool::new(false),
            prepare_calls: AtomicUsize::new(0),
            queried_before_prepare: AtomicBool::new(false),
            context_queried_after_prepare: AtomicBool::new(false),
        }
    }

    fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }

    fn record_capability_query(&self) -> bool {
        let prepared = self.prepared.load(Ordering::SeqCst);
        if self.armed.load(Ordering::SeqCst) && !prepared {
            self.queried_before_prepare.store(true, Ordering::SeqCst);
        }
        prepared
    }
}

#[async_trait]
impl Provider for ActivationSensitiveProvider {
    fn get_name(&self) -> &str {
        "cursor-activation-test"
    }

    async fn prepare_session(
        &self,
        _provider_session_id: Option<&str>,
        _has_provider_history: bool,
    ) -> Result<(), ProviderError> {
        self.prepare_calls.fetch_add(1, Ordering::SeqCst);
        self.prepared.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        if !self.prepared.load(Ordering::SeqCst) {
            return Err(ProviderError::RequestFailed(
                "stream entered before session activation".to_string(),
            ));
        }
        Ok(Box::pin(async_stream::try_stream! {
            yield (Some(Message::assistant().with_text("activated")), None);
        }))
    }

    async fn get_context_limit(&self, _model_config: &ModelConfig) -> Result<usize, ProviderError> {
        if self.record_capability_query() {
            self.context_queried_after_prepare
                .store(true, Ordering::SeqCst);
            Ok(262_144)
        } else {
            Ok(8_192)
        }
    }

    fn manages_own_context(&self) -> bool {
        self.record_capability_query()
    }

    fn permission_routing(&self) -> PermissionRouting {
        if self.record_capability_query() {
            PermissionRouting::ActionRequired
        } else {
            PermissionRouting::Noop
        }
    }
}

struct PrepareFailureProvider {
    prepare_calls: AtomicUsize,
    stream_calls: AtomicUsize,
    saw_saved_session: AtomicBool,
}

impl PrepareFailureProvider {
    fn new() -> Self {
        Self {
            prepare_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
            saw_saved_session: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Provider for PrepareFailureProvider {
    fn get_name(&self) -> &str {
        "cursor-resume-failure-test"
    }

    async fn prepare_session(
        &self,
        provider_session_id: Option<&str>,
        has_provider_history: bool,
    ) -> Result<(), ProviderError> {
        self.prepare_calls.fetch_add(1, Ordering::SeqCst);
        self.saw_saved_session.store(
            provider_session_id == Some("saved-cursor-session") && has_provider_history,
            Ordering::SeqCst,
        );
        Err(ProviderError::RequestFailed(
            "Cursor ACP resume failed".to_string(),
        ))
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        Err(ProviderError::RequestFailed(
            "stream must not follow a failed resume".to_string(),
        ))
    }
}

async fn agent_with_dummy_api() -> Result<(Agent, Arc<DummyApi>, String, tempfile::TempDir)> {
    let api = Arc::new(DummyApi::start(ProviderFeatures::default()).await);
    let api_client = goose_providers::api_client::ApiClient::new_with_tls(
        api.uri(),
        goose_providers::api_client::AuthMethod::NoAuth,
        None,
    )?;
    let provider: Arc<dyn Provider> = Arc::new(
        goose_providers::openai::OpenAiProviderBuilder::new(api_client)
            .name("openai")
            .build(),
    );

    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "state-machine-reply".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let agent = Agent::with_config(AgentConfig::new(
        session_manager,
        PermissionManager::instance(),
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    ));
    agent
        .update_provider(
            provider,
            ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                .with_canonical_limits("openai"),
            &session.id,
        )
        .await?;

    Ok((agent, api, session.id, temp_dir))
}

async fn agent_with_nested_elicitation_provider() -> Result<(Agent, String, tempfile::TempDir)> {
    let provider: Arc<dyn Provider> = Arc::new(NestedElicitationProvider::new());
    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "state-machine-elicitation".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let agent = Agent::with_config(AgentConfig::new(
        session_manager,
        PermissionManager::instance(),
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    ));
    agent
        .update_provider(provider, ModelConfig::new("nested-acp-test"), &session.id)
        .await?;

    Ok((agent, session.id, temp_dir))
}

async fn agent_with_activation_sensitive_provider() -> Result<(
    Agent,
    Arc<ActivationSensitiveProvider>,
    String,
    tempfile::TempDir,
)> {
    let provider = Arc::new(ActivationSensitiveProvider::new());
    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "cursor-session-activation".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let agent = Agent::with_config(AgentConfig::new(
        session_manager,
        PermissionManager::instance(),
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    ));
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("cursor-activation-test"),
            &session.id,
        )
        .await?;
    provider.arm();

    Ok((agent, provider, session.id, temp_dir))
}

async fn agent_with_prepare_failure_provider() -> Result<(
    Agent,
    Arc<PrepareFailureProvider>,
    String,
    tempfile::TempDir,
)> {
    let provider = Arc::new(PrepareFailureProvider::new());
    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "cursor-resume-failure".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    session_manager
        .add_message(
            &session.id,
            &Message::assistant()
                .with_text("prior Cursor response")
                .with_inference(InferenceMetadata {
                    provider: "cursor-resume-failure-test".to_string(),
                    requested_model: "cursor-resume-failure-test".to_string(),
                    resolved_model: None,
                    provider_session_id: Some("saved-cursor-session".to_string()),
                }),
        )
        .await?;
    let agent = Agent::with_config(AgentConfig::new(
        session_manager,
        PermissionManager::instance(),
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    ));
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("cursor-resume-failure-test"),
            &session.id,
        )
        .await?;

    Ok((agent, provider, session.id, temp_dir))
}

#[tokio::test]
async fn reply_streams_the_turn_and_ends() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("are you there?").reply("still here");

    let session_config = SessionConfig {
        id: session_id.clone(),
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let stream = agent
        .reply_with_state_machine(
            Message::user().with_text("are you there?"),
            session_config,
            Some(CancellationToken::new()),
        )
        .await?;

    let replies = tokio::time::timeout(Duration::from_secs(30), async move {
        tokio::pin!(stream);
        let mut replies = Vec::new();
        while let Some(event) = stream.next().await {
            if let AgentEvent::Message(message) = event? {
                replies.push(message.as_concat_text());
            }
        }
        anyhow::Ok(replies)
    })
    .await??;

    assert!(
        replies.iter().any(|reply| reply == "still here"),
        "expected the scripted reply, got {replies:?}"
    );
    assert_eq!(api.call_count(), 1);

    Ok(())
}

async fn assert_nested_provider_elicitation_order(use_state_machine: bool) -> Result<()> {
    let (agent, session_id, _temp_dir) = agent_with_nested_elicitation_provider().await?;
    let session_config = SessionConfig {
        id: session_id.clone(),
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let user_message = Message::user().with_text("interview me");
    let cancel = Some(CancellationToken::new());
    let stream = if use_state_machine {
        agent
            .reply_with_state_machine(user_message, session_config, cancel)
            .await?
    } else {
        agent.reply(user_message, session_config, cancel).await?
    };
    tokio::pin!(stream);

    let mut saw_request = false;
    let mut saw_continuation = false;
    while let Some(event) = stream.next().await {
        let AgentEvent::Message(message) = event? else {
            continue;
        };
        if message.content.iter().any(|content| {
            matches!(
                content,
                MessageContent::ActionRequired(action)
                    if matches!(
                        &action.data,
                        ActionRequiredData::Elicitation { id, .. }
                            if id == NESTED_ELICITATION_ID
                    )
            )
        }) {
            saw_request = true;
            let (first, second) = tokio::join!(
                agent.submit_elicitation_response(
                    &session_id,
                    NESTED_ELICITATION_ID,
                    ElicitationOutcome::Accept(serde_json::json!({
                        "direction": "local"
                    })),
                    None,
                ),
                agent.submit_elicitation_response(
                    &session_id,
                    NESTED_ELICITATION_ID,
                    ElicitationOutcome::Accept(serde_json::json!({
                        "direction": "upstream"
                    })),
                    None,
                ),
            );
            assert_eq!(
                [first?, second?]
                    .into_iter()
                    .filter(|submitted| *submitted)
                    .count(),
                1,
                "only one concurrent response may consume the live elicitation"
            );
        }
        saw_continuation |= message.as_concat_text() == "continued after answer";
    }

    assert!(saw_request);
    assert!(saw_continuation);
    let session = agent
        .config
        .session_manager
        .get_session(&session_id, true)
        .await?;
    let conversation = session.conversation.expect("conversation");
    let messages = conversation.messages();
    let request_index = messages
        .iter()
        .position(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    MessageContent::ActionRequired(action)
                        if matches!(
                            &action.data,
                            ActionRequiredData::Elicitation { id, .. }
                                if id == NESTED_ELICITATION_ID
                        )
                )
            })
        })
        .expect("persisted elicitation request");
    let response_index = messages
        .iter()
        .position(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    MessageContent::ActionRequired(action)
                        if matches!(
                            &action.data,
                            ActionRequiredData::ElicitationResponse { id, .. }
                                if id == NESTED_ELICITATION_ID
                        )
                )
            })
        })
        .expect("persisted elicitation response");
    let continuation_index = messages
        .iter()
        .position(|message| message.as_concat_text() == "continued after answer")
        .expect("persisted continuation");

    assert!(request_index < response_index);
    assert!(response_index < continuation_index);
    assert_eq!(
        messages
            .iter()
            .filter(|message| crate::acp::is_provider_form_elicitation(message))
            .count(),
        1
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.content.iter().any(|content| {
                    matches!(
                        content,
                        MessageContent::ActionRequired(action)
                            if matches!(
                                &action.data,
                                ActionRequiredData::ElicitationResponse { id, .. }
                                    if id == NESTED_ELICITATION_ID
                            )
                    )
                })
            })
            .count(),
        1
    );

    Ok(())
}

#[tokio::test]
async fn nested_provider_elicitation_persists_before_response_in_state_machine() -> Result<()> {
    assert_nested_provider_elicitation_order(true).await
}

#[tokio::test]
async fn nested_provider_elicitation_persists_before_response_in_legacy_loop() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", None::<&str>)]);
    assert_nested_provider_elicitation_order(false).await
}

async fn assert_cursor_transport_precedes_capability_queries(
    use_state_machine: bool,
) -> Result<()> {
    let (agent, provider, session_id, _temp_dir) =
        agent_with_activation_sensitive_provider().await?;
    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let user_message = Message::user().with_text("use the selected transport");
    let cancel = Some(CancellationToken::new());
    let stream = if use_state_machine {
        agent
            .reply_with_state_machine(user_message, session_config, cancel)
            .await?
    } else {
        agent.reply(user_message, session_config, cancel).await?
    };
    let saw_activated = tokio::time::timeout(Duration::from_secs(30), async move {
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            if let AgentEvent::Message(message) = event? {
                if message.as_concat_text() == "activated" {
                    return anyhow::Ok(true);
                }
            }
        }
        anyhow::Ok(false)
    })
    .await??;

    assert!(saw_activated);
    assert!(provider.prepared.load(Ordering::SeqCst));
    assert_eq!(provider.prepare_calls.load(Ordering::SeqCst), 1);
    assert!(provider
        .context_queried_after_prepare
        .load(Ordering::SeqCst));
    assert!(!provider.queried_before_prepare.load(Ordering::SeqCst));

    Ok(())
}

#[tokio::test]
async fn cursor_transport_precedes_capability_queries_in_state_machine() -> Result<()> {
    assert_cursor_transport_precedes_capability_queries(true).await
}

#[tokio::test]
async fn cursor_transport_precedes_capability_queries_in_legacy_loop() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", None::<&str>)]);
    assert_cursor_transport_precedes_capability_queries(false).await
}

async fn assert_cursor_resume_failure_is_terminal(use_state_machine: bool) -> Result<()> {
    let (agent, provider, session_id, _temp_dir) = agent_with_prepare_failure_provider().await?;
    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let user_message = Message::user().with_text("resume the Cursor session");
    let cancel = Some(CancellationToken::new());
    let result = if use_state_machine {
        agent
            .reply_with_state_machine(user_message, session_config, cancel)
            .await
    } else {
        agent.reply(user_message, session_config, cancel).await
    };
    let error = match result {
        Ok(_) => panic!("failed session preparation must stop the reply"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("Cursor ACP resume failed"));
    assert_eq!(provider.prepare_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 0);
    assert!(provider.saw_saved_session.load(Ordering::SeqCst));

    Ok(())
}

#[tokio::test]
async fn cursor_resume_failure_is_terminal_in_state_machine() -> Result<()> {
    assert_cursor_resume_failure_is_terminal(true).await
}

#[tokio::test]
async fn cursor_resume_failure_is_terminal_in_legacy_loop() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", None::<&str>)]);
    assert_cursor_resume_failure_is_terminal(false).await
}

#[tokio::test]
async fn bang_shell_uses_the_state_machine_when_the_flag_is_disabled() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", None::<&str>)]);
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let stream = agent
        .reply(
            Message::user().with_text("!echo hello"),
            session_config,
            Some(CancellationToken::new()),
        )
        .await?;
    tokio::pin!(stream);
    let mut requested_shell = false;
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            requested_shell |= message.content.iter().any(|content| {
                matches!(
                    content,
                    crate::conversation::message::MessageContent::ToolRequest(request)
                        if request.tool_call.as_ref().is_ok_and(|call| call.name == "shell")
                )
            });
        }
    }

    assert!(requested_shell);
    assert_eq!(api.call_count(), 0);

    Ok(())
}
