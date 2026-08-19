//! Covers `Agent::reply_with_state_machine`, the entry point the CLI and desktop
//! reach when the state machine is enabled.

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
use crate::conversation::message::{ActionRequiredData, Message, MessageContent};
use crate::providers::base::{MessageStream, Provider};
use crate::session::{SessionManager, SessionType};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;

const NESTED_ELICITATION_ID: &str = "acp-provider:state-machine-test";

struct NestedElicitationProvider {
    pending: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl NestedElicitationProvider {
    fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(None)),
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
        request_id == NESTED_ELICITATION_ID && self.pending.lock().await.is_some()
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
        let Some(response_tx) = self.pending.lock().await.take() else {
            return false;
        };
        response_tx.send(()).is_ok()
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
