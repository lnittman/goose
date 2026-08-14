//! With `GOOSE_COMPACTION_CACHE_PREFIX` enabled, the compaction request must
//! replay the last routed request's own prefix (system prompt, tools,
//! projected messages) with the compaction instruction as the final user
//! message; otherwise the standalone summarizer shape is used unchanged.

use std::sync::Mutex;

use async_trait::async_trait;
use goose::context_mgmt::{compact_messages, request_header, CompactionResult};
use goose::conversation::message::Message;
use goose::conversation::{fix_conversation, merge_consecutive_messages_for_request, Conversation};
use goose::providers::base::Provider;
use goose_providers::base::{stream_from_single_message, MessageStream};
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::formats::anthropic::{self, AnthropicFormatOptions};
use goose_providers::formats::openai;
use goose_providers::images::ImageFormat;
use goose_providers::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, Tool};
use rmcp::object;
use serde_json::Value;
use serial_test::serial;

const SYSTEM: &str = "You are goose, a careful coding assistant.";

struct EnvGuard {
    keys: Vec<&'static str>,
}

impl EnvGuard {
    fn set(vars: &[(&'static str, &str)]) -> Self {
        for (key, value) in vars {
            std::env::set_var(key, value);
        }
        Self {
            keys: vars.iter().map(|(key, _)| *key).collect(),
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for key in &self.keys {
            std::env::remove_var(key);
        }
    }
}

// A blank GOOSE_COMPACTION_MODEL shadows any model pinned in the local user
// config, keeping the compaction model equal to the main model.
fn prefix_env() -> EnvGuard {
    EnvGuard::set(&[
        ("GOOSE_COMPACTION_CACHE_PREFIX", "true"),
        ("GOOSE_COMPACTION_MODEL", ""),
    ])
}

#[derive(Clone)]
struct CapturedRequest {
    model_config: ModelConfig,
    system: String,
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

struct CapturingProvider {
    response: Message,
    usage: Usage,
    captured: Mutex<Vec<CapturedRequest>>,
}

impl CapturingProvider {
    fn new(response: Message) -> Self {
        Self::with_usage(response, Usage::default())
    }

    fn with_usage(response: Message, usage: Usage) -> Self {
        Self {
            response,
            usage,
            captured: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<CapturedRequest> {
        self.captured.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for CapturingProvider {
    fn get_name(&self) -> &str {
        "anthropic"
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.captured.lock().unwrap().push(CapturedRequest {
            model_config: model_config.clone(),
            system: system.to_string(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
        });
        Ok(stream_from_single_message(
            self.response.clone(),
            ProviderUsage::new("claude-test".to_string(), self.usage),
        ))
    }
}

fn tools() -> Vec<Tool> {
    vec![Tool::new(
        "read_file",
        "Read a file from disk",
        object!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }),
    )]
}

fn conversation() -> Conversation {
    Conversation::new_unvalidated(vec![
        Message::user().with_text("What does the main entrypoint do?"),
        Message::assistant().with_tool_request(
            "call_1",
            Ok(CallToolRequestParams::new("read_file")
                .with_arguments(object!({ "path": "src/main.rs" }))),
        ),
        Message::user().with_tool_response(
            "call_1",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "fn main() { run(); }",
            )])),
        ),
        Message::assistant().with_text("It calls `run()`."),
    ])
}

/// The projection every request goes through in
/// `stream_response_from_provider` before any formatter.
fn provider_view(messages: &[Message]) -> Vec<Message> {
    let projected =
        Conversation::new_unvalidated(messages.iter().cloned()).agent_visible_messages();
    let (fixed, _) = fix_conversation(Conversation::new_unvalidated(projected));
    merge_consecutive_messages_for_request(fixed.messages().clone())
}

fn main_model_config() -> ModelConfig {
    ModelConfig::new("claude-test").with_context_limit(Some(200_000))
}

fn record_header(session_id: &str, model_name: &str) {
    request_header::record(
        session_id,
        request_header::RequestHeader {
            provider_name: "anthropic".to_string(),
            model_name: model_name.to_string(),
            system_prompt: SYSTEM.to_string(),
            tools: tools(),
        },
    );
}

async fn run_compaction(
    provider: &CapturingProvider,
    session_id: &str,
    conversation: &Conversation,
) -> CompactionResult {
    compact_messages(
        provider,
        &main_model_config(),
        session_id,
        conversation,
        true,
    )
    .await
    .unwrap()
}

#[tokio::test]
#[serial(compaction_cache_prefix_env)]
async fn prefix_compaction_replays_the_recorded_request_header() {
    let _env = prefix_env();
    let session_id = "prefix-session";
    record_header(session_id, "claude-test");

    let provider = CapturingProvider::new(Message::assistant().with_text("<summary>"));
    let conversation = conversation();
    run_compaction(&provider, session_id, &conversation).await;

    let requests = provider.requests();
    assert_eq!(requests.len(), 1, "prefix shape must not need retries");
    let request = &requests[0];

    assert_eq!(request.system, SYSTEM);
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "read_file");
    assert!(
        !request.model_config.prompt_cache_disabled(),
        "the prefix request must keep prompt-cache breakpoints"
    );

    let expected_prefix = provider_view(conversation.messages());
    assert_eq!(&request.messages[..expected_prefix.len()], &expected_prefix);
    assert_eq!(request.messages.len(), expected_prefix.len() + 2);
    let instruction = request.messages.last().unwrap();
    assert_eq!(instruction.role, rmcp::model::Role::User);
    assert!(instruction
        .as_concat_text()
        .contains("distill the conversation so far"));
}

#[tokio::test]
#[serial(compaction_cache_prefix_env)]
async fn inapplicable_states_fall_back_to_the_standalone_shape() {
    let scenarios: [(&str, &str, &str, Option<&str>); 4] = [
        ("gate off", "false", "", Some("claude-test")),
        ("no recorded header", "true", "", None),
        (
            "compaction model override",
            "true",
            "other-model",
            Some("claude-test"),
        ),
        (
            "header from another model",
            "true",
            "",
            Some("claude-previous"),
        ),
    ];

    for (name, gate, compaction_model, header_model) in scenarios {
        let _env = EnvGuard::set(&[
            ("GOOSE_COMPACTION_CACHE_PREFIX", gate),
            ("GOOSE_COMPACTION_MODEL", compaction_model),
        ]);
        let session_id = format!("fallback-{name}");
        if let Some(model) = header_model {
            record_header(&session_id, model);
        }

        let provider = CapturingProvider::new(Message::assistant().with_text("<summary>"));
        run_compaction(&provider, &session_id, &conversation()).await;

        let requests = provider.requests();
        assert_eq!(requests.len(), 1, "{name}");
        assert!(requests[0].tools.is_empty(), "{name}: standalone shape");
        assert!(
            requests[0]
                .system
                .contains("What does the main entrypoint do?"),
            "{name}: transcript flattened into the system prompt"
        );
        assert!(
            requests[0].model_config.prompt_cache_disabled(),
            "{name}: one-shot semantics"
        );
    }
}

#[tokio::test]
#[serial(compaction_cache_prefix_env)]
async fn a_rejected_prefix_response_falls_back_and_keeps_its_usage() {
    let _env = prefix_env();
    let session_id = "prefix-session-rejected";
    record_header(session_id, "claude-test");

    let response = Message::assistant().with_tool_request(
        "call_2",
        Ok(CallToolRequestParams::new("read_file")
            .with_arguments(object!({ "path": "src/lib.rs" }))),
    );
    let provider = CapturingProvider::with_usage(response, Usage::new(Some(100), Some(10), None));
    let result = run_compaction(&provider, session_id, &conversation()).await;

    let requests = provider.requests();
    assert_eq!(
        requests.len(),
        2,
        "prefix attempt, then standalone fallback"
    );
    assert!(requests[1].tools.is_empty(), "fallback is standalone");
    assert_eq!(
        result.usage.usage.input_tokens,
        Some(200),
        "the rejected prefix call's usage must be counted"
    );
}

/// Strip `cache_control` markers: caches key on the normalized bytes, not the
/// marker encoding.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| k.as_str() != "cache_control")
                .map(|(k, v)| (k.clone(), canonical(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn assert_message_prefix(prev: &[Value], next: &[Value], formatter: &str) {
    assert!(next.len() > prev.len(), "{formatter}: request must extend");
    for (i, (p, n)) in prev.iter().zip(next).enumerate() {
        assert_eq!(p, n, "{formatter}: cached bytes changed at message {i}");
    }
}

#[tokio::test]
#[serial(compaction_cache_prefix_env)]
async fn compaction_request_extends_the_conversation_request_in_both_formats() {
    let _env = prefix_env();
    let session_id = "prefix-session-invariance";
    record_header(session_id, "claude-test");

    let provider = CapturingProvider::new(Message::assistant().with_text("<summary>"));
    let conversation = conversation();
    run_compaction(&provider, session_id, &conversation).await;
    let compaction = &provider.requests()[0];

    // The conversation's last routed request: everything up to the final
    // assistant reply it produced.
    let last_request_messages = provider_view(&conversation.messages()[..3]);

    let anthropic_config = ModelConfig::new("claude-test");
    let format_anthropic = |messages: &[Message]| {
        canonical(
            &anthropic::create_request(
                "anthropic",
                &anthropic_config,
                SYSTEM,
                messages,
                &tools(),
                AnthropicFormatOptions::default(),
            )
            .unwrap(),
        )
    };
    let prev = format_anthropic(&last_request_messages);
    let next = format_anthropic(&compaction.messages);
    assert_eq!(prev["tools"], next["tools"], "anthropic: tools changed");
    assert_eq!(prev["system"], next["system"], "anthropic: system changed");
    assert_message_prefix(
        prev["messages"].as_array().unwrap(),
        next["messages"].as_array().unwrap(),
        "anthropic",
    );

    let openai_config = ModelConfig::new("gpt-4.1-mini");
    let format_openai = |messages: &[Message]| {
        openai::create_request(
            &openai_config,
            SYSTEM,
            messages,
            &tools(),
            &ImageFormat::OpenAi,
            false,
        )
        .unwrap()["messages"]
            .as_array()
            .unwrap()
            .clone()
    };
    assert_message_prefix(
        &format_openai(&last_request_messages),
        &format_openai(&compaction.messages),
        "openai",
    );
}
