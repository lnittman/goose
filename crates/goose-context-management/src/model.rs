use std::sync::Arc;

use async_trait::async_trait;
use goose_providers::base::Provider;
use goose_providers::conversation::message::Message;
use goose_providers::conversation::token_usage::ProviderUsage;
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;

/// The single completion call compaction needs. Implementations decide model
/// selection, fallbacks and session plumbing.
#[async_trait]
pub trait CompactionModel: Send + Sync {
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
    ) -> Result<(Message, ProviderUsage), ProviderError>;

    /// Replays the conversation's own request prefix. Implementations must
    /// keep it cache-compatible with the last routed request (same model,
    /// cache on, same thinking config); the default reports unsupported so
    /// callers fall back.
    async fn complete_prefix(
        &self,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        Err(ProviderError::ExecutionError(
            "prefix compaction is not supported by this CompactionModel".to_string(),
        ))
    }
}

/// Counts tokens for usage estimation and retained-context reporting.
#[async_trait]
pub trait TokenEstimator: Send + Sync {
    async fn count_chat_tokens(&self, system: &str, messages: &[Message]) -> usize;
    /// Like [`Self::count_chat_tokens`] but including tool schemas; the
    /// default ignores them.
    async fn count_chat_tokens_with_tools(
        &self,
        system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> usize {
        self.count_chat_tokens(system, messages).await
    }
    async fn count_text_tokens(&self, text: &str) -> usize;
}

pub struct ProviderModel {
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
}

impl ProviderModel {
    pub fn new(provider: Arc<dyn Provider>, model_config: ModelConfig) -> Self {
        Self {
            provider,
            model_config,
        }
    }
}

#[async_trait]
impl CompactionModel for ProviderModel {
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        self.provider
            .complete(&self.model_config, system, messages, &[])
            .await
    }

    async fn complete_prefix(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        self.provider
            .complete(&self.model_config, system, messages, tools)
            .await
    }
}
