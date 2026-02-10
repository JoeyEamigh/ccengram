mod ollama;
mod openai_compat;
mod rate_limit;
mod resilient;
pub mod validation;

#[cfg(feature = "llama-cpp")]
pub mod llamacpp;

use std::sync::Arc;

pub use ollama::OllamaProvider;
pub use openai_compat::OpenAiCompatibleProvider;
use resilient::{ResilientProvider, RetryConfig};

use crate::domain::config::{EmbeddingConfig, EmbeddingProvider as ConfigEmbeddingProvider};

/// Embedding mode determines how text is formatted before embedding.
///
/// qwen3-embedding (and similar instruction-following embedding models) produce
/// better results when queries are prefixed with a task instruction, while
/// documents are embedded without any prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingMode {
  /// Embedding a document for storage/indexing.
  /// Text is embedded as-is without any prefix.
  #[default]
  Document,
  /// Embedding a query for retrieval/search.
  /// Text is prefixed with a task instruction for better retrieval.
  Query,
}

#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
  fn name(&self) -> &str;
  fn model_id(&self) -> &str;
  fn dimensions(&self) -> usize;

  async fn embed(&self, text: &str, mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError>;
  async fn embed_batch(&self, texts: &[&str], mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

impl dyn EmbeddingProvider {
  pub async fn from_config(config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingProvider>, EmbeddingError> {
    match config.provider {
      ConfigEmbeddingProvider::Ollama => {
        let provider = OllamaProvider::new(config)?;

        Ok(Arc::new(provider))
      }
      ConfigEmbeddingProvider::OpenRouter => {
        let provider = OpenAiCompatibleProvider::from_embedding_config_openrouter(config)?;

        let resilient = ResilientProvider::with_config(provider, RetryConfig::for_cloud());
        Ok(Arc::new(resilient))
      }
      ConfigEmbeddingProvider::DeepInfra => {
        let provider = OpenAiCompatibleProvider::from_embedding_config_deepinfra(config)?;

        let resilient = ResilientProvider::with_config(provider, RetryConfig::for_cloud());
        Ok(Arc::new(resilient))
      }
      #[cfg(feature = "llama-cpp")]
      ConfigEmbeddingProvider::LlamaCpp => {
        let provider = llamacpp::LlamaCppEmbeddingProvider::new(config).await?;
        Ok(Arc::new(provider))
      }
      #[cfg(not(feature = "llama-cpp"))]
      ConfigEmbeddingProvider::LlamaCpp => {
        let provider = OpenAiCompatibleProvider::from_embedding_config_llamacpp(config);
        Ok(Arc::new(provider))
      }
    }
  }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
  #[error("No api key configured for provider")]
  NoApiKey,
  #[error("Request failed: {0}")]
  Request(#[from] reqwest::Error),
  #[error("Provider error: {0}")]
  ProviderError(String),
  #[error("Network error: {0}")]
  Network(String),
  #[error("Request timed out")]
  Timeout,
  #[error("Upstream provider timeout (received only keep-alive data)")]
  UpstreamTimeout,
  #[error("Rate limit exhausted after waiting {0:?}")]
  RateLimitExhausted(std::time::Duration),
  #[error("Response parsing failed: {0}")]
  ParseError(String),
  #[error("Batch size mismatch: expected {expected}, got {got}")]
  BatchSizeMismatch { expected: usize, got: usize },
}

impl EmbeddingError {
  /// Returns true if this error warrants backoff before retry.
  /// Upstream timeouts don't need backoff - the provider already waited.
  pub fn needs_backoff(&self) -> bool {
    !matches!(self, EmbeddingError::UpstreamTimeout)
  }
}
