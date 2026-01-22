use async_trait::async_trait;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
  fn name(&self) -> &str;
  fn model_id(&self) -> &str;
  fn dimensions(&self) -> usize;

  async fn embed(&self, text: &str) -> Result<Vec<f32>, crate::EmbeddingError>;
  async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, crate::EmbeddingError>;
  async fn is_available(&self) -> bool;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
  #[error("Provider not available")]
  NotAvailable,
  #[error("Request failed: {0}")]
  Request(#[from] reqwest::Error),
  #[error("Provider error: {0}")]
  ProviderError(String),
  #[error("Network error: {0}")]
  Network(String),
  #[error("Request timed out")]
  Timeout,
}
