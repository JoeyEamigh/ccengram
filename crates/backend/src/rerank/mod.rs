mod deepinfra;
#[cfg(feature = "llama-cpp")]
pub mod llamacpp;

pub use deepinfra::DeepInfraReranker;

pub struct RerankCandidate {
  pub id: String,
  pub text: String,
}

pub struct RerankResult {
  pub id: String,
  pub score: f32,
}

pub struct RerankRequest {
  pub query: String,
  pub instruction: Option<String>,
  pub candidates: Vec<RerankCandidate>,
  pub top_n: Option<usize>,
}

pub struct RerankResponse {
  pub results: Vec<RerankResult>,
  pub duration_ms: u64,
}

#[async_trait::async_trait]
pub trait RerankerProvider: Send + Sync {
  fn name(&self) -> &str;
  fn is_available(&self) -> bool;
  fn max_candidates(&self) -> usize;
  async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, RerankerError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RerankerError {
  #[error("Reranker not available")]
  NotAvailable,
  #[error("Request failed: {0}")]
  Request(#[from] reqwest::Error),
  #[error("Provider error: {0}")]
  ProviderError(String),
  #[error("Reranker request timed out")]
  Timeout,
}
