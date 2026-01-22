use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const DEFAULT_MODEL: &str = "openai/text-embedding-3-small";
const DEFAULT_DIMENSIONS: usize = 1536;

#[derive(Debug, Clone)]
pub struct OpenRouterProvider {
  client: reqwest::Client,
  api_key: String,
  model: String,
  dimensions: usize,
}

impl OpenRouterProvider {
  pub fn new(api_key: impl Into<String>) -> Self {
    Self {
      client: reqwest::Client::new(),
      api_key: api_key.into(),
      model: DEFAULT_MODEL.to_string(),
      dimensions: DEFAULT_DIMENSIONS,
    }
  }

  pub fn with_model(mut self, model: impl Into<String>, dimensions: usize) -> Self {
    self.model = model.into();
    self.dimensions = dimensions;
    self
  }

  pub fn from_env() -> Option<Self> {
    std::env::var("OPENROUTER_API_KEY").ok().map(Self::new)
  }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
  model: &'a str,
  input: EmbeddingInput<'a>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum EmbeddingInput<'a> {
  Single(&'a str),
  Batch(Vec<&'a str>),
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
  data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
  embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OpenRouterProvider {
  fn name(&self) -> &str {
    "openrouter"
  }

  fn model_id(&self) -> &str {
    &self.model
  }

  fn dimensions(&self) -> usize {
    self.dimensions
  }

  async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
    let request = EmbeddingRequest {
      model: &self.model,
      input: EmbeddingInput::Single(text),
    };

    debug!("Embedding text with OpenRouter: {} chars", text.len());

    let response = self
      .client
      .post(OPENROUTER_URL)
      .header("Authorization", format!("Bearer {}", self.api_key))
      .header("Content-Type", "application/json")
      .json(&request)
      .send()
      .await?;

    if !response.status().is_success() {
      let status = response.status();
      let body = response.text().await.unwrap_or_default();
      warn!("OpenRouter embedding failed: {} - {}", status, body);
      return Err(EmbeddingError::ProviderError(format!(
        "OpenRouter returned {}: {}",
        status, body
      )));
    }

    let result: EmbeddingResponse = response.json().await?;

    result
      .data
      .into_iter()
      .next()
      .map(|d| d.embedding)
      .ok_or_else(|| EmbeddingError::ProviderError("No embedding in response".into()))
  }

  async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }

    let request = EmbeddingRequest {
      model: &self.model,
      input: EmbeddingInput::Batch(texts.to_vec()),
    };

    debug!("Embedding {} texts with OpenRouter", texts.len());

    let response = self
      .client
      .post(OPENROUTER_URL)
      .header("Authorization", format!("Bearer {}", self.api_key))
      .header("Content-Type", "application/json")
      .json(&request)
      .send()
      .await?;

    if !response.status().is_success() {
      let status = response.status();
      let body = response.text().await.unwrap_or_default();
      warn!("OpenRouter batch embedding failed: {} - {}", status, body);
      return Err(EmbeddingError::ProviderError(format!(
        "OpenRouter returned {}: {}",
        status, body
      )));
    }

    let result: EmbeddingResponse = response.json().await?;

    Ok(result.data.into_iter().map(|d| d.embedding).collect())
  }

  async fn is_available(&self) -> bool {
    // OpenRouter is a cloud service, just check we have an API key
    !self.api_key.is_empty()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_provider_new() {
    let provider = OpenRouterProvider::new("test-key");
    assert_eq!(provider.name(), "openrouter");
    assert_eq!(provider.model_id(), DEFAULT_MODEL);
    assert_eq!(provider.dimensions(), DEFAULT_DIMENSIONS);
  }

  #[test]
  fn test_provider_customization() {
    let provider = OpenRouterProvider::new("test-key").with_model("custom/model", 512);

    assert_eq!(provider.model_id(), "custom/model");
    assert_eq!(provider.dimensions(), 512);
  }

  #[test]
  fn test_from_env_missing() {
    // Clear any existing env var for this test
    unsafe {
      std::env::remove_var("OPENROUTER_API_KEY");
    }
    assert!(OpenRouterProvider::from_env().is_none());
  }

  #[tokio::test]
  async fn test_is_available_with_key() {
    let provider = OpenRouterProvider::new("test-key");
    assert!(provider.is_available().await);
  }

  #[tokio::test]
  async fn test_is_available_without_key() {
    let provider = OpenRouterProvider::new("");
    assert!(!provider.is_available().await);
  }

  // Integration tests require valid API key
  #[tokio::test]
  #[ignore = "requires OPENROUTER_API_KEY"]
  async fn test_embed_text() {
    let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");

    let embedding = provider.embed("Hello, world!").await.unwrap();
    assert_eq!(embedding.len(), provider.dimensions());
  }

  #[tokio::test]
  #[ignore = "requires OPENROUTER_API_KEY"]
  async fn test_embed_batch() {
    let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");

    let texts = vec!["Hello", "World", "Test"];
    let embeddings = provider.embed_batch(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 3);
    for embedding in &embeddings {
      assert_eq!(embedding.len(), provider.dimensions());
    }
  }
}
