use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen3-embedding";
const DEFAULT_DIMENSIONS: usize = 4096;
const DEFAULT_CONTEXT_LENGTH: usize = 32768;
const DEFAULT_MAX_BATCH_SIZE: usize = 64;
/// Average tokens per chunk estimate (used for batch size calculation)
const AVG_CHUNK_TOKENS: usize = 512;

/// Calculate max batch size based on context length
/// Formula: clamp(context_length / avg_chunk_tokens, 1, 64)
fn calculate_max_batch_size(context_length: usize) -> usize {
  let calculated = context_length / AVG_CHUNK_TOKENS;
  calculated.clamp(1, DEFAULT_MAX_BATCH_SIZE)
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
  client: reqwest::Client,
  base_url: String,
  model: String,
  dimensions: usize,
  /// Context length for batch size calculation
  context_length: usize,
  /// Maximum batch size (auto-calculated from context_length if not set)
  max_batch_size: usize,
}

impl Default for OllamaProvider {
  fn default() -> Self {
    Self::new()
  }
}

impl OllamaProvider {
  pub fn new() -> Self {
    let max_batch_size = calculate_max_batch_size(DEFAULT_CONTEXT_LENGTH);
    Self {
      client: reqwest::Client::new(),
      base_url: DEFAULT_OLLAMA_URL.to_string(),
      model: DEFAULT_MODEL.to_string(),
      dimensions: DEFAULT_DIMENSIONS,
      context_length: DEFAULT_CONTEXT_LENGTH,
      max_batch_size,
    }
  }

  pub fn with_url(mut self, url: impl Into<String>) -> Self {
    self.base_url = url.into();
    self
  }

  pub fn with_model(mut self, model: impl Into<String>, dimensions: usize) -> Self {
    self.model = model.into();
    self.dimensions = dimensions;
    self
  }

  /// Set context length for batch size calculation
  pub fn with_context_length(mut self, context_length: usize) -> Self {
    self.context_length = context_length;
    self.max_batch_size = calculate_max_batch_size(context_length);
    self
  }

  /// Set explicit max batch size (overrides auto-calculation)
  pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
    self.max_batch_size = max_batch_size;
    self
  }

  /// Get the current max batch size
  pub fn max_batch_size(&self) -> usize {
    self.max_batch_size
  }

  /// Get the configured context length
  pub fn context_length(&self) -> usize {
    self.context_length
  }

  /// Single embedding endpoint (legacy)
  fn embeddings_url(&self) -> String {
    format!("{}/api/embeddings", self.base_url)
  }

  /// Batch embedding endpoint (new)
  fn embed_url(&self) -> String {
    format!("{}/api/embed", self.base_url)
  }

  fn tags_url(&self) -> String {
    format!("{}/api/tags", self.base_url)
  }

  fn show_url(&self) -> String {
    format!("{}/api/show", self.base_url)
  }

  /// Check if Ollama is available and return the list of models
  pub async fn check_health(&self) -> OllamaHealthStatus {
    let available = match self
      .client
      .get(&self.base_url)
      .timeout(std::time::Duration::from_secs(5))
      .send()
      .await
    {
      Ok(response) => response.status().is_success(),
      Err(_) => false,
    };

    if !available {
      return OllamaHealthStatus {
        available: false,
        models: vec![],
        configured_model: self.model.clone(),
        configured_model_available: false,
      };
    }

    // Get list of available models
    let models: Vec<String> = match self.client.get(self.tags_url()).send().await {
      Ok(response) if response.status().is_success() => {
        #[derive(Deserialize)]
        struct TagsResponse {
          models: Vec<ModelInfo>,
        }
        #[derive(Deserialize)]
        struct ModelInfo {
          name: String,
        }
        response
          .json::<TagsResponse>()
          .await
          .map(|t| t.models.into_iter().map(|m| m.name).collect())
          .unwrap_or_default()
      }
      _ => vec![],
    };

    let configured_model_available = models
      .iter()
      .any(|m| m.starts_with(&self.model) || self.model.starts_with(m));

    OllamaHealthStatus {
      available,
      models,
      configured_model: self.model.clone(),
      configured_model_available,
    }
  }

  /// Query Ollama for actual model context length
  pub async fn get_model_context_length(&self) -> Option<usize> {
    let request = ShowRequest { name: &self.model };

    match self.client.post(self.show_url()).json(&request).send().await {
      Ok(response) if response.status().is_success() => {
        if let Ok(show) = response.json::<ShowResponse>().await {
          show
            .model_info
            .and_then(|info| info.context_length)
            .map(|len| len as usize)
        } else {
          None
        }
      }
      _ => None,
    }
  }

  /// Native batch embedding using /api/embed endpoint (single HTTP request)
  async fn embed_batch_native(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    // Split into sub-batches based on max_batch_size
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(self.max_batch_size) {
      let request = BatchEmbeddingRequest {
        model: &self.model,
        input: chunk.to_vec(),
      };

      debug!(
        "Batch embedding {} texts with Ollama (batch size: {})",
        chunk.len(),
        self.max_batch_size
      );

      let response = self.client.post(self.embed_url()).json(&request).send().await?;

      if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        warn!("Ollama batch embedding failed: {} - {}", status, body);
        return Err(EmbeddingError::ProviderError(format!(
          "Ollama returned {}: {}",
          status, body
        )));
      }

      let result: BatchEmbeddingResponse = response.json().await?;

      if result.embeddings.len() != chunk.len() {
        warn!(
          "Batch size mismatch: got {} embeddings for {} inputs",
          result.embeddings.len(),
          chunk.len()
        );
        return Err(EmbeddingError::ProviderError(format!(
          "Batch size mismatch: got {} embeddings for {} inputs",
          result.embeddings.len(),
          chunk.len()
        )));
      }

      for embedding in result.embeddings {
        if embedding.len() != self.dimensions {
          warn!(
            "Unexpected embedding dimensions: got {}, expected {}",
            embedding.len(),
            self.dimensions
          );
        }
        all_embeddings.push(embedding);
      }
    }

    info!(
      "Batch embedded {} texts in {} sub-batches",
      texts.len(),
      texts.len().div_ceil(self.max_batch_size)
    );

    Ok(all_embeddings)
  }

  /// Parallel batch embedding (fallback) - uses semaphore to limit concurrency
  async fn embed_batch_parallel(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    debug!("Using parallel fallback for {} texts", texts.len());

    let semaphore = Arc::new(Semaphore::new(4)); // Max 4 concurrent requests

    let futures: Vec<_> = texts
      .iter()
      .map(|text| {
        let permit = semaphore.clone();
        let text = text.to_string();
        let provider = self.clone();
        async move {
          let _permit = match permit.acquire().await {
            Ok(permit) => permit,
            Err(_) => return Err(EmbeddingError::ProviderError("semaphore closed".to_string())),
          };
          provider.embed(&text).await
        }
      })
      .collect();

    let results: Vec<Result<Vec<f32>, EmbeddingError>> = futures::future::join_all(futures).await;

    // Collect results, propagating first error
    results.into_iter().collect()
  }
}

/// Health status for Ollama
#[derive(Debug, Clone, serde::Serialize)]
pub struct OllamaHealthStatus {
  pub available: bool,
  pub models: Vec<String>,
  pub configured_model: String,
  pub configured_model_available: bool,
}

/// Request for single embedding (legacy /api/embeddings endpoint)
#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
  model: &'a str,
  prompt: &'a str,
}

/// Response from single embedding (legacy /api/embeddings endpoint)
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
  embedding: Vec<f32>,
}

/// Request for batch embedding (/api/embed endpoint)
#[derive(Debug, Serialize)]
struct BatchEmbeddingRequest<'a> {
  model: &'a str,
  input: Vec<&'a str>,
}

/// Response from batch embedding (/api/embed endpoint)
#[derive(Debug, Deserialize)]
struct BatchEmbeddingResponse {
  embeddings: Vec<Vec<f32>>,
}

/// Request for model info (/api/show endpoint)
#[derive(Debug, Serialize)]
struct ShowRequest<'a> {
  name: &'a str,
}

/// Response from model info (partial - only fields we need)
#[derive(Debug, Deserialize)]
struct ShowResponse {
  #[serde(default)]
  model_info: Option<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
  #[serde(rename = "general.context_length", default)]
  context_length: Option<u64>,
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
  fn name(&self) -> &str {
    "ollama"
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
      prompt: text,
    };

    debug!("Embedding text with Ollama: {} chars", text.len());

    let response = self.client.post(self.embeddings_url()).json(&request).send().await?;

    if !response.status().is_success() {
      let status = response.status();
      let body = response.text().await.unwrap_or_default();
      warn!("Ollama embedding failed: {} - {}", status, body);
      return Err(EmbeddingError::ProviderError(format!(
        "Ollama returned {}: {}",
        status, body
      )));
    }

    let result: EmbeddingResponse = response.json().await?;

    if result.embedding.len() != self.dimensions {
      warn!(
        "Unexpected embedding dimensions: got {}, expected {}",
        result.embedding.len(),
        self.dimensions
      );
    }

    Ok(result.embedding)
  }

  async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }

    // Try native batch API first, fall back to parallel on error
    match self.embed_batch_native(texts).await {
      Ok(embeddings) => Ok(embeddings),
      Err(e) => {
        warn!("Native batch embedding failed ({}), falling back to parallel", e);
        self.embed_batch_parallel(texts).await
      }
    }
  }

  async fn is_available(&self) -> bool {
    // Try a simple health check
    match self.client.get(&self.base_url).send().await {
      Ok(response) => response.status().is_success(),
      Err(_) => false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_provider_defaults() {
    let provider = OllamaProvider::new();
    assert_eq!(provider.name(), "ollama");
    assert_eq!(provider.model_id(), DEFAULT_MODEL);
    assert_eq!(provider.dimensions(), DEFAULT_DIMENSIONS);
  }

  #[test]
  fn test_provider_customization() {
    let provider = OllamaProvider::new()
      .with_url("http://custom:8080")
      .with_model("custom-model", 1024);

    assert_eq!(provider.base_url, "http://custom:8080");
    assert_eq!(provider.model_id(), "custom-model");
    assert_eq!(provider.dimensions(), 1024);
  }

  #[test]
  fn test_embeddings_url() {
    let provider = OllamaProvider::new();
    assert_eq!(provider.embeddings_url(), "http://localhost:11434/api/embeddings");
  }

  // Integration tests require a running Ollama instance
  #[tokio::test]
  async fn test_embed_text() {
    let provider = OllamaProvider::new();

    if !provider.is_available().await {
      eprintln!("Ollama not available, skipping test");
      return;
    }

    let embedding = provider.embed("Hello, world!").await.unwrap();
    assert_eq!(embedding.len(), provider.dimensions());
  }

  #[tokio::test]
  async fn test_embed_batch() {
    let provider = OllamaProvider::new();

    if !provider.is_available().await {
      eprintln!("Ollama not available, skipping test");
      return;
    }

    let texts = vec!["Hello", "World", "Test"];
    let embeddings = provider.embed_batch(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 3);
    for embedding in &embeddings {
      assert_eq!(embedding.len(), provider.dimensions());
    }
  }

  #[tokio::test]
  async fn test_embed_batch_empty_input() {
    let provider = OllamaProvider::new();
    // Empty input should return empty vec, not error (no network call needed)
    let result = provider.embed_batch(&[]).await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
  }

  #[test]
  fn test_max_batch_size_calculation() {
    // 32K context / 512 tokens per chunk = 64 (capped at max)
    assert_eq!(calculate_max_batch_size(32768), 64);

    // 16K context / 512 tokens per chunk = 32
    assert_eq!(calculate_max_batch_size(16384), 32);

    // 8K context / 512 tokens per chunk = 16
    assert_eq!(calculate_max_batch_size(8192), 16);

    // 4K context / 512 tokens per chunk = 8
    assert_eq!(calculate_max_batch_size(4096), 8);

    // Very small context should still return at least 1
    assert_eq!(calculate_max_batch_size(256), 1);
  }

  #[test]
  fn test_context_length_configuration() {
    let provider = OllamaProvider::new().with_context_length(8192);
    assert_eq!(provider.context_length(), 8192);
    assert_eq!(provider.max_batch_size(), 16); // 8192 / 512 = 16
  }

  #[test]
  fn test_explicit_max_batch_size() {
    let provider = OllamaProvider::new()
      .with_context_length(32768)
      .with_max_batch_size(10); // Override auto-calculation
    assert_eq!(provider.max_batch_size(), 10);
  }

  #[test]
  fn test_embed_url() {
    let provider = OllamaProvider::new();
    assert_eq!(provider.embed_url(), "http://localhost:11434/api/embed");
  }

  #[tokio::test]
  async fn test_embed_batch_native_success() {
    // Integration test: verify native batch API works when Ollama is available
    let provider = OllamaProvider::new();

    if !provider.is_available().await {
      eprintln!("Ollama not available, skipping test");
      return;
    }

    // Test with 5 texts as specified in acceptance criteria
    let texts = vec![
      "First sentence to embed",
      "Second sentence to embed",
      "Third sentence to embed",
      "Fourth sentence to embed",
      "Fifth sentence to embed",
    ];

    let embeddings = provider.embed_batch_native(&texts).await.unwrap();

    // Verify we got 5 embeddings
    assert_eq!(embeddings.len(), 5);

    // Verify each has correct dimensions
    for embedding in &embeddings {
      assert_eq!(embedding.len(), provider.dimensions());
    }
  }

  #[tokio::test]
  async fn test_embed_batch_with_fallback() {
    // Integration test: verify fallback works
    // This test uses a non-existent URL to force fallback
    let provider = OllamaProvider::new().with_url("http://localhost:99999");

    // Should fail (no server at that port)
    let result = provider.embed_batch(&["test"]).await;
    assert!(result.is_err());
  }

  #[tokio::test]
  async fn test_embed_batch_sub_batching() {
    // Integration test: verify large batches are split correctly
    let provider = OllamaProvider::new().with_max_batch_size(3); // Small batch size for testing

    if !provider.is_available().await {
      eprintln!("Ollama not available, skipping test");
      return;
    }

    // 7 texts should be split into 3 sub-batches (3 + 3 + 1)
    let texts: Vec<&str> = (0..7).map(|i| match i {
      0 => "Text zero",
      1 => "Text one",
      2 => "Text two",
      3 => "Text three",
      4 => "Text four",
      5 => "Text five",
      6 => "Text six",
      _ => unreachable!(),
    }).collect();

    let embeddings = provider.embed_batch(&texts).await.unwrap();
    assert_eq!(embeddings.len(), 7);
  }

  #[test]
  fn test_batch_split_calculation() {
    // Verify batch splitting logic
    // With max_batch_size=16 and context_length=8192:
    // 100 chunks should be split into ceil(100/16) = 7 sub-batches
    let provider = OllamaProvider::new().with_context_length(8192);
    assert_eq!(provider.max_batch_size(), 16);

    let num_batches = 100_usize.div_ceil(provider.max_batch_size());
    assert_eq!(num_batches, 7);
  }
}
