use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, error, info, trace, warn};

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen3-embedding";
const DEFAULT_DIMENSIONS: usize = 4096;
const DEFAULT_CONTEXT_LENGTH: usize = 32768;
const DEFAULT_MAX_BATCH_SIZE: usize = 64;
/// Average tokens per chunk estimate (used for batch size calculation)
const AVG_CHUNK_TOKENS: usize = 512;
/// Default maximum concurrent sub-batch requests for Ollama.
/// Limited to avoid overwhelming local GPU memory/compute.
const DEFAULT_MAX_CONCURRENT: usize = 4;

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
    info!(
      url = DEFAULT_OLLAMA_URL,
      model = DEFAULT_MODEL,
      dimensions = DEFAULT_DIMENSIONS,
      context_length = DEFAULT_CONTEXT_LENGTH,
      max_batch_size = max_batch_size,
      "Ollama provider initialized"
    );
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
    debug!(url = %self.base_url, model = %self.model, "Checking Ollama health");
    let start = Instant::now();

    let available = match self
      .client
      .get(&self.base_url)
      .timeout(std::time::Duration::from_secs(5))
      .send()
      .await
    {
      Ok(response) => response.status().is_success(),
      Err(e) => {
        debug!(err = %e, elapsed_ms = start.elapsed().as_millis(), "Ollama health check failed");
        false
      }
    };

    if !available {
      debug!(elapsed_ms = start.elapsed().as_millis(), "Ollama not available");
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

    debug!(
      available = available,
      model_count = models.len(),
      configured_model = %self.model,
      configured_model_available = configured_model_available,
      elapsed_ms = start.elapsed().as_millis(),
      "Ollama health check complete"
    );

    OllamaHealthStatus {
      available,
      models,
      configured_model: self.model.clone(),
      configured_model_available,
    }
  }

  /// Query Ollama for actual model context length
  pub async fn get_model_context_length(&self) -> Option<usize> {
    trace!(model = %self.model, "Querying model context length");
    let request = ShowRequest { name: &self.model };
    let start = Instant::now();

    match self.client.post(self.show_url()).json(&request).send().await {
      Ok(response) if response.status().is_success() => {
        if let Ok(show) = response.json::<ShowResponse>().await {
          let context_length = show
            .model_info
            .and_then(|info| info.context_length)
            .map(|len| len as usize);
          trace!(
            model = %self.model,
            context_length = ?context_length,
            elapsed_ms = start.elapsed().as_millis(),
            "Model context length query complete"
          );
          context_length
        } else {
          trace!(model = %self.model, "Failed to parse model info response");
          None
        }
      }
      Ok(response) => {
        trace!(model = %self.model, status = %response.status(), "Model info request failed");
        None
      }
      Err(e) => {
        trace!(model = %self.model, err = %e, "Model info request error");
        None
      }
    }
  }

  /// Native batch embedding using /api/embed endpoint.
  ///
  /// Processes sub-batches concurrently using semaphore-limited parallelism.
  /// This significantly improves throughput when multiple sub-batches are needed,
  /// while respecting the server's capacity limits.
  async fn embed_batch_native(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    let num_batches = texts.len().div_ceil(self.max_batch_size);
    let start = Instant::now();

    // For single batch, no concurrency overhead needed
    if num_batches <= 1 {
      return self.embed_single_batch(texts).await;
    }

    debug!(
      batch_size = texts.len(),
      sub_batches = num_batches,
      max_batch_size = self.max_batch_size,
      model = %self.model,
      "Processing batch with concurrent sub-batches"
    );

    // Limit concurrent requests to avoid overwhelming local GPU
    let semaphore = Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT));

    // Create indexed sub-batch tasks
    let futures: Vec<_> = texts
      .chunks(self.max_batch_size)
      .enumerate()
      .map(|(batch_idx, chunk)| {
        let permit = semaphore.clone();
        let provider = self.clone();
        let chunk_owned: Vec<String> = chunk.iter().map(|s| s.to_string()).collect();
        async move {
          let _permit = match permit.acquire().await {
            Ok(permit) => permit,
            Err(_) => return Err(EmbeddingError::ProviderError("semaphore closed".to_string())),
          };
          let chunk_refs: Vec<&str> = chunk_owned.iter().map(|s| s.as_str()).collect();
          let embeddings = provider.embed_single_batch(&chunk_refs).await?;
          Ok((batch_idx, embeddings))
        }
      })
      .collect();

    // Wait for all batches concurrently
    #[allow(clippy::type_complexity)]
    let results: Vec<Result<(usize, Vec<Vec<f32>>), EmbeddingError>> = futures::future::join_all(futures).await;

    // Collect and sort results by batch index to maintain order
    let mut indexed_results: Vec<(usize, Vec<Vec<f32>>)> = Vec::with_capacity(num_batches);
    for result in results {
      indexed_results.push(result?);
    }
    indexed_results.sort_by_key(|(idx, _)| *idx);

    // Flatten into final result
    let mut all_embeddings = Vec::with_capacity(texts.len());
    for (_, embeddings) in indexed_results {
      all_embeddings.extend(embeddings);
    }

    debug!(
      batch_size = texts.len(),
      sub_batches = num_batches,
      elapsed_ms = start.elapsed().as_millis(),
      "Batch embedding complete"
    );

    Ok(all_embeddings)
  }

  /// Embed a single batch of texts (used internally by embed_batch_native)
  async fn embed_single_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let request = BatchEmbeddingRequest {
      model: &self.model,
      input: texts.to_vec(),
    };

    trace!(batch_size = texts.len(), model = %self.model, "Sending batch embedding request");
    let start = Instant::now();

    let response = self.client.post(self.embed_url()).json(&request).send().await?;

    trace!(
      status = %response.status(),
      elapsed_ms = start.elapsed().as_millis(),
      "Received batch embedding response"
    );

    if !response.status().is_success() {
      let status = response.status();
      let body = response.text().await.unwrap_or_default();
      warn!(
        status = %status,
        batch_size = texts.len(),
        model = %self.model,
        "Ollama batch embedding failed"
      );
      return Err(EmbeddingError::ProviderError(format!(
        "Ollama returned {}: {}",
        status, body
      )));
    }

    let result: BatchEmbeddingResponse = response.json().await?;
    trace!(
      embeddings_count = result.embeddings.len(),
      elapsed_ms = start.elapsed().as_millis(),
      "Parsed batch embedding response"
    );

    if result.embeddings.len() != texts.len() {
      error!(
        expected = texts.len(),
        got = result.embeddings.len(),
        model = %self.model,
        "Batch size mismatch in embedding response"
      );
      return Err(EmbeddingError::ProviderError(format!(
        "Batch size mismatch: got {} embeddings for {} inputs",
        result.embeddings.len(),
        texts.len()
      )));
    }

    for (i, embedding) in result.embeddings.iter().enumerate() {
      if embedding.len() != self.dimensions {
        warn!(
          index = i,
          expected = self.dimensions,
          got = embedding.len(),
          model = %self.model,
          "Unexpected embedding dimensions"
        );
      }
    }

    Ok(result.embeddings)
  }

  /// Parallel batch embedding (fallback) - uses semaphore to limit concurrency
  async fn embed_batch_parallel(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    debug!(
      batch_size = texts.len(),
      max_concurrent = DEFAULT_MAX_CONCURRENT,
      "Using parallel fallback for batch embedding"
    );
    let start = Instant::now();

    let semaphore = Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT));

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
    let collected: Result<Vec<Vec<f32>>, EmbeddingError> = results.into_iter().collect();

    match &collected {
      Ok(_) => {
        debug!(
          batch_size = texts.len(),
          elapsed_ms = start.elapsed().as_millis(),
          "Parallel fallback embedding complete"
        );
      }
      Err(e) => {
        warn!(
          batch_size = texts.len(),
          elapsed_ms = start.elapsed().as_millis(),
          err = %e,
          "Parallel fallback embedding failed"
        );
      }
    }

    collected
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

    trace!(text_len = text.len(), model = %self.model, "Sending single embedding request");
    let start = Instant::now();

    let response = self.client.post(self.embeddings_url()).json(&request).send().await?;

    trace!(
      status = %response.status(),
      elapsed_ms = start.elapsed().as_millis(),
      "Received single embedding response"
    );

    if !response.status().is_success() {
      let status = response.status();
      let body = response.text().await.unwrap_or_default();
      warn!(
        status = %status,
        text_len = text.len(),
        model = %self.model,
        "Ollama single embedding failed"
      );
      return Err(EmbeddingError::ProviderError(format!(
        "Ollama returned {}: {}",
        status, body
      )));
    }

    let result: EmbeddingResponse = response.json().await?;

    if result.embedding.len() != self.dimensions {
      warn!(
        expected = self.dimensions,
        got = result.embedding.len(),
        model = %self.model,
        "Unexpected embedding dimensions"
      );
    }

    trace!(
      dimensions = result.embedding.len(),
      elapsed_ms = start.elapsed().as_millis(),
      "Single embedding complete"
    );

    Ok(result.embedding)
  }

  async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      trace!("Empty batch, returning immediately");
      return Ok(Vec::new());
    }

    debug!(batch_size = texts.len(), model = %self.model, "Embedding batch");

    // Try native batch API first, fall back to parallel on error
    match self.embed_batch_native(texts).await {
      Ok(embeddings) => Ok(embeddings),
      Err(e) => {
        warn!(
          batch_size = texts.len(),
          err = %e,
          "Native batch embedding failed, falling back to parallel"
        );
        self.embed_batch_parallel(texts).await
      }
    }
  }

  async fn is_available(&self) -> bool {
    // Try a simple health check
    debug!(url = %self.base_url, "Checking Ollama availability");
    match self.client.get(&self.base_url).send().await {
      Ok(response) => {
        let available = response.status().is_success();
        debug!(available = available, "Ollama availability check complete");
        available
      }
      Err(e) => {
        debug!(err = %e, "Ollama availability check failed");
        false
      }
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
    let provider = OllamaProvider::new().with_context_length(32768).with_max_batch_size(10); // Override auto-calculation
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
    let texts: Vec<&str> = (0..7)
      .map(|i| match i {
        0 => "Text zero",
        1 => "Text one",
        2 => "Text two",
        3 => "Text three",
        4 => "Text four",
        5 => "Text five",
        6 => "Text six",
        _ => unreachable!(),
      })
      .collect();

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
