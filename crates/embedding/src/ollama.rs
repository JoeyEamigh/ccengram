use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen3-embedding";
const DEFAULT_DIMENSIONS: usize = 4096;

#[derive(Debug, Clone)]
pub struct OllamaProvider {
  client: reqwest::Client,
  base_url: String,
  model: String,
  dimensions: usize,
}

impl Default for OllamaProvider {
  fn default() -> Self {
    Self::new()
  }
}

impl OllamaProvider {
  pub fn new() -> Self {
    Self {
      client: reqwest::Client::new(),
      base_url: DEFAULT_OLLAMA_URL.to_string(),
      model: DEFAULT_MODEL.to_string(),
      dimensions: DEFAULT_DIMENSIONS,
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

  fn embeddings_url(&self) -> String {
    format!("{}/api/embeddings", self.base_url)
  }

  fn tags_url(&self) -> String {
    format!("{}/api/tags", self.base_url)
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
}

/// Health status for Ollama
#[derive(Debug, Clone, serde::Serialize)]
pub struct OllamaHealthStatus {
  pub available: bool,
  pub models: Vec<String>,
  pub configured_model: String,
  pub configured_model_available: bool,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
  model: &'a str,
  prompt: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
  embedding: Vec<f32>,
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
    // Ollama doesn't have native batch support, so we parallelize with bounded concurrency
    use std::sync::Arc;
    use tokio::sync::Semaphore;

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
}
