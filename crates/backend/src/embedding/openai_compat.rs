use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, trace, warn};

use super::{
  EmbeddingError, EmbeddingMode, EmbeddingProvider,
  rate_limit::{FifoRateLimiter, RateLimitConfig, RateLimitToken},
};
use crate::config::EmbeddingConfig;

pub struct OpenAiCompatibleConfig {
  pub name: String,
  pub base_url: String,
  pub api_key: Option<String>,
  pub model: String,
  pub dimensions: usize,
  pub max_batch_size: usize,
  pub query_instruction: Option<String>,
  pub rate_limit: Option<RateLimitConfig>,
}

#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
  client: reqwest::Client,
  name: String,
  base_url: String,
  api_key: Option<String>,
  model: String,
  dimensions: usize,
  max_batch_size: usize,
  rate_limiter: Option<Arc<FifoRateLimiter>>,
  query_instruction: Option<String>,
}

impl OpenAiCompatibleProvider {
  pub fn new(config: OpenAiCompatibleConfig) -> Self {
    let has_instruction = config.query_instruction.as_ref().is_some_and(|s| !s.is_empty());
    let has_rate_limit = config.rate_limit.is_some();
    info!(
      name = %config.name,
      base_url = %config.base_url,
      model = %config.model,
      dimensions = config.dimensions,
      max_batch_size = config.max_batch_size,
      has_query_instruction = has_instruction,
      has_rate_limit,
      "OpenAI-compatible provider initialized"
    );

    let rate_limiter = config.rate_limit.map(|rl| Arc::new(FifoRateLimiter::new(rl)));

    Self {
      client: reqwest::Client::new(),
      name: config.name,
      base_url: config.base_url,
      api_key: config.api_key,
      model: config.model,
      dimensions: config.dimensions,
      max_batch_size: config.max_batch_size,
      rate_limiter,
      query_instruction: config.query_instruction,
    }
  }

  pub fn from_embedding_config_openrouter(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
    let api_key = config
      .openrouter_api_key
      .clone()
      .or_else(|| key_from_env("OPENROUTER_API_KEY"))
      .ok_or(EmbeddingError::NoApiKey)?;

    Ok(Self::new(OpenAiCompatibleConfig {
      name: "openrouter".to_string(),
      base_url: "https://openrouter.ai/api/v1".to_string(),
      api_key: Some(api_key),
      model: config.model.clone(),
      dimensions: config.dimensions,
      max_batch_size: config.max_batch_size.unwrap_or(512),
      query_instruction: config.query_instruction.clone(),
      rate_limit: Some(RateLimitConfig::for_openrouter()),
    }))
  }

  pub fn from_embedding_config_deepinfra(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
    let api_key = config
      .deepinfra_api_key
      .clone()
      .or_else(|| key_from_env("DEEPINFRA_API_KEY"))
      .ok_or(EmbeddingError::NoApiKey)?;

    Ok(Self::new(OpenAiCompatibleConfig {
      name: "deepinfra".to_string(),
      base_url: "https://api.deepinfra.com/v1/openai".to_string(),
      api_key: Some(api_key),
      model: config.model.clone(),
      dimensions: config.dimensions,
      max_batch_size: config.max_batch_size.unwrap_or(512),
      query_instruction: config.query_instruction.clone(),
      rate_limit: None,
    }))
  }

  #[cfg(not(feature = "llama-cpp"))]
  pub fn from_embedding_config_llamacpp(config: &EmbeddingConfig) -> Self {
    Self::new(OpenAiCompatibleConfig {
      name: "llamacpp".to_string(),
      base_url: "http://localhost:8080/v1".to_string(),
      api_key: None,
      model: config.model.clone(),
      dimensions: config.dimensions,
      max_batch_size: config.max_batch_size.unwrap_or(64),
      query_instruction: config.query_instruction.clone(),
      rate_limit: None,
    })
  }

  fn embeddings_url(&self) -> String {
    format!("{}/embeddings", self.base_url)
  }

  fn format_for_embedding(&self, text: &str, mode: EmbeddingMode) -> String {
    match mode {
      EmbeddingMode::Query => {
        if let Some(ref instruction) = self.query_instruction
          && !instruction.is_empty()
        {
          return format!("Instruct: {}\nQuery:{}", instruction, text);
        }
        text.to_string()
      }
      EmbeddingMode::Document => text.to_string(),
    }
  }

  async fn acquire_rate_limit_slot(&self) -> Result<Option<RateLimitToken>, EmbeddingError> {
    match &self.rate_limiter {
      Some(limiter) => limiter.acquire().await.map(Some),
      None => Ok(None),
    }
  }

  async fn refund_rate_limit_slot(&self, token: Option<RateLimitToken>) {
    if let (Some(limiter), Some(token)) = (&self.rate_limiter, token) {
      limiter.refund(token).await;
    }
  }

  #[tracing::instrument(level = "trace", skip(self, texts), fields(batch_size = texts.len(), provider = %self.name))]
  async fn embed_single_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }

    let token = self.acquire_rate_limit_slot().await?;

    let request = EmbeddingRequest {
      model: &self.model,
      input: EmbeddingInput::Batch(texts.to_vec()),
      encoding_format: "float",
    };

    trace!(
      batch_size = texts.len(),
      model = %self.model,
      provider = %self.name,
      "Sending batch embedding request"
    );
    let start = Instant::now();

    let mut req = self
      .client
      .post(self.embeddings_url())
      .header("Content-Type", "application/json")
      .json(&request);

    if let Some(ref key) = self.api_key {
      req = req.header("Authorization", format!("Bearer {}", key));
    }

    let response = match req.send().await {
      Ok(resp) => resp,
      Err(e) => {
        warn!(
          error = %e,
          batch_size = texts.len(),
          provider = %self.name,
          "Network error sending batch embedding request, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;

        if e.is_timeout() {
          return Err(EmbeddingError::Timeout);
        }
        return Err(EmbeddingError::Network(e.to_string()));
      }
    };

    let status = response.status();
    trace!(
      status = %status,
      elapsed_ms = start.elapsed().as_millis(),
      provider = %self.name,
      "Received response"
    );

    if !status.is_success() {
      let status_code = status.as_u16();
      let body = response.text().await.unwrap_or_default();
      let body_preview: String = body.trim_start().chars().take(500).collect();

      if status_code >= 500 {
        warn!(
          status = %status,
          batch_size = texts.len(),
          model = %self.model,
          provider = %self.name,
          body_preview = %body_preview,
          "Server error, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;
      } else if status_code == 401 || status_code == 403 {
        error!(
          status = %status,
          model = %self.model,
          provider = %self.name,
          body_preview = %body_preview,
          "Authentication failed"
        );
      } else if status_code == 429 {
        warn!(
          status = %status,
          batch_size = texts.len(),
          model = %self.model,
          provider = %self.name,
          body_preview = %body_preview,
          "Rate limit exceeded"
        );
      } else {
        warn!(
          status = %status,
          batch_size = texts.len(),
          model = %self.model,
          provider = %self.name,
          body_preview = %body_preview,
          "Batch embedding failed"
        );
      }

      return Err(EmbeddingError::ProviderError(format!(
        "{} returned {}: {}",
        self.name,
        status,
        body_preview.chars().take(300).collect::<String>()
      )));
    }

    let body_text = response.text().await.map_err(|e| {
      warn!(
        error = %e,
        batch_size = texts.len(),
        model = %self.model,
        provider = %self.name,
        elapsed_ms = start.elapsed().as_millis(),
        "Failed to read response body"
      );
      EmbeddingError::Network(format!("Failed to read response body: {}", e))
    })?;

    // Check for error response (OpenRouter can return errors with 200 OK)
    if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&body_text) {
      let provider = error_resp
        .error
        .metadata
        .as_ref()
        .and_then(|m| m.provider_name.as_deref())
        .unwrap_or("unknown");

      let body_preview: String = body_text.trim_start().chars().take(500).collect();
      warn!(
        provider = %provider,
        message = %error_resp.error.message,
        code = ?error_resp.error.code,
        batch_size = texts.len(),
        model = %self.model,
        name = %self.name,
        body_preview = %body_preview,
        "Error response from provider"
      );

      return Err(EmbeddingError::ProviderError(format!(
        "Provider '{}' failed: {}",
        provider, error_resp.error.message
      )));
    }

    let result: EmbeddingResponse = match serde_json::from_str(&body_text) {
      Ok(r) => r,
      Err(e) => {
        let printable_count = body_text.chars().filter(|c| !c.is_whitespace()).count();

        if printable_count == 0 && !body_text.is_empty() {
          debug!(
            batch_size = texts.len(),
            model = %self.model,
            provider = %self.name,
            elapsed_ms = start.elapsed().as_millis(),
            "Upstream provider timeout (received only keep-alive data), will retry"
          );
          return Err(EmbeddingError::UpstreamTimeout);
        }

        let trimmed = body_text.trim();
        let body_preview: String = trimmed.chars().take(500).collect();
        warn!(
          error = %e,
          batch_size = texts.len(),
          model = %self.model,
          provider = %self.name,
          elapsed_ms = start.elapsed().as_millis(),
          printable_count,
          body_preview = %body_preview,
          "Failed to parse response JSON"
        );
        return Err(EmbeddingError::ParseError(format!(
          "JSON parse error: {}. Response preview: {}",
          e,
          body_preview.chars().take(300).collect::<String>()
        )));
      }
    };

    trace!(
      embeddings_count = result.data.len(),
      elapsed_ms = start.elapsed().as_millis(),
      provider = %self.name,
      "Parsed response"
    );

    let expected = texts.len();
    let got = result.data.len();

    if got < expected {
      error!(
        expected,
        got,
        model = %self.model,
        provider = %self.name,
        "Returned fewer embeddings than inputs"
      );
      return Err(EmbeddingError::BatchSizeMismatch { expected, got });
    }

    if got > expected {
      warn!(
        expected,
        got,
        extra = got - expected,
        model = %self.model,
        provider = %self.name,
        "Returned more embeddings than inputs, using first {} (discarding {} extra)",
        expected,
        got - expected
      );
    }

    Ok(result.data.into_iter().take(expected).map(|d| d.embedding).collect())
  }

  async fn embed_batch_concurrent(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let num_batches = texts.len().div_ceil(self.max_batch_size);
    let start = Instant::now();

    if num_batches <= 1 {
      return self.embed_single_batch(texts).await;
    }

    debug!(
      batch_size = texts.len(),
      sub_batches = num_batches,
      max_batch_size = self.max_batch_size,
      model = %self.model,
      provider = %self.name,
      "Processing batch with concurrent sub-batches"
    );

    let futures: Vec<_> = texts
      .chunks(self.max_batch_size)
      .enumerate()
      .map(|(batch_idx, chunk)| {
        let provider = self.clone();
        let chunk_owned: Vec<String> = chunk.iter().map(|s| s.to_string()).collect();
        async move {
          let chunk_refs: Vec<&str> = chunk_owned.iter().map(|s| s.as_str()).collect();
          let embeddings = provider.embed_single_batch(&chunk_refs).await?;
          Ok::<_, EmbeddingError>((batch_idx, embeddings))
        }
      })
      .collect();

    #[allow(clippy::type_complexity)]
    let results: Vec<Result<(usize, Vec<Vec<f32>>), EmbeddingError>> = futures::future::join_all(futures).await;

    let mut indexed_results: Vec<(usize, Vec<Vec<f32>>)> = Vec::with_capacity(num_batches);
    for result in results {
      indexed_results.push(result?);
    }
    indexed_results.sort_by_key(|(idx, _)| *idx);

    let mut all_embeddings = Vec::with_capacity(texts.len());
    for (_, embeddings) in indexed_results {
      all_embeddings.extend(embeddings);
    }

    debug!(
      batch_size = texts.len(),
      sub_batches = num_batches,
      elapsed_ms = start.elapsed().as_millis(),
      provider = %self.name,
      "Batch embedding complete"
    );

    Ok(all_embeddings)
  }
}

fn key_from_env(var: &str) -> Option<String> {
  match std::env::var(var) {
    Ok(key) => {
      debug!("{} found in environment", var);
      Some(key)
    }
    Err(_) => {
      debug!("{} not set", var);
      None
    }
  }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
  model: &'a str,
  input: EmbeddingInput<'a>,
  encoding_format: &'a str,
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

#[derive(Debug, Deserialize)]
struct ErrorResponse {
  error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
  message: String,
  #[serde(default)]
  code: Option<i32>,
  #[serde(default)]
  metadata: Option<ErrorMetadata>,
}

#[derive(Debug, Deserialize)]
struct ErrorMetadata {
  #[serde(default)]
  provider_name: Option<String>,
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatibleProvider {
  fn name(&self) -> &str {
    &self.name
  }

  fn model_id(&self) -> &str {
    &self.model
  }

  fn dimensions(&self) -> usize {
    self.dimensions
  }

  async fn embed(&self, text: &str, mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
    let formatted = self.format_for_embedding(text, mode);

    let token = self.acquire_rate_limit_slot().await?;

    let request = EmbeddingRequest {
      model: &self.model,
      input: EmbeddingInput::Single(&formatted),
      encoding_format: "float",
    };

    trace!(text_len = text.len(), mode = ?mode, model = %self.model, provider = %self.name, "Sending single embedding request");
    let start = Instant::now();

    let mut req = self
      .client
      .post(self.embeddings_url())
      .header("Content-Type", "application/json")
      .json(&request);

    if let Some(ref key) = self.api_key {
      req = req.header("Authorization", format!("Bearer {}", key));
    }

    let response = match req.send().await {
      Ok(resp) => resp,
      Err(e) => {
        warn!(
          error = %e,
          text_len = text.len(),
          provider = %self.name,
          "Network error sending single embedding request, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;

        if e.is_timeout() {
          return Err(EmbeddingError::Timeout);
        }
        return Err(EmbeddingError::Network(e.to_string()));
      }
    };

    let status = response.status();
    trace!(
      status = %status,
      elapsed_ms = start.elapsed().as_millis(),
      provider = %self.name,
      "Received single embedding response"
    );

    if !status.is_success() {
      let status_code = status.as_u16();
      let body = response.text().await.unwrap_or_default();

      if status_code >= 500 {
        warn!(
          status = %status,
          text_len = text.len(),
          model = %self.model,
          provider = %self.name,
          "Server error, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;
      } else if status_code == 401 || status_code == 403 {
        error!(
          status = %status,
          model = %self.model,
          provider = %self.name,
          "Authentication failed"
        );
      } else if status_code == 429 {
        warn!(
          status = %status,
          text_len = text.len(),
          model = %self.model,
          provider = %self.name,
          "Rate limit exceeded"
        );
      } else {
        warn!(
          status = %status,
          text_len = text.len(),
          model = %self.model,
          provider = %self.name,
          "Single embedding failed"
        );
      }

      return Err(EmbeddingError::ProviderError(format!(
        "{} returned {}: {}",
        self.name, status, body
      )));
    }

    let result: EmbeddingResponse = response.json().await?;

    let embedding = result.data.into_iter().next().map(|d| d.embedding).ok_or_else(|| {
      error!(model = %self.model, provider = %self.name, "Returned empty response");
      EmbeddingError::ProviderError("No embedding in response".into())
    })?;

    trace!(
      dimensions = embedding.len(),
      elapsed_ms = start.elapsed().as_millis(),
      provider = %self.name,
      "Single embedding complete"
    );

    Ok(embedding)
  }

  async fn embed_batch(&self, texts: &[&str], mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      trace!("Empty batch, returning immediately");
      return Ok(Vec::new());
    }

    let formatted: Vec<String> = texts.iter().map(|t| self.format_for_embedding(t, mode)).collect();
    let formatted_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();

    debug!(batch_size = texts.len(), mode = ?mode, model = %self.model, provider = %self.name, "Embedding batch");
    self.embed_batch_concurrent(&formatted_refs).await
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::{Config, EmbeddingConfig};

  #[tokio::test]
  async fn test_embed_text_document() {
    let config = Config::default();
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_openrouter(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let embedding = provider.embed("Hello, world!", EmbeddingMode::Document).await.unwrap();
    assert_eq!(
      embedding.len(),
      provider.dimensions(),
      "embedding dimensions should match config"
    );
  }

  #[tokio::test]
  async fn test_embed_text_query() {
    let config = Config::default();
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_openrouter(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let embedding = provider.embed("Hello, world!", EmbeddingMode::Query).await.unwrap();
    assert_eq!(
      embedding.len(),
      provider.dimensions(),
      "embedding dimensions should match config"
    );
  }

  #[tokio::test]
  async fn test_embed_batch() {
    let config = Config::default();
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_openrouter(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let texts = vec!["Hello", "World", "Test"];
    let embeddings = provider.embed_batch(&texts, EmbeddingMode::Document).await.unwrap();

    assert_eq!(embeddings.len(), 3, "should return one embedding per input text");
    for embedding in &embeddings {
      assert_eq!(
        embedding.len(),
        provider.dimensions(),
        "each embedding should have correct dimensions"
      );
    }
  }

  #[tokio::test]
  async fn test_embed_batch_with_subbatching() {
    let config = Config {
      embedding: EmbeddingConfig {
        max_batch_size: Some(2),
        ..Default::default()
      },
      ..Default::default()
    };
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_openrouter(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let texts = vec!["One", "Two", "Three", "Four", "Five"];
    let embeddings = provider.embed_batch(&texts, EmbeddingMode::Document).await.unwrap();

    assert_eq!(embeddings.len(), 5, "sub-batching should still return all embeddings");
    for embedding in &embeddings {
      assert_eq!(
        embedding.len(),
        provider.dimensions(),
        "each embedding should have correct dimensions"
      );
    }
  }

  #[tokio::test]
  async fn test_embed_batch_empty() {
    let config = Config::default();
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_openrouter(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };
    let result = provider.embed_batch(&[], EmbeddingMode::Document).await;
    assert!(result.is_ok(), "empty batch should succeed");
    assert!(result.unwrap().is_empty(), "empty batch should return empty vec");
  }

  #[test]
  fn test_format_for_embedding_query_with_instruction() {
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
      name: "test".to_string(),
      base_url: "http://localhost".to_string(),
      api_key: None,
      model: "test".to_string(),
      dimensions: 4096,
      max_batch_size: 512,
      query_instruction: Some("Test instruction".to_string()),
      rate_limit: None,
    });
    let formatted = provider.format_for_embedding("test query", EmbeddingMode::Query);
    assert!(
      formatted.starts_with("Instruct:"),
      "Query should have instruction prefix"
    );
    assert!(
      formatted.contains("Test instruction"),
      "Query should contain custom instruction"
    );
    assert!(
      formatted.contains("Query:test query"),
      "Query should contain the query text"
    );
  }

  #[test]
  fn test_format_for_embedding_query_no_instruction() {
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
      name: "test".to_string(),
      base_url: "http://localhost".to_string(),
      api_key: None,
      model: "test".to_string(),
      dimensions: 4096,
      max_batch_size: 512,
      query_instruction: None,
      rate_limit: None,
    });
    let formatted = provider.format_for_embedding("test query", EmbeddingMode::Query);
    assert_eq!(formatted, "test query", "Query without instruction should be unchanged");
  }

  #[test]
  fn test_format_for_embedding_query_empty_instruction() {
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
      name: "test".to_string(),
      base_url: "http://localhost".to_string(),
      api_key: None,
      model: "test".to_string(),
      dimensions: 4096,
      max_batch_size: 512,
      query_instruction: Some(String::new()),
      rate_limit: None,
    });
    let formatted = provider.format_for_embedding("test query", EmbeddingMode::Query);
    assert_eq!(
      formatted, "test query",
      "Query with empty instruction should be unchanged"
    );
  }

  #[test]
  fn test_format_for_embedding_document() {
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
      name: "test".to_string(),
      base_url: "http://localhost".to_string(),
      api_key: None,
      model: "test".to_string(),
      dimensions: 4096,
      max_batch_size: 512,
      query_instruction: Some("Test instruction".to_string()),
      rate_limit: None,
    });
    let formatted = provider.format_for_embedding("test document", EmbeddingMode::Document);
    assert_eq!(
      formatted, "test document",
      "Document should be unchanged regardless of instruction"
    );
  }

  #[test]
  fn test_request_body_json_format_single() {
    // Verify the actual JSON structure matches OpenAI API format
    let request = EmbeddingRequest {
      model: "text-embedding-3-small",
      input: EmbeddingInput::Single("hello world"),
      encoding_format: "float",
    };

    let json = serde_json::to_value(&request).expect("should serialize");

    assert_eq!(json["model"], "text-embedding-3-small", "model field should be present");
    assert_eq!(
      json["input"], "hello world",
      "single input should be a string, not array"
    );
    assert_eq!(json["encoding_format"], "float", "encoding_format should be 'float'");
  }

  #[test]
  fn test_request_body_json_format_batch() {
    // Verify batch input serializes as array (OpenAI batch format)
    let request = EmbeddingRequest {
      model: "qwen/qwen3-embedding-8b",
      input: EmbeddingInput::Batch(vec!["text one", "text two", "text three"]),
      encoding_format: "float",
    };

    let json = serde_json::to_value(&request).expect("should serialize");

    assert!(json["input"].is_array(), "batch input should be an array");
    let input_arr = json["input"].as_array().unwrap();
    assert_eq!(input_arr.len(), 3, "batch should have 3 elements");
    assert_eq!(input_arr[0], "text one", "first element should match");
    assert_eq!(input_arr[2], "text three", "third element should match");
  }

  #[test]
  fn test_embeddings_url_construction() {
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
      name: "deepinfra".to_string(),
      base_url: "https://api.deepinfra.com/v1/openai".to_string(),
      api_key: Some("test-key".to_string()),
      model: "Qwen/Qwen3-Embedding-8B".to_string(),
      dimensions: 4096,
      max_batch_size: 512,
      query_instruction: None,
      rate_limit: None,
    });

    assert_eq!(
      provider.embeddings_url(),
      "https://api.deepinfra.com/v1/openai/embeddings",
      "embeddings URL should be base_url + /embeddings"
    );
  }

  #[test]
  fn test_embeddings_url_no_trailing_slash() {
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
      name: "openrouter".to_string(),
      base_url: "https://openrouter.ai/api/v1".to_string(),
      api_key: None,
      model: "test".to_string(),
      dimensions: 4096,
      max_batch_size: 512,
      query_instruction: None,
      rate_limit: None,
    });

    assert_eq!(
      provider.embeddings_url(),
      "https://openrouter.ai/api/v1/embeddings",
      "URL should work without trailing slash"
    );
  }

  #[test]
  fn test_response_deserialization() {
    // Verify we can deserialize a real OpenAI-format response
    let json = r#"{
      "object": "list",
      "data": [
        {
          "object": "embedding",
          "index": 0,
          "embedding": [0.1, 0.2, 0.3, 0.4]
        },
        {
          "object": "embedding",
          "index": 1,
          "embedding": [0.5, 0.6, 0.7, 0.8]
        }
      ],
      "model": "text-embedding-3-small",
      "usage": {"prompt_tokens": 5, "total_tokens": 5}
    }"#;

    let response: EmbeddingResponse = serde_json::from_str(json).expect("should deserialize OpenAI response");
    assert_eq!(response.data.len(), 2, "should have 2 embeddings");
    assert_eq!(
      response.data[0].embedding,
      vec![0.1, 0.2, 0.3, 0.4],
      "first embedding should match"
    );
    assert_eq!(
      response.data[1].embedding,
      vec![0.5, 0.6, 0.7, 0.8],
      "second embedding should match"
    );
  }

  #[test]
  fn test_error_response_deserialization() {
    // Verify we can detect OpenRouter-style error responses
    let json = r#"{
      "error": {
        "message": "No available provider for model",
        "code": 503,
        "metadata": {
          "provider_name": "deepinfra"
        }
      }
    }"#;

    let error: ErrorResponse = serde_json::from_str(json).expect("should deserialize error");
    assert_eq!(error.error.message, "No available provider for model");
    assert_eq!(error.error.code, Some(503));
    assert_eq!(
      error.error.metadata.as_ref().unwrap().provider_name.as_deref(),
      Some("deepinfra")
    );
  }

  #[tokio::test]
  async fn test_deepinfra_provider_construction() {
    // Test that DeepInfra provider builds the right URL
    let config = EmbeddingConfig {
      deepinfra_api_key: Some("test-key".to_string()),
      model: "Qwen/Qwen3-Embedding-8B".to_string(),
      dimensions: 4096,
      ..Default::default()
    };

    let provider = OpenAiCompatibleProvider::from_embedding_config_deepinfra(&config)
      .expect("should create provider with explicit key");

    assert_eq!(provider.name(), "deepinfra", "name should be deepinfra");
    assert_eq!(provider.model_id(), "Qwen/Qwen3-Embedding-8B", "model should match");
    assert_eq!(provider.dimensions(), 4096, "dimensions should match");
  }

  fn deepinfra_config() -> EmbeddingConfig {
    EmbeddingConfig {
      provider: crate::config::EmbeddingProvider::DeepInfra,
      model: "Qwen/Qwen3-Embedding-8B".to_string(),
      dimensions: 4096,
      query_instruction: Some(
        "Given a code search query, retrieve relevant code snippets and documentation that match the query".to_string(),
      ),
      ..Default::default()
    }
  }

  fn assert_normalized(embedding: &[f32], label: &str) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
      (norm - 1.0).abs() < 0.05,
      "{}: L2 norm should be ~1.0 but was {}",
      label,
      norm
    );
    assert!(
      embedding.iter().all(|x| x.is_finite()),
      "{}: all values should be finite floats",
      label
    );
  }

  #[tokio::test]
  async fn test_deepinfra_single_embedding() {
    let config = deepinfra_config();
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_deepinfra(&config) else {
      eprintln!("DEEPINFRA_API_KEY not set, skipping test");
      return;
    };

    let doc_embedding = provider
      .embed("fn main() { println!(\"hello world\"); }", EmbeddingMode::Document)
      .await
      .expect("DeepInfra document embedding should succeed");

    assert_eq!(
      doc_embedding.len(),
      4096,
      "document embedding should have 4096 dimensions"
    );
    assert_normalized(&doc_embedding, "document");

    let query_embedding = provider
      .embed("find the main function", EmbeddingMode::Query)
      .await
      .expect("DeepInfra query embedding should succeed");

    assert_eq!(
      query_embedding.len(),
      4096,
      "query embedding should have 4096 dimensions"
    );
    assert_normalized(&query_embedding, "query");
  }

  #[tokio::test]
  async fn test_deepinfra_batch_embedding() {
    let config = deepinfra_config();
    let Ok(provider) = OpenAiCompatibleProvider::from_embedding_config_deepinfra(&config) else {
      eprintln!("DEEPINFRA_API_KEY not set, skipping test");
      return;
    };

    let texts = vec![
      "struct Config { timeout: u64 }",
      "impl Display for Error { fn fmt(&self, f: &mut Formatter) -> Result { write!(f, \"{}\", self.0) } }",
      "async fn fetch_data(url: &str) -> Result<Response, Error> { client.get(url).send().await }",
      "use std::collections::HashMap;",
      "#[derive(Debug, Clone, Serialize, Deserialize)]",
    ];

    let embeddings = provider
      .embed_batch(&texts, EmbeddingMode::Document)
      .await
      .expect("DeepInfra batch embedding should succeed");

    assert_eq!(
      embeddings.len(),
      texts.len(),
      "should return exactly one embedding per input text"
    );

    for (i, emb) in embeddings.iter().enumerate() {
      assert_eq!(emb.len(), 4096, "embedding {} should have 4096 dimensions", i);
      assert_normalized(emb, &format!("batch[{}]", i));
    }
  }

  #[tokio::test]
  async fn test_deepinfra_vs_openrouter_both_valid() {
    let deepinfra_config = deepinfra_config();
    let Ok(deepinfra) = OpenAiCompatibleProvider::from_embedding_config_deepinfra(&deepinfra_config) else {
      eprintln!("DEEPINFRA_API_KEY not set, skipping test");
      return;
    };

    let openrouter_config = Config::default();
    let Ok(openrouter) = OpenAiCompatibleProvider::from_embedding_config_openrouter(&openrouter_config.embedding)
    else {
      eprintln!("OPENROUTER_API_KEY not set, skipping cross-provider test");
      return;
    };

    let text = "pub fn search(query: &str, limit: usize) -> Vec<Result> { vec![] }";

    let (di_result, or_result) = tokio::join!(
      deepinfra.embed(text, EmbeddingMode::Document),
      openrouter.embed(text, EmbeddingMode::Document),
    );

    let di_emb = di_result.expect("DeepInfra embedding should succeed");
    let or_emb = or_result.expect("OpenRouter embedding should succeed");

    assert_eq!(di_emb.len(), 4096, "DeepInfra should return 4096 dimensions");
    assert_eq!(or_emb.len(), 4096, "OpenRouter should return 4096 dimensions");

    assert_normalized(&di_emb, "deepinfra");
    assert_normalized(&or_emb, "openrouter");
  }
}
