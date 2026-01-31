use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, trace, warn};

use super::{
  EmbeddingError, EmbeddingMode, EmbeddingProvider,
  rate_limit::{FifoRateLimiter, RateLimitConfig, RateLimitToken},
};
use crate::config::EmbeddingConfig;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/embeddings";

#[derive(Clone)]
pub struct OpenRouterProvider {
  client: reqwest::Client,
  api_key: String,
  model: String,
  dimensions: usize,
  /// Maximum texts per batch request
  max_batch_size: usize,
  /// FIFO rate limiter for HTTP requests (shared across clones)
  rate_limiter: Arc<FifoRateLimiter>,
  /// Optional task instruction to prepend to queries.
  /// When Some and non-empty, queries are formatted as `Instruct: {instruction}\nQuery:{query}`.
  /// When None or empty, queries are embedded as-is.
  query_instruction: Option<String>,
}

impl OpenRouterProvider {
  pub fn new(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
    let api_key = if let Some(key) = &config.openrouter_api_key {
      key.clone()
    } else if let Some(key) = Self::key_from_env() {
      key
    } else {
      return Err(EmbeddingError::NoApiKey);
    };

    let model = config.model.clone();
    let dimensions = config.dimensions;
    // OpenRouter embedding API can handle large batches - unlike Ollama, the entire
    // request doesn't need to fit in context, just each individual text.
    // Higher batch sizes = fewer HTTP requests = better throughput with rate limiting.
    let max_batch_size = config.max_batch_size.unwrap_or(512);
    let query_instruction = config.query_instruction.clone();

    let has_instruction = query_instruction.as_ref().is_some_and(|s| !s.is_empty());
    info!(
      model,
      dimensions,
      max_batch_size,
      has_query_instruction = has_instruction,
      "OpenRouter provider initialized"
    );

    Ok(Self {
      client: reqwest::Client::new(),
      api_key,
      model,
      dimensions,
      max_batch_size,
      rate_limiter: Arc::new(FifoRateLimiter::new(RateLimitConfig::for_openrouter())),
      query_instruction,
    })
  }

  fn key_from_env() -> Option<String> {
    match std::env::var("OPENROUTER_API_KEY") {
      Ok(key) => {
        debug!("OPENROUTER_API_KEY found in environment");
        Some(key)
      }
      Err(_) => {
        debug!("OPENROUTER_API_KEY not set");
        None
      }
    }
  }

  /// Format text for embedding based on mode.
  ///
  /// For queries (if query_instruction is set): `Instruct: {instruction}\nQuery:{text}`
  /// For documents or if no instruction: text as-is
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

  /// Acquire a rate limit slot with FIFO ordering, waiting if necessary.
  ///
  /// Returns a `RateLimitToken` that can be used to refund the slot if the
  /// request fails in a way that didn't consume API rate limit capacity.
  async fn acquire_rate_limit_slot(&self) -> Result<RateLimitToken, EmbeddingError> {
    self.rate_limiter.acquire().await
  }

  /// Refund a rate limit slot when a request fails without consuming API capacity.
  ///
  /// Call this for:
  /// - Network errors (request never reached OpenRouter)
  /// - Timeouts (request may not have been processed)
  /// - 5xx server errors (server failed before rate limiting)
  ///
  /// Do NOT call for:
  /// - 429 errors (OpenRouter counted the request against rate limit)
  /// - 4xx errors (request was processed)
  async fn refund_rate_limit_slot(&self, token: RateLimitToken) {
    self.rate_limiter.refund(token).await;
  }

  /// Embed a single batch of texts (internal helper)
  /// Rate limiting is applied here at the HTTP request level.
  ///
  /// Automatically refunds rate limit slots on network errors and 5xx server errors,
  /// as these failures didn't consume OpenRouter's rate limit capacity.
  async fn embed_single_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }

    // Acquire rate limit slot before making HTTP request
    let token = self.acquire_rate_limit_slot().await?;

    let request = EmbeddingRequest {
      model: &self.model,
      input: EmbeddingInput::Batch(texts.to_vec()),
    };

    trace!(
      batch_size = texts.len(),
      model = %self.model,
      "Sending batch embedding request to OpenRouter"
    );
    let start = Instant::now();

    let response = match self
      .client
      .post(OPENROUTER_URL)
      .header("Authorization", format!("Bearer {}", self.api_key))
      .header("Content-Type", "application/json")
      .json(&request)
      .send()
      .await
    {
      Ok(resp) => resp,
      Err(e) => {
        // Network error - request never reached OpenRouter, refund the slot
        warn!(
          error = %e,
          batch_size = texts.len(),
          "Network error sending batch embedding request, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;

        // Check if it was a timeout
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
      "Received response from OpenRouter"
    );

    if !status.is_success() {
      let status_code = status.as_u16();
      let body = response.text().await.unwrap_or_default();
      let body_preview: String = body.trim_start().chars().take(500).collect();

      // Refund for 5xx server errors - these didn't hit OpenRouter's rate limiter
      if status_code >= 500 {
        warn!(
          status = %status,
          batch_size = texts.len(),
          model = %self.model,
          body_preview = %body_preview,
          "OpenRouter server error, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;
      } else if status_code == 401 || status_code == 403 {
        error!(
          status = %status,
          model = %self.model,
          body_preview = %body_preview,
          "OpenRouter authentication failed"
        );
        // Don't refund - auth errors still count against rate limit
      } else if status_code == 429 {
        warn!(
          status = %status,
          batch_size = texts.len(),
          model = %self.model,
          body_preview = %body_preview,
          "OpenRouter rate limit exceeded"
        );
        // Don't refund - 429 means OpenRouter counted this request
      } else {
        warn!(
          status = %status,
          batch_size = texts.len(),
          model = %self.model,
          body_preview = %body_preview,
          "OpenRouter batch embedding failed"
        );
        // Don't refund 4xx errors - request was processed
      }

      return Err(EmbeddingError::ProviderError(format!(
        "OpenRouter returned {}: {}",
        status,
        body_preview.chars().take(300).collect::<String>()
      )));
    }

    // Read body as text first so we can log it on parse failure
    let body_text = response.text().await.map_err(|e| {
      warn!(
        error = %e,
        batch_size = texts.len(),
        model = %self.model,
        elapsed_ms = start.elapsed().as_millis(),
        "Failed to read OpenRouter response body"
      );
      EmbeddingError::Network(format!("Failed to read response body: {}", e))
    })?;

    // First, check if OpenRouter returned an error response (can happen with 200 OK)
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
        body_preview = %body_preview,
        "OpenRouter returned error response from provider"
      );

      // TODO(debug): remove after debugging batch failures
      // Log detailed stats about the failing batch
      let total_chars: usize = texts.iter().map(|t| t.len()).sum();
      let max_len = texts.iter().map(|t| t.len()).max().unwrap_or(0);
      let min_len = texts.iter().map(|t| t.len()).min().unwrap_or(0);
      let empty_count = texts.iter().filter(|t| t.is_empty()).count();
      let very_long_count = texts.iter().filter(|t| t.len() > 8000).count();
      let has_nulls = texts.iter().any(|t| t.contains('\0'));
      warn!(
        total_chars,
        avg_len = total_chars / texts.len().max(1),
        max_len,
        min_len,
        empty_count,
        very_long_count,
        has_nulls,
        "[DEBUG] Batch stats for failed OpenRouter request"
      );

      return Err(EmbeddingError::ProviderError(format!(
        "Provider '{}' failed: {}",
        provider, error_resp.error.message
      )));
    }

    let result: EmbeddingResponse = match serde_json::from_str(&body_text) {
      Ok(r) => r,
      Err(e) => {
        // Check if this is a whitespace-only response (upstream provider timeout)
        let printable_count = body_text.chars().filter(|c| !c.is_whitespace()).count();

        if printable_count == 0 && !body_text.is_empty() {
          // Upstream provider timed out - only sent keep-alive whitespace
          // This is a transient issue, retry immediately without backoff
          debug!(
            batch_size = texts.len(),
            model = %self.model,
            elapsed_ms = start.elapsed().as_millis(),
            "Upstream provider timeout (received only keep-alive data), will retry"
          );
          return Err(EmbeddingError::UpstreamTimeout);
        }

        // Actual parse error - log details for debugging
        let trimmed = body_text.trim();
        let body_preview: String = trimmed.chars().take(500).collect();
        warn!(
          error = %e,
          batch_size = texts.len(),
          model = %self.model,
          elapsed_ms = start.elapsed().as_millis(),
          printable_count,
          body_preview = %body_preview,
          "Failed to parse OpenRouter response JSON"
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
      "Parsed OpenRouter response"
    );

    let expected = texts.len();
    let got = result.data.len();

    if got < expected {
      // Got fewer embeddings than inputs - this is a real error, we're missing data
      error!(
        expected,
        got,
        model = %self.model,
        "OpenRouter returned fewer embeddings than inputs"
      );
      return Err(EmbeddingError::BatchSizeMismatch { expected, got });
    }

    if got > expected {
      // Got more embeddings than inputs - provider quirk, use first N with warning
      warn!(
        expected,
        got,
        extra = got - expected,
        model = %self.model,
        "OpenRouter returned more embeddings than inputs, using first {} (discarding {} extra)",
        expected,
        got - expected
      );
    }

    // Take only the embeddings we need (handles both exact match and overflow)
    Ok(result.data.into_iter().take(expected).map(|d| d.embedding).collect())
  }

  /// Embed texts with sub-batching and full concurrent processing.
  ///
  /// Splits large batches into sub-batches of max_batch_size and processes
  /// them concurrently. Rate limiting is handled at the HTTP request level
  /// inside embed_single_batch(), so we can safely send all sub-batches
  /// concurrently - the rate limiter will naturally throttle them.
  async fn embed_batch_concurrent(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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

    // Create indexed sub-batch tasks - NO semaphore limit, rate limiter handles throttling
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

    // Wait for all batches concurrently - rate limiter inside embed_single_batch
    // will naturally throttle to stay within OpenRouter's 70 req/10s limit
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
      "OpenRouter batch embedding complete"
    );

    Ok(all_embeddings)
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

/// OpenRouter error response (returned with 200 OK when provider fails)
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

  async fn embed(&self, text: &str, mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
    // Format text based on mode (query vs document)
    let formatted = self.format_for_embedding(text, mode);

    // Acquire rate limit slot before making HTTP request
    let token = self.acquire_rate_limit_slot().await?;

    let request = EmbeddingRequest {
      model: &self.model,
      input: EmbeddingInput::Single(&formatted),
    };

    trace!(text_len = text.len(), mode = ?mode, model = %self.model, "Sending single embedding request to OpenRouter");
    let start = Instant::now();

    let response = match self
      .client
      .post(OPENROUTER_URL)
      .header("Authorization", format!("Bearer {}", self.api_key))
      .header("Content-Type", "application/json")
      .json(&request)
      .send()
      .await
    {
      Ok(resp) => resp,
      Err(e) => {
        // Network error - request never reached OpenRouter, refund the slot
        warn!(
          error = %e,
          text_len = text.len(),
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
      "Received single embedding response from OpenRouter"
    );

    if !status.is_success() {
      let status_code = status.as_u16();
      let body = response.text().await.unwrap_or_default();

      // Refund for 5xx server errors
      if status_code >= 500 {
        warn!(
          status = %status,
          text_len = text.len(),
          model = %self.model,
          "OpenRouter server error, refunding rate limit slot"
        );
        self.refund_rate_limit_slot(token).await;
      } else if status_code == 401 || status_code == 403 {
        error!(
          status = %status,
          model = %self.model,
          "OpenRouter authentication failed"
        );
      } else if status_code == 429 {
        warn!(
          status = %status,
          text_len = text.len(),
          model = %self.model,
          "OpenRouter rate limit exceeded"
        );
      } else {
        warn!(
          status = %status,
          text_len = text.len(),
          model = %self.model,
          "OpenRouter single embedding failed"
        );
      }

      return Err(EmbeddingError::ProviderError(format!(
        "OpenRouter returned {}: {}",
        status, body
      )));
    }

    let result: EmbeddingResponse = response.json().await?;

    let embedding = result.data.into_iter().next().map(|d| d.embedding).ok_or_else(|| {
      error!(model = %self.model, "OpenRouter returned empty response");
      EmbeddingError::ProviderError("No embedding in response".into())
    })?;

    trace!(
      dimensions = embedding.len(),
      elapsed_ms = start.elapsed().as_millis(),
      "Single embedding complete"
    );

    Ok(embedding)
  }

  async fn embed_batch(&self, texts: &[&str], mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      trace!("Empty batch, returning immediately");
      return Ok(Vec::new());
    }

    // Format all texts based on mode
    let formatted: Vec<String> = texts.iter().map(|t| self.format_for_embedding(t, mode)).collect();
    let formatted_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();

    debug!(batch_size = texts.len(), mode = ?mode, model = %self.model, "Embedding batch with OpenRouter");
    self.embed_batch_concurrent(&formatted_refs).await
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::{Config, EmbeddingConfig};

  // Integration tests - run when OPENROUTER_API_KEY is set, skip otherwise
  #[tokio::test]
  async fn test_embed_text_document() {
    let config = Config::default();
    let Ok(provider) = OpenRouterProvider::new(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let embedding = provider.embed("Hello, world!", EmbeddingMode::Document).await.unwrap();
    assert_eq!(embedding.len(), provider.dimensions());
  }

  #[tokio::test]
  async fn test_embed_text_query() {
    let config = Config::default();
    let Ok(provider) = OpenRouterProvider::new(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let embedding = provider.embed("Hello, world!", EmbeddingMode::Query).await.unwrap();
    assert_eq!(embedding.len(), provider.dimensions());
  }

  #[tokio::test]
  async fn test_embed_batch() {
    let config = Config::default();
    let Ok(provider) = OpenRouterProvider::new(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    let texts = vec!["Hello", "World", "Test"];
    let embeddings = provider.embed_batch(&texts, EmbeddingMode::Document).await.unwrap();

    assert_eq!(embeddings.len(), 3);
    for embedding in &embeddings {
      assert_eq!(embedding.len(), provider.dimensions());
    }
  }

  #[tokio::test]
  async fn test_embed_batch_with_subbatching() {
    let config = Config {
      embedding: EmbeddingConfig {
        max_batch_size: Some(2), // Force sub-batching
        ..Default::default()
      },
      ..Default::default()
    };
    let Ok(provider) = OpenRouterProvider::new(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };

    // 5 texts with batch size 2 = 3 sub-batches
    let texts = vec!["One", "Two", "Three", "Four", "Five"];
    let embeddings = provider.embed_batch(&texts, EmbeddingMode::Document).await.unwrap();

    assert_eq!(embeddings.len(), 5);
    for embedding in &embeddings {
      assert_eq!(embedding.len(), provider.dimensions());
    }
  }

  #[tokio::test]
  async fn test_embed_batch_empty() {
    let config = Config::default();
    let Ok(provider) = OpenRouterProvider::new(&config.embedding) else {
      eprintln!("OPENROUTER_API_KEY not set, skipping test");
      return;
    };
    let result = provider.embed_batch(&[], EmbeddingMode::Document).await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
  }

  #[test]
  fn test_format_for_embedding_query_with_instruction() {
    let config = EmbeddingConfig {
      query_instruction: Some("Test instruction".to_string()),
      ..Default::default()
    };
    let provider = OpenRouterProvider::new(&config).expect("create provider");
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
    let config = EmbeddingConfig {
      query_instruction: None,
      ..Default::default()
    };
    let provider = OpenRouterProvider::new(&config).expect("create provider");
    let formatted = provider.format_for_embedding("test query", EmbeddingMode::Query);
    assert_eq!(formatted, "test query", "Query without instruction should be unchanged");
  }

  #[test]
  fn test_format_for_embedding_query_empty_instruction() {
    let config = EmbeddingConfig {
      query_instruction: Some(String::new()),
      ..Default::default()
    };
    let provider = OpenRouterProvider::new(&config).expect("create provider");
    let formatted = provider.format_for_embedding("test query", EmbeddingMode::Query);
    assert_eq!(
      formatted, "test query",
      "Query with empty instruction should be unchanged"
    );
  }

  #[test]
  fn test_format_for_embedding_document() {
    let config = EmbeddingConfig {
      query_instruction: Some("Test instruction".to_string()),
      ..Default::default()
    };
    let provider = OpenRouterProvider::new(&config).expect("create provider");
    let formatted = provider.format_for_embedding("test document", EmbeddingMode::Document);
    assert_eq!(
      formatted, "test document",
      "Document should be unchanged regardless of instruction"
    );
  }
}
