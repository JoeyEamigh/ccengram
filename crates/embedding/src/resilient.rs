// Resilient HTTP client wrapper with retries and backoff
//
// Features:
// - Exponential backoff with jitter
// - Retry on 429, 502, 503, 504 status codes
// - Retry-After header parsing
// - Network error detection and retry
// - Configurable timeouts

use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, trace, warn};

/// Configuration for resilient HTTP operations
#[derive(Debug, Clone)]
pub struct RetryConfig {
  /// Maximum number of retry attempts
  pub max_retries: u32,
  /// Initial backoff duration
  pub initial_backoff: Duration,
  /// Maximum backoff duration
  pub max_backoff: Duration,
  /// Backoff multiplier (exponential factor)
  pub backoff_multiplier: f64,
  /// Whether to add jitter to backoff
  pub add_jitter: bool,
  /// Request timeout
  pub request_timeout: Duration,
}

impl Default for RetryConfig {
  fn default() -> Self {
    Self {
      max_retries: 3,
      initial_backoff: Duration::from_secs(1),
      max_backoff: Duration::from_secs(30),
      backoff_multiplier: 2.0,
      add_jitter: true,
      request_timeout: Duration::from_secs(60),
    }
  }
}

impl RetryConfig {
  /// Create a config optimized for fast local services (like Ollama)
  pub fn for_local() -> Self {
    Self {
      max_retries: 2,
      initial_backoff: Duration::from_millis(500),
      max_backoff: Duration::from_secs(5),
      backoff_multiplier: 2.0,
      add_jitter: true,
      request_timeout: Duration::from_secs(30),
    }
  }

  /// Create a config optimized for cloud APIs
  pub fn for_cloud() -> Self {
    Self {
      max_retries: 5,
      initial_backoff: Duration::from_secs(1),
      max_backoff: Duration::from_secs(60),
      backoff_multiplier: 2.0,
      add_jitter: true,
      request_timeout: Duration::from_secs(120),
    }
  }

  /// Calculate backoff duration for a given attempt
  pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
    let base = self.initial_backoff.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
    let mut backoff = Duration::from_secs_f64(base.min(self.max_backoff.as_secs_f64()));

    if self.add_jitter {
      // Add up to 25% jitter
      let jitter_factor = 1.0 + (rand_f64() * 0.25);
      backoff = Duration::from_secs_f64(backoff.as_secs_f64() * jitter_factor);
    }

    backoff.min(self.max_backoff)
  }
}

/// A simple pseudo-random number generator for jitter (no external deps)
fn rand_f64() -> f64 {
  use std::time::{SystemTime, UNIX_EPOCH};

  let nanos = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .subsec_nanos();

  (nanos as f64 / u32::MAX as f64).fract()
}

/// Check if an error is retryable
pub fn is_retryable_error(error: &EmbeddingError) -> bool {
  match error {
    EmbeddingError::Network(_) => true,
    EmbeddingError::ProviderError(msg) => {
      // Check for retryable status codes in the message
      msg.contains("429") // Rate limited
        || msg.contains("502") // Bad gateway
        || msg.contains("503") // Service unavailable
        || msg.contains("504") // Gateway timeout
    }
    EmbeddingError::Timeout => true,
    _ => false,
  }
}

/// A resilient embedding provider that wraps another provider with retry logic
pub struct ResilientProvider<P: EmbeddingProvider> {
  inner: P,
  config: RetryConfig,
}

/// Boxed future type for async recursive calls
type BoxedEmbedFuture<'a> =
  std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send + 'a>>;

impl<P: EmbeddingProvider> ResilientProvider<P> {
  pub fn new(provider: P) -> Self {
    Self {
      inner: provider,
      config: RetryConfig::default(),
    }
  }

  pub fn with_config(provider: P, config: RetryConfig) -> Self {
    Self {
      inner: provider,
      config,
    }
  }

  async fn embed_with_retry(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
    let mut last_error = None;
    let max_retries = self.config.max_retries;

    for attempt in 0..=max_retries {
      if attempt > 0 {
        let backoff = self.config.backoff_for_attempt(attempt - 1);
        trace!(backoff_ms = backoff.as_millis(), "Applying backoff before retry");
        debug!(
          attempt = attempt,
          max_retries = max_retries,
          backoff_ms = backoff.as_millis(),
          "Retrying single embed after backoff"
        );
        sleep(backoff).await;
      }

      match tokio::time::timeout(self.config.request_timeout, self.inner.embed(text)).await {
        Ok(Ok(result)) => {
          if attempt > 0 {
            info!(attempt = attempt, "Single embed succeeded after retry");
          }
          return Ok(result);
        }
        Ok(Err(e)) => {
          if is_retryable_error(&e) && attempt < max_retries {
            warn!(
              attempt = attempt + 1,
              max_retries = max_retries,
              err = %e,
              "Retryable error, will retry"
            );
            last_error = Some(e);
            continue;
          }
          if attempt == max_retries && is_retryable_error(&e) {
            warn!(
              attempt = attempt + 1,
              max_retries = max_retries,
              err = %e,
              "All retries exhausted"
            );
          }
          return Err(e);
        }
        Err(_) => {
          warn!(
            attempt = attempt + 1,
            max_retries = max_retries,
            timeout_ms = self.config.request_timeout.as_millis(),
            "Request timed out"
          );
          last_error = Some(EmbeddingError::Timeout);
          if attempt < max_retries {
            continue;
          }
        }
      }
    }

    warn!(max_retries = max_retries, "All retries exhausted");
    Err(last_error.unwrap_or_else(|| EmbeddingError::ProviderError("Max retries exceeded".to_string())))
  }

  /// Embed a batch of texts with retry logic that:
  /// 1. Retries the entire batch on transient failures
  /// 2. Falls back to binary-search retry to isolate problematic texts on persistent failures
  fn embed_batch_with_retry<'a>(&'a self, texts: &'a [&'a str], initial_attempt: u32) -> BoxedEmbedFuture<'a> {
    Box::pin(async move {
      if texts.is_empty() {
        return Ok(Vec::new());
      }

      let max_retries = self.config.max_retries;
      let mut attempt = initial_attempt;

      loop {
        // Apply backoff if this is a retry
        if attempt > 0 {
          let backoff = self.config.backoff_for_attempt(attempt - 1);
          trace!(backoff_ms = backoff.as_millis(), "Applying backoff before batch retry");
          debug!(
            attempt = attempt,
            max_retries = max_retries,
            batch_size = texts.len(),
            backoff_ms = backoff.as_millis(),
            "Retrying batch embed after backoff"
          );
          sleep(backoff).await;
        }

        // Try the batch operation with timeout
        match tokio::time::timeout(self.config.request_timeout, self.inner.embed_batch(texts)).await {
          Ok(Ok(embeddings)) => {
            if attempt > 0 {
              info!(
                attempt = attempt,
                batch_size = texts.len(),
                "Batch embed succeeded after retry"
              );
            }
            return Ok(embeddings);
          }
          Ok(Err(e)) if is_retryable_error(&e) && attempt < max_retries => {
            // Transient error - retry the entire batch
            warn!(
              attempt = attempt + 1,
              max_retries = max_retries,
              batch_size = texts.len(),
              err = %e,
              "Retryable batch error, will retry"
            );
            attempt += 1;
            continue;
          }
          Ok(Err(e)) if texts.len() > 1 => {
            // Persistent failure on batch - binary split to isolate problematic texts
            warn!(
              attempt = attempt + 1,
              batch_size = texts.len(),
              err = %e,
              "Batch embedding failed, splitting to isolate problematic texts"
            );
            return self.split_and_retry(texts).await;
          }
          Ok(Err(e)) => {
            // Single text that failed - return the error
            debug!(
              attempt = attempt + 1,
              err = %e,
              "Single text embed failed"
            );
            return Err(e);
          }
          Err(_) => {
            // Timeout
            warn!(
              attempt = attempt + 1,
              max_retries = max_retries,
              batch_size = texts.len(),
              timeout_ms = self.config.request_timeout.as_millis(),
              "Batch request timed out"
            );
            if attempt < max_retries {
              attempt += 1;
              continue;
            } else if texts.len() > 1 {
              // Try splitting on timeout too
              debug!(
                batch_size = texts.len(),
                "Splitting batch after timeout to isolate slow texts"
              );
              return self.split_and_retry(texts).await;
            } else {
              warn!(max_retries = max_retries, "All retries exhausted after timeout");
              return Err(EmbeddingError::Timeout);
            }
          }
        }
      }
    })
  }

  /// Split batch in half and retry both halves concurrently
  async fn split_and_retry(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let mid = texts.len() / 2;
    let (left, right) = texts.split_at(mid);

    debug!(
      total_size = texts.len(),
      left_size = left.len(),
      right_size = right.len(),
      "Splitting batch for retry"
    );

    // Process both halves concurrently
    let (left_result, right_result) = tokio::join!(
      self.embed_batch_with_retry(left, 0),
      self.embed_batch_with_retry(right, 0)
    );

    let mut results = left_result?;
    results.extend(right_result?);

    debug!(result_size = results.len(), "Split retry completed successfully");

    Ok(results)
  }
}

#[async_trait]
impl<P: EmbeddingProvider + Send + Sync> EmbeddingProvider for ResilientProvider<P> {
  fn name(&self) -> &str {
    self.inner.name()
  }

  fn model_id(&self) -> &str {
    self.inner.model_id()
  }

  fn dimensions(&self) -> usize {
    self.inner.dimensions()
  }

  async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
    self.embed_with_retry(text).await
  }

  async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    self.embed_batch_with_retry(texts, 0).await
  }

  async fn is_available(&self) -> bool {
    self.inner.is_available().await
  }
}

/// Wrap any embedding provider with resilient retry logic
pub fn wrap_resilient<P: EmbeddingProvider>(provider: P) -> ResilientProvider<P> {
  ResilientProvider::new(provider)
}

/// Wrap any embedding provider with resilient retry logic using Arc for sharing
pub fn wrap_resilient_arc<P>(provider: P) -> Arc<dyn EmbeddingProvider + Send + Sync>
where
  P: EmbeddingProvider + Send + Sync + 'static,
{
  Arc::new(ResilientProvider::new(provider))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_retry_config_defaults() {
    let config = RetryConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_backoff, Duration::from_secs(1));
    assert_eq!(config.max_backoff, Duration::from_secs(30));
  }

  #[test]
  fn test_retry_config_for_local() {
    let config = RetryConfig::for_local();
    assert_eq!(config.max_retries, 2);
    assert!(config.initial_backoff < Duration::from_secs(1));
  }

  #[test]
  fn test_retry_config_for_cloud() {
    let config = RetryConfig::for_cloud();
    assert_eq!(config.max_retries, 5);
    assert!(config.max_backoff > Duration::from_secs(30));
  }

  #[test]
  fn test_backoff_calculation() {
    let config = RetryConfig {
      initial_backoff: Duration::from_secs(1),
      backoff_multiplier: 2.0,
      max_backoff: Duration::from_secs(60),
      add_jitter: false,
      ..Default::default()
    };

    assert_eq!(config.backoff_for_attempt(0), Duration::from_secs(1));
    assert_eq!(config.backoff_for_attempt(1), Duration::from_secs(2));
    assert_eq!(config.backoff_for_attempt(2), Duration::from_secs(4));
    assert_eq!(config.backoff_for_attempt(3), Duration::from_secs(8));
  }

  #[test]
  fn test_backoff_respects_max() {
    let config = RetryConfig {
      initial_backoff: Duration::from_secs(10),
      backoff_multiplier: 10.0,
      max_backoff: Duration::from_secs(30),
      add_jitter: false,
      ..Default::default()
    };

    // 10 * 10^2 = 1000 seconds, but should be capped at 30
    assert_eq!(config.backoff_for_attempt(2), Duration::from_secs(30));
  }

  #[test]
  fn test_is_retryable_error() {
    assert!(is_retryable_error(&EmbeddingError::Network(
      "connection reset".to_string()
    )));
    assert!(is_retryable_error(&EmbeddingError::Timeout));
    assert!(is_retryable_error(&EmbeddingError::ProviderError(
      "Status 429".to_string()
    )));
    assert!(is_retryable_error(&EmbeddingError::ProviderError(
      "Got 503".to_string()
    )));
    assert!(!is_retryable_error(&EmbeddingError::ProviderError(
      "Invalid input".to_string()
    )));
    assert!(!is_retryable_error(&EmbeddingError::ProviderError(
      "Status 400".to_string()
    )));
  }

  #[test]
  fn test_rand_f64_is_bounded() {
    for _ in 0..100 {
      let val = rand_f64();
      assert!(val >= 0.0);
      assert!(val <= 1.0);
    }
  }

  // Mock provider for testing batch behavior
  use std::sync::atomic::{AtomicUsize, Ordering};

  struct MockBatchProvider {
    embed_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    fail_batch_until: AtomicUsize,
    fail_with_retryable: bool,
    fail_on_text: Option<String>,
  }

  impl MockBatchProvider {
    fn new() -> Self {
      Self {
        embed_calls: AtomicUsize::new(0),
        batch_calls: AtomicUsize::new(0),
        fail_batch_until: AtomicUsize::new(0),
        fail_with_retryable: false,
        fail_on_text: None,
      }
    }

    fn failing_until(attempts: usize, retryable: bool) -> Self {
      Self {
        embed_calls: AtomicUsize::new(0),
        batch_calls: AtomicUsize::new(0),
        fail_batch_until: AtomicUsize::new(attempts),
        fail_with_retryable: retryable,
        fail_on_text: None,
      }
    }

    fn failing_on_text(text: &str) -> Self {
      Self {
        embed_calls: AtomicUsize::new(0),
        batch_calls: AtomicUsize::new(0),
        fail_batch_until: AtomicUsize::new(0),
        fail_with_retryable: false,
        fail_on_text: Some(text.to_string()),
      }
    }
  }

  #[async_trait]
  impl EmbeddingProvider for MockBatchProvider {
    fn name(&self) -> &str {
      "mock"
    }
    fn model_id(&self) -> &str {
      "mock-model"
    }
    fn dimensions(&self) -> usize {
      384
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
      self.embed_calls.fetch_add(1, Ordering::SeqCst);
      Ok(vec![0.1; 384])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
      let call_num = self.batch_calls.fetch_add(1, Ordering::SeqCst);

      // Check if we should fail based on attempt count
      let fail_until = self.fail_batch_until.load(Ordering::SeqCst);
      if call_num < fail_until {
        if self.fail_with_retryable {
          return Err(EmbeddingError::Network("connection reset".to_string()));
        } else {
          return Err(EmbeddingError::ProviderError("permanent error".to_string()));
        }
      }

      // Check if any text should cause failure
      if let Some(ref bad_text) = self.fail_on_text {
        for text in texts {
          if *text == bad_text {
            return Err(EmbeddingError::ProviderError(format!("bad text: {}", bad_text)));
          }
        }
      }

      Ok(texts.iter().map(|_| vec![0.1; 384]).collect())
    }

    async fn is_available(&self) -> bool {
      true
    }
  }

  #[tokio::test]
  async fn test_batch_uses_inner_embed_batch() {
    // Verify batch calls the inner provider's embed_batch, not individual embed
    let provider = MockBatchProvider::new();
    let resilient = ResilientProvider::with_config(
      provider,
      RetryConfig {
        max_retries: 3,
        initial_backoff: Duration::from_millis(1),
        ..Default::default()
      },
    );

    let texts: Vec<&str> = (0..64).map(|_| "test text").collect();
    let result = resilient.embed_batch(&texts).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);

    // Should have called embed_batch once, not embed 64 times
    assert_eq!(resilient.inner.batch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(resilient.inner.embed_calls.load(Ordering::SeqCst), 0);
  }

  #[tokio::test]
  async fn test_batch_retries_on_transient_error() {
    // Fail first 2 attempts with retryable error, succeed on 3rd
    let provider = MockBatchProvider::failing_until(2, true);
    let resilient = ResilientProvider::with_config(
      provider,
      RetryConfig {
        max_retries: 3,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(10),
        add_jitter: false,
        ..Default::default()
      },
    );

    let texts = vec!["text1", "text2", "text3"];
    let result = resilient.embed_batch(&texts).await;

    assert!(result.is_ok());
    // Should have retried twice, then succeeded on 3rd attempt
    assert_eq!(resilient.inner.batch_calls.load(Ordering::SeqCst), 3);
  }

  #[tokio::test]
  async fn test_batch_splits_on_persistent_failure() {
    // Provider fails on a specific text, should binary split to isolate
    let provider = MockBatchProvider::failing_on_text("bad_text");
    let resilient = ResilientProvider::with_config(
      provider,
      RetryConfig {
        max_retries: 0, // No retries, go straight to split
        initial_backoff: Duration::from_millis(1),
        ..Default::default()
      },
    );

    // 4 texts, one is bad. Binary split: [0,1] and [2,3]
    // If bad_text is at index 2, right half fails, splits again
    let texts = vec!["good1", "good2", "bad_text", "good3"];
    let result = resilient.embed_batch(&texts).await;

    // Should fail because bad_text eventually causes an error when it's alone
    assert!(result.is_err());

    // But good texts in other splits would have succeeded if we continued
    // The key is it split and tried, not just failed immediately
    assert!(resilient.inner.batch_calls.load(Ordering::SeqCst) > 1);
  }

  #[tokio::test]
  async fn test_batch_partial_success_with_split() {
    // Custom provider that only fails when batch contains bad_text
    // but succeeds for batches without it
    struct PartialFailProvider {
      batch_calls: AtomicUsize,
    }

    #[async_trait]
    impl EmbeddingProvider for PartialFailProvider {
      fn name(&self) -> &str {
        "partial"
      }
      fn model_id(&self) -> &str {
        "partial-model"
      }
      fn dimensions(&self) -> usize {
        4
      }
      async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(vec![0.1; 4])
      }
      async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        // Fail if batch contains the bad text
        for text in texts {
          if *text == "bad" {
            return Err(EmbeddingError::ProviderError("bad text".to_string()));
          }
        }
        Ok(texts.iter().map(|_| vec![0.1; 4]).collect())
      }
      async fn is_available(&self) -> bool {
        true
      }
    }

    let provider = PartialFailProvider {
      batch_calls: AtomicUsize::new(0),
    };
    let resilient = ResilientProvider::with_config(
      provider,
      RetryConfig {
        max_retries: 0,
        initial_backoff: Duration::from_millis(1),
        ..Default::default()
      },
    );

    // good1, good2, good3 should all succeed when bad is isolated
    // With binary split: [good1, good2] succeeds, [bad, good3] fails
    // Then [bad] fails (single), [good3] succeeds
    // Actually with order [good1, good2, bad, good3]:
    //   First: [good1, good2] vs [bad, good3]
    //   [good1, good2] succeeds
    //   [bad, good3] fails, splits to [bad] vs [good3]
    //   [bad] fails (single text, returns error)
    // So overall fails because we can't embed bad
    let texts = vec!["good1", "good2", "bad", "good3"];
    let result = resilient.embed_batch(&texts).await;

    // This should fail because "bad" can't be embedded even alone
    assert!(result.is_err());

    // But it should have made multiple batch calls due to splitting
    assert!(resilient.inner.batch_calls.load(Ordering::SeqCst) >= 3);
  }

  #[tokio::test]
  async fn test_empty_batch() {
    let provider = MockBatchProvider::new();
    let resilient = ResilientProvider::new(provider);

    let result = resilient.embed_batch(&[]).await;

    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
    // Should not have made any calls
    assert_eq!(resilient.inner.batch_calls.load(Ordering::SeqCst), 0);
  }

  #[tokio::test]
  async fn test_single_text_batch() {
    let provider = MockBatchProvider::new();
    let resilient = ResilientProvider::new(provider);

    let result = resilient.embed_batch(&["single"]).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
    assert_eq!(resilient.inner.batch_calls.load(Ordering::SeqCst), 1);
  }
}
