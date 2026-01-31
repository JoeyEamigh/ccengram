// Resilient HTTP client wrapper with retries and backoff
//
// Features:
// - Exponential backoff with jitter
// - Retry on 429, 502, 503, 504 status codes
// - Retry-After header parsing
// - Network error detection and retry
// - Configurable timeouts

use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;
use tracing::{debug, trace, warn};

use super::{EmbeddingError, EmbeddingMode, EmbeddingProvider};

// TODO(debug): remove after debugging batch failures
/// Analyze batch content characteristics for debugging
fn debug_batch_stats(texts: &[&str]) -> BatchStats {
  let mut stats = BatchStats {
    count: texts.len(),
    ..Default::default()
  };

  for (i, text) in texts.iter().enumerate() {
    let len = text.len();
    stats.total_chars += len;
    stats.max_len = stats.max_len.max(len);
    stats.min_len = stats.min_len.min(len);

    // Check for unusual characteristics
    let has_null = text.contains('\0');
    let has_high_unicode = text.chars().any(|c| c as u32 > 0xFFFF);
    let non_ascii_ratio = text.chars().filter(|c| !c.is_ascii()).count() as f64 / text.len().max(1) as f64;
    let whitespace_ratio = text.chars().filter(|c| c.is_whitespace()).count() as f64 / text.len().max(1) as f64;

    if has_null {
      stats.texts_with_null.push(i);
    }
    if has_high_unicode {
      stats.texts_with_high_unicode.push(i);
    }
    if non_ascii_ratio > 0.5 {
      stats.texts_high_non_ascii.push((i, non_ascii_ratio));
    }
    if whitespace_ratio > 0.8 {
      stats.texts_high_whitespace.push((i, whitespace_ratio));
    }
    if len > 8000 {
      stats.texts_very_long.push((i, len));
    }
    if len == 0 {
      stats.texts_empty.push(i);
    }
  }

  stats
}

// TODO(debug): remove after debugging batch failures
#[derive(Default)]
struct BatchStats {
  count: usize,
  total_chars: usize,
  max_len: usize,
  min_len: usize,
  texts_with_null: Vec<usize>,
  texts_with_high_unicode: Vec<usize>,
  texts_high_non_ascii: Vec<(usize, f64)>,
  texts_high_whitespace: Vec<(usize, f64)>,
  texts_very_long: Vec<(usize, usize)>,
  texts_empty: Vec<usize>,
}

// TODO(debug): remove after debugging batch failures
impl BatchStats {
  fn has_anomalies(&self) -> bool {
    !self.texts_with_null.is_empty()
      || !self.texts_with_high_unicode.is_empty()
      || !self.texts_high_non_ascii.is_empty()
      || !self.texts_high_whitespace.is_empty()
      || !self.texts_very_long.is_empty()
      || !self.texts_empty.is_empty()
  }
}

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
      request_timeout: Duration::from_secs(600),
    }
  }
}

impl RetryConfig {
  /// Create a config optimized for cloud APIs
  pub fn for_cloud() -> Self {
    Self {
      max_retries: 5,
      initial_backoff: Duration::from_secs(1),
      max_backoff: Duration::from_secs(60),
      backoff_multiplier: 2.0,
      add_jitter: true,
      request_timeout: Duration::from_secs(600),
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
      let msg_lower = msg.to_lowercase();
      // Retryable status codes
      msg.contains("429") // Rate limited
        || msg.contains("502") // Bad gateway
        || msg.contains("503") // Service unavailable
        || msg.contains("504") // Gateway timeout
        || msg.contains("404") // OpenRouter "no successful provider" often has code 404
        // Transient OpenRouter errors
        || msg_lower.contains("no successful provider")
        || msg_lower.contains("temporarily unavailable")
        || msg_lower.contains("try again")
        || msg_lower.contains("overloaded")
    }
    EmbeddingError::Timeout => true,
    // Upstream timeout (whitespace-only response) - provider timed out, retry immediately
    EmbeddingError::UpstreamTimeout => true,
    // Rate limit exhaustion is retryable (wait and try again)
    EmbeddingError::RateLimitExhausted(_) => true,
    // Parse errors might be transient (server returned garbage once)
    EmbeddingError::ParseError(_) => true,
    // Batch size mismatch with fewer embeddings is retryable (might be transient)
    EmbeddingError::BatchSizeMismatch { .. } => true,
    _ => false,
  }
}

/// Check if an error is a provider-side bug that batch splitting won't help with.
///
/// These errors indicate the provider is misbehaving in a way that isn't caused
/// by our input, so splitting the batch won't isolate any "problematic text".
fn is_provider_bug(error: &EmbeddingError) -> bool {
  matches!(
    error,
    EmbeddingError::BatchSizeMismatch { .. } | EmbeddingError::UpstreamTimeout
  )
}

/// Check if an error is a rate limit exhaustion.
///
/// Rate limit exhaustion should NEVER trigger batch splitting because:
/// - Splitting creates MORE requests, making the problem worse
/// - The error is about throughput, not problematic text content
fn is_rate_limit_error(error: &EmbeddingError) -> bool {
  matches!(error, EmbeddingError::RateLimitExhausted(_))
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
  #[allow(dead_code)]
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

  async fn embed_with_retry(&self, text: &str, mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
    let mut last_error: Option<EmbeddingError> = None;
    let max_retries = self.config.max_retries;

    for attempt in 0..=max_retries {
      if attempt > 0 {
        // Skip backoff if previous error doesn't need it (e.g., upstream timeout)
        let needs_backoff = last_error.as_ref().is_none_or(|e| e.needs_backoff());
        if needs_backoff {
          let backoff = self.config.backoff_for_attempt(attempt - 1);
          trace!(backoff_ms = backoff.as_millis(), "Applying backoff before retry");
          debug!(
            attempt = attempt,
            max_retries = max_retries,
            backoff_ms = backoff.as_millis(),
            "Retrying single embed after backoff"
          );
          sleep(backoff).await;
        } else {
          debug!(
            attempt = attempt,
            max_retries = max_retries,
            "Retrying single embed immediately (no backoff needed)"
          );
        }
      }

      match tokio::time::timeout(self.config.request_timeout, self.inner.embed(text, mode)).await {
        Ok(Ok(result)) => {
          if attempt > 0 {
            debug!(attempt = attempt, "Single embed succeeded after retry");
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
  fn embed_batch_with_retry<'a>(
    &'a self,
    texts: &'a [&'a str],
    mode: EmbeddingMode,
    initial_attempt: u32,
  ) -> BoxedEmbedFuture<'a> {
    Box::pin(async move {
      if texts.is_empty() {
        return Ok(Vec::new());
      }

      let max_retries = self.config.max_retries;
      let mut attempt = initial_attempt;

      // TODO(debug): remove after debugging batch failures
      // Log initial batch fingerprint for correlation
      if initial_attempt == 0 {
        let batch_fingerprint: u64 = texts.iter().map(|t| t.len() as u64).sum::<u64>()
          ^ (texts.len() as u64 * 31)
          ^ texts
            .first()
            .map(|t| t.chars().take(10).map(|c| c as u64).sum::<u64>())
            .unwrap_or(0);
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        trace!(
          batch_size = texts.len(),
          total_chars,
          fingerprint = format!("{:016x}", batch_fingerprint),
          "[DEBUG] Starting batch embed"
        );
      }

      let mut last_batch_error: Option<EmbeddingError> = None;

      loop {
        // Apply backoff if this is a retry (unless previous error doesn't need it)
        if attempt > 0 {
          let needs_backoff = last_batch_error.as_ref().is_none_or(|e| e.needs_backoff());
          if needs_backoff {
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
          } else {
            debug!(
              attempt = attempt,
              max_retries = max_retries,
              batch_size = texts.len(),
              "Retrying batch embed immediately (upstream timeout, no backoff needed)"
            );
          }
        }

        // Try the batch operation with timeout
        match tokio::time::timeout(self.config.request_timeout, self.inner.embed_batch(texts, mode)).await {
          Ok(Ok(embeddings)) => {
            if attempt > 0 {
              debug!(
                attempt = attempt,
                batch_size = texts.len(),
                "Batch embed succeeded after retry"
              );
            }
            return Ok(embeddings);
          }
          Ok(Err(e)) if is_rate_limit_error(&e) => {
            // Rate limit exhaustion - do NOT split or retry, just fail
            // Splitting would create MORE requests, making things worse
            warn!(
              attempt = attempt + 1,
              batch_size = texts.len(),
              err = %e,
              "Rate limit exhausted, failing batch (splitting would make it worse)"
            );
            return Err(e);
          }
          Ok(Err(e)) if is_provider_bug(&e) => {
            // Provider bug (e.g., batch size mismatch) - splitting won't help
            // The provider is misbehaving regardless of our input
            warn!(
              attempt = attempt + 1,
              batch_size = texts.len(),
              err = %e,
              "Provider bug detected, failing batch (splitting won't help)"
            );
            return Err(e);
          }
          Ok(Err(e)) => {
            // Any other error - decide whether to retry, split, or fail
            let min_retries_before_split = 2;
            let should_retry = is_retryable_error(&e) && attempt < max_retries;
            let can_split = texts.len() > 1 && attempt >= min_retries_before_split;

            if should_retry || attempt < min_retries_before_split {
              // Retry the batch (either it's retryable, or we haven't tried enough times)
              warn!(
                attempt = attempt + 1,
                max_retries = max_retries,
                batch_size = texts.len(),
                err = %e,
                retryable = is_retryable_error(&e),
                "Batch error, will retry"
              );
              last_batch_error = Some(e);
              attempt += 1;
              continue;
            } else if can_split {
              // We've retried enough times, try splitting to isolate problematic texts
              // TODO(debug): remove after debugging batch failures
              let stats = debug_batch_stats(texts);
              warn!(
                attempt = attempt + 1,
                batch_size = texts.len(),
                err = %e,
                total_chars = stats.total_chars,
                avg_len = stats.total_chars / stats.count.max(1),
                max_len = stats.max_len,
                min_len = stats.min_len,
                has_anomalies = stats.has_anomalies(),
                texts_with_null = ?stats.texts_with_null,
                texts_with_high_unicode = ?stats.texts_with_high_unicode,
                texts_high_non_ascii = ?stats.texts_high_non_ascii,
                texts_high_whitespace = ?stats.texts_high_whitespace,
                texts_very_long = ?stats.texts_very_long,
                texts_empty = ?stats.texts_empty,
                "Batch embedding failed after retries, splitting to isolate problematic texts"
              );
              // TODO(debug): Log first few text previews to identify content patterns
              for (i, text) in texts.iter().take(3).enumerate() {
                let preview: String = text.chars().take(200).collect();
                let preview_escaped = preview.replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
                warn!(
                  text_index = i,
                  text_len = text.len(),
                  preview = %preview_escaped,
                  "[DEBUG] Sample text from failing batch"
                );
              }
              return self.split_and_retry(texts, mode).await;
            } else {
              // Single text that failed after retries - return the error
              // TODO(debug): remove after debugging batch failures
              let text = texts.first().copied().unwrap_or("");
              let preview: String = text.chars().take(500).collect();
              let preview_escaped = preview.replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
              let has_null = text.contains('\0');
              let has_high_unicode = text.chars().any(|c| c as u32 > 0xFFFF);
              let non_ascii_count = text.chars().filter(|c| !c.is_ascii()).count();
              warn!(
                attempt = attempt + 1,
                err = %e,
                text_len = text.len(),
                has_null,
                has_high_unicode,
                non_ascii_count,
                preview = %preview_escaped,
                "[DEBUG] Single text that failed all retries"
              );
              return Err(e);
            }
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
              return self.split_and_retry(texts, mode).await;
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
  async fn split_and_retry(&self, texts: &[&str], mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let mid = texts.len() / 2;
    let (left, right) = texts.split_at(mid);

    // TODO(debug): remove after debugging batch failures
    // Create a simple fingerprint of batch content to track lineage
    let batch_fingerprint: u64 = texts.iter().map(|t| t.len() as u64).sum::<u64>()
      ^ (texts.len() as u64 * 31)
      ^ texts
        .first()
        .map(|t| t.chars().take(10).map(|c| c as u64).sum::<u64>())
        .unwrap_or(0);
    let left_fingerprint: u64 = left.iter().map(|t| t.len() as u64).sum::<u64>()
      ^ (left.len() as u64 * 31)
      ^ left
        .first()
        .map(|t| t.chars().take(10).map(|c| c as u64).sum::<u64>())
        .unwrap_or(0);
    let right_fingerprint: u64 = right.iter().map(|t| t.len() as u64).sum::<u64>()
      ^ (right.len() as u64 * 31)
      ^ right
        .first()
        .map(|t| t.chars().take(10).map(|c| c as u64).sum::<u64>())
        .unwrap_or(0);

    debug!(
      total_size = texts.len(),
      left_size = left.len(),
      right_size = right.len(),
      batch_fingerprint = format!("{:016x}", batch_fingerprint),
      left_fingerprint = format!("{:016x}", left_fingerprint),
      right_fingerprint = format!("{:016x}", right_fingerprint),
      "Splitting batch for retry"
    );

    // Process both halves concurrently
    let (left_result, right_result) = tokio::join!(
      self.embed_batch_with_retry(left, mode, 0),
      self.embed_batch_with_retry(right, mode, 0)
    );

    // TODO(debug): remove after debugging batch failures
    let left_ok = left_result.is_ok();
    let right_ok = right_result.is_ok();
    if !left_ok || !right_ok {
      warn!(
        left_ok,
        right_ok,
        left_size = left.len(),
        right_size = right.len(),
        left_fingerprint = format!("{:016x}", left_fingerprint),
        right_fingerprint = format!("{:016x}", right_fingerprint),
        "[DEBUG] Split results - at least one half failed"
      );
    }

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

  async fn embed(&self, text: &str, mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
    self.embed_with_retry(text, mode).await
  }

  async fn embed_batch(&self, texts: &[&str], mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    self.embed_batch_with_retry(texts, mode, 0).await
  }
}

#[cfg(test)]
mod tests {
  use super::*;

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
    // Rate limit exhaustion is retryable (but should NOT trigger splitting)
    assert!(is_retryable_error(&EmbeddingError::RateLimitExhausted(
      Duration::from_secs(60)
    )));
    // Parse errors are retryable (might be transient server issue)
    assert!(is_retryable_error(&EmbeddingError::ParseError(
      "JSON parse error".to_string()
    )));
    // Batch size mismatch is retryable
    assert!(is_retryable_error(&EmbeddingError::BatchSizeMismatch {
      expected: 10,
      got: 5
    }));
    // OpenRouter "no successful provider" errors are retryable
    assert!(is_retryable_error(&EmbeddingError::ProviderError(
      "Provider 'unknown' failed: No successful provider responses.".to_string()
    )));
    // 404 errors from OpenRouter are retryable
    assert!(is_retryable_error(&EmbeddingError::ProviderError(
      "OpenRouter returned 404: no providers".to_string()
    )));
  }

  #[test]
  fn test_is_rate_limit_error() {
    assert!(is_rate_limit_error(&EmbeddingError::RateLimitExhausted(
      Duration::from_secs(60)
    )));
    assert!(!is_rate_limit_error(&EmbeddingError::Timeout));
    assert!(!is_rate_limit_error(&EmbeddingError::Network("test".to_string())));
    assert!(!is_rate_limit_error(&EmbeddingError::ProviderError("429".to_string())));
  }

  #[test]
  fn test_is_provider_bug() {
    // Batch size mismatch is a provider bug - splitting won't help
    assert!(is_provider_bug(&EmbeddingError::BatchSizeMismatch {
      expected: 10,
      got: 5
    }));
    // Other errors are not provider bugs
    assert!(!is_provider_bug(&EmbeddingError::Timeout));
    assert!(!is_provider_bug(&EmbeddingError::Network("test".to_string())));
    assert!(!is_provider_bug(&EmbeddingError::ParseError("parse error".to_string())));
    assert!(!is_provider_bug(&EmbeddingError::ProviderError("error".to_string())));
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

    async fn embed(&self, _text: &str, _mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
      self.embed_calls.fetch_add(1, Ordering::SeqCst);
      Ok(vec![0.1; 384])
    }

    async fn embed_batch(&self, texts: &[&str], _mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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
    let result = resilient.embed_batch(&texts, EmbeddingMode::Document).await;

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
    let result = resilient.embed_batch(&texts, EmbeddingMode::Document).await;

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
    let result = resilient.embed_batch(&texts, EmbeddingMode::Document).await;

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
      async fn embed(&self, _text: &str, _mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
        Ok(vec![0.1; 4])
      }
      async fn embed_batch(&self, texts: &[&str], _mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        // Fail if batch contains the bad text
        for text in texts {
          if *text == "bad" {
            return Err(EmbeddingError::ProviderError("bad text".to_string()));
          }
        }
        Ok(texts.iter().map(|_| vec![0.1; 4]).collect())
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
    let result = resilient.embed_batch(&texts, EmbeddingMode::Document).await;

    // This should fail because "bad" can't be embedded even alone
    assert!(result.is_err());

    // But it should have made multiple batch calls due to splitting
    assert!(resilient.inner.batch_calls.load(Ordering::SeqCst) >= 3);
  }

  #[tokio::test]
  async fn test_empty_batch() {
    let provider = MockBatchProvider::new();
    let resilient = ResilientProvider::new(provider);

    let result = resilient.embed_batch(&[], EmbeddingMode::Document).await;

    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
    // Should not have made any calls
    assert_eq!(resilient.inner.batch_calls.load(Ordering::SeqCst), 0);
  }

  #[tokio::test]
  async fn test_single_text_batch() {
    let provider = MockBatchProvider::new();
    let resilient = ResilientProvider::new(provider);

    let result = resilient.embed_batch(&["single"], EmbeddingMode::Document).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
    assert_eq!(resilient.inner.batch_calls.load(Ordering::SeqCst), 1);
  }
}
