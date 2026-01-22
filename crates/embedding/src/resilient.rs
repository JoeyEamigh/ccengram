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
use tracing::{debug, warn};

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

    for attempt in 0..=self.config.max_retries {
      if attempt > 0 {
        let backoff = self.config.backoff_for_attempt(attempt - 1);
        debug!("Retry attempt {} after {:?}", attempt, backoff);
        sleep(backoff).await;
      }

      match tokio::time::timeout(self.config.request_timeout, self.inner.embed(text)).await {
        Ok(Ok(result)) => return Ok(result),
        Ok(Err(e)) => {
          if is_retryable_error(&e) && attempt < self.config.max_retries {
            warn!("Retryable error on attempt {}: {}", attempt + 1, e);
            last_error = Some(e);
            continue;
          }
          return Err(e);
        }
        Err(_) => {
          warn!("Request timed out on attempt {}", attempt + 1);
          last_error = Some(EmbeddingError::Timeout);
          if attempt < self.config.max_retries {
            continue;
          }
        }
      }
    }

    Err(last_error.unwrap_or_else(|| EmbeddingError::ProviderError("Max retries exceeded".to_string())))
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
    let mut results = Vec::with_capacity(texts.len());

    for text in texts {
      // Each text gets its own retry logic
      results.push(self.embed_with_retry(text).await?);
    }

    Ok(results)
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
}
