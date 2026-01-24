// Rate limiter for API providers with sliding window rate limiting
//
// Implements a sliding window rate limiter that tracks requests over a
// configurable time window and delays requests when the limit is reached.

use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::{debug, trace, warn};

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
  /// Maximum requests allowed in the window
  pub max_requests: usize,
  /// Time window duration
  pub window: Duration,
  /// Maximum time to wait for a slot before failing
  pub max_wait: Duration,
}

impl Default for RateLimitConfig {
  fn default() -> Self {
    Self {
      max_requests: 70,
      window: Duration::from_secs(10),
      max_wait: Duration::from_secs(30),
    }
  }
}

impl RateLimitConfig {
  /// Create a config for OpenRouter (50 requests per 10s sliding window)
  /// OpenRouter's actual limit is 70/10s, but we use 50 for safety margin.
  pub fn for_openrouter() -> Self {
    Self {
      max_requests: 50,
      window: Duration::from_secs(10),
      max_wait: Duration::from_secs(60),
    }
  }

  /// Create a config with custom limits
  pub fn new(max_requests: usize, window: Duration) -> Self {
    Self {
      max_requests,
      window,
      max_wait: Duration::from_secs(30),
    }
  }

  /// Set the maximum wait time
  pub fn with_max_wait(mut self, max_wait: Duration) -> Self {
    self.max_wait = max_wait;
    self
  }
}

/// Sliding window rate limiter
#[derive(Debug)]
pub struct SlidingWindowLimiter {
  config: RateLimitConfig,
  /// Timestamps of recent requests (within the window)
  request_times: VecDeque<Instant>,
}

impl SlidingWindowLimiter {
  pub fn new(config: RateLimitConfig) -> Self {
    let capacity = config.max_requests + 1;
    Self {
      config,
      request_times: VecDeque::with_capacity(capacity),
    }
  }

  /// Remove expired timestamps from the window
  fn prune_expired(&mut self) {
    let cutoff = Instant::now() - self.config.window;
    let before_count = self.request_times.len();
    while let Some(&oldest) = self.request_times.front() {
      if oldest < cutoff {
        self.request_times.pop_front();
      } else {
        break;
      }
    }
    let pruned = before_count - self.request_times.len();
    if pruned > 0 {
      trace!(
        pruned = pruned,
        remaining = self.request_times.len(),
        window_ms = self.config.window.as_millis(),
        "Rate limit window reset, pruned expired timestamps"
      );
    }
  }

  /// Check if we can make a request now, and if not, how long to wait
  fn check_and_wait_time(&mut self) -> Option<Duration> {
    self.prune_expired();

    if self.request_times.len() < self.config.max_requests {
      // Under the limit, can proceed immediately
      None
    } else {
      // At the limit - calculate when the oldest request will expire
      if let Some(&oldest) = self.request_times.front() {
        let expires_at = oldest + self.config.window;
        let now = Instant::now();
        if expires_at > now { Some(expires_at - now) } else { None }
      } else {
        None
      }
    }
  }

  /// Record that a request was made
  fn record_request(&mut self) {
    self.request_times.push_back(Instant::now());
  }

  /// Check if a slot is available and record the request if so.
  /// Returns None if slot was acquired, or Some(wait_time) if we need to wait.
  pub fn check_and_record(&mut self) -> Option<Duration> {
    let wait = self.check_and_wait_time();
    if wait.is_none() {
      self.record_request();
    }
    wait
  }

  /// Get the current request count in the window
  fn current_count(&mut self) -> usize {
    self.prune_expired();
    self.request_times.len()
  }
}

/// A rate-limited embedding provider that wraps another provider
pub struct RateLimitedProvider<P: EmbeddingProvider> {
  inner: P,
  limiter: Arc<Mutex<SlidingWindowLimiter>>,
  config: RateLimitConfig,
}

impl<P: EmbeddingProvider> RateLimitedProvider<P> {
  pub fn new(provider: P) -> Self {
    let config = RateLimitConfig::default();
    Self {
      inner: provider,
      limiter: Arc::new(Mutex::new(SlidingWindowLimiter::new(config.clone()))),
      config,
    }
  }

  pub fn with_config(provider: P, config: RateLimitConfig) -> Self {
    Self {
      inner: provider,
      limiter: Arc::new(Mutex::new(SlidingWindowLimiter::new(config.clone()))),
      config,
    }
  }

  /// Wait for a rate limit slot, returning error if max wait time exceeded
  async fn acquire_slot(&self) -> Result<(), EmbeddingError> {
    let start = Instant::now();

    loop {
      let wait_time = {
        let mut limiter = self.limiter.lock().await;
        limiter.check_and_wait_time()
      };

      match wait_time {
        None => {
          // Slot available, record the request
          let mut limiter = self.limiter.lock().await;
          limiter.record_request();
          let current = limiter.current_count();
          debug!(
            current = current,
            max = self.config.max_requests,
            "Rate limit slot acquired"
          );
          return Ok(());
        }
        Some(wait) => {
          // Check if we've exceeded max wait time
          if start.elapsed() + wait > self.config.max_wait {
            warn!(
              max_wait_ms = self.config.max_wait.as_millis(),
              elapsed_ms = start.elapsed().as_millis(),
              "Rate limiter max wait time exceeded"
            );
            return Err(EmbeddingError::ProviderError(format!(
              "Rate limit wait time exceeded ({:?})",
              self.config.max_wait
            )));
          }

          debug!(
            wait_ms = wait.as_millis(),
            max_requests = self.config.max_requests,
            "Waiting for rate limit slot"
          );
          sleep(wait).await;
        }
      }
    }
  }
}

#[async_trait]
impl<P: EmbeddingProvider + Send + Sync> EmbeddingProvider for RateLimitedProvider<P> {
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
    self.acquire_slot().await?;
    self.inner.embed(text).await
  }

  async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    // Each batch request counts as one API call
    self.acquire_slot().await?;
    self.inner.embed_batch(texts).await
  }

  async fn is_available(&self) -> bool {
    self.inner.is_available().await
  }
}

/// Wrap any embedding provider with rate limiting
pub fn wrap_rate_limited<P: EmbeddingProvider>(provider: P, config: RateLimitConfig) -> RateLimitedProvider<P> {
  RateLimitedProvider::with_config(provider, config)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::atomic::{AtomicUsize, Ordering};

  #[test]
  fn test_rate_limit_config_defaults() {
    let config = RateLimitConfig::default();
    assert_eq!(config.max_requests, 70);
    assert_eq!(config.window, Duration::from_secs(10));
  }

  #[test]
  fn test_rate_limit_config_openrouter() {
    let config = RateLimitConfig::for_openrouter();
    assert_eq!(config.max_requests, 50); // 50/10s with safety margin (actual limit is 70)
    assert_eq!(config.window, Duration::from_secs(10));
  }

  #[test]
  fn test_sliding_window_under_limit() {
    let config = RateLimitConfig::new(5, Duration::from_secs(1));
    let mut limiter = SlidingWindowLimiter::new(config);

    // First 5 requests should go through immediately
    for _ in 0..5 {
      assert!(limiter.check_and_wait_time().is_none());
      limiter.record_request();
    }
  }

  #[test]
  fn test_sliding_window_at_limit() {
    let config = RateLimitConfig::new(5, Duration::from_secs(10));
    let mut limiter = SlidingWindowLimiter::new(config);

    // Fill up the window
    for _ in 0..5 {
      limiter.record_request();
    }

    // 6th request should need to wait
    let wait = limiter.check_and_wait_time();
    assert!(wait.is_some());
    assert!(wait.unwrap() <= Duration::from_secs(10));
  }

  #[test]
  fn test_sliding_window_prune_expired() {
    let config = RateLimitConfig::new(5, Duration::from_millis(10));
    let mut limiter = SlidingWindowLimiter::new(config);

    // Add some requests
    for _ in 0..5 {
      limiter.record_request();
    }
    assert_eq!(limiter.current_count(), 5);

    // Wait for window to expire
    std::thread::sleep(Duration::from_millis(15));

    // Requests should be pruned
    assert_eq!(limiter.current_count(), 0);
  }

  // Mock provider for testing
  struct MockProvider {
    call_count: AtomicUsize,
  }

  impl MockProvider {
    fn new() -> Self {
      Self {
        call_count: AtomicUsize::new(0),
      }
    }
  }

  #[async_trait]
  impl EmbeddingProvider for MockProvider {
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
      self.call_count.fetch_add(1, Ordering::SeqCst);
      Ok(vec![0.1; 384])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
      self.call_count.fetch_add(1, Ordering::SeqCst);
      Ok(texts.iter().map(|_| vec![0.1; 384]).collect())
    }

    async fn is_available(&self) -> bool {
      true
    }
  }

  #[tokio::test]
  async fn test_rate_limited_provider_passthrough() {
    let provider = MockProvider::new();
    let config = RateLimitConfig::new(10, Duration::from_secs(1));
    let limited = RateLimitedProvider::with_config(provider, config);

    // Should pass through to inner provider
    let result = limited.embed("test").await;
    assert!(result.is_ok());
    assert_eq!(limited.inner.call_count.load(Ordering::SeqCst), 1);
  }

  #[tokio::test]
  async fn test_rate_limited_provider_batch() {
    let provider = MockProvider::new();
    let config = RateLimitConfig::new(10, Duration::from_secs(1));
    let limited = RateLimitedProvider::with_config(provider, config);

    let texts = vec!["a", "b", "c"];
    let result = limited.embed_batch(&texts).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 3);
    // Batch counts as one call
    assert_eq!(limited.inner.call_count.load(Ordering::SeqCst), 1);
  }

  #[tokio::test]
  async fn test_rate_limited_respects_limit() {
    let provider = MockProvider::new();
    // Very restrictive: 3 requests per 100ms
    let config = RateLimitConfig::new(3, Duration::from_millis(100)).with_max_wait(Duration::from_millis(500));
    let limited = RateLimitedProvider::with_config(provider, config);

    let start = Instant::now();

    // First 3 should be immediate
    for _ in 0..3 {
      limited.embed("test").await.unwrap();
    }

    let after_first_batch = start.elapsed();
    assert!(after_first_batch < Duration::from_millis(50), "First 3 should be fast");

    // 4th should wait
    limited.embed("test").await.unwrap();

    let after_fourth = start.elapsed();
    assert!(
      after_fourth >= Duration::from_millis(100),
      "4th request should have waited"
    );
  }

  #[tokio::test]
  async fn test_rate_limited_max_wait_exceeded() {
    let provider = MockProvider::new();
    // Very restrictive with short max wait
    let config = RateLimitConfig::new(1, Duration::from_secs(10)).with_max_wait(Duration::from_millis(10));
    let limited = RateLimitedProvider::with_config(provider, config);

    // First request should succeed
    limited.embed("test").await.unwrap();

    // Second request should fail (can't wait 10s with 10ms max wait)
    let result = limited.embed("test").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Rate limit"));
  }
}
