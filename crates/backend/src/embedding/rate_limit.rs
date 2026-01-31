// Rate limiter for API providers with sliding window rate limiting
//
// Implements a sliding window rate limiter that tracks requests over a
// configurable time window and delays requests when the limit is reached.
//
// The limiter supports a token-based refund mechanism for failed requests
// that didn't actually consume API rate limit capacity (network errors,
// server errors, etc.).
//
// FifoRateLimiter provides FIFO ordering using tokio::sync::Semaphore,
// ensuring fair queuing under high load.

use std::{
  collections::HashMap,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
  time::Duration,
};

use tokio::{sync::Semaphore, task::AbortHandle};
use tracing::{debug, trace};

/// Token returned when recording a request, used for potential refunds.
///
/// When a request fails due to network errors or server errors (5xx),
/// the request likely didn't count against the API provider's rate limit.
/// Use this token to refund the slot and keep our local limiter accurate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitToken {
  id: u64,
}

impl RateLimitToken {
  fn new(id: u64) -> Self {
    Self { id }
  }
}

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
      max_requests: 65,
      window: Duration::from_secs(10),
      max_wait: Duration::from_secs(600),
    }
  }
}

impl RateLimitConfig {
  /// Create a config for OpenRouter (65 requests per 10s sliding window)
  /// OpenRouter's actual limit is 70/10s, but we use 65 for safety margin.
  /// max_wait is 600s to handle sustained high throughput during initial indexing.
  pub fn for_openrouter() -> Self {
    Self {
      max_requests: 65,
      window: Duration::from_secs(10),
      max_wait: Duration::from_secs(600),
    }
  }

  #[allow(dead_code)]
  /// Create a config with custom limits
  pub fn new(max_requests: usize, window: Duration) -> Self {
    Self {
      max_requests,
      window,
      max_wait: Duration::from_secs(600),
    }
  }
}

// ============================================================================
// FIFO Rate Limiter (recommended for production use)
// ============================================================================

/// FIFO rate limiter using tokio Semaphore for fair queuing.
///
/// This implementation guarantees FIFO ordering: requests are processed
/// in the order they arrive, preventing starvation under high load.
///
/// The sliding window is implemented by:
/// 1. Acquiring a semaphore permit (FIFO ordered)
/// 2. Forgetting the permit (not dropping it)
/// 3. Scheduling permit restoration after window duration
///
/// Refunds work by canceling the scheduled restoration and immediately
/// restoring the permit.
pub struct FifoRateLimiter {
  semaphore: Arc<Semaphore>,
  config: RateLimitConfig,
  next_token_id: AtomicU64,
  /// Track active tokens for refund support (maps token_id -> abort handle)
  active_tokens: tokio::sync::Mutex<HashMap<u64, AbortHandle>>,
}

impl FifoRateLimiter {
  pub fn new(config: RateLimitConfig) -> Self {
    debug!(
      max_requests = config.max_requests,
      window_ms = config.window.as_millis(),
      max_wait_ms = config.max_wait.as_millis(),
      "FIFO rate limiter initialized"
    );
    Self {
      semaphore: Arc::new(Semaphore::new(config.max_requests)),
      config,
      next_token_id: AtomicU64::new(0),
      active_tokens: tokio::sync::Mutex::new(HashMap::new()),
    }
  }

  /// Acquire a rate limit slot with FIFO ordering.
  ///
  /// Returns a token that can be used to refund the slot if the request
  /// fails without consuming API rate limit capacity.
  pub async fn acquire(&self) -> Result<RateLimitToken, super::EmbeddingError> {
    // Acquire permit with FIFO ordering (semaphore guarantees this)
    let permit = match tokio::time::timeout(self.config.max_wait, self.semaphore.acquire()).await {
      Ok(Ok(permit)) => permit,
      Ok(Err(_)) => return Err(super::EmbeddingError::ProviderError("Rate limiter closed".into())),
      Err(_) => return Err(super::EmbeddingError::RateLimitExhausted(self.config.max_wait)),
    };

    // Forget the permit - we'll restore it after window duration
    permit.forget();

    // Generate token ID
    let token_id = self.next_token_id.fetch_add(1, Ordering::Relaxed);

    // Schedule permit restoration after window duration
    let semaphore = self.semaphore.clone();
    let window = self.config.window;
    let handle = tokio::spawn(async move {
      tokio::time::sleep(window).await;
      semaphore.add_permits(1);
      trace!("Rate limit slot restored after window expiry");
    });

    // Store abort handle for potential refund
    {
      let mut tokens = self.active_tokens.lock().await;
      tokens.insert(token_id, handle.abort_handle());
    }

    trace!(token_id, "Rate limit slot acquired (FIFO)");
    Ok(RateLimitToken::new(token_id))
  }

  /// Refund a rate limit slot when a request fails without consuming API capacity.
  ///
  /// Call this for network errors, timeouts, and 5xx server errors.
  /// Do NOT call for 429 or other 4xx errors.
  pub async fn refund(&self, token: RateLimitToken) {
    let mut tokens = self.active_tokens.lock().await;
    if let Some(abort_handle) = tokens.remove(&token.id) {
      // Cancel the scheduled permit restoration
      abort_handle.abort();
      // Immediately restore the permit
      self.semaphore.add_permits(1);
      trace!(token_id = token.id, "Rate limit slot refunded");
    } else {
      trace!(
        token_id = token.id,
        "Rate limit refund: token not found (may have expired)"
      );
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_fifo_under_limit() {
    let config = RateLimitConfig::new(5, Duration::from_secs(1));
    let limiter = FifoRateLimiter::new(config);

    // First 5 requests should go through immediately
    for _ in 0..5 {
      let result = limiter.acquire().await;
      assert!(result.is_ok(), "Should acquire slot under limit");
    }
  }

  #[tokio::test]
  async fn test_fifo_refund_restores_capacity() {
    let config = RateLimitConfig {
      max_requests: 2,
      window: Duration::from_secs(10),
      max_wait: Duration::from_millis(100),
    };
    let limiter = FifoRateLimiter::new(config);

    // Fill to capacity
    let token1 = limiter.acquire().await.expect("first acquire");
    let _token2 = limiter.acquire().await.expect("second acquire");

    // Third should timeout (no slots available)
    let result3 = limiter.acquire().await;
    assert!(result3.is_err(), "Should fail when at capacity");

    // Refund first token
    limiter.refund(token1).await;

    // Now should succeed
    let result4 = limiter.acquire().await;
    assert!(result4.is_ok(), "Should succeed after refund");
  }

  #[tokio::test]
  async fn test_fifo_window_expiry() {
    let config = RateLimitConfig {
      max_requests: 2,
      window: Duration::from_millis(50),
      max_wait: Duration::from_millis(200),
    };
    let limiter = FifoRateLimiter::new(config);

    // Fill to capacity
    let _token1 = limiter.acquire().await.expect("first acquire");
    let _token2 = limiter.acquire().await.expect("second acquire");

    // Wait for window to expire
    tokio::time::sleep(Duration::from_millis(60)).await;

    // Should succeed now that slots have expired
    let result3 = limiter.acquire().await;
    assert!(result3.is_ok(), "Should succeed after window expiry");
  }
}
