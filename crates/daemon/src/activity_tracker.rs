//! Activity tracking for daemon idle detection.
//!
//! Tracks the last activity time to determine when the daemon
//! has been idle long enough to auto-shutdown.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::trace;

/// Tracks daemon activity for idle timeout detection.
///
/// Uses atomic operations for thread-safe, lock-free activity tracking.
/// The touch() method should be called on every RPC request.
pub struct ActivityTracker {
  /// Last activity timestamp (Unix millis for atomic storage)
  last_activity_millis: AtomicU64,

  /// Daemon start time (for uptime calculation)
  started_at: Instant,
}

impl ActivityTracker {
  /// Create a new activity tracker. Sets initial activity time to now.
  pub fn new() -> Self {
    let now_millis = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_millis() as u64;

    Self {
      last_activity_millis: AtomicU64::new(now_millis),
      started_at: Instant::now(),
    }
  }

  /// Record activity (called on any RPC request).
  pub fn touch(&self) {
    let now_millis = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_millis() as u64;
    self.last_activity_millis.store(now_millis, Ordering::Relaxed);
    trace!("Activity recorded");
  }

  /// Get duration since last activity.
  pub fn idle_duration(&self) -> Duration {
    let last_millis = self.last_activity_millis.load(Ordering::Relaxed);
    let now_millis = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_millis() as u64;
    Duration::from_millis(now_millis.saturating_sub(last_millis))
  }

  /// Get daemon uptime.
  pub fn uptime(&self) -> Duration {
    self.started_at.elapsed()
  }

  /// Get the last activity timestamp as Unix milliseconds.
  pub fn last_activity_millis(&self) -> u64 {
    self.last_activity_millis.load(Ordering::Relaxed)
  }
}

impl Default for ActivityTracker {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_new_tracker_has_recent_activity() {
    let tracker = ActivityTracker::new();
    let idle = tracker.idle_duration();

    // Should be very recently active (within 100ms)
    assert!(idle.as_millis() < 100);
  }

  #[test]
  fn test_touch_updates_activity() {
    let tracker = ActivityTracker::new();
    let initial_millis = tracker.last_activity_millis();

    // Sleep briefly
    std::thread::sleep(Duration::from_millis(10));

    tracker.touch();
    let after_touch_millis = tracker.last_activity_millis();

    assert!(after_touch_millis >= initial_millis);
  }

  #[test]
  fn test_idle_duration_increases() {
    let tracker = ActivityTracker::new();

    // Sleep briefly
    std::thread::sleep(Duration::from_millis(50));

    let idle = tracker.idle_duration();
    assert!(idle.as_millis() >= 50);
  }

  #[test]
  fn test_touch_resets_idle() {
    let tracker = ActivityTracker::new();

    // Sleep to create idle time
    std::thread::sleep(Duration::from_millis(50));

    // Touch to reset
    tracker.touch();

    // Idle should be very small now
    let idle = tracker.idle_duration();
    assert!(idle.as_millis() < 20);
  }

  #[test]
  fn test_uptime_increases() {
    let tracker = ActivityTracker::new();

    std::thread::sleep(Duration::from_millis(50));

    let uptime = tracker.uptime();
    assert!(uptime.as_millis() >= 50);
  }

  #[test]
  fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let tracker = Arc::new(ActivityTracker::new());
    let mut handles = vec![];

    // Spawn multiple threads touching the tracker
    for _ in 0..10 {
      let tracker = Arc::clone(&tracker);
      handles.push(thread::spawn(move || {
        for _ in 0..100 {
          tracker.touch();
        }
      }));
    }

    for handle in handles {
      handle.join().unwrap();
    }

    // Tracker should still be functional
    let idle = tracker.idle_duration();
    assert!(idle.as_millis() < 100);
  }
}
