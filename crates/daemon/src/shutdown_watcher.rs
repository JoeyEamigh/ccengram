//! Automatic shutdown watcher for daemon lifecycle management.
//!
//! Periodically checks if the daemon should shut down based on:
//! - No active sessions
//! - Idle timeout exceeded

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::{debug, info};

use crate::activity_tracker::ActivityTracker;
use crate::server::ShutdownHandle;
use crate::session_tracker::SessionTracker;

/// Watches for conditions that trigger automatic daemon shutdown.
///
/// Runs a background loop that:
/// 1. Cleans up stale sessions periodically
/// 2. Triggers shutdown when no sessions are active AND idle timeout is exceeded
pub struct ShutdownWatcher {
  session_tracker: Arc<SessionTracker>,
  activity_tracker: Arc<ActivityTracker>,
  shutdown_handle: ShutdownHandle,
  idle_timeout: Duration,
  check_interval: Duration,
}

impl ShutdownWatcher {
  /// Create a new shutdown watcher.
  ///
  /// # Arguments
  /// * `session_tracker` - Tracks active Claude Code sessions
  /// * `activity_tracker` - Tracks daemon activity for idle detection
  /// * `shutdown_handle` - Handle to trigger daemon shutdown
  /// * `idle_timeout_secs` - Seconds of idle time before shutdown (0 = immediate)
  pub fn new(
    session_tracker: Arc<SessionTracker>,
    activity_tracker: Arc<ActivityTracker>,
    shutdown_handle: ShutdownHandle,
    idle_timeout_secs: u64,
  ) -> Self {
    Self {
      session_tracker,
      activity_tracker,
      shutdown_handle,
      idle_timeout: Duration::from_secs(idle_timeout_secs),
      check_interval: Duration::from_secs(30),
    }
  }

  /// Set a custom check interval (primarily for testing).
  #[cfg(test)]
  pub fn with_check_interval(mut self, interval: Duration) -> Self {
    self.check_interval = interval;
    self
  }

  /// Run the shutdown watcher loop.
  ///
  /// This should be spawned as a background task. It returns when:
  /// - Shutdown is triggered (either by this watcher or externally)
  /// - An external shutdown signal is received on shutdown_rx
  pub async fn run(self, mut shutdown_rx: broadcast::Receiver<()>) {
    let mut interval = tokio::time::interval(self.check_interval);

    info!(
      idle_timeout_secs = self.idle_timeout.as_secs(),
      check_interval_secs = self.check_interval.as_secs(),
      "ShutdownWatcher started"
    );

    loop {
      tokio::select! {
          _ = interval.tick() => {
              if self.check_shutdown().await {
                  break;
              }
          }
          _ = shutdown_rx.recv() => {
              debug!("ShutdownWatcher received external shutdown signal");
              break;
          }
      }
    }

    info!("ShutdownWatcher stopped");
  }

  /// Check if shutdown conditions are met and trigger if so.
  ///
  /// Returns true if shutdown was triggered.
  async fn check_shutdown(&self) -> bool {
    // 1. Cleanup stale sessions
    let stale = self.session_tracker.cleanup_stale().await;
    if !stale.is_empty() {
      debug!(count = stale.len(), "Cleaned up stale sessions");
    }

    // 2. Check if we should shutdown
    let has_sessions = self.session_tracker.has_active_sessions().await;
    let idle_duration = self.activity_tracker.idle_duration();
    let active_count = self.session_tracker.active_count().await;

    debug!(
      has_sessions = has_sessions,
      active_count = active_count,
      idle_secs = idle_duration.as_secs(),
      timeout_secs = self.idle_timeout.as_secs(),
      "Shutdown check"
    );

    // Only shutdown if:
    // - No active sessions AND
    // - Idle for longer than timeout
    if !has_sessions && idle_duration >= self.idle_timeout {
      info!(
        idle_secs = idle_duration.as_secs(),
        "No active sessions and idle timeout reached, shutting down"
      );
      self.shutdown_handle.shutdown();
      return true;
    }

    // Log when approaching timeout (within 60 seconds)
    if !has_sessions && idle_duration.as_secs() + 60 >= self.idle_timeout.as_secs() && self.idle_timeout.as_secs() > 60
    {
      debug!(
        idle_secs = idle_duration.as_secs(),
        timeout_secs = self.idle_timeout.as_secs(),
        remaining_secs = self.idle_timeout.as_secs().saturating_sub(idle_duration.as_secs()),
        "Idle timeout approaching"
      );
    }

    false
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::session_tracker::SessionId;

  fn create_test_components(
    session_timeout_secs: u64,
  ) -> (
    Arc<SessionTracker>,
    Arc<ActivityTracker>,
    ShutdownHandle,
    broadcast::Sender<()>,
  ) {
    let session_tracker = Arc::new(SessionTracker::new(session_timeout_secs));
    let activity_tracker = Arc::new(ActivityTracker::new());
    let (shutdown_tx, _) = broadcast::channel(1);
    let shutdown_handle = ShutdownHandle::from_sender(shutdown_tx.clone());
    (session_tracker, activity_tracker, shutdown_handle, shutdown_tx)
  }

  #[tokio::test]
  async fn test_no_shutdown_with_active_sessions() {
    let (session_tracker, activity_tracker, shutdown_handle, _) = create_test_components(1800);

    // Register a session
    session_tracker.register(SessionId::from("test-session")).await;

    let watcher = ShutdownWatcher::new(
      Arc::clone(&session_tracker),
      Arc::clone(&activity_tracker),
      shutdown_handle,
      0, // Immediate shutdown when idle
    );

    // Check should not trigger shutdown because we have an active session
    let triggered = watcher.check_shutdown().await;
    assert!(!triggered);
  }

  #[tokio::test]
  async fn test_shutdown_when_no_sessions_and_idle() {
    let (session_tracker, activity_tracker, shutdown_handle, _shutdown_tx) = create_test_components(1800);

    let watcher = ShutdownWatcher::new(
      Arc::clone(&session_tracker),
      Arc::clone(&activity_tracker),
      shutdown_handle,
      0, // Immediate shutdown when idle
    );

    // No sessions registered, idle timeout is 0
    let triggered = watcher.check_shutdown().await;
    assert!(triggered);

    // Note: We don't verify the shutdown signal here because the shutdown_handle.shutdown()
    // was already called before we could subscribe. The triggered=true return value confirms
    // the shutdown logic executed.
  }

  #[tokio::test]
  async fn test_no_shutdown_when_not_idle_enough() {
    let (session_tracker, activity_tracker, shutdown_handle, _) = create_test_components(1800);

    let watcher = ShutdownWatcher::new(
      Arc::clone(&session_tracker),
      Arc::clone(&activity_tracker),
      shutdown_handle,
      3600, // 1 hour timeout - won't be reached
    );

    // No sessions, but not idle long enough
    let triggered = watcher.check_shutdown().await;
    assert!(!triggered);
  }

  #[tokio::test]
  async fn test_stale_session_cleanup() {
    let (session_tracker, activity_tracker, shutdown_handle, _) = create_test_components(0); // 0 session timeout = immediate stale

    // Register a session
    session_tracker.register(SessionId::from("stale-session")).await;

    // Wait a tiny bit so the session becomes stale
    tokio::time::sleep(Duration::from_millis(10)).await;

    let watcher = ShutdownWatcher::new(
      Arc::clone(&session_tracker),
      Arc::clone(&activity_tracker),
      shutdown_handle,
      0,
    );

    // This should clean up the stale session AND trigger shutdown
    let triggered = watcher.check_shutdown().await;
    assert!(triggered);

    // Session should be cleaned up
    assert!(!session_tracker.has_active_sessions().await);
  }

  #[tokio::test]
  async fn test_external_shutdown_signal_stops_watcher() {
    let (session_tracker, activity_tracker, shutdown_handle, shutdown_tx) = create_test_components(1800);

    // Register a session so it won't auto-shutdown
    session_tracker.register(SessionId::from("active-session")).await;

    let watcher = ShutdownWatcher::new(session_tracker, activity_tracker, shutdown_handle, 3600)
      .with_check_interval(Duration::from_millis(50));

    let shutdown_rx = shutdown_tx.subscribe();

    // Spawn watcher in background
    let watcher_handle = tokio::spawn(async move {
      watcher.run(shutdown_rx).await;
    });

    // Give watcher time to start
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Send external shutdown signal
    let _ = shutdown_tx.send(());

    // Watcher should stop
    tokio::time::timeout(Duration::from_millis(200), watcher_handle)
      .await
      .expect("Watcher should stop within timeout")
      .expect("Watcher task should complete");
  }
}
