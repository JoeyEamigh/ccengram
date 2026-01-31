//! Session lifecycle tracking for daemon auto-shutdown.
//!
//! Tracks active Claude Code sessions to determine when the daemon
//! can safely shut down.

use std::{
  collections::{HashMap, HashSet},
  time::{Duration, Instant},
};

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Unique identifier for a Claude Code session
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct SessionId(pub String);

impl std::fmt::Display for SessionId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl From<String> for SessionId {
  fn from(s: String) -> Self {
    Self(s)
  }
}

impl From<&str> for SessionId {
  fn from(s: &str) -> Self {
    Self(s.to_string())
  }
}

/// Tracks active Claude Code sessions for daemon lifecycle management.
///
/// Sessions are registered on SessionStart hook and unregistered on SessionEnd.
/// Stale sessions (those that haven't been touched recently) are cleaned up
/// periodically to handle cases where SessionEnd wasn't received.
#[derive(Debug)]
pub struct SessionTracker {
  /// Currently active sessions
  active_sessions: RwLock<HashSet<SessionId>>,

  /// Last activity time per session
  session_last_seen: RwLock<HashMap<SessionId, Instant>>,

  /// Session timeout duration - sessions older than this are considered stale
  session_timeout: Duration,
}

impl SessionTracker {
  /// Create a new session tracker with the given session timeout in seconds.
  pub fn new(session_timeout_secs: u64) -> Self {
    Self {
      active_sessions: RwLock::new(HashSet::new()),
      session_last_seen: RwLock::new(HashMap::new()),
      session_timeout: Duration::from_secs(session_timeout_secs),
    }
  }

  /// Register a new session (called on SessionStart hook).
  pub async fn register(&self, session_id: SessionId) {
    let mut sessions = self.active_sessions.write().await;
    let mut last_seen = self.session_last_seen.write().await;

    sessions.insert(session_id.clone());
    last_seen.insert(session_id.clone(), Instant::now());

    info!(session_id = %session_id, "Session registered");
    debug!(active_count = sessions.len(), "Active sessions");
  }

  /// Unregister a session (called on SessionEnd hook).
  pub async fn unregister(&self, session_id: &SessionId) {
    let mut sessions = self.active_sessions.write().await;
    let mut last_seen = self.session_last_seen.write().await;

    sessions.remove(session_id);
    last_seen.remove(session_id);

    info!(session_id = %session_id, "Session unregistered");
    debug!(active_count = sessions.len(), "Active sessions");
  }

  /// Touch a session to update its last-seen timestamp (called on any hook from that session).
  pub async fn touch(&self, session_id: &SessionId) {
    let mut last_seen = self.session_last_seen.write().await;
    if let Some(ts) = last_seen.get_mut(session_id) {
      *ts = Instant::now();
      debug!(session_id = %session_id, "Session touched");
    }
  }

  /// Clean up stale sessions that haven't been seen within session_timeout.
  ///
  /// Returns the list of session IDs that were removed.
  pub async fn cleanup_stale(&self) -> Vec<SessionId> {
    let now = Instant::now();
    let mut stale = Vec::new();

    // First pass: identify stale sessions
    {
      let last_seen = self.session_last_seen.read().await;
      for (id, ts) in last_seen.iter() {
        let idle_duration = now.duration_since(*ts);
        if idle_duration > self.session_timeout {
          debug!(
            session_id = %id,
            idle_secs = idle_duration.as_secs(),
            timeout_secs = self.session_timeout.as_secs(),
            "Session timed out"
          );
          stale.push(id.clone());
        }
      }
    }

    // Second pass: remove stale sessions
    if !stale.is_empty() {
      let mut sessions = self.active_sessions.write().await;
      let mut last_seen = self.session_last_seen.write().await;

      for id in &stale {
        sessions.remove(id);
        last_seen.remove(id);
        warn!(session_id = %id, "Removed stale session");
      }
    }

    stale
  }

  /// Check if any sessions are currently active.
  pub async fn has_active_sessions(&self) -> bool {
    !self.active_sessions.read().await.is_empty()
  }

  /// Get the count of active sessions.
  pub async fn active_count(&self) -> usize {
    self.active_sessions.read().await.len()
  }

  /// Get a list of all active session IDs.
  pub async fn list_sessions(&self) -> Vec<SessionId> {
    let sessions: Vec<SessionId> = self.active_sessions.read().await.iter().cloned().collect();
    debug!(count = sessions.len(), "Listed active sessions");
    sessions
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_register_unregister() {
    let tracker = SessionTracker::new(1800);

    let session_id = SessionId::from("test-session-1");
    tracker.register(session_id.clone()).await;

    assert!(tracker.has_active_sessions().await);
    assert_eq!(tracker.active_count().await, 1);

    tracker.unregister(&session_id).await;

    assert!(!tracker.has_active_sessions().await);
    assert_eq!(tracker.active_count().await, 0);
  }

  #[tokio::test]
  async fn test_multiple_sessions() {
    let tracker = SessionTracker::new(1800);

    let session1 = SessionId::from("session-1");
    let session2 = SessionId::from("session-2");

    tracker.register(session1.clone()).await;
    tracker.register(session2.clone()).await;

    assert_eq!(tracker.active_count().await, 2);

    tracker.unregister(&session1).await;

    assert_eq!(tracker.active_count().await, 1);
    assert!(tracker.has_active_sessions().await);
  }

  #[tokio::test]
  async fn test_touch_updates_timestamp() {
    let tracker = SessionTracker::new(1800);

    let session_id = SessionId::from("test-session");
    tracker.register(session_id.clone()).await;

    // Touch should not fail
    tracker.touch(&session_id).await;

    assert!(tracker.has_active_sessions().await);
  }

  #[tokio::test]
  async fn test_cleanup_stale_sessions() {
    // Use a very short timeout for testing
    let tracker = SessionTracker::new(0);

    let session_id = SessionId::from("stale-session");
    tracker.register(session_id.clone()).await;

    // Wait a tiny bit so the session becomes stale
    tokio::time::sleep(Duration::from_millis(10)).await;

    let stale = tracker.cleanup_stale().await;

    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].0, "stale-session");
    assert!(!tracker.has_active_sessions().await);
  }

  #[tokio::test]
  async fn test_cleanup_preserves_recent_sessions() {
    let tracker = SessionTracker::new(3600); // 1 hour timeout

    let session_id = SessionId::from("recent-session");
    tracker.register(session_id.clone()).await;

    let stale = tracker.cleanup_stale().await;

    assert!(stale.is_empty());
    assert!(tracker.has_active_sessions().await);
  }

  #[tokio::test]
  async fn test_list_sessions() {
    let tracker = SessionTracker::new(1800);

    let session1 = SessionId::from("session-1");
    let session2 = SessionId::from("session-2");

    tracker.register(session1.clone()).await;
    tracker.register(session2.clone()).await;

    let sessions = tracker.list_sessions().await;
    assert_eq!(sessions.len(), 2);
  }
}
