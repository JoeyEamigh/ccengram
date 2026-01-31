use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use super::{
  lifecycle::{activity::KeepAlive, session::SessionTracker},
  router::ProjectRouter,
};
use crate::domain::config::{DaemonConfig, DecayConfig};

/// Configuration for idle shutdown behavior (background mode only).
///
/// When the daemon runs in background mode, it monitors for idle conditions
/// and automatically shuts down to conserve resources.
#[derive(Debug)]
pub struct IdleShutdownConfig {
  /// Seconds of idle time before triggering shutdown
  pub timeout_secs: u64,
  /// Activity tracker for idle detection
  pub activity: Arc<KeepAlive>,
  /// Session tracker for active session detection
  pub sessions: Arc<SessionTracker>,
}

/// Scheduler configuration for the actor-based daemon.
///
/// References the decay and daemon config sections directly instead of
/// duplicating values with hardcoded defaults.
#[derive(Debug)]
pub struct SchedulerConfig {
  /// Decay and memory lifecycle settings
  pub decay: DecayConfig,
  /// Daemon lifecycle settings (log retention, idle check interval, etc.)
  pub daemon: DaemonConfig,
  /// Optional idle shutdown configuration (background mode only)
  pub idle_shutdown: Option<IdleShutdownConfig>,
}

/// Background task scheduler for daemon operations.
///
/// Handles:
/// - Memory decay (periodic salience reduction)
/// - Stale session cleanup
/// - Log file rotation
/// - Idle shutdown check (background mode only)
///
/// This version uses `ProjectRouter` instead of `ProjectRegistry` and
/// `CancellationToken` instead of broadcast channels.
pub struct Scheduler {
  router: Arc<ProjectRouter>,
  config: SchedulerConfig,
}

impl Scheduler {
  /// Create a new scheduler.
  pub fn new(router: Arc<ProjectRouter>, config: SchedulerConfig) -> Self {
    Self { router, config }
  }

  /// Run the scheduler until cancelled.
  pub async fn run(self, cancel: CancellationToken) {
    use std::time::Duration;

    use tokio::time::interval;

    let decay_interval = Duration::from_secs(self.config.decay.decay_interval_hours * 3600);
    let cleanup_interval = Duration::from_secs(self.config.decay.session_cleanup_hours * 3600);
    let log_cleanup_interval = Duration::from_secs(24 * 3600); // Once per day
    let idle_check_interval = Duration::from_secs(self.config.daemon.idle_check_interval_secs);

    let mut decay_timer = interval(decay_interval);
    let mut cleanup_timer = interval(cleanup_interval);
    let mut log_cleanup_timer = interval(log_cleanup_interval);
    let mut idle_timer = interval(idle_check_interval);

    // Skip the immediate ticks
    decay_timer.tick().await;
    cleanup_timer.tick().await;
    log_cleanup_timer.tick().await;
    idle_timer.tick().await;

    // Run log cleanup once at startup if retention is enabled
    if self.config.daemon.log_retention_days > 0 {
      let deleted = self.cleanup_old_logs();
      if deleted > 0 {
        debug!("Cleaned up {} old log files at startup", deleted);
      }
    }

    info!("Scheduler started");

    loop {
      tokio::select! {
        biased;

        _ = cancel.cancelled() => {
          debug!("Scheduler shutting down (cancelled)");
          break;
        }

        _ = decay_timer.tick() => {
          debug!("Running scheduled decay");
          self.apply_decay().await;
        }

        _ = cleanup_timer.tick() => {
          debug!("Running scheduled session cleanup");
          self.cleanup_stale_sessions().await;
        }

        _ = log_cleanup_timer.tick() => {
          if self.config.daemon.log_retention_days > 0 {
            let deleted = self.cleanup_old_logs();
            if deleted > 0 {
              debug!("Cleaned up {} old log files", deleted);
            }
          }
        }

        _ = idle_timer.tick() => {
            if self.check_idle_shutdown(&cancel).await {
                break;
            }
        }
      }
    }

    info!("Scheduler stopped");
  }

  /// Apply memory decay to all projects.
  async fn apply_decay(&self) {
    let project_ids = self.router.list();
    if project_ids.is_empty() {
      return;
    }

    tracing::debug!("Applying decay to {} projects", project_ids.len());

    for id in &project_ids {
      if let Some(handle) = self.router.get(id) {
        match handle
          .request(format!("decay-{}", id), super::message::ProjectActorPayload::ApplyDecay)
          .await
        {
          Ok(_) => tracing::trace!(project_id = %id, "Decay applied"),
          Err(e) => tracing::warn!(project_id = %id, error = %e, "Failed to apply decay"),
        }
      }
    }
  }

  /// Cleanup stale sessions in all projects.
  async fn cleanup_stale_sessions(&self) {
    // Clean up stale sessions from the session tracker if configured
    if let Some(ref idle_config) = self.config.idle_shutdown {
      let stale = idle_config.sessions.cleanup_stale().await;
      if !stale.is_empty() {
        tracing::debug!("Cleaned up {} stale sessions from tracker", stale.len());
      }
    }

    // Send cleanup messages to all ProjectActors for DB session cleanup
    let project_ids = self.router.list();
    if project_ids.is_empty() {
      return;
    }

    let max_age_hours = self.config.decay.max_session_age_hours;
    tracing::debug!("Cleaning up stale sessions in {} projects", project_ids.len());

    for id in &project_ids {
      if let Some(handle) = self.router.get(id) {
        match handle
          .request(
            format!("cleanup-{}", id),
            super::message::ProjectActorPayload::CleanupSessions { max_age_hours },
          )
          .await
        {
          Ok(_) => tracing::trace!(project_id = %id, "Session cleanup complete"),
          Err(e) => {
            tracing::warn!(project_id = %id, error = %e, "Failed to cleanup sessions")
          }
        }
      }
    }
  }

  /// Cleanup old log files based on retention policy.
  fn cleanup_old_logs(&self) -> usize {
    use std::time::SystemTime;

    let retention_secs = self.config.daemon.log_retention_days * 24 * 3600;
    let now = SystemTime::now();
    let data_dir = crate::dirs::default_data_dir();
    let mut deleted = 0;

    let entries = match std::fs::read_dir(&data_dir) {
      Ok(e) => e,
      Err(e) => {
        tracing::warn!("Failed to read log directory {:?}: {}", data_dir, e);
        return 0;
      }
    };

    for entry in entries.flatten() {
      let path = entry.path();

      // Skip directories
      if path.is_dir() {
        continue;
      }

      // Only consider log files
      let file_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => continue,
      };

      if !file_name.starts_with("ccengram.log") {
        continue;
      }

      // Check file age
      let metadata = match entry.metadata() {
        Ok(m) => m,
        Err(_) => continue,
      };

      let modified = match metadata.modified() {
        Ok(t) => t,
        Err(_) => continue,
      };

      let age = match now.duration_since(modified) {
        Ok(d) => d,
        Err(_) => continue,
      };

      // Delete if older than retention period
      if age.as_secs() > retention_secs {
        if let Err(e) = std::fs::remove_file(&path) {
          tracing::warn!("Failed to delete old log file {:?}: {}", path, e);
        } else {
          tracing::debug!("Deleted old log file: {:?}", path);
          deleted += 1;
        }
      }
    }

    deleted
  }

  /// Check if idle shutdown conditions are met.
  ///
  /// Returns true if shutdown was triggered.
  async fn check_idle_shutdown(&self, cancel: &CancellationToken) -> bool {
    let Some(ref idle_config) = self.config.idle_shutdown else {
      return false;
    };

    // Clean up stale sessions first
    let stale = idle_config.sessions.cleanup_stale().await;
    if !stale.is_empty() {
      tracing::debug!("Cleaned up {} stale sessions during idle check", stale.len());
    }

    let has_sessions = idle_config.sessions.has_active_sessions().await;
    let idle_duration = idle_config.activity.idle_duration();
    let timeout = std::time::Duration::from_secs(idle_config.timeout_secs);

    tracing::trace!(
      has_sessions = has_sessions,
      idle_secs = idle_duration.as_secs(),
      timeout_secs = timeout.as_secs(),
      "Idle shutdown check"
    );

    // Only shutdown if:
    // - No active sessions AND
    // - Idle for longer than timeout
    if !has_sessions && idle_duration >= timeout {
      info!(
        idle_secs = idle_duration.as_secs(),
        "No active sessions and idle timeout reached, shutting down"
      );
      cancel.cancel();
      return true;
    }

    false
  }
}
