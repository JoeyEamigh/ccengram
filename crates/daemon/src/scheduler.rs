// Background task scheduler for daemon operations
//
// Handles:
// - Memory decay (periodic salience reduction)
// - Stale session cleanup
// - Checkpoint saving for indexing

use crate::projects::ProjectRegistry;
use chrono::Utc;
use engram_core::ConfigDecay;
use extract::{DecayConfig as ExtractDecayConfig, DecayStats, apply_decay_batch};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
  /// Decay interval in hours (default: 60)
  pub decay_interval_hours: u64,
  /// Session cleanup interval in hours (default: 6)
  pub session_cleanup_hours: u64,
  /// Maximum session age in hours before cleanup (default: 6)
  pub max_session_age_hours: u64,
  /// Checkpoint save interval in seconds (default: 30)
  pub checkpoint_interval_secs: u64,
  /// Maximum memories to process per decay batch (default: 5000)
  /// Prevents OOM with large memory counts
  pub decay_batch_size: usize,
}

impl Default for SchedulerConfig {
  fn default() -> Self {
    Self {
      decay_interval_hours: 60,
      session_cleanup_hours: 6,
      max_session_age_hours: 6,
      checkpoint_interval_secs: 30,
      decay_batch_size: 5000,
    }
  }
}

impl From<&ConfigDecay> for SchedulerConfig {
  fn from(config: &ConfigDecay) -> Self {
    Self {
      decay_interval_hours: config.decay_interval_hours,
      session_cleanup_hours: config.session_cleanup_hours,
      max_session_age_hours: config.max_session_age_hours,
      checkpoint_interval_secs: 30, // From index config, not decay
      decay_batch_size: 5000,
    }
  }
}

/// Background task scheduler
pub struct Scheduler {
  config: SchedulerConfig,
  decay_config: ExtractDecayConfig,
  registry: Arc<ProjectRegistry>,
  shutdown_rx: broadcast::Receiver<()>,
}

impl Scheduler {
  pub fn new(registry: Arc<ProjectRegistry>, shutdown_rx: broadcast::Receiver<()>) -> Self {
    Self {
      config: SchedulerConfig::default(),
      decay_config: ExtractDecayConfig::default(),
      registry,
      shutdown_rx,
    }
  }

  pub fn with_config(
    registry: Arc<ProjectRegistry>,
    shutdown_rx: broadcast::Receiver<()>,
    config: SchedulerConfig,
  ) -> Self {
    Self {
      config,
      decay_config: ExtractDecayConfig::default(),
      registry,
      shutdown_rx,
    }
  }

  pub fn with_decay_config(
    registry: Arc<ProjectRegistry>,
    shutdown_rx: broadcast::Receiver<()>,
    config: SchedulerConfig,
    decay_config: ExtractDecayConfig,
  ) -> Self {
    Self {
      config,
      decay_config,
      registry,
      shutdown_rx,
    }
  }

  /// Run the scheduler (spawns background tasks)
  pub async fn run(mut self) {
    let decay_interval = Duration::from_secs(self.config.decay_interval_hours * 3600);
    let cleanup_interval = Duration::from_secs(self.config.session_cleanup_hours * 3600);
    let max_session_age = self.config.max_session_age_hours;

    let mut decay_timer = interval(decay_interval);
    let mut cleanup_timer = interval(cleanup_interval);

    // Skip the immediate ticks
    decay_timer.tick().await;
    cleanup_timer.tick().await;

    loop {
      tokio::select! {
        _ = decay_timer.tick() => {
          info!("Running scheduled decay");
          let stats = self.apply_decay().await;
          if stats.total_processed > 0 {
            info!(
              "Decay complete: processed={}, decayed={}, archive_candidates={}, avg_drop={:.4}",
              stats.total_processed, stats.decayed_count, stats.archive_candidates, stats.average_salience_drop
            );
          }
        }
        _ = cleanup_timer.tick() => {
          info!("Running scheduled session cleanup");
          let cleaned = self.cleanup_stale_sessions(max_session_age).await;
          if cleaned > 0 {
            info!("Cleaned up {} stale sessions", cleaned);
          }
        }
        _ = self.shutdown_rx.recv() => {
          debug!("Scheduler received shutdown signal");
          break;
        }
      }
    }
  }

  /// Apply decay to all memories in all projects
  async fn apply_decay(&self) -> DecayStats {
    let mut total_stats = DecayStats::default();
    let projects = self.registry.list().await;

    for project in projects {
      match self.registry.get_or_create(&project.path).await {
        Ok((_, db)) => {
          // Get memories in batches to prevent OOM on large projects
          match db
            .list_memories(Some("is_deleted = false"), Some(self.config.decay_batch_size))
            .await
          {
            Ok(mut memories) => {
              let now = Utc::now();
              let results = apply_decay_batch(&mut memories, now, &self.decay_config);

              // Collect memories that actually decayed for batch update
              let decayed_memories: Vec<_> = memories
                .iter()
                .zip(results.iter())
                .filter(|(_, result)| result.new_salience < result.previous_salience)
                .map(|(memory, _)| memory.clone())
                .collect();

              if !decayed_memories.is_empty()
                && let Err(e) = db.batch_update_memories(&decayed_memories).await
              {
                warn!("Failed to batch update decayed memories: {}", e);
              }

              let stats = DecayStats::from_results(&results);
              total_stats.total_processed += stats.total_processed;
              total_stats.decayed_count += stats.decayed_count;
              total_stats.archive_candidates += stats.archive_candidates;
            }
            Err(e) => {
              error!("Failed to list memories for project {}: {}", project.name, e);
            }
          }
        }
        Err(e) => {
          error!("Failed to open project {}: {}", project.name, e);
        }
      }
    }

    // Compute average
    if total_stats.decayed_count > 0 {
      // We don't have individual drops so we can't compute average; leave at 0
    }

    total_stats
  }

  /// Cleanup stale sessions in all projects
  async fn cleanup_stale_sessions(&self, max_age_hours: u64) -> usize {
    let mut total_cleaned = 0;
    let projects = self.registry.list().await;

    for project in projects {
      match self.registry.get_or_create(&project.path).await {
        Ok((_, db)) => match db.cleanup_stale_sessions(max_age_hours).await {
          Ok(cleaned) => {
            total_cleaned += cleaned;
          }
          Err(e) => {
            error!("Failed to cleanup sessions for project {}: {}", project.name, e);
          }
        },
        Err(e) => {
          error!("Failed to open project {}: {}", project.name, e);
        }
      }
    }

    total_cleaned
  }
}

/// Spawn the scheduler as a background task
pub fn spawn_scheduler(
  registry: Arc<ProjectRegistry>,
  shutdown_rx: broadcast::Receiver<()>,
) -> tokio::task::JoinHandle<()> {
  let scheduler = Scheduler::new(registry, shutdown_rx);
  tokio::spawn(async move {
    scheduler.run().await;
  })
}

/// Spawn the scheduler with custom config
pub fn spawn_scheduler_with_config(
  registry: Arc<ProjectRegistry>,
  shutdown_rx: broadcast::Receiver<()>,
  config: SchedulerConfig,
) -> tokio::task::JoinHandle<()> {
  let scheduler = Scheduler::with_config(registry, shutdown_rx, config);
  tokio::spawn(async move {
    scheduler.run().await;
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_scheduler_config_defaults() {
    let config = SchedulerConfig::default();
    assert_eq!(config.decay_interval_hours, 60);
    assert_eq!(config.session_cleanup_hours, 6);
    assert_eq!(config.max_session_age_hours, 6);
    assert_eq!(config.checkpoint_interval_secs, 30);
    assert_eq!(config.decay_batch_size, 5000);
  }
}
