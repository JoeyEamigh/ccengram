use crate::activity_tracker::ActivityTracker;
use crate::projects::ProjectRegistry;
use crate::router::Router;
use crate::scheduler::{SchedulerConfig, spawn_scheduler_with_config};
use crate::server::{Server, ShutdownHandle};
use crate::session_tracker::SessionTracker;
use crate::shutdown_watcher::ShutdownWatcher;
use embedding::{EmbeddingProvider, OllamaProvider, OpenRouterProvider};
use engram_core::{Config, ConfigEmbeddingProvider, EmbeddingConfig, HooksConfig};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum LifecycleError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Server error: {0}")]
  Server(#[from] crate::server::ServerError),
}

/// Daemon configuration
#[derive(Debug, Clone)]
pub struct DaemonConfig {
  /// Socket path for IPC
  pub socket_path: PathBuf,
  /// Data directory for storage
  pub data_dir: PathBuf,
  /// Idle timeout in seconds before auto-shutdown (0 = immediate after last session)
  pub idle_timeout_secs: u64,
  /// Session timeout in seconds - sessions without activity are considered dead
  pub session_timeout_secs: u64,
  /// Whether to run in foreground mode (disables auto-shutdown)
  pub foreground: bool,
  /// Embedding provider configuration
  pub embedding: EmbeddingConfig,
  /// Log retention in days (0 = keep forever)
  pub log_retention_days: u64,
  /// Hook behavior configuration
  pub hooks: HooksConfig,
}

impl Default for DaemonConfig {
  fn default() -> Self {
    // Load from effective config if available
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config = Config::load_for_project(&cwd);

    Self {
      socket_path: crate::server::default_socket_path(),
      data_dir: db::default_data_dir(),
      idle_timeout_secs: config.daemon.idle_timeout_secs,
      session_timeout_secs: config.daemon.session_timeout_secs,
      foreground: false,
      embedding: config.embedding,
      log_retention_days: config.daemon.log_retention_days,
      hooks: config.hooks,
    }
  }
}

impl DaemonConfig {
  /// Create a new config with foreground mode enabled
  pub fn foreground() -> Self {
    Self {
      foreground: true,
      ..Self::default()
    }
  }

  /// Create a new config for background mode (auto-shutdown enabled)
  pub fn background() -> Self {
    Self {
      foreground: false,
      ..Self::default()
    }
  }
}

/// Create an embedding provider from config
fn create_embedding_provider(config: &EmbeddingConfig) -> Arc<dyn EmbeddingProvider> {
  match config.provider {
    ConfigEmbeddingProvider::Ollama => {
      let mut provider = OllamaProvider::new()
        .with_url(&config.ollama_url)
        .with_model(&config.model, config.dimensions)
        .with_context_length(config.context_length);

      // Apply explicit max_batch_size if configured
      if let Some(max_batch_size) = config.max_batch_size {
        provider = provider.with_max_batch_size(max_batch_size);
      }

      Arc::new(provider)
    }
    ConfigEmbeddingProvider::OpenRouter => {
      // Try config first, then env var
      let api_key = config
        .openrouter_api_key
        .clone()
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .unwrap_or_default();

      if api_key.is_empty() {
        warn!("OpenRouter API key not configured, falling back to Ollama");
        let mut provider = OllamaProvider::new()
          .with_url(&config.ollama_url)
          .with_model(&config.model, config.dimensions)
          .with_context_length(config.context_length);

        if let Some(max_batch_size) = config.max_batch_size {
          provider = provider.with_max_batch_size(max_batch_size);
        }

        Arc::new(provider)
      } else {
        let provider = OpenRouterProvider::new(api_key).with_model(&config.model, config.dimensions);
        Arc::new(provider)
      }
    }
  }
}

/// Daemon lifecycle manager
pub struct Daemon {
  config: DaemonConfig,
  registry: Arc<ProjectRegistry>,
  shutdown: Option<ShutdownHandle>,
  scheduler_shutdown_tx: Option<broadcast::Sender<()>>,
  /// Session tracker for lifecycle management
  session_tracker: Arc<SessionTracker>,
  /// Activity tracker for idle detection
  activity_tracker: Arc<ActivityTracker>,
}

impl Daemon {
  pub fn new(config: DaemonConfig) -> Self {
    let registry = Arc::new(ProjectRegistry::with_data_dir(config.data_dir.clone()));
    let session_tracker = Arc::new(SessionTracker::new(config.session_timeout_secs));
    let activity_tracker = Arc::new(ActivityTracker::new());

    Self {
      config,
      registry,
      shutdown: None,
      scheduler_shutdown_tx: None,
      session_tracker,
      activity_tracker,
    }
  }

  /// Run the daemon
  pub async fn run(&mut self) -> Result<(), LifecycleError> {
    info!("Starting CCEngram daemon");
    info!("Socket: {:?}", self.config.socket_path);
    info!("Data dir: {:?}", self.config.data_dir);
    info!(
      "Mode: {}",
      if self.config.foreground {
        "foreground (auto-shutdown disabled)"
      } else {
        "background (auto-shutdown enabled)"
      }
    );

    // Create embedding provider from config
    let embedding = create_embedding_provider(&self.config.embedding);
    info!(
      "Using embedding provider: {} ({}, {} dims)",
      embedding.name(),
      embedding.model_id(),
      embedding.dimensions()
    );

    // Check if embedding provider is available
    if embedding.is_available().await {
      info!("Embedding provider is available");
    } else {
      warn!("Embedding provider is not available - falling back to text search");
    }

    // Create router with our registry, embedding provider, and configs
    let router = Router::with_embedding_and_config(
      Arc::clone(&self.registry),
      embedding,
      &self.config.hooks,
      &self.config.embedding,
    );
    let router = Arc::new(router);

    // Log hooks configuration
    if !self.config.hooks.enabled {
      info!("Automatic memory capture is DISABLED");
    } else {
      if !self.config.hooks.llm_extraction {
        info!("LLM extraction is disabled, using basic summary extraction");
      }
      if !self.config.hooks.tool_observations {
        info!("Tool observation memories are disabled");
      }
    }

    // Create server
    let server = Server::with_socket_path(Arc::clone(&router), self.config.socket_path.clone());
    let shutdown = server.shutdown_handle();
    self.shutdown = Some(shutdown.clone());

    // Give the router the shutdown handle and trackers
    router.set_shutdown_handle(shutdown.clone()).await;
    router.set_session_tracker(Arc::clone(&self.session_tracker)).await;
    router.set_activity_tracker(Arc::clone(&self.activity_tracker)).await;

    // Create shutdown channel for scheduler and watcher
    let (scheduler_shutdown_tx, scheduler_shutdown_rx) = broadcast::channel(1);
    self.scheduler_shutdown_tx = Some(scheduler_shutdown_tx.clone());

    // Spawn the background scheduler for decay and cleanup with log retention config
    let scheduler_config = SchedulerConfig {
      log_retention_days: self.config.log_retention_days,
      ..SchedulerConfig::default()
    };
    let _scheduler_handle =
      spawn_scheduler_with_config(Arc::clone(&self.registry), scheduler_shutdown_rx, scheduler_config);
    info!(
      "Started background scheduler (log retention: {} days)",
      self.config.log_retention_days
    );

    // Spawn shutdown watcher only in background mode
    let watcher_handle = if !self.config.foreground {
      let watcher = ShutdownWatcher::new(
        Arc::clone(&self.session_tracker),
        Arc::clone(&self.activity_tracker),
        shutdown.clone(),
        self.config.idle_timeout_secs,
      );
      let watcher_shutdown_rx = scheduler_shutdown_tx.subscribe();
      info!(
        "Auto-shutdown enabled: idle timeout {} seconds",
        self.config.idle_timeout_secs
      );
      Some(tokio::spawn(async move {
        watcher.run(watcher_shutdown_rx).await;
      }))
    } else {
      info!("Foreground mode: auto-shutdown disabled");
      None
    };

    // Handle ctrl-c gracefully
    let shutdown_clone = shutdown.clone();
    let scheduler_tx = scheduler_shutdown_tx;
    tokio::spawn(async move {
      if let Err(e) = signal::ctrl_c().await {
        warn!("Failed to listen for ctrl-c: {}", e);
        return;
      }
      info!("Received ctrl-c, shutting down...");
      let _ = scheduler_tx.send(());
      shutdown_clone.shutdown();
    });

    // Run server
    server.run().await?;

    // Cleanup: wait for watcher to stop
    if let Some(handle) = watcher_handle {
      let _ = handle.await;
    }

    // Cleanup: stop watchers first, then close connections
    self.registry.stop_all_watchers().await;
    self.registry.close_all().await;
    info!("Daemon shutdown complete");

    Ok(())
  }

  /// Shutdown the daemon
  pub fn shutdown(&self) {
    if let Some(ref shutdown) = self.shutdown {
      shutdown.shutdown();
    }
  }

  /// Get the project registry
  pub fn registry(&self) -> Arc<ProjectRegistry> {
    Arc::clone(&self.registry)
  }

  /// Get the session tracker
  pub fn session_tracker(&self) -> Arc<SessionTracker> {
    Arc::clone(&self.session_tracker)
  }

  /// Get the activity tracker
  pub fn activity_tracker(&self) -> Arc<ActivityTracker> {
    Arc::clone(&self.activity_tracker)
  }
}

/// Check if daemon is already running
pub fn is_running(socket_path: &std::path::Path) -> bool {
  // Try to connect to the socket
  std::os::unix::net::UnixStream::connect(socket_path).is_ok()
}

/// Get the PID file path
pub fn pid_file_path() -> PathBuf {
  if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
    PathBuf::from(runtime_dir).join("ccengram.pid")
  } else {
    let uid = std::process::id();
    PathBuf::from(format!("/tmp/{}.pid", uid))
  }
}

/// Write PID file
pub fn write_pid_file() -> Result<(), std::io::Error> {
  let pid_path = pid_file_path();
  std::fs::write(&pid_path, std::process::id().to_string())
}

/// Remove PID file
pub fn remove_pid_file() {
  let pid_path = pid_file_path();
  let _ = std::fs::remove_file(pid_path);
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_default_config() {
    let config = DaemonConfig::default();
    assert!(!config.socket_path.to_string_lossy().is_empty());
    // Default idle timeout comes from config (5 minutes = 300 seconds)
    assert_eq!(config.idle_timeout_secs, 300);
    // Default session timeout is 30 minutes = 1800 seconds
    assert_eq!(config.session_timeout_secs, 1800);
    // Default is background mode (not foreground)
    assert!(!config.foreground);
  }

  #[test]
  fn test_foreground_config() {
    let config = DaemonConfig::foreground();
    assert!(config.foreground);
  }

  #[test]
  fn test_background_config() {
    let config = DaemonConfig::background();
    assert!(!config.foreground);
  }

  #[test]
  fn test_is_running_no_socket() {
    let dir = TempDir::new().unwrap();
    let socket_path = dir.path().join("nonexistent.sock");
    assert!(!is_running(&socket_path));
  }
}
