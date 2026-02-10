//! Daemon lifecycle management using the actor-based architecture.
//!
//! The daemon is the main entry point for the CCEngram background process.
//! It orchestrates all components using the actor model with message passing
//! instead of shared-state concurrency.
//!
//! # Architecture
//!
//! ```text
//! Daemon (Supervisor)
//!   ├── Server (IPC listener, spawns connection tasks)
//!   ├── Scheduler (decay, cleanup, log rotation, idle shutdown)
//!   └── ProjectRouter
//!         └── ProjectActor (per-project, spawned on demand)
//!               ├── IndexerActor
//!               └── WatcherTask
//! ```
//!
//! # Lifecycle
//!
//! 1. Create master `CancellationToken`
//! 2. Create embedding provider (shared, immutable)
//! 3. Create `ProjectRouter` with child token
//! 4. Create lifecycle trackers (activity, sessions)
//! 5. Create `Server` with all dependencies (no two-phase init)
//! 6. Spawn `Scheduler` for background tasks
//! 7. Run server until cancelled
//! 8. Graceful shutdown: cancel children, wait for tasks, shutdown projects

use std::{path::PathBuf, sync::Arc};

use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

#[cfg(feature = "llama-cpp")]
use crate::rerank::llamacpp::LlamaCppReranker;
use crate::{
  actor::{
    IdleShutdownConfig, ProjectRouter, Scheduler, SchedulerConfig,
    lifecycle::{activity::KeepAlive, session::SessionTracker},
  },
  dirs,
  domain::config::{Config, DaemonSettings, RerankerProviderKind},
  embedding::EmbeddingProvider,
  ipc::{Client, IpcError},
  rerank::{DeepInfraReranker, RerankerProvider},
  server::{DaemonState, Server, ServerConfig},
};

// ============================================================================
// Configuration
// ============================================================================

/// Daemon runtime configuration.
///
/// This struct contains all configuration needed to run the daemon.
/// It's constructed from the global config file with optional overrides.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
  /// Path to the Unix socket for IPC
  pub socket_path: PathBuf,
  /// Base directory for project data (databases, logs, etc.)
  pub data_dir: PathBuf,
  /// Run in foreground mode (disables auto-shutdown)
  pub foreground: bool,
  /// Full configuration (embedding, daemon, hooks, decay, etc.)
  pub config: Config,
}

impl RuntimeConfig {
  pub async fn load() -> Self {
    // Auto-create user config on first run if it doesn't exist
    Self::ensure_user_config().await;

    let config = Config::load_global().await;

    Self {
      socket_path: dirs::default_socket_path(),
      data_dir: dirs::default_data_dir(),
      foreground: false,
      config,
    }
  }

  /// Ensure user config file exists, creating it with defaults if not.
  async fn ensure_user_config() {
    use crate::domain::config::ToolPreset;

    let Some(user_config_path) = Config::user_config_path() else {
      return;
    };

    if user_config_path.exists() {
      return;
    }

    // Create parent directory if needed
    if let Some(parent) = user_config_path.parent()
      && let Err(e) = tokio::fs::create_dir_all(parent).await
    {
      warn!("Failed to create config directory: {}", e);
      return;
    }

    // Write default config
    let template = Config::generate_template(ToolPreset::Standard);
    match tokio::fs::write(&user_config_path, &template).await {
      Ok(()) => info!("Created user config: {:?}", user_config_path),
      Err(e) => warn!("Failed to create user config: {}", e),
    }
  }
}

/// The CCEngram daemon - manages the entire application lifecycle.
///
/// The daemon is responsible for:
/// - Starting and supervising all actors
/// - Managing the IPC server
/// - Coordinating graceful shutdown
///
/// # Usage
///
/// ```ignore
/// let daemon = Daemon::with_defaults();
/// daemon.run().await;
/// ```
pub struct Daemon {
  runtime_config: RuntimeConfig,
}

impl Daemon {
  /// Create a new daemon with the given configuration.
  pub fn new(runtime_config: RuntimeConfig) -> Self {
    Self { runtime_config }
  }

  /// Create a daemon with default configuration.
  pub async fn with_defaults() -> Self {
    Self::new(RuntimeConfig::load().await)
  }

  /// Connect to an existing daemon, starting one if necessary.
  ///
  /// This function:
  /// 1. Tries to connect to an existing daemon
  /// 2. If that fails, spawns a new daemon in background mode
  /// 3. Polls for the socket to become available (up to 5 seconds)
  /// 4. Returns a connected client
  ///
  /// # Errors
  ///
  /// Returns an error if:
  /// - The daemon fails to start within the timeout
  /// - Connection to the daemon fails after startup
  pub async fn connect_or_start(cwd: PathBuf) -> Result<Client, IpcError> {
    let running = dirs::is_daemon_running();
    if running {
      debug!("Daemon is already running, connecting...");
      return Client::connect(cwd).await;
    }

    info!("Daemon is not running, starting in background...");
    let pid = Self::spawn_background().await?;
    debug!("Spawned daemon with PID {}", pid);

    // Poll for socket to become available (up to 5 seconds)
    let delay = std::time::Duration::from_millis(500);
    tokio::time::sleep(delay).await;

    let socket_path = dirs::default_socket_path();
    let mut attempts = 0;
    let max_attempts = 10;

    while attempts < max_attempts {
      if let Ok(client) = Client::connect_to(cwd.clone(), &socket_path).await {
        info!("Successfully connected to daemon");
        return Ok(client);
      }

      attempts += 1;
      debug!("Waiting for daemon to start... (attempt {}/{})", attempts, max_attempts);
      tokio::time::sleep(delay).await;
    }

    Err(IpcError::Connection("Failed to connect to daemon after startup".into()))
  }

  /// Spawn the daemon process.
  ///
  /// If `config.foreground` is false, this spawns a detached child process
  /// running `ccengram daemon --background` and returns the child PID.
  /// The child process will run with its own fresh Tokio runtime.
  ///
  /// If `config.foreground` is true, runs the daemon in the current process.
  pub async fn spawn(config: RuntimeConfig) -> std::io::Result<i32> {
    if !config.foreground {
      // Spawn a detached child process instead of forking.
      // This avoids issues with Tokio runtime not surviving fork.
      return Self::spawn_detached().await;
    }

    // Foreground mode: run directly in this process
    let daemon = Self::new(config);
    daemon.run().await;
    std::process::exit(0);
  }

  /// Spawn a detached daemon process using the current executable.
  ///
  /// This re-executes the current binary with `daemon --background`,
  /// ensuring a clean process with its own Tokio runtime.
  async fn spawn_detached() -> std::io::Result<i32> {
    use std::process::{Command, Stdio};

    let exe = std::env::current_exe()?;

    let child = Command::new(&exe)
      .arg("daemon")
      .arg("--background")
      .stdin(Stdio::null())
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .spawn()?;

    let pid = child.id() as i32;
    info!("Spawned daemon process with PID {}", pid);

    Ok(pid)
  }

  /// Spawn the daemon in foreground mode.
  ///
  /// The daemon runs in the current process and blocks until shutdown.
  pub async fn spawn_foreground() -> std::io::Result<()> {
    let config = RuntimeConfig {
      foreground: true,
      ..RuntimeConfig::load().await
    };

    let daemon = Self::new(config);
    daemon.run().await;
    Ok(())
  }

  /// Spawn the daemon in background mode.
  ///
  /// Spawns a detached child process and returns its PID.
  pub async fn spawn_background() -> std::io::Result<i32> {
    Self::spawn_detached().await
  }

  /// Run the daemon directly in this process (background mode).
  ///
  /// Called when the process was spawned with `--background`.
  /// This runs the daemon without trying to spawn another process.
  pub async fn run_background() {
    let config = RuntimeConfig {
      foreground: false,
      ..RuntimeConfig::load().await
    };

    let daemon = Self::new(config);
    daemon.run().await;
  }

  /// Run the daemon (blocking until shutdown).
  ///
  /// This is the main entry point that:
  /// 1. Creates all components with full configuration (no two-phase init)
  /// 2. Spawns background tasks
  /// 3. Runs the IPC server
  /// 4. Handles graceful shutdown
  async fn run(self) {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    #[cfg(feature = "dhat-ad-hoc")]
    let _profiler = dhat::Profiler::new_ad_hoc();

    info!("Starting CCEngram daemon");
    info!("Socket: {:?}", self.runtime_config.socket_path);
    info!("Data dir: {:?}", self.runtime_config.data_dir);

    // Master cancellation token - propagates to all children
    let cancel = CancellationToken::new();

    // Create embedding provider (shared, immutable)
    let Ok(embedding) = <dyn EmbeddingProvider>::from_config(&self.runtime_config.config.embedding).await else {
      error!("Failed to create embedding provider, shutting down daemon");
      panic!("Failed to create embedding provider");
    };

    info!(
      "Embedding provider: {} ({}, {} dims)",
      embedding.name(),
      embedding.model_id(),
      embedding.dimensions()
    );

    // Create reranker provider if configured
    let reranker: Option<Arc<dyn RerankerProvider>> = if self.runtime_config.config.reranker.enabled {
      match self.runtime_config.config.reranker.provider {
        RerankerProviderKind::DeepInfra => match DeepInfraReranker::new(&self.runtime_config.config.reranker) {
          Ok(r) => {
            info!(
              "Reranker provider: {} (max_candidates: {})",
              r.name(),
              r.max_candidates()
            );
            Some(Arc::new(r))
          }
          Err(e) => {
            warn!("Failed to create DeepInfra reranker: {}, reranking disabled", e);
            None
          }
        },
        RerankerProviderKind::LlamaCpp => {
          #[cfg(feature = "llama-cpp")]
          {
            match LlamaCppReranker::new(&self.runtime_config.config.reranker).await {
              Ok(r) => {
                info!(
                  "Reranker provider: {} (max_candidates: {})",
                  r.name(),
                  r.max_candidates()
                );
                Some(Arc::new(r) as Arc<dyn RerankerProvider>)
              }
              Err(e) => {
                warn!("Failed to create LlamaCpp reranker: {}, reranking disabled", e);
                None
              }
            }
          }
          #[cfg(not(feature = "llama-cpp"))]
          {
            warn!("LlamaCpp reranker requires the 'llama-cpp' feature, reranking disabled");
            None
          }
        }
        RerankerProviderKind::None => None,
      }
    } else {
      None
    };

    // Create daemon-level settings to pass to project actors
    let daemon_settings = DaemonSettings::from_config(&self.runtime_config.config);

    // Create the project router (replaces ProjectRegistry)
    let router = Arc::new(ProjectRouter::new(
      self.runtime_config.data_dir.clone(),
      embedding,
      reranker,
      daemon_settings,
      cancel.child_token(),
    ));

    // Create lifecycle trackers
    let activity = Arc::new(KeepAlive::new());
    let sessions = Arc::new(SessionTracker::new(
      self.runtime_config.config.daemon.session_timeout_secs,
    ));

    // Log hooks configuration
    if !self.runtime_config.config.hooks.enabled || !cfg!(feature = "automemory") {
      info!("Automatic memory capture is DISABLED");
    }

    // Create daemon state for Status/Metrics requests
    let auto_shutdown = !self.runtime_config.foreground;
    let daemon_state = Arc::new(DaemonState::new(self.runtime_config.foreground, auto_shutdown));

    let server_config = ServerConfig {
      socket_path: self.runtime_config.socket_path.clone(),
      router: Arc::clone(&router),
      activity: Arc::clone(&activity),
      sessions: Arc::clone(&sessions),
      daemon_state,
    };

    // Create server (fully configured, no mutation needed)
    let server = Server::new(server_config);

    // Build scheduler configuration
    let idle_shutdown = if self.runtime_config.foreground {
      info!("Foreground mode: auto-shutdown disabled");
      None
    } else {
      info!(
        "Auto-shutdown enabled: {} second idle timeout",
        self.runtime_config.config.daemon.idle_timeout_secs
      );
      Some(IdleShutdownConfig {
        timeout_secs: self.runtime_config.config.daemon.idle_timeout_secs,
        activity: Arc::clone(&activity),
        sessions: Arc::clone(&sessions),
      })
    };

    let scheduler_config = SchedulerConfig {
      decay: self.runtime_config.config.decay.clone(),
      daemon: self.runtime_config.config.daemon.clone(),
      idle_shutdown,
    };

    // Spawn scheduler for background tasks (decay, cleanup, idle shutdown)
    let scheduler_handle = {
      let router = Arc::clone(&router);
      let cancel = cancel.clone();
      tokio::spawn(async move {
        Scheduler::new(router, scheduler_config).run(cancel).await;
      })
    };
    info!(
      "Started background scheduler (log retention: {} days)",
      self.runtime_config.config.daemon.log_retention_days
    );

    // Handle ctrl-c gracefully
    let cancel_for_signal = cancel.clone();
    tokio::spawn(async move {
      if let Err(e) = signal::ctrl_c().await {
        warn!("Failed to listen for ctrl-c: {}", e);
        return;
      }
      info!("Received ctrl-c, shutting down...");
      cancel_for_signal.cancel();
    });

    // Run server until cancelled
    if let Err(e) = server.run(cancel.child_token()).await {
      warn!("Server error: {}", e);
    }

    info!("Shutting down...");
    cancel.cancel();

    let _ = scheduler_handle.await;
    router.shutdown_all().await;

    info!("Daemon shutdown complete");
  }
}
