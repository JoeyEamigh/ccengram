use crate::projects::ProjectRegistry;
use crate::router::Router;
use crate::scheduler::spawn_scheduler;
use crate::server::{Server, ShutdownHandle};
use embedding::{EmbeddingProvider, OllamaProvider, OpenRouterProvider};
use engram_core::{ConfigEmbeddingProvider, EmbeddingConfig};
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
  /// Idle timeout in seconds before auto-shutdown
  pub idle_timeout_secs: u64,
  /// Whether to daemonize (run in background)
  pub daemonize: bool,
  /// Embedding provider configuration
  pub embedding: EmbeddingConfig,
}

impl Default for DaemonConfig {
  fn default() -> Self {
    Self {
      socket_path: crate::server::default_socket_path(),
      data_dir: db::default_data_dir(),
      idle_timeout_secs: 1800, // 30 minutes
      daemonize: false,
      embedding: EmbeddingConfig::default(),
    }
  }
}

/// Create an embedding provider from config
fn create_embedding_provider(config: &EmbeddingConfig) -> Arc<dyn EmbeddingProvider> {
  match config.provider {
    ConfigEmbeddingProvider::Ollama => {
      let provider = OllamaProvider::new()
        .with_url(&config.ollama_url)
        .with_model(&config.model, config.dimensions);
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
        let provider = OllamaProvider::new()
          .with_url(&config.ollama_url)
          .with_model(&config.model, config.dimensions);
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
}

impl Daemon {
  pub fn new(config: DaemonConfig) -> Self {
    let registry = Arc::new(ProjectRegistry::with_data_dir(config.data_dir.clone()));

    Self {
      config,
      registry,
      shutdown: None,
      scheduler_shutdown_tx: None,
    }
  }

  /// Run the daemon
  pub async fn run(&mut self) -> Result<(), LifecycleError> {
    info!("Starting CCEngram daemon");
    info!("Socket: {:?}", self.config.socket_path);
    info!("Data dir: {:?}", self.config.data_dir);

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

    // Create router with our registry and embedding provider
    let router = Router::with_embedding(Arc::clone(&self.registry), embedding);
    let router = Arc::new(router);

    // Create server
    let server = Server::with_socket_path(Arc::clone(&router), self.config.socket_path.clone());
    let shutdown = server.shutdown_handle();
    self.shutdown = Some(shutdown.clone());

    // Give the router the shutdown handle so it can process shutdown requests
    router.set_shutdown_handle(shutdown.clone()).await;

    // Create shutdown channel for scheduler
    let (scheduler_shutdown_tx, scheduler_shutdown_rx) = broadcast::channel(1);
    self.scheduler_shutdown_tx = Some(scheduler_shutdown_tx.clone());

    // Spawn the background scheduler for decay and cleanup
    let _scheduler_handle = spawn_scheduler(Arc::clone(&self.registry), scheduler_shutdown_rx);
    info!("Started background scheduler");

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
    assert_eq!(config.idle_timeout_secs, 1800);
  }

  #[test]
  fn test_is_running_no_socket() {
    let dir = TempDir::new().unwrap();
    let socket_path = dir.path().join("nonexistent.sock");
    assert!(!is_running(&socket_path));
  }
}
