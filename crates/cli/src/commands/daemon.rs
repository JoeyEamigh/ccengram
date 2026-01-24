//! Daemon command

use anyhow::{Context, Result, bail};
use daemon::{Daemon, DaemonConfig, default_socket_path, is_running};
use engram_core::config::EmbeddingProvider;
use tracing::info;

/// Start the daemon
///
/// # Arguments
/// * `foreground` - Run in foreground mode (disables auto-shutdown, logs to console)
/// * `background` - Run in background mode (enables auto-shutdown, for auto-start)
/// * `embedding_provider` - Override embedding provider (ollama or openrouter)
/// * `openrouter_api_key` - Override OpenRouter API key
pub async fn cmd_daemon(
  foreground: bool,
  background: bool,
  embedding_provider: Option<String>,
  openrouter_api_key: Option<String>,
) -> Result<()> {
  // --foreground and --background are mutually exclusive
  if foreground && background {
    bail!("Cannot specify both --foreground and --background");
  }

  // If --background, we're being spawned by auto-start
  // Check if already running and exit silently
  let socket_path = default_socket_path();
  if is_running(&socket_path) {
    if background {
      // Another instance is handling it, exit silently
      return Ok(());
    }
    bail!("Daemon is already running at {:?}", socket_path);
  }

  // Create config based on mode
  let mut config = if foreground {
    DaemonConfig::foreground()
  } else {
    DaemonConfig::background()
  };

  // Apply embedding provider overrides
  if let Some(provider) = embedding_provider {
    match provider.to_lowercase().as_str() {
      "ollama" => {
        config.embedding.provider = EmbeddingProvider::Ollama;
        info!("Using Ollama embedding provider (override)");
      }
      "openrouter" => {
        config.embedding.provider = EmbeddingProvider::OpenRouter;
        // Also set default model for openrouter if using default ollama model
        if config.embedding.model == "qwen3-embedding" {
          config.embedding.model = "qwen/qwen3-embedding-8b".to_string();
          config.embedding.dimensions = 4096;
        }
        info!("Using OpenRouter embedding provider (override)");
      }
      other => bail!("Unknown embedding provider: {}. Use 'ollama' or 'openrouter'", other),
    }
  }

  if let Some(key) = openrouter_api_key {
    config.embedding.openrouter_api_key = Some(key);
  }

  let mut daemon = Daemon::new(config);

  info!("Starting CCEngram daemon");
  daemon.run().await.context("Failed to run daemon")?;

  Ok(())
}
