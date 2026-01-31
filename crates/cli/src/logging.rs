//! Logging utilities for CLI commands and daemon

use ccengram::config::Config;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize logging for CLI commands (console only)
pub fn init_cli_logging() {
  // Build env filter (allows RUST_LOG override)
  let env_filter = EnvFilter::builder()
    .with_default_directive(tracing::Level::INFO.into())
    .from_env_lossy();

  tracing_subscriber::fmt()
    .with_env_filter(env_filter)
    .with_span_events(FmtSpan::CLOSE)
    .init();
}

/// Parse log level from config string
fn parse_log_level(level: &str) -> tracing::Level {
  match level.to_lowercase().as_str() {
    "off" | "error" => tracing::Level::ERROR,
    "warn" => tracing::Level::WARN,
    "info" => tracing::Level::INFO,
    "debug" => tracing::Level::DEBUG,
    "trace" => tracing::Level::TRACE,
    _ => tracing::Level::INFO,
  }
}

/// Initialize logging for daemon with config-driven settings.
///
/// In foreground mode: Logs to console only with colors
/// In background mode: Logs to file only (no ANSI)
///
/// Returns the guard that must be kept alive for the duration of the program
pub async fn init_daemon_logging_with_config(foreground: bool) -> Option<WorkerGuard> {
  // Load config from current directory (or defaults)
  let config = Config::load_global().await;
  let daemon_config = &config.daemon;

  // Parse log level from config
  let level = parse_log_level(&daemon_config.log_level);
  let default_directive = level.into();

  // directives for debug builds
  #[cfg(debug_assertions)]
  let filter_directives = if let Ok(filter) = std::env::var("RUST_LOG") {
    filter
  } else {
    "ccengram=trace,llm=trace".to_string()
  };

  #[cfg(not(debug_assertions))]
  let filter_directives = if let Ok(filter) = std::env::var("RUST_LOG") {
    filter
  } else {
    format!("ccengram={},llm={}", &daemon_config.log_level, &daemon_config.log_level)
  };

  let env_filter = EnvFilter::builder()
    .with_default_directive(default_directive)
    .parse_lossy(filter_directives);

  // Setup file logging
  let log_dir = ccengram::dirs::default_data_dir();
  if tokio::fs::create_dir_all(&log_dir).await.is_err() {
    // Fall back to console-only logging
    init_cli_logging();
    return None;
  }

  // Create rolling file appender based on config
  let file_appender = match daemon_config.log_rotation.as_str() {
    "hourly" => tracing_appender::rolling::hourly(&log_dir, "ccengram.log"),
    "never" => tracing_appender::rolling::never(&log_dir, "ccengram.log"),
    _ => tracing_appender::rolling::daily(&log_dir, "ccengram.log"),
  };

  let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

  if foreground {
    // Foreground mode: both console (with colors) and file logging
    let subscriber = tracing_subscriber::registry()
      .with(env_filter)
      .with(
        tracing_subscriber::fmt::layer()
          .with_ansi(true)
          .with_writer(std::io::stdout)
          .with_span_events(FmtSpan::CLOSE)
          .with_target(true),
      )
      .with(
        tracing_subscriber::fmt::layer()
          .with_ansi(false)
          .with_writer(file_writer)
          .with_span_events(FmtSpan::CLOSE)
          .with_target(true),
      );

    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default());

    subscriber.init();
  } else {
    // Background mode: file logging only (no ANSI)
    let subscriber = tracing_subscriber::registry().with(env_filter).with(
      tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(file_writer)
        .with_span_events(FmtSpan::CLOSE)
        .with_target(true),
    );

    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default());

    subscriber.init();
  }

  Some(guard)
}
