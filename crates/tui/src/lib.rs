pub mod app;
pub mod daemon_client;
pub mod event;
pub mod theme;
pub mod views;
pub mod widgets;

use anyhow::Result;
use std::path::PathBuf;

pub use app::{App, InputMode, View};
pub use daemon_client::DaemonClient;

/// Run the TUI application
pub async fn run(project_path: PathBuf) -> Result<()> {
  app::run(project_path).await
}
