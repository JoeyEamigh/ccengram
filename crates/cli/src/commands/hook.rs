//! Hook command for handling hook events

use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::{Client, HookEvent, HookHandler, ProjectRegistry, default_socket_path, is_running};
use ipc::{HookParams, Method, Request};
use std::sync::Arc;
use tracing::error;

/// Handle a hook event
pub async fn cmd_hook(name: &str) -> Result<()> {
  // Parse hook event name
  let event: HookEvent = name.parse().map_err(|e| anyhow::anyhow!("Unknown hook: {}", e))?;

  // Read input from stdin
  let input = daemon::hooks::read_hook_input().context("Failed to read hook input")?;

  // Try to connect to running daemon first
  let socket_path = default_socket_path();
  if is_running(&socket_path) {
    let mut client = Client::connect_to(&socket_path)
      .await
      .context("Failed to connect to daemon")?;

    let params = HookParams {
      hook_name: name.to_string(),
      session_id: None,
      cwd: None,
      data: input,
    };

    let request = Request {
      id: Some(1),
      method: Method::Hook,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to send hook to daemon")?;

    if let Some(err) = response.error {
      error!("Hook error: {}", err.message);
    }
  } else {
    // Handle hook directly (stateless mode)
    let registry = Arc::new(ProjectRegistry::new());
    let handler = HookHandler::new(registry);

    match handler.handle(event, input).await {
      Ok(result) => {
        println!("{}", serde_json::to_string(&result)?);
      }
      Err(e) => {
        error!("Hook error: {}", e);
        std::process::exit(1);
      }
    }
  }

  Ok(())
}
