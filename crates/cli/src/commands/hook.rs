//! Hook command for handling hook events
//!
//! Hooks are fire-and-forget: we send the request to the daemon and exit immediately
//! without waiting for a response. This ensures hooks don't block Claude Code.

use std::io::Read;

use anyhow::{Context, Result};
use ccengram::ipc::hook::HookParams;

/// Read hook input from stdin (JSON parameters from Claude Code)
fn read_hook_input() -> Result<serde_json::Value> {
  let mut input = String::new();
  std::io::stdin().read_to_string(&mut input)?;

  if input.trim().is_empty() {
    return Ok(serde_json::Value::Object(serde_json::Map::new()));
  }

  serde_json::from_str(&input).context("Invalid JSON in hook input")
}

/// Handle a hook event (fire-and-forget)
pub async fn cmd_hook(name: &str) -> Result<()> {
  // Read input from stdin
  let mut input = read_hook_input().context("Failed to read hook input")?;

  // Extract session_id and cwd from the input JSON to avoid duplicate fields
  // (HookParams uses #[serde(flatten)] which would create duplicates)
  let session_id = input
    .as_object()
    .and_then(|obj| obj.get("session_id"))
    .and_then(|v| v.as_str())
    .map(String::from);

  let cwd = input
    .as_object()
    .and_then(|obj| obj.get("cwd"))
    .and_then(|v| v.as_str())
    .map(String::from)
    .or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()));

  // Remove session_id and cwd from data to avoid duplicates when flattened
  if let Some(obj) = input.as_object_mut() {
    obj.remove("session_id");
    obj.remove("cwd");
  }

  let params = HookParams {
    hook_name: name.to_string(),
    session_id,
    cwd: cwd.clone(),
    data: input,
  };

  // Fire-and-forget: send request without waiting for response
  // This ensures hooks don't block Claude Code
  let cwd_path = cwd
    .map(std::path::PathBuf::from)
    .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));

  // Auto-start daemon if not running
  let client = match ccengram::Daemon::connect_or_start(cwd_path).await {
    Ok(c) => c,
    Err(e) => {
      eprintln!("ccengram: failed to start daemon: {}", e);
      return Ok(());
    }
  };

  if let Err(e) = client.fire_and_forget(params).await {
    eprintln!("ccengram: hook send failed: {}", e);
  }

  Ok(())
}
