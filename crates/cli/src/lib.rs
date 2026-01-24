//! CCEngram CLI library - shared tools and definitions

pub mod tools;

pub use tools::{all_tool_definitions, get_filtered_tool_definitions, get_tool_definitions_for_cwd};

use serde::Serialize;

/// Convert a typed ipc::Request<P> to daemon::Request for wire transmission.
///
/// The ipc crate uses typed params and Method enum, while the daemon crate
/// uses String method names and serde_json::Value params for wire format.
pub fn to_daemon_request<P: Serialize>(request: ipc::Request<P>) -> daemon::Request {
  // Serialize the Method enum to get the snake_case name
  let method_name = serde_json::to_value(request.method)
    .ok()
    .and_then(|v| v.as_str().map(|s| s.to_string()))
    .unwrap_or_else(|| format!("{:?}", request.method).to_lowercase());

  daemon::Request {
    id: request.id.map(|id| serde_json::Value::Number(id.into())),
    method: method_name,
    params: serde_json::to_value(&request.params).unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
  }
}
