use crate::hooks::{HookEvent, HookHandler};
use crate::projects::ProjectRegistry;
use crate::server::ShutdownHandle;
use crate::tools::ToolHandler;
use embedding::EmbeddingProvider;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// JSON-RPC style request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
  #[serde(default)]
  pub id: Option<serde_json::Value>,
  pub method: String,
  #[serde(default)]
  pub params: serde_json::Value,
}

/// JSON-RPC style response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub id: Option<serde_json::Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub result: Option<serde_json::Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
  pub code: i32,
  pub message: String,
}

impl Response {
  pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
    Self {
      id,
      result: Some(result),
      error: None,
    }
  }

  pub fn error(id: Option<serde_json::Value>, code: i32, message: &str) -> Self {
    Self {
      id,
      result: None,
      error: Some(RpcError {
        code,
        message: message.to_string(),
      }),
    }
  }
}

/// Request router for the daemon
pub struct Router {
  registry: Arc<ProjectRegistry>,
  tool_handler: Arc<ToolHandler>,
  hook_handler: Arc<HookHandler>,
  shutdown_handle: Arc<Mutex<Option<ShutdownHandle>>>,
}

impl Router {
  pub fn new() -> Self {
    let registry = Arc::new(ProjectRegistry::new());
    let tool_handler = Arc::new(ToolHandler::new(Arc::clone(&registry)));
    let hook_handler = Arc::new(HookHandler::new(Arc::clone(&registry)));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
    }
  }

  pub fn with_registry(registry: Arc<ProjectRegistry>) -> Self {
    let tool_handler = Arc::new(ToolHandler::new(Arc::clone(&registry)));
    let hook_handler = Arc::new(HookHandler::new(Arc::clone(&registry)));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
    }
  }

  pub fn with_embedding(registry: Arc<ProjectRegistry>, embedding: Arc<dyn EmbeddingProvider>) -> Self {
    let tool_handler = Arc::new(ToolHandler::with_embedding(
      Arc::clone(&registry),
      Arc::clone(&embedding),
    ));
    let hook_handler = Arc::new(HookHandler::with_embedding(Arc::clone(&registry), embedding));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
    }
  }

  /// Set the shutdown handle (called after server is created)
  pub async fn set_shutdown_handle(&self, handle: ShutdownHandle) {
    let mut guard = self.shutdown_handle.lock().await;
    *guard = Some(handle);
  }

  /// Get the project registry (for testing/cleanup)
  pub fn registry(&self) -> &Arc<ProjectRegistry> {
    &self.registry
  }

  /// Handle an incoming request
  pub async fn handle(&self, request: Request) -> Response {
    debug!("Handling request: {}", request.method);

    match request.method.as_str() {
      // Health/meta commands
      "ping" => Response::success(request.id, serde_json::json!("pong")),
      "status" => self.handle_status(request).await,
      "shutdown" => self.handle_shutdown(request).await,

      // Memory tools
      "memory_search" => self.tool_handler.memory_search(request).await,
      "memory_get" => self.tool_handler.memory_get(request).await,
      "memory_list" => self.tool_handler.memory_list(request).await,
      "memory_add" => self.tool_handler.memory_add(request).await,
      "memory_reinforce" => self.tool_handler.memory_reinforce(request).await,
      "memory_deemphasize" => self.tool_handler.memory_deemphasize(request).await,
      "memory_delete" => self.tool_handler.memory_delete(request).await,
      "memory_supersede" => self.tool_handler.memory_supersede(request).await,
      "memory_timeline" => self.tool_handler.memory_timeline(request).await,

      // Code tools
      "code_search" => self.tool_handler.code_search(request).await,
      "code_index" => self.tool_handler.code_index(request).await,
      "code_list" => self.tool_handler.code_list(request).await,
      "code_import_chunk" => self.tool_handler.code_import_chunk(request).await,
      "code_stats" => self.tool_handler.code_stats(request).await,

      // Watch tools
      "watch_start" => self.tool_handler.watch_start(request).await,
      "watch_stop" => self.tool_handler.watch_stop(request).await,
      "watch_status" => self.tool_handler.watch_status(request).await,

      // Document tools
      "docs_search" => self.tool_handler.docs_search(request).await,
      "docs_ingest" => self.tool_handler.docs_ingest(request).await,

      // Entity tools
      "entity_list" => self.tool_handler.entity_list(request).await,
      "entity_get" => self.tool_handler.entity_get(request).await,
      "entity_top" => self.tool_handler.entity_top(request).await,

      // Relationship tools
      "relationship_add" => self.tool_handler.relationship_add(request).await,
      "relationship_list" => self.tool_handler.relationship_list(request).await,
      "relationship_delete" => self.tool_handler.relationship_delete(request).await,
      "relationship_related" => self.tool_handler.relationship_related(request).await,

      // Statistics & Health
      "project_stats" => self.tool_handler.project_stats(request).await,
      "health_check" => self.tool_handler.health_check(request).await,

      // Migration
      "migrate_embedding" => self.tool_handler.migrate_embedding(request).await,

      // Hook events
      "hook" => self.handle_hook(request).await,

      // Unknown method
      _ => {
        warn!("Unknown method: {}", request.method);
        Response::error(request.id, -32601, &format!("Method not found: {}", request.method))
      }
    }
  }

  async fn handle_status(&self, request: Request) -> Response {
    let projects = self.registry.list().await;
    let status = serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "projects": projects.len(),
    });
    Response::success(request.id, status)
  }

  async fn handle_shutdown(&self, request: Request) -> Response {
    info!("Shutdown requested via RPC");
    let guard = self.shutdown_handle.lock().await;
    if let Some(ref handle) = *guard {
      handle.shutdown();
      Response::success(request.id, serde_json::json!({"status": "shutting_down"}))
    } else {
      Response::error(request.id, -32000, "Shutdown handle not available")
    }
  }

  async fn handle_hook(&self, request: Request) -> Response {
    let event_str = request
      .params
      .get("event")
      .and_then(|v| v.as_str())
      .unwrap_or("unknown");
    debug!("Received hook event: {}", event_str);

    // Parse the event type
    let event: HookEvent = match event_str.parse() {
      Ok(e) => e,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid hook event: {}", e)),
    };

    // Get the params for the hook
    let params = request.params.get("params").cloned().unwrap_or(serde_json::json!({}));

    // Delegate to hook handler
    match self.hook_handler.handle(event, params).await {
      Ok(result) => Response::success(request.id, result),
      Err(e) => Response::error(request.id, -32000, &format!("Hook error: {}", e)),
    }
  }
}

impl Default for Router {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_response_success() {
    let response = Response::success(Some(serde_json::json!(1)), serde_json::json!("test"));
    assert!(response.result.is_some());
    assert!(response.error.is_none());
  }

  #[test]
  fn test_response_error() {
    let response = Response::error(Some(serde_json::json!(1)), -1, "test error");
    assert!(response.result.is_none());
    assert!(response.error.is_some());
    assert_eq!(response.error.as_ref().unwrap().code, -1);
  }

  #[tokio::test]
  async fn test_ping() {
    let router = Router::new();
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "ping".to_string(),
      params: serde_json::json!({}),
    };

    let response = router.handle(request).await;
    assert!(response.result.is_some());
    assert_eq!(response.result.unwrap(), serde_json::json!("pong"));
  }

  #[tokio::test]
  async fn test_unknown_method() {
    let router = Router::new();
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "unknown_method".to_string(),
      params: serde_json::json!({}),
    };

    let response = router.handle(request).await;
    assert!(response.error.is_some());
    assert_eq!(response.error.as_ref().unwrap().code, -32601);
  }
}
