use crate::activity_tracker::ActivityTracker;
use crate::hooks::{HookEvent, HookHandler};
use crate::projects::ProjectRegistry;
use crate::server::ShutdownHandle;
use crate::session_tracker::SessionTracker;
use crate::tools::ToolHandler;
use embedding::EmbeddingProvider;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
  /// Session tracker for lifecycle management
  session_tracker: Arc<Mutex<Option<Arc<SessionTracker>>>>,
  /// Activity tracker for idle detection
  activity_tracker: Arc<Mutex<Option<Arc<ActivityTracker>>>>,
  /// Whether daemon is in foreground mode
  foreground: Arc<Mutex<bool>>,
  /// Embedding provider reference for metrics
  embedding_provider: Arc<Mutex<Option<Arc<dyn EmbeddingProvider>>>>,
  /// Total requests handled (for metrics)
  request_count: AtomicU64,
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
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(None)),
      request_count: AtomicU64::new(0),
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
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(None)),
      request_count: AtomicU64::new(0),
    }
  }

  pub fn with_embedding(registry: Arc<ProjectRegistry>, embedding: Arc<dyn EmbeddingProvider>) -> Self {
    let tool_handler = Arc::new(ToolHandler::with_embedding(
      Arc::clone(&registry),
      Arc::clone(&embedding),
    ));
    let hook_handler = Arc::new(HookHandler::with_embedding(
      Arc::clone(&registry),
      Arc::clone(&embedding),
    ));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(Some(embedding))),
      request_count: AtomicU64::new(0),
    }
  }

  /// Create a router with embedding provider and hooks configuration
  pub fn with_embedding_and_config(
    registry: Arc<ProjectRegistry>,
    embedding: Arc<dyn EmbeddingProvider>,
    hooks_config: &engram_core::HooksConfig,
  ) -> Self {
    let tool_handler = Arc::new(ToolHandler::with_embedding(
      Arc::clone(&registry),
      Arc::clone(&embedding),
    ));
    let hook_handler =
      Arc::new(HookHandler::with_embedding(Arc::clone(&registry), Arc::clone(&embedding)).with_config(hooks_config));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(Some(embedding))),
      request_count: AtomicU64::new(0),
    }
  }

  /// Set the shutdown handle (called after server is created)
  pub async fn set_shutdown_handle(&self, handle: ShutdownHandle) {
    let mut guard = self.shutdown_handle.lock().await;
    *guard = Some(handle);
  }

  /// Set the session tracker for lifecycle management
  pub async fn set_session_tracker(&self, tracker: Arc<SessionTracker>) {
    let mut guard = self.session_tracker.lock().await;
    *guard = Some(tracker.clone());
    // Also pass to hook handler
    self.hook_handler.set_session_tracker(tracker).await;
  }

  /// Set the activity tracker for idle detection
  pub async fn set_activity_tracker(&self, tracker: Arc<ActivityTracker>) {
    let mut guard = self.activity_tracker.lock().await;
    *guard = Some(tracker);
  }

  /// Set foreground mode flag
  pub async fn set_foreground(&self, foreground: bool) {
    let mut guard = self.foreground.lock().await;
    *guard = foreground;
  }

  /// Get the project registry (for testing/cleanup)
  pub fn registry(&self) -> &Arc<ProjectRegistry> {
    &self.registry
  }

  /// Handle an incoming request
  pub async fn handle(&self, request: Request) -> Response {
    debug!("Handling request: {}", request.method);

    // Increment request counter
    self.request_count.fetch_add(1, Ordering::Relaxed);

    // Touch activity tracker on every request
    {
      let guard = self.activity_tracker.lock().await;
      if let Some(ref tracker) = *guard {
        tracker.touch();
      }
    }

    match request.method.as_str() {
      // Health/meta commands
      "ping" => Response::success(request.id, serde_json::json!("pong")),
      "status" => self.handle_status(request).await,
      "metrics" => self.handle_metrics(request).await,
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
      "code_context" => self.tool_handler.code_context(request).await,
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
      "doc_context" => self.tool_handler.doc_context(request).await,
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

      // Memory restore/deleted
      "memory_restore" => self.tool_handler.memory_restore(request).await,
      "memory_list_deleted" => self.tool_handler.memory_list_deleted(request).await,

      // Project management
      "projects_list" => self.handle_projects_list(request).await,
      "project_info" => self.handle_project_info(request).await,
      "project_clean" => self.handle_project_clean(request).await,
      "projects_clean_all" => self.handle_projects_clean_all(request).await,

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

    // Get session count
    let active_sessions = {
      let guard = self.session_tracker.lock().await;
      match &*guard {
        Some(tracker) => tracker.active_count().await,
        None => 0,
      }
    };

    // Get activity info
    let (idle_seconds, uptime_seconds) = {
      let guard = self.activity_tracker.lock().await;
      match &*guard {
        Some(tracker) => (tracker.idle_duration().as_secs(), tracker.uptime().as_secs()),
        None => (0, 0),
      }
    };

    // Get foreground mode
    let foreground = *self.foreground.lock().await;

    let status = serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "projects": projects.len(),
        "active_sessions": active_sessions,
        "idle_seconds": idle_seconds,
        "uptime_seconds": uptime_seconds,
        "foreground": foreground,
        "auto_shutdown": !foreground,
    });
    Response::success(request.id, status)
  }

  /// Handle metrics request - returns detailed daemon metrics for monitoring
  async fn handle_metrics(&self, request: Request) -> Response {
    let projects = self.registry.list().await;

    // Get session details
    let (active_sessions, session_list) = {
      let guard = self.session_tracker.lock().await;
      match &*guard {
        Some(tracker) => {
          let sessions = tracker.list_sessions().await;
          let session_ids: Vec<String> = sessions.iter().map(|s| s.0.clone()).collect();
          (sessions.len(), session_ids)
        }
        None => (0, vec![]),
      }
    };

    // Get activity info
    let (idle_seconds, uptime_seconds) = {
      let guard = self.activity_tracker.lock().await;
      match &*guard {
        Some(tracker) => (tracker.idle_duration().as_secs(), tracker.uptime().as_secs()),
        None => (0, 0),
      }
    };

    // Get request count
    let total_requests = self.request_count.load(Ordering::Relaxed);

    // Get foreground mode
    let foreground = *self.foreground.lock().await;

    // Get embedding provider info
    let embedding_info = {
      let guard = self.embedding_provider.lock().await;
      match &*guard {
        Some(provider) => serde_json::json!({
          "name": provider.name(),
          "model": provider.model_id(),
          "dimensions": provider.dimensions(),
        }),
        None => serde_json::json!(null),
      }
    };

    // Get process memory (if available on Linux)
    let memory_kb = Self::get_process_memory_kb();

    let metrics = serde_json::json!({
      "daemon": {
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime_seconds,
        "idle_seconds": idle_seconds,
        "foreground": foreground,
        "auto_shutdown": !foreground,
      },
      "requests": {
        "total": total_requests,
        "per_second": if uptime_seconds > 0 {
          total_requests as f64 / uptime_seconds as f64
        } else {
          0.0
        },
      },
      "sessions": {
        "active": active_sessions,
        "ids": session_list,
      },
      "projects": {
        "count": projects.len(),
        "names": projects.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
      },
      "embedding": embedding_info,
      "memory": {
        "rss_kb": memory_kb,
      },
    });

    Response::success(request.id, metrics)
  }

  /// Get process RSS memory in KB (Linux only, returns None on other platforms)
  fn get_process_memory_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
      // Read /proc/self/statm
      if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        let parts: Vec<&str> = statm.split_whitespace().collect();
        if parts.len() >= 2 {
          // Second field is RSS in pages, page size is typically 4KB
          if let Ok(rss_pages) = parts[1].parse::<u64>() {
            return Some(rss_pages * 4); // Convert to KB
          }
        }
      }
      None
    }
    #[cfg(not(target_os = "linux"))]
    {
      None
    }
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

  /// List all projects in the registry
  async fn handle_projects_list(&self, request: Request) -> Response {
    let projects = self.registry.list().await;

    let project_list: Vec<serde_json::Value> = projects
      .iter()
      .map(|p| {
        serde_json::json!({
          "id": p.id.as_str(),
          "path": p.path.to_string_lossy(),
          "name": p.name,
        })
      })
      .collect();

    Response::success(request.id, serde_json::json!(project_list))
  }

  /// Get detailed info for a specific project
  async fn handle_project_info(&self, request: Request) -> Response {
    let project_identifier = request.params.get("project").and_then(|v| v.as_str()).unwrap_or("");

    if project_identifier.is_empty() {
      return Response::error(request.id, -32602, "Missing project parameter");
    }

    // Try to find the project by path or ID
    let path = std::path::Path::new(project_identifier);
    let result = if path.exists() {
      self.registry.get_or_create(path).await
    } else {
      // Try to find by ID prefix in the registry
      let projects = self.registry.list().await;
      let found = projects.iter().find(|p| {
        p.id.as_str().starts_with(project_identifier) || p.path.to_string_lossy().contains(project_identifier)
      });
      match found {
        Some(p) => self.registry.get_or_create(&p.path).await,
        None => {
          return Response::error(
            request.id,
            -32000,
            &format!("Project not found: {}", project_identifier),
          );
        }
      }
    };

    match result {
      Ok((info, db)) => {
        // Get statistics
        let memory_count = db.count_memories(Some("is_deleted = false")).await.unwrap_or(0);
        let code_chunk_count = db.count_code_chunks(None).await.unwrap_or(0);
        let document_count = db.count_document_chunks(None).await.unwrap_or(0);

        // Get project UUID for session count
        let project_uuid = uuid::Uuid::parse_str(info.id.as_str()).unwrap_or_else(|_| uuid::Uuid::nil());
        let session_count = db.count_sessions(&project_uuid).await.unwrap_or(0);

        Response::success(
          request.id,
          serde_json::json!({
            "id": info.id.as_str(),
            "path": info.path.to_string_lossy(),
            "name": info.name,
            "memory_count": memory_count,
            "code_chunk_count": code_chunk_count,
            "document_count": document_count,
            "session_count": session_count,
            "db_path": db.path.to_string_lossy(),
          }),
        )
      }
      Err(e) => Response::error(request.id, -32000, &format!("Failed to get project info: {}", e)),
    }
  }

  /// Clean (remove) a specific project's data
  async fn handle_project_clean(&self, request: Request) -> Response {
    let project_identifier = request.params.get("project").and_then(|v| v.as_str()).unwrap_or("");

    if project_identifier.is_empty() {
      return Response::error(request.id, -32602, "Missing project parameter");
    }

    // Find the project
    let path = std::path::Path::new(project_identifier);
    let project_info = if path.exists() {
      match crate::projects::ProjectInfo::from_path(path) {
        Ok(info) => info,
        Err(e) => return Response::error(request.id, -32000, &format!("Invalid project path: {}", e)),
      }
    } else {
      // Try to find by ID prefix
      let projects = self.registry.list().await;
      match projects.iter().find(|p| {
        p.id.as_str().starts_with(project_identifier) || p.path.to_string_lossy().contains(project_identifier)
      }) {
        Some(p) => p.clone(),
        None => {
          return Response::error(
            request.id,
            -32000,
            &format!("Project not found: {}", project_identifier),
          );
        }
      }
    };

    // Get counts before deletion for reporting
    let counts = match self.registry.get_or_create(&project_info.path).await {
      Ok((_, db)) => {
        let memories = db.count_memories(None).await.unwrap_or(0);
        let code_chunks = db.count_code_chunks(None).await.unwrap_or(0);
        let documents = db.count_document_chunks(None).await.unwrap_or(0);
        (memories, code_chunks, documents)
      }
      Err(_) => (0, 0, 0),
    };

    // Close the project connection
    self.registry.close(project_info.id.as_str()).await;

    // Delete the project data directory
    let data_dir = project_info.id.data_dir(self.registry.data_dir());
    if data_dir.exists()
      && let Err(e) = std::fs::remove_dir_all(&data_dir)
    {
      return Response::error(request.id, -32000, &format!("Failed to remove project data: {}", e));
    }

    Response::success(
      request.id,
      serde_json::json!({
        "path": project_info.path.to_string_lossy(),
        "memories_deleted": counts.0,
        "code_chunks_deleted": counts.1,
        "documents_deleted": counts.2,
      }),
    )
  }

  /// Clean all projects
  async fn handle_projects_clean_all(&self, request: Request) -> Response {
    let projects = self.registry.list().await;
    let count = projects.len();

    // Close all connections first
    self.registry.close_all().await;

    // Remove all project data directories
    let data_dir = self.registry.data_dir();
    let projects_dir = data_dir.join("projects");
    if projects_dir.exists()
      && let Err(e) = std::fs::remove_dir_all(&projects_dir)
    {
      return Response::error(
        request.id,
        -32000,
        &format!("Failed to remove projects directory: {}", e),
      );
    }

    Response::success(
      request.id,
      serde_json::json!({
        "projects_removed": count,
      }),
    )
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

  #[tokio::test]
  async fn test_metrics() {
    let router = Router::new();

    // Make a few requests to increment the counter
    for _ in 0..3 {
      let request = Request {
        id: Some(serde_json::json!(1)),
        method: "ping".to_string(),
        params: serde_json::json!({}),
      };
      router.handle(request).await;
    }

    // Now request metrics
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "metrics".to_string(),
      params: serde_json::json!({}),
    };

    let response = router.handle(request).await;
    assert!(response.result.is_some());

    let result = response.result.unwrap();

    // Check daemon info
    assert!(result.get("daemon").is_some());
    assert!(result["daemon"]["version"].is_string());
    assert!(result["daemon"]["uptime_seconds"].is_u64());

    // Check requests info
    assert!(result.get("requests").is_some());
    // 4 total requests: 3 pings + 1 metrics
    assert_eq!(result["requests"]["total"], 4);

    // Check sessions info
    assert!(result.get("sessions").is_some());
    assert_eq!(result["sessions"]["active"], 0);

    // Check projects info
    assert!(result.get("projects").is_some());
    assert_eq!(result["projects"]["count"], 0);

    // Check memory info
    assert!(result.get("memory").is_some());
  }
}
