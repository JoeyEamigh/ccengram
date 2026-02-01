//! IPC server for the actor-based daemon architecture.
//!
//! The server accepts connections on a Unix socket and routes requests
//! to `ProjectActor` instances via the `ProjectRouter`. It supports
//! response streaming for long-running operations.
//!
//! # Design Principles
//!
//! - **No two-phase initialization**: All dependencies are passed to `Server::new()`
//! - **No `set_*` methods**: Configuration is immutable after construction
//! - **Actor-based routing**: Requests go through `ProjectRouter` → `ProjectActor`
//! - **Streaming support**: Response channel supports multiple messages per request
//!
//! # Example
//!
//! ```ignore
//! let config = ServerConfig {
//!     socket_path: PathBuf::from("/tmp/ccengram.sock"),
//!     router: Arc::new(project_router),
//!     activity: Arc::new(activity_tracker),
//!     sessions: Arc::new(session_tracker),
//!     hooks_config: HooksConfig::default(),
//! };
//!
//! let server = Server::new(config);
//! server.run(cancel_token).await?;
//! ```

use std::{
  path::PathBuf,
  sync::{Arc, atomic::AtomicU64},
};

use futures::{SinkExt, StreamExt};
use tokio::net::{UnixListener, UnixStream};
use tokio_util::{
  codec::{Framed, LinesCodec},
  sync::CancellationToken,
};
use tracing::{debug, error, info, trace, warn};

use crate::{
  actor::{
    ProjectRouter,
    lifecycle::{
      activity::KeepAlive,
      session::{SessionId, SessionTracker},
    },
    message::{ProjectActorPayload, ProjectActorResponse},
  },
  ipc::{
    IpcError, Request, RequestData, Response, ResponseData,
    system::{
      DaemonMetrics, EmbeddingProviderInfo, MemoryUsageMetrics, MetricsResult, ProjectsMetrics, RequestsMetrics,
      SessionsMetrics, StatusResult, SystemRequest, SystemResponse,
    },
  },
};

// ============================================================================
// Server Configuration
// ============================================================================

/// Daemon-level state for handling Status and Metrics requests.
///
/// This is shared across all connections and provides info about the daemon itself,
/// not individual projects.
pub struct DaemonState {
  /// Daemon process ID
  pub pid: u32,
  /// Daemon start time (for uptime calculation)
  pub start_time: std::time::Instant,
  /// Whether running in foreground mode
  pub foreground: bool,
  /// Whether auto-shutdown is enabled
  pub auto_shutdown: bool,
}

impl DaemonState {
  /// Create new daemon state with current process info.
  pub fn new(foreground: bool, auto_shutdown: bool) -> Self {
    Self {
      pid: std::process::id(),
      start_time: std::time::Instant::now(),
      foreground,
      auto_shutdown,
    }
  }
}

/// Configuration for the IPC server.
///
/// Contains all dependencies the server needs, eliminating the need
/// for two-phase initialization with `set_*` methods. All fields are
/// immutable after construction.
pub struct ServerConfig {
  /// Path to the Unix socket for IPC
  pub socket_path: PathBuf,

  /// Project router for dispatching requests to ProjectActors
  pub router: Arc<ProjectRouter>,

  /// Activity tracker for idle detection
  pub activity: Arc<KeepAlive>,

  /// Session tracker for lifecycle management
  pub sessions: Arc<SessionTracker>,

  /// Daemon-level state for Status/Metrics requests
  pub daemon_state: Arc<DaemonState>,
}

// ============================================================================
// Server
// ============================================================================

/// IPC server that accepts connections and routes requests to ProjectActors.
///
/// The server listens on a Unix socket and spawns a task for each connection.
/// Requests are routed to `ProjectActor` instances via the `ProjectRouter`,
/// which spawns actors on demand.
///
/// # Lifecycle
///
/// 1. `Server::new()` creates the server with all dependencies
/// 2. `Server::run()` binds the socket and accepts connections
/// 3. Each connection spawns a `handle_connection` task
/// 4. On cancellation, cleanup and exit
///
/// # Threading Model
///
/// - Server accepts connections on main task
/// - Each connection runs in its own spawned task
/// - All tasks share the `ProjectRouter` via `Arc`
pub struct Server {
  config: ServerConfig,
  /// Total requests handled across all connections (for metrics)
  request_count: AtomicU64,
}

impl Server {
  /// Create a new server with the given configuration.
  ///
  /// All dependencies must be provided upfront - there are no `set_*` methods.
  pub fn new(config: ServerConfig) -> Self {
    Self {
      config,
      request_count: AtomicU64::new(0),
    }
  }

  /// Run the server until the cancellation token is triggered.
  ///
  /// This method:
  /// 1. Removes any stale socket file
  /// 2. Creates the socket parent directory if needed
  /// 3. Binds to the socket and accepts connections
  /// 4. Spawns a task for each connection
  /// 5. Cleans up on shutdown
  pub async fn run(&self, cancel: CancellationToken) -> Result<(), IpcError> {
    // Remove stale socket file
    if self.config.socket_path.exists() {
      tokio::fs::remove_file(&self.config.socket_path).await?;
    }

    // Create parent directory if needed
    if let Some(parent) = self.config.socket_path.parent() {
      tokio::fs::create_dir_all(parent).await?;
    }

    let listener = UnixListener::bind(&self.config.socket_path)?;
    info!("Server listening on {:?}", self.config.socket_path);

    #[cfg(all(not(target_env = "msvc"), feature = "jemalloc-pprof"))]
    {
      let pprof_sock = if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        std::path::PathBuf::from(runtime_dir).join("ccengram-pprof.sock")
      } else {
        let uid = unsafe { libc::getuid() };
        std::path::PathBuf::from(format!("/tmp/{}-pprof.sock", uid))
      };

      let pprof_listener = UnixListener::bind(&pprof_sock)?;

      tokio::spawn(async move {
        loop {
          match pprof_listener.accept().await {
            Ok((stream, _)) => {
              let framed = Framed::new(stream, tokio_util::codec::BytesCodec::new());
              let (mut sink, _) = framed.split();

              let mut prof_ctl = jemalloc_pprof::PROF_CTL.as_ref().unwrap().lock().await;
              let Ok(pprof) = prof_ctl.dump_pprof() else {
                warn!("failed to get pprof data");
                continue;
              };

              sink
                .send(Into::<tokio_util::bytes::Bytes>::into(pprof))
                .await
                .unwrap_or_else(|e| {
                  error!("failed to send pprof data: {}", e);
                });
            }
            Err(e) => {
              error!("Accept error on pprof socket: {}", e);
              break;
            }
          }
        }
      });
    }

    loop {
      tokio::select! {
        biased;

        _ = cancel.cancelled() => {
            info!("Server shutting down (cancelled)");
            break;
        }

        result = listener.accept() => {
          match result {
            Ok((stream, _)) => {
              // Touch activity tracker on any connection
              self.config.activity.touch();

              let router = Arc::clone(&self.config.router);
              let activity = Arc::clone(&self.config.activity);
              let sessions = Arc::clone(&self.config.sessions);
              let daemon_state = Arc::clone(&self.config.daemon_state);
              let cancel_token = cancel.clone();
              let request_count = &self.request_count;

              // Increment connection count (we track requests inside handle_connection)
              let _ = request_count;

              tokio::spawn(handle_connection(stream, router, activity, sessions, daemon_state, cancel_token));
            }
            Err(e) => {
              error!("Accept error: {}", e);
            }
          }
        }
      }
    }

    // Cleanup socket file
    if self.config.socket_path.exists() {
      tokio::fs::remove_file(&self.config.socket_path).await?;
    }

    Ok(())
  }
}

// ============================================================================
// Connection Handler
// ============================================================================

/// Handle a single client connection.
///
/// This function:
/// 1. Reads newline-delimited JSON requests from the client
/// 2. Routes each request to the appropriate ProjectActor via the router
/// 3. Streams responses back to the client until a final response
/// 4. Touches the activity tracker on each request
///
/// # Protocol
///
/// - Requests: JSON objects, one per line
/// - Responses: JSON objects, one per line (may be multiple for streaming)
/// - A response with `is_final()` == true ends the request
///
/// # Error Handling
///
/// - Parse errors return an error response but don't close the connection
/// - Actor errors return an error response but don't close the connection
/// - IO errors close the connection
async fn handle_connection(
  stream: UnixStream,
  router: Arc<ProjectRouter>,
  activity: Arc<KeepAlive>,
  sessions: Arc<SessionTracker>,
  daemon_state: Arc<DaemonState>,
  cancel: CancellationToken,
) -> Result<(), IpcError> {
  debug!("Client connected");
  let framed = Framed::new(stream, LinesCodec::new());
  let (mut sink, mut stream) = framed.split();
  let mut request_count = 0u64;

  while let Some(result) = stream.next().await {
    let line = match result {
      Ok(l) => l,
      Err(e) => {
        warn!(error = %e, "Error reading from client");
        break;
      }
    };

    // Touch activity tracker on every request
    activity.touch();
    request_count += 1;

    let trimmed = line.trim();
    if trimmed.is_empty() {
      continue;
    }

    // Parse request
    let request: Request = match serde_json::from_str(trimmed) {
      Ok(r) => r,
      Err(e) => {
        warn!("Invalid request JSON: {}", e);
        let response = Response::rpc_error("unknown", -32700, format!("Parse error: {}", e));
        let json = serde_json::to_string(&response)?;
        sink.send(json).await?;
        continue;
      }
    };

    let start = std::time::Instant::now();
    trace!(method = ?request.data, id = %request.id, cwd = %request.cwd, "Processing request");

    // Track sessions for lifecycle management
    if let RequestData::Hook(ref params) = request.data
      && let Some(ref session_id) = params.session_id
    {
      let sid = SessionId::from(session_id.as_str());
      match params.hook_name.as_str() {
        "SessionStart" => {
          sessions.register(sid).await;
        }
        "SessionEnd" => {
          sessions.unregister(&sid).await;
        }
        _ => {
          // Touch session on any other hook to keep it alive
          sessions.touch(&sid).await;
        }
      }
    }

    // Handle daemon-level system requests directly (Status, Metrics, Shutdown)
    // These don't need a project context
    if let RequestData::System(ref sys_req) = request.data
      && let Some(response) = handle_daemon_request(
        &request.id,
        sys_req,
        &daemon_state,
        &router,
        &activity,
        &sessions,
        &cancel,
      )
      .await
    {
      let json = serde_json::to_string(&response)?;
      sink.send(json).await?;
      let elapsed = start.elapsed();
      debug!(id = %request.id, elapsed_ms = elapsed.as_millis() as u64, "Daemon request completed");
      continue;
    }

    // Get or create project actor for this request's cwd
    let project_path = PathBuf::from(&request.cwd);
    let handle = match router.get_or_create(&project_path).await {
      Ok(h) => h,
      Err(e) => {
        let response = Response::rpc_error(&request.id, -32000, format!("Failed to get project: {}", e));
        let json = serde_json::to_string(&response)?;
        sink.send(json).await?;
        continue;
      }
    };

    // Convert IPC request to actor message payload
    let payload = ProjectActorPayload::Request(request.data);

    // Send request to project actor and get response channel
    let mut reply_rx = match handle.send(request.id.clone(), payload).await {
      Ok(rx) => rx,
      Err(e) => {
        let response = Response::rpc_error(&request.id, -32000, format!("Failed to send to actor: {}", e));
        let json = serde_json::to_string(&response)?;
        sink.send(json).await?;
        continue;
      }
    };

    // Stream responses until we get a final one
    while let Some(response) = reply_rx.recv().await {
      let ipc_response = convert_actor_response(&request.id, response.clone());
      let json = serde_json::to_string(&ipc_response)?;
      sink.send(json).await?;

      if response.is_final() {
        break;
      }
    }

    let elapsed = start.elapsed();
    debug!(
        id = %request.id,
        elapsed_ms = elapsed.as_millis() as u64,
        "Request completed"
    );
  }

  debug!(requests_handled = request_count, "Client disconnected");
  Ok(())
}

/// Convert an actor response to an IPC response.
///
/// This handles the different response types:
/// - `Progress` → stream chunk with status info
/// - `Stream` → stream chunk with data
/// - `Done` → success response
/// - `Error` → error response
fn convert_actor_response(request_id: &str, response: ProjectActorResponse) -> Response {
  match response {
    ProjectActorResponse::Progress {
      message,
      percent,
      stage,
      processed,
      total,
      current_file,
      chunks_created,
    } => Response::stream_progress_full(
      request_id,
      crate::ipc::StreamProgress {
        message,
        percent,
        stage,
        processed,
        total,
        current_file,
        chunks_created,
      },
    ),
    ProjectActorResponse::Stream { data } => Response::stream_chunk(request_id, data),
    ProjectActorResponse::Done(data) => Response::success(request_id, data),
    ProjectActorResponse::Error { code, message } => Response::rpc_error(request_id, code, message),
  }
}

/// Handle daemon-level system requests that don't need a project context.
///
/// Returns `Some(Response)` if the request was handled, `None` if it should
/// be routed to a ProjectActor.
async fn handle_daemon_request(
  request_id: &str,
  sys_req: &SystemRequest,
  daemon_state: &DaemonState,
  router: &ProjectRouter,
  activity: &KeepAlive,
  sessions: &SessionTracker,
  cancel: &CancellationToken,
) -> Option<Response> {
  match sys_req {
    SystemRequest::Status(_) => {
      let uptime = daemon_state.start_time.elapsed().as_secs();
      let idle_secs = activity.idle_duration().as_secs();
      let active_sessions = sessions.active_count().await;
      let projects = router.list().len();

      let result = StatusResult {
        status: "running".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        pid: daemon_state.pid,
        projects,
        active_sessions,
        idle_seconds: idle_secs,
        uptime_seconds: uptime,
        foreground: daemon_state.foreground,
        auto_shutdown: daemon_state.auto_shutdown,
      };

      Some(Response::success(
        request_id,
        ResponseData::System(SystemResponse::Status(result)),
      ))
    }
    SystemRequest::Metrics(_) => {
      let uptime = daemon_state.start_time.elapsed().as_secs();
      let idle_secs = activity.idle_duration().as_secs();

      let session_list = sessions.list_sessions().await;
      let session_ids: Vec<String> = session_list.iter().map(|s| s.0.clone()).collect();

      let project_ids = router.list();
      let project_names: Vec<String> = project_ids.iter().map(|id| id.as_str().to_string()).collect();

      let (emb_name, emb_model, emb_dims) = router.embedding_info();

      // Get RSS from /proc/self/statm on Linux
      let rss_kb = get_rss_kb().await;

      let result = MetricsResult {
        daemon: DaemonMetrics {
          version: env!("CARGO_PKG_VERSION").to_string(),
          uptime_seconds: uptime,
          idle_seconds: idle_secs,
          foreground: daemon_state.foreground,
          auto_shutdown: daemon_state.auto_shutdown,
        },
        requests: RequestsMetrics {
          total: 0, // TODO: add request counter if needed
          per_second: 0.0,
        },
        sessions: SessionsMetrics {
          active: session_ids.len(),
          ids: session_ids,
        },
        projects: ProjectsMetrics {
          count: project_names.len(),
          names: project_names,
        },
        embedding: Some(EmbeddingProviderInfo {
          name: emb_name,
          model: emb_model,
          dimensions: emb_dims,
        }),
        memory: MemoryUsageMetrics { rss_kb },
      };

      Some(Response::success(
        request_id,
        ResponseData::System(SystemResponse::Metrics(result)),
      ))
    }
    SystemRequest::Shutdown(_) => {
      info!("Shutdown requested via RPC");
      cancel.cancel();
      Some(Response::success(
        request_id,
        ResponseData::System(SystemResponse::Shutdown {
          message: "Daemon shutting down".to_string(),
        }),
      ))
    }
    // Other requests fall through to ProjectActor
    _ => None,
  }
}

/// Get RSS memory usage in KB from /proc/self/statm on Linux.
/// Returns None on non-Linux or if reading fails.
async fn get_rss_kb() -> Option<u64> {
  #[cfg(target_os = "linux")]
  {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open("/proc/self/statm").await.ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await.ok()?;
    let fields: Vec<&str> = contents.split_whitespace().collect();
    // statm format: size resident shared text lib data dt (all in pages)
    // We want resident (index 1), multiply by page size (usually 4KB)
    let resident_pages: u64 = fields.get(1)?.parse().ok()?;
    let page_size_kb = 4; // Assume 4KB pages
    Some(resident_pages * page_size_kb)
  }
  #[cfg(not(target_os = "linux"))]
  {
    None
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ipc::system::SystemResponse;

  #[test]
  fn test_convert_actor_response_done() {
    let response = ProjectActorResponse::Done(ResponseData::System(SystemResponse::Ping("pong".to_string())));
    let ipc = convert_actor_response("test-1", response);

    assert_eq!(ipc.id, "test-1");
    match ipc.scenario {
      crate::ipc::ResponseScenario::Result { data } => {
        assert!(matches!(data, ResponseData::System(SystemResponse::Ping(_))));
      }
      _ => panic!("Expected Result scenario"),
    }
  }

  #[test]
  fn test_convert_actor_response_error() {
    let response = ProjectActorResponse::Error {
      code: -32000,
      message: "test error".to_string(),
    };
    let ipc = convert_actor_response("test-2", response);

    assert_eq!(ipc.id, "test-2");
    match ipc.scenario {
      crate::ipc::ResponseScenario::Error { error } => {
        assert!(matches!(error, IpcError::Rpc { code: -32000, .. }));
      }
      _ => panic!("Expected Error scenario"),
    }
  }

  #[test]
  fn test_convert_actor_response_stream() {
    let response = ProjectActorResponse::Stream {
      data: ResponseData::System(SystemResponse::Ping("streaming".to_string())),
    };
    let ipc = convert_actor_response("test-3", response);

    assert_eq!(ipc.id, "test-3");
    match ipc.scenario {
      crate::ipc::ResponseScenario::Stream { chunk, done, .. } => {
        assert!(chunk.is_some());
        assert!(!done);
      }
      _ => panic!("Expected Stream scenario"),
    }
  }

  #[test]
  fn test_convert_actor_response_progress() {
    let response = ProjectActorResponse::Progress {
      message: "Indexing files".to_string(),
      percent: Some(50),
      stage: Some("embedding".to_string()),
      processed: Some(25),
      total: Some(50),
      current_file: Some("src/main.rs".to_string()),
      chunks_created: Some(100),
    };
    let ipc = convert_actor_response("test-4", response);

    assert_eq!(ipc.id, "test-4");
    match ipc.scenario {
      crate::ipc::ResponseScenario::Stream { chunk, progress, done } => {
        assert!(chunk.is_none());
        assert!(!done);
        let p = progress.expect("Expected progress");
        assert_eq!(p.message, "Indexing files");
        assert_eq!(p.percent, Some(50));
        assert_eq!(p.stage, Some("embedding".to_string()));
        assert_eq!(p.processed, Some(25));
        assert_eq!(p.total, Some(50));
      }
      _ => panic!("Expected Stream scenario"),
    }
  }
}
