use crate::router::{Request, Response, Router};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info, warn};

/// Sender type for streaming progress updates
pub type ProgressSender = mpsc::Sender<Response>;

#[derive(Error, Debug)]
pub enum ServerError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("JSON error: {0}")]
  Json(#[from] serde_json::Error),
  #[error("Server shutdown")]
  Shutdown,
  #[error("Connection error: {0}")]
  Connection(String),
}

/// Get the default socket path
pub fn default_socket_path() -> PathBuf {
  // Try XDG_RUNTIME_DIR first, fallback to /tmp
  if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
    PathBuf::from(runtime_dir).join("ccengram.sock")
  } else {
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/{}.sock", uid))
  }
}

/// Unix socket server for the daemon
pub struct Server {
  socket_path: PathBuf,
  router: Arc<Router>,
  shutdown_tx: broadcast::Sender<()>,
}

impl Server {
  pub fn new(router: Router) -> Self {
    Self::with_socket_path(Arc::new(router), default_socket_path())
  }

  pub fn with_socket_path(router: Arc<Router>, socket_path: PathBuf) -> Self {
    let (shutdown_tx, _) = broadcast::channel(1);
    Self {
      socket_path,
      router,
      shutdown_tx,
    }
  }

  /// Get a shutdown handle to signal server shutdown
  pub fn shutdown_handle(&self) -> ShutdownHandle {
    ShutdownHandle {
      tx: self.shutdown_tx.clone(),
    }
  }

  /// Get the socket path
  pub fn socket_path(&self) -> &Path {
    &self.socket_path
  }

  /// Run the server
  pub async fn run(&self) -> Result<(), ServerError> {
    // Remove stale socket file
    if self.socket_path.exists() {
      std::fs::remove_file(&self.socket_path)?;
    }

    // Create parent directory if needed
    if let Some(parent) = self.socket_path.parent() {
      std::fs::create_dir_all(parent)?;
    }

    let listener = UnixListener::bind(&self.socket_path)?;
    info!("Daemon listening on {:?}", self.socket_path);

    let mut shutdown_rx = self.shutdown_tx.subscribe();

    loop {
      tokio::select! {
          result = listener.accept() => {
              match result {
                  Ok((stream, _)) => {
                      let router = Arc::clone(&self.router);
                      tokio::spawn(async move {
                          if let Err(e) = handle_connection(stream, router).await {
                              error!("Connection error: {}", e);
                          }
                      });
                  }
                  Err(e) => {
                      error!("Accept error: {}", e);
                  }
              }
          }
          _ = shutdown_rx.recv() => {
              info!("Shutdown signal received");
              break;
          }
      }
    }

    // Cleanup socket file
    if self.socket_path.exists() {
      std::fs::remove_file(&self.socket_path)?;
    }

    Ok(())
  }
}

/// Handle to signal server shutdown
#[derive(Clone)]
pub struct ShutdownHandle {
  tx: broadcast::Sender<()>,
}

impl ShutdownHandle {
  /// Trigger shutdown
  pub fn shutdown(&self) {
    let _ = self.tx.send(());
  }

  /// Create a ShutdownHandle from a broadcast sender (for testing)
  pub fn from_sender(tx: broadcast::Sender<()>) -> Self {
    Self { tx }
  }

  /// Subscribe to shutdown signals
  pub fn subscribe(&self) -> broadcast::Receiver<()> {
    self.tx.subscribe()
  }
}

/// Check if a request is for a streaming method
fn is_streaming_request(request: &Request) -> bool {
  // Check if the request has stream: true in params
  if let Some(stream) = request.params.get("stream") {
    return stream.as_bool().unwrap_or(false);
  }
  false
}

/// Handle a single client connection
async fn handle_connection(stream: UnixStream, router: Arc<Router>) -> Result<(), ServerError> {
  debug!("Client connected");
  let (reader, mut writer) = stream.into_split();
  let mut reader = BufReader::new(reader);
  let mut line = String::new();
  let mut request_count = 0u64;

  loop {
    line.clear();
    let n = reader.read_line(&mut line).await?;

    if n == 0 {
      // Client disconnected
      debug!(requests_handled = request_count, "Client disconnected");
      break;
    }

    let trimmed = line.trim();
    if trimmed.is_empty() {
      continue;
    }

    // Parse request
    let request: Request = match serde_json::from_str(trimmed) {
      Ok(r) => r,
      Err(e) => {
        warn!("Invalid request JSON: {}", e);
        let response = Response::error(None, -32700, &format!("Parse error: {}", e));
        let json = serde_json::to_string(&response)?;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
        continue;
      }
    };

    let start = std::time::Instant::now();
    request_count += 1;
    debug!(method = %request.method, id = ?request.id, "Processing request");

    // Check if this is a streaming request
    if is_streaming_request(&request) {
      // Handle streaming request
      let (tx, mut rx) = mpsc::channel::<Response>(32);

      // Spawn the handler with progress sender
      let router_clone = Arc::clone(&router);
      let request_clone = request.clone();
      tokio::spawn(async move {
        router_clone.handle_streaming(request_clone, tx).await;
      });

      // Read from channel and write to socket until channel closes
      while let Some(response) = rx.recv().await {
        let json = serde_json::to_string(&response)?;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;

        // If this response has a result or error, it's the final one
        if response.result.is_some() || response.error.is_some() {
          break;
        }
      }
    } else {
      // Route and handle normally (single response)
      let response = router.handle(request.clone()).await;

      // Send response
      let json = serde_json::to_string(&response)?;
      writer.write_all(json.as_bytes()).await?;
      writer.write_all(b"\n").await?;
      writer.flush().await?;

      let elapsed = start.elapsed();
      debug!(
        method = %request.method,
        elapsed_ms = elapsed.as_millis() as u64,
        has_error = response.error.is_some(),
        "Request completed"
      );
    }
  }

  Ok(())
}

/// Client for connecting to the daemon
pub struct Client {
  stream: UnixStream,
}

impl Client {
  /// Connect to daemon at the default socket path
  pub async fn connect() -> Result<Self, ServerError> {
    Self::connect_to(&default_socket_path()).await
  }

  /// Connect to daemon at a specific socket path
  pub async fn connect_to(socket_path: &Path) -> Result<Self, ServerError> {
    let stream = UnixStream::connect(socket_path).await?;
    Ok(Self { stream })
  }

  /// Send a request and receive response
  pub async fn request(&mut self, request: Request) -> Result<Response, ServerError> {
    let (reader, mut writer) = self.stream.split();

    // Send request
    let json = serde_json::to_string(&request)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;

    // Read response
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    let response: Response = serde_json::from_str(&line)?;
    Ok(response)
  }

  /// Send a request with a method and params
  pub async fn call(&mut self, method: &str, params: serde_json::Value) -> Result<Response, ServerError> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let request = Request {
      id: Some(serde_json::Value::Number(id.into())),
      method: method.to_string(),
      params,
    };

    self.request(request).await
  }

  /// Send a streaming request and return a StreamingResponse for reading progress.
  pub async fn request_streaming(self, request: Request) -> Result<StreamingResponse, ServerError> {
    let (reader, mut writer) = self.stream.into_split();

    // Send request
    let json = serde_json::to_string(&request)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;

    Ok(StreamingResponse {
      reader: BufReader::new(reader),
      line_buf: String::new(),
    })
  }
}

/// A streaming response reader that yields progress updates then final result
pub struct StreamingResponse {
  reader: BufReader<tokio::net::unix::OwnedReadHalf>,
  line_buf: String,
}

impl StreamingResponse {
  /// Receive the next response (progress or final result).
  /// Returns None when the stream is closed.
  pub async fn recv(&mut self) -> Option<Response> {
    self.line_buf.clear();

    match self.reader.read_line(&mut self.line_buf).await {
      Ok(0) => None, // EOF
      Ok(_) => {
        let trimmed = self.line_buf.trim();
        if trimmed.is_empty() {
          // Skip empty lines, try again
          return Box::pin(self.recv()).await;
        }

        match serde_json::from_str::<Response>(trimmed) {
          Ok(response) => Some(response),
          Err(e) => {
            warn!("Failed to parse streaming response: {}", e);
            None
          }
        }
      }
      Err(e) => {
        warn!("Error reading streaming response: {}", e);
        None
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_default_socket_path() {
    let path = default_socket_path();
    assert!(path.to_string_lossy().contains("ccengram"));
  }

  #[tokio::test]
  async fn test_server_client_roundtrip() {
    let dir = TempDir::new().unwrap();
    let socket_path = dir.path().join("test.sock");

    let router = Arc::new(Router::new());
    let server = Server::with_socket_path(router, socket_path.clone());
    let shutdown = server.shutdown_handle();

    // Start server in background
    let server_handle = tokio::spawn(async move { server.run().await });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Connect client
    let mut client = Client::connect_to(&socket_path).await.unwrap();

    // Send a ping request
    let response = client.call("ping", serde_json::json!({})).await.unwrap();

    // ping returns "pong"
    assert!(response.result.is_some() || response.error.is_some());

    // Shutdown server
    shutdown.shutdown();
    let _ = server_handle.await;
  }
}
