use std::{
  collections::HashMap,
  path::{Path, PathBuf},
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

use futures::{SinkExt, StreamExt};
use tokio::{net::UnixStream, sync::mpsc};
use tokio_util::codec::{Framed, LinesCodec};
use tracing::{debug, error, warn};

use super::{IpcError, Request, RequestData, Response, ResponseData, ResponseScenario};

type FramedStream = Framed<UnixStream, LinesCodec>;

/// Progress info for a pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct StageProgress {
  /// Pipeline stage name (scanning, reading, parsing, embedding, writing)
  pub stage: Option<String>,
  /// Files processed in this stage
  pub processed: Option<usize>,
  /// Total files to process in this stage
  pub total: Option<usize>,
  /// Current file being processed
  pub current_file: Option<String>,
  /// Chunks created so far (populated during writing)
  pub chunks_created: Option<usize>,
}

/// Update from a streaming request.
#[derive(Debug, Clone)]
pub enum StreamUpdate<T> {
  /// Progress update with optional message, percent, and stage info.
  Progress {
    message: String,
    percent: Option<u8>,
    stage: StageProgress,
  },
  /// Final result (success or error).
  Done(Result<T, IpcError>),
}

struct OutboundRequest {
  request: Request,
  response_tx: mpsc::Sender<Response>,
}

/// Trait for typed IPC requests that know their response type.
///
/// Implement this trait for request parameter types to enable type-safe
/// request/response handling via `Client::call()`.
pub trait IpcRequest: Into<RequestData> + Clone {
  /// The expected response type for this request.
  type Response;

  /// Extract the typed response from a ResponseData.
  /// Returns `Err(IpcError::NoResult)` if the response variant doesn't match.
  fn extract(data: ResponseData) -> Result<Self::Response, IpcError>;
}

/// Client for connecting to the daemon.
#[derive(Clone)]
pub struct Client {
  cwd: PathBuf,
  request_tx: mpsc::Sender<OutboundRequest>,
  counter: Arc<AtomicU64>,
}

impl Client {
  pub async fn connect(cwd: PathBuf) -> Result<Self, IpcError> {
    Self::connect_to(cwd, &crate::dirs::default_socket_path()).await
  }

  pub async fn connect_to(cwd: PathBuf, socket_path: &Path) -> Result<Self, IpcError> {
    let stream = UnixStream::connect(socket_path).await?;
    let framed = Framed::new(stream, LinesCodec::new());
    let (sink, read_stream) = framed.split();

    let (request_tx, request_rx) = mpsc::channel(64);
    tokio::spawn(Self::multiplexer(sink, read_stream, request_rx));

    Ok(Self {
      cwd,
      request_tx,
      counter: Arc::new(AtomicU64::new(1)),
    })
  }

  async fn multiplexer(
    mut sink: futures::stream::SplitSink<FramedStream, String>,
    mut stream: futures::stream::SplitStream<FramedStream>,
    mut request_rx: mpsc::Receiver<OutboundRequest>,
  ) {
    let mut pending: HashMap<String, mpsc::Sender<Response>> = HashMap::new();

    loop {
      tokio::select! {
        Some(outbound) = request_rx.recv() => {
          let id = outbound.request.id.clone();
          match serde_json::to_string(&outbound.request) {
            Ok(json) => {
              pending.insert(id.clone(), outbound.response_tx);
              if let Err(e) = sink.send(json).await {
                error!("failed to send request: {e}");
                if let Some(tx) = pending.remove(&id) {
                  let _ = tx.send(Response::error(id, IpcError::Connection(e.to_string()))).await;
                }
              }
            }
            Err(e) => {
              let _ = outbound.response_tx.send(Response::error(id, IpcError::Serde(e.to_string()))).await;
            }
          }
        }

        result = stream.next() => {
          match result {
            Some(Ok(line)) => {
              match serde_json::from_str::<Response>(&line) {
                Ok(response) => {
                  let id = response.id.clone();
                  let is_final = match &response.scenario {
                    ResponseScenario::Stream { done, .. } => *done,
                    _ => true,
                  };

                  if let Some(tx) = pending.get(&id) {
                    if tx.send(response).await.is_err() {
                      debug!("receiver dropped for request {id}");
                      pending.remove(&id);
                    } else if is_final {
                      pending.remove(&id);
                    }
                  } else {
                    warn!("received response for unknown request id: {id}");
                  }
                }
                Err(e) => {
                  error!("failed to parse response: {e}");
                }
              }
            }
            Some(Err(e)) => {
              error!("connection error: {e}");
              break;
            }
            None => {
              debug!("connection closed");
              break;
            }
          }
        }
      }
    }

    for (id, tx) in pending {
      let _ = tx
        .send(Response::error(id, IpcError::Connection("connection closed".into())))
        .await;
    }

    debug!("multiplexer exited");
  }

  /// Send a typed request and receive a typed response.
  ///
  /// This is the preferred API when you want compile-time type safety.
  /// The response type is determined by the request type via `IpcRequest`.
  pub async fn call<R: IpcRequest>(&self, req: R) -> Result<R::Response, IpcError> {
    let response = self.request(req).await?;
    match response.scenario {
      ResponseScenario::Result { data } => R::extract(data),
      ResponseScenario::Error { error } => Err(error),
      ResponseScenario::Stream { chunk: Some(data), .. } => R::extract(data),
      ResponseScenario::Stream { chunk: None, .. } => Err(IpcError::NoResult),
    }
  }

  /// Send a typed request and receive a stream of progress updates.
  ///
  /// Returns a receiver that yields `StreamUpdate` items containing either:
  /// - Progress messages (percent complete, message)
  /// - The final typed result
  ///
  /// Use this for long-running operations like indexing where you want to
  /// show progress to the user.
  pub async fn call_streaming<R>(&self, req: R) -> Result<mpsc::Receiver<StreamUpdate<R::Response>>, IpcError>
  where
    R: IpcRequest + Send + 'static,
    R::Response: Send + 'static,
  {
    let rx = self.request_stream(req).await?;
    let (update_tx, update_rx) = mpsc::channel(16);

    tokio::spawn(async move {
      let mut rx = rx;
      while let Some(response) = rx.recv().await {
        let update = match response.scenario {
          ResponseScenario::Result { data } => match R::extract(data) {
            Ok(result) => StreamUpdate::Done(Ok(result)),
            Err(e) => StreamUpdate::Done(Err(e)),
          },
          ResponseScenario::Error { error } => StreamUpdate::Done(Err(error)),
          ResponseScenario::Stream { chunk, progress, done } => {
            if done {
              // Final chunk
              match chunk {
                Some(data) => match R::extract(data) {
                  Ok(result) => StreamUpdate::Done(Ok(result)),
                  Err(e) => StreamUpdate::Done(Err(e)),
                },
                None => StreamUpdate::Done(Err(IpcError::NoResult)),
              }
            } else {
              // Progress update with stage info
              let stage = progress
                .as_ref()
                .map(|p| StageProgress {
                  stage: p.stage.clone(),
                  processed: p.processed,
                  total: p.total,
                  current_file: p.current_file.clone(),
                  chunks_created: p.chunks_created,
                })
                .unwrap_or_default();

              StreamUpdate::Progress {
                message: progress.as_ref().map(|p| p.message.clone()).unwrap_or_default(),
                percent: progress.as_ref().and_then(|p| p.percent),
                stage,
              }
            }
          }
        };

        let is_done = matches!(update, StreamUpdate::Done(_));
        if update_tx.send(update).await.is_err() {
          break;
        }
        if is_done {
          break;
        }
      }
    });

    Ok(update_rx)
  }

  /// Send a request and receive a single untyped response.
  async fn request(&self, data: impl Into<RequestData>) -> Result<Response, IpcError> {
    let mut rx = self.request_stream(data).await?;
    rx.recv()
      .await
      .ok_or_else(|| IpcError::Connection("no response received".into()))
  }

  /// Send a request and receive a stream of responses.
  async fn request_stream(&self, data: impl Into<RequestData>) -> Result<mpsc::Receiver<Response>, IpcError> {
    let id = self.counter.fetch_add(1, Ordering::Relaxed);

    let request = Request {
      id: id.to_string(),
      cwd: self.cwd.to_string_lossy().to_string(),
      data: data.into(),
    };

    self.raw_request_stream(&request).await
  }

  /// Send a raw request and receive a stream of responses.
  async fn raw_request_stream(&self, request: &Request) -> Result<mpsc::Receiver<Response>, IpcError> {
    let (response_tx, response_rx) = mpsc::channel(16);

    self
      .request_tx
      .send(OutboundRequest {
        request: request.clone(),
        response_tx,
      })
      .await
      .map_err(|_| IpcError::Connection("multiplexer died".into()))?;

    Ok(response_rx)
  }

  pub fn cwd(&self) -> &Path {
    &self.cwd
  }

  pub fn change_cwd(&mut self, new_cwd: PathBuf) {
    self.cwd = new_cwd;
  }

  /// Send a typed request without waiting for a response.
  ///
  /// This is useful for fire-and-forget operations like hooks where
  /// blocking on the response would delay the caller unnecessarily.
  ///
  /// Opens a direct connection to avoid race conditions with process exit -
  /// ensures the request is fully written before returning.
  pub async fn fire_and_forget<R: Into<RequestData>>(&self, req: R) -> Result<(), IpcError> {
    Self::fire_and_forget_to(self.cwd.clone(), &crate::dirs::default_socket_path(), req).await
  }

  /// Send a typed request to a specific socket without waiting for a response.
  pub async fn fire_and_forget_to<R: Into<RequestData>>(
    cwd: PathBuf,
    socket_path: &Path,
    req: R,
  ) -> Result<(), IpcError> {
    use futures::SinkExt;
    use tokio_util::codec::{Framed, LinesCodec};

    let stream = UnixStream::connect(socket_path).await?;
    let mut framed = Framed::new(stream, LinesCodec::new());

    let request = Request {
      id: "fire-and-forget".to_string(),
      cwd: cwd.to_string_lossy().to_string(),
      data: req.into(),
    };

    let json = serde_json::to_string(&request).map_err(|e| IpcError::Serde(e.to_string()))?;

    framed
      .send(json)
      .await
      .map_err(|e| IpcError::Connection(e.to_string()))?;

    Ok(())
  }
}

/// Helper to collect all stream chunks into a Vec.
pub async fn collect_stream(mut rx: mpsc::Receiver<Response>) -> Vec<Response> {
  let mut responses = Vec::new();
  while let Some(response) = rx.recv().await {
    let is_done = matches!(response.scenario, ResponseScenario::Stream { done: true, .. });
    responses.push(response);
    if is_done {
      break;
    }
  }
  responses
}

/// Macro to implement IpcRequest for a request type.
///
/// Full form (4 args) - generates IpcRequest, From<Req> for RequestData, From<Resp> for ResponseData:
/// ```ignore
/// impl_ipc_request!(
///   MemorySearchParams => MemorySearchResult,
///   ResponseData::Memory(MemoryResponse::Search(v)) => v,
///   v => RequestData::Memory(MemoryRequest::Search(v)),
///   v => ResponseData::Memory(MemoryResponse::Search(v))
/// );
/// ```
///
/// Short form (3 args) - skips From<Resp> for ResponseData (use when response type is shared):
/// ```ignore
/// impl_ipc_request!(
///   MemoryDeemphasizeParams => MemoryUpdateResult,
///   ResponseData::Memory(MemoryResponse::Update(v)) => v,
///   v => RequestData::Memory(MemoryRequest::Deemphasize(v))
/// );
/// ```
#[macro_export]
macro_rules! impl_ipc_request {
  // Full form: generates all three impls
  ($req:ty => $resp:ty, $resp_pattern:pat => $resp_extract:expr, $req_ident:ident => $req_construct:expr, $resp_ident:ident => $resp_construct:expr) => {
    impl $crate::ipc::client::IpcRequest for $req {
      type Response = $resp;

      fn extract(data: $crate::ipc::ResponseData) -> Result<Self::Response, $crate::ipc::IpcError> {
        match data {
          $resp_pattern => Ok($resp_extract),
          _ => Err($crate::ipc::IpcError::NoResult),
        }
      }
    }

    impl From<$req> for $crate::ipc::RequestData {
      fn from($req_ident: $req) -> Self {
        $req_construct
      }
    }

    impl From<$resp> for $crate::ipc::ResponseData {
      fn from($resp_ident: $resp) -> Self {
        $resp_construct
      }
    }
  };

  // Short form: skips From<Response> for ResponseData (for shared response types)
  ($req:ty => $resp:ty, $resp_pattern:pat => $resp_extract:expr, $req_ident:ident => $req_construct:expr) => {
    impl $crate::ipc::client::IpcRequest for $req {
      type Response = $resp;

      fn extract(data: $crate::ipc::ResponseData) -> Result<Self::Response, $crate::ipc::IpcError> {
        match data {
          $resp_pattern => Ok($resp_extract),
          _ => Err($crate::ipc::IpcError::NoResult),
        }
      }
    }

    impl From<$req> for $crate::ipc::RequestData {
      fn from($req_ident: $req) -> Self {
        $req_construct
      }
    }
  };
}

pub use impl_ipc_request;
