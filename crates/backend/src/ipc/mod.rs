//! IPC module - client/server communication and type definitions
use serde::{Deserialize, Serialize};

pub mod types;

pub mod client;

pub use client::{Client, IpcRequest, StreamUpdate, collect_stream};
pub use types::*;

#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum IpcError {
  #[error("Ser/de error: {0}")]
  Serde(String),
  #[error("RPC error {code}: {message}")]
  Rpc { code: i32, message: String },
  #[error("No result in response")]
  NoResult,
  #[error("IO error: {0}")]
  Io(String),
  #[error("Server shutdown")]
  Shutdown,
  #[error("Connection error: {0}")]
  Connection(String),
  #[error("Codec error: {0}")]
  Codec(String),
}

impl From<serde_json::Error> for IpcError {
  fn from(err: serde_json::Error) -> Self {
    IpcError::Serde(err.to_string())
  }
}

impl From<std::io::Error> for IpcError {
  fn from(err: std::io::Error) -> Self {
    IpcError::Io(err.to_string())
  }
}

impl From<tokio_util::codec::LinesCodecError> for IpcError {
  fn from(err: tokio_util::codec::LinesCodecError) -> Self {
    IpcError::Codec(err.to_string())
  }
}

// ============================================================================
// Request/Response envelopes (top-level IPC protocol)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
  pub id: String,
  pub cwd: String, // path of the project making the request
  #[serde(flatten)]
  pub data: RequestData,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "method", content = "params")]
pub enum RequestData {
  System(system::SystemRequest),
  Memory(memory::MemoryRequest),
  Code(code::CodeRequest),
  Watch(watch::WatchRequest),
  Docs(docs::DocsRequest),
  Relationship(relationship::RelationshipRequest),
  Project(project::ProjectRequest),
  Hook(hook::HookParams),
  // Unified Search
  Explore(search::ExploreParams),
  Context(search::ContextParams),
}

// ============================================================================
// Response envelope
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
  pub id: String,
  #[serde(flatten)]
  pub scenario: ResponseScenario,
}

impl Response {
  pub fn is_ok(&self) -> bool {
    !self.is_error()
  }

  pub fn is_error(&self) -> bool {
    matches!(self.scenario, ResponseScenario::Error { .. })
  }

  pub fn is_success(&self) -> bool {
    matches!(self.scenario, ResponseScenario::Result { .. })
  }

  pub fn is_stream(&self) -> bool {
    matches!(self.scenario, ResponseScenario::Stream { .. })
  }

  pub fn get_data(&self) -> Option<&ResponseData> {
    match &self.scenario {
      ResponseScenario::Result { data } => Some(data),
      _ => None,
    }
  }

  pub fn get_error(&self) -> Option<&IpcError> {
    match &self.scenario {
      ResponseScenario::Error { error } => Some(error),
      _ => None,
    }
  }

  pub fn scenario(self) -> ResponseScenario {
    self.scenario
  }

  /// Create a success response with typed data
  pub fn success(id: impl Into<String>, data: ResponseData) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Result { data },
    }
  }

  /// Create an error response
  pub fn error(id: impl Into<String>, error: IpcError) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Error { error },
    }
  }

  /// Create an RPC error response with code and message
  pub fn rpc_error(id: impl Into<String>, code: i32, message: impl Into<String>) -> Self {
    Self::error(
      id,
      IpcError::Rpc {
        code,
        message: message.into(),
      },
    )
  }

  /// Create a stream chunk response
  pub fn stream_chunk(id: impl Into<String>, data: ResponseData) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Stream {
        chunk: Some(data),
        progress: None,
        done: false,
      },
    }
  }

  /// Create a stream progress response (no data, just progress info)
  pub fn stream_progress(id: impl Into<String>, message: impl Into<String>, percent: Option<u8>) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Stream {
        chunk: None,
        progress: Some(StreamProgress {
          message: message.into(),
          percent,
          stage: None,
          processed: None,
          total: None,
          current_file: None,
          chunks_created: None,
        }),
        done: false,
      },
    }
  }

  /// Create a rich stream progress response with full stage info
  pub fn stream_progress_full(id: impl Into<String>, progress: StreamProgress) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Stream {
        chunk: None,
        progress: Some(progress),
        done: false,
      },
    }
  }

  /// Create a stream done response
  pub fn stream_done(id: impl Into<String>) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Stream {
        chunk: None,
        progress: None,
        done: true,
      },
    }
  }

  /// Create a stream done response with final data
  pub fn stream_done_with_data(id: impl Into<String>, data: ResponseData) -> Self {
    Self {
      id: id.into(),
      scenario: ResponseScenario::Stream {
        chunk: Some(data),
        progress: None,
        done: true,
      },
    }
  }
}

/// Progress information for streaming responses.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProgress {
  /// Human-readable progress message
  pub message: String,
  /// Percent complete (0-100) for this stage
  pub percent: Option<u8>,
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

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseScenario {
  Error {
    error: IpcError,
  },
  Result {
    #[serde(flatten)]
    data: ResponseData,
  },
  Stream {
    chunk: Option<ResponseData>,
    progress: Option<StreamProgress>,
    done: bool,
  },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "method", content = "params")]
pub enum ResponseData {
  System(system::SystemResponse),
  Memory(memory::MemoryResponse),
  Code(code::CodeResponse),
  Watch(watch::WatchResponse),
  Docs(docs::DocsResponse),
  Relationship(relationship::RelationshipResponse),
  Project(project::ProjectResponse),
  Hook(hook::HookResult),
  // Unified Search
  Explore(search::ExploreResult),
  Context(Vec<search::ContextItem>),
}
