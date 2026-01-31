//! Actor message types for the daemon architecture
//!
//! All requests include an `mpsc::Sender` for responses, enabling streaming.
//!
//! ## Pipeline Message Types
//!
//! The streaming pipeline uses bounded channels with backpressure for file indexing:
//!
//! ```text
//! Scanner → Reader → Parser → Embedder → Writer
//!   256      128      256       64       flush
//! ```
//!
//! Each stage has its own message type defined in the pipeline modules.

use std::{path::PathBuf, sync::Arc};

use tokio::sync::mpsc;

use crate::ipc::{RequestData, ResponseData};

/// Unique identifier for a request (for correlation in logs and responses)
pub type RequestId = String;

// ============================================================================
// Project Actor Messages
// ============================================================================

/// A message sent to a ProjectActor
#[derive(Debug)]
pub struct ProjectActorMessage {
  /// Request ID for correlation
  pub id: RequestId,
  /// Channel to send responses (supports streaming via multiple sends)
  pub reply: mpsc::Sender<ProjectActorResponse>,
  /// The actual request payload
  pub payload: ProjectActorPayload,
}

#[allow(clippy::large_enum_variant)]
/// The payload of a project request
#[derive(Debug, Clone)]
pub enum ProjectActorPayload {
  /// Standard request from IPC layer
  Request(RequestData),
  /// Apply memory decay (scheduler-triggered)
  ApplyDecay,
  /// Cleanup stale sessions (scheduler-triggered)
  CleanupSessions {
    /// Maximum session age in hours
    max_age_hours: u64,
  },
  /// Shutdown this project actor
  Shutdown,
}

/// Response from a ProjectActor
#[derive(Debug, Clone)]
pub enum ProjectActorResponse {
  /// Streaming progress update (not final)
  Progress {
    message: String,
    percent: Option<u8>,
    /// Pipeline stage (scanning, reading, parsing, embedding, writing)
    stage: Option<String>,
    /// Files processed in this stage
    processed: Option<usize>,
    /// Total files to process in this stage
    total: Option<usize>,
    /// Current file being processed
    current_file: Option<String>,
    /// Chunks created so far
    chunks_created: Option<usize>,
  },
  #[allow(dead_code)]
  /// Streaming data chunk (not final)
  Stream { data: ResponseData },
  /// Final success response
  Done(ResponseData),
  /// Error response (final)
  Error { code: i32, message: String },
}

impl ProjectActorResponse {
  /// Returns true if this is a final response (Done or Error)
  pub fn is_final(&self) -> bool {
    matches!(self, Self::Done(_) | Self::Error { .. })
  }

  /// Create a simple progress response (for backward compatibility)
  pub fn progress(message: impl Into<String>, percent: Option<u8>) -> Self {
    Self::Progress {
      message: message.into(),
      percent,
      stage: None,
      processed: None,
      total: None,
      current_file: None,
      chunks_created: None,
    }
  }

  /// Create a progress response from IndexProgress
  pub fn from_index_progress(progress: &IndexProgress) -> Self {
    Self::Progress {
      message: progress.current_file.clone().unwrap_or_default(),
      percent: Some(progress.percent()),
      stage: Some(progress.stage.as_str().to_string()),
      processed: Some(progress.processed),
      total: Some(progress.total),
      current_file: progress.current_file.clone(),
      chunks_created: if progress.chunks_created > 0 {
        Some(progress.chunks_created)
      } else {
        None
      },
    }
  }

  /// Create an error response
  pub fn error(code: i32, message: impl Into<String>) -> Self {
    Self::Error {
      code,
      message: message.into(),
    }
  }

  /// Create a "method not found" error (JSON-RPC standard code)
  pub fn method_not_found(method: &str) -> Self {
    Self::Error {
      code: -32601,
      message: format!("Method not found: {}", method),
    }
  }

  /// Create an internal error
  pub fn internal_error(message: impl Into<String>) -> Self {
    Self::Error {
      code: -32000,
      message: message.into(),
    }
  }
}

// ============================================================================
// Indexer Actor Messages
// ============================================================================

/// A job for the IndexerActor
#[derive(Debug)]
pub enum IndexJob {
  /// Index a single file
  File {
    path: PathBuf,
    /// Previous content for incremental parsing
    old_content: Option<String>,
  },
  /// Delete all chunks for a file
  Delete { path: PathBuf },
  /// Rename a file (preserves embeddings)
  Rename { from: PathBuf, to: PathBuf },
  /// Batch index multiple files
  Batch {
    files: Vec<PathBuf>,
    /// Optional progress channel
    progress: Option<mpsc::Sender<IndexProgress>>,
  },
  /// Shutdown the indexer
  Shutdown,
}

/// Pipeline stage for progress reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
  Scanning,
  Reading,
  Parsing,
  Embedding,
  Writing,
}

impl PipelineStage {
  pub fn as_str(&self) -> &'static str {
    match self {
      Self::Scanning => "scanning",
      Self::Reading => "reading",
      Self::Parsing => "parsing",
      Self::Embedding => "embedding",
      Self::Writing => "writing",
    }
  }
}

/// Progress update from batch indexing
#[derive(Debug, Clone)]
pub struct IndexProgress {
  /// Which pipeline stage is reporting
  pub stage: PipelineStage,
  /// Number of files processed so far in this stage
  pub processed: usize,
  /// Total number of files to process in this stage
  pub total: usize,
  /// Current file being processed (if any)
  pub current_file: Option<String>,
  /// Number of chunks created so far (populated during writing stage)
  pub chunks_created: usize,
}

impl IndexProgress {
  /// Create a new progress update for a specific stage
  pub fn new(stage: PipelineStage, processed: usize, total: usize) -> Self {
    Self {
      stage,
      processed,
      total,
      current_file: None,
      chunks_created: 0,
    }
  }

  /// Set the current file being processed
  pub fn with_current_file(mut self, file: impl Into<String>) -> Self {
    self.current_file = Some(file.into());
    self
  }

  /// Set the number of chunks created
  pub fn with_chunks_created(mut self, count: usize) -> Self {
    self.chunks_created = count;
    self
  }

  /// Calculate completion percentage for this stage
  pub fn percent(&self) -> u8 {
    if self.total == 0 {
      100
    } else {
      ((self.processed as f64 / self.total as f64) * 100.0).min(100.0) as u8
    }
  }

  /// Check if processing is complete
  pub fn is_complete(&self) -> bool {
    self.processed >= self.total
  }
}

// ============================================================================
// Pipeline Message Types
// ============================================================================

/// File discovered by scanner or watcher
///
/// The scanner produces these for bulk indexing. The watcher sends them
/// directly to the reader stage for low-latency incremental updates.
#[derive(Debug, Clone)]
pub enum PipelineFile {
  /// A file to be read and indexed
  File {
    /// Absolute path to the file
    path: PathBuf,
    /// Path relative to project root
    relative: String,
    /// Previous content for incremental parsing (reuses unchanged chunks)
    old_content: Option<Arc<String>>,
  },
  /// Signals the scanner is done producing files
  Done,
}

impl PipelineFile {
  /// Create a new file entry for the pipeline
  pub fn file(path: PathBuf, relative: String) -> Self {
    Self::File {
      path,
      relative,
      old_content: None,
    }
  }
}

/// File content loaded by reader stage
///
/// The reader stage reads file content from disk and forwards it to the parser.
/// Failed reads are logged and skipped (no error variant).
#[derive(Debug, Clone)]
pub enum PipelineContent {
  /// File content successfully read
  File {
    /// Path relative to project root
    relative: String,
    /// File content as string
    content: String,
    /// Previous content for incremental parsing
    old_content: Option<Arc<String>>,
  },
  /// Signals the reader is done
  Done,
}

impl PipelineContent {
  /// Create a new file content entry
  pub fn file(relative: String, content: String) -> Self {
    Self::File {
      relative,
      content,
      old_content: None,
    }
  }

  /// Create a file content entry with previous content
  pub fn file_with_old_content(relative: String, content: String, old_content: Arc<String>) -> Self {
    Self::File {
      relative,
      content,
      old_content: Some(old_content),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ipc::system::SystemResponse;

  #[test]
  fn test_project_response_is_final() {
    let progress = ProjectActorResponse::Progress {
      message: "test".to_string(),
      percent: Some(50),
      chunks_created: None,
      current_file: None,
      processed: None,
      stage: None,
      total: None,
    };
    assert!(!progress.is_final());

    let stream = ProjectActorResponse::Stream {
      data: ResponseData::System(SystemResponse::Ping("pong".to_string())),
    };
    assert!(!stream.is_final());

    let done = ProjectActorResponse::Done(ResponseData::System(SystemResponse::Ping("done".to_string())));
    assert!(done.is_final());

    let error = ProjectActorResponse::Error {
      code: -32000,
      message: "error".to_string(),
    };
    assert!(error.is_final());
  }
}
