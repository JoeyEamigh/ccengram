//! Actor handles for communicating with actors
//!
//! Handles are cheap to clone and provide a way to send messages to actors.
//! They encapsulate the channel sender and provide convenient methods for
//! request/response patterns.

use std::sync::{
  Arc,
  atomic::{AtomicUsize, Ordering},
};

use tokio::sync::mpsc;

use super::message::{IndexJob, IndexProgress, ProjectActorMessage, ProjectActorPayload, ProjectActorResponse};

// ============================================================================
// Project Handle
// ============================================================================

/// Handle to communicate with a ProjectActor
///
/// The handle is cheap to clone and can be shared across tasks.
/// Each request creates a new response channel for streaming support.
#[derive(Clone, Debug)]
pub struct ProjectHandle {
  pub tx: mpsc::Sender<ProjectActorMessage>,
}

impl ProjectHandle {
  /// Create a new handle from a sender
  pub fn new(tx: mpsc::Sender<ProjectActorMessage>) -> Self {
    Self { tx }
  }

  /// Send a request and get a receiver for responses
  ///
  /// The receiver may yield multiple responses (for streaming) before
  /// a final `Done` or `Error` response.
  pub async fn send(
    &self,
    id: String,
    payload: ProjectActorPayload,
  ) -> Result<mpsc::Receiver<ProjectActorResponse>, SendError> {
    let (reply_tx, reply_rx) = mpsc::channel(32);
    let msg = ProjectActorMessage {
      id,
      reply: reply_tx,
      payload,
    };
    self.tx.send(msg).await.map_err(|_| SendError::ActorGone)?;
    Ok(reply_rx)
  }

  /// Send a request and wait for the final response (ignoring streaming)
  ///
  /// This is a convenience method for simple request/response patterns
  /// where streaming is not needed.
  pub async fn request(&self, id: String, payload: ProjectActorPayload) -> Result<ProjectActorResponse, SendError> {
    let mut rx = self.send(id, payload).await?;

    // Drain until we get a final response
    loop {
      match rx.recv().await {
        Some(response) if response.is_final() => return Ok(response),
        Some(_) => continue, // Skip non-final responses
        None => return Err(SendError::ActorGone),
      }
    }
  }
}

// ============================================================================
// Indexer Handle
// ============================================================================

/// Handle to communicate with an IndexerActor
///
/// The indexer handle is simpler than ProjectHandle because index jobs
/// are fire-and-forget (progress is sent through a separate channel if needed).
///
/// Tracks pending job count for backpressure and status reporting.
#[derive(Clone, Debug)]
pub struct IndexerHandle {
  pub tx: mpsc::Sender<IndexJob>,
  pending: Arc<AtomicUsize>,
}

impl IndexerHandle {
  #[allow(dead_code)]
  /// Create a new handle from a sender
  pub fn new(tx: mpsc::Sender<IndexJob>) -> Self {
    Self {
      tx,
      pending: Arc::new(AtomicUsize::new(0)),
    }
  }

  /// Create a new handle with shared pending counter (for actor to decrement)
  pub fn with_pending(tx: mpsc::Sender<IndexJob>, pending: Arc<AtomicUsize>) -> Self {
    Self { tx, pending }
  }

  /// Get current pending job count
  pub fn pending_count(&self) -> usize {
    self.pending.load(Ordering::Relaxed)
  }

  /// Send an index job to the actor
  pub async fn send(&self, job: IndexJob) -> Result<(), SendError> {
    // Don't count shutdown as pending
    if !matches!(job, IndexJob::Shutdown) {
      self.pending.fetch_add(1, Ordering::Relaxed);
    }
    self.tx.send(job).await.map_err(|e| {
      // Decrement on failure
      if !matches!(e.0, IndexJob::Shutdown) {
        self.pending.fetch_sub(1, Ordering::Relaxed);
      }
      SendError::ActorGone
    })
  }

  /// Queue a batch of files for indexing with optional progress reporting
  pub async fn index_batch(
    &self,
    files: Vec<std::path::PathBuf>,
    progress: Option<mpsc::Sender<IndexProgress>>,
  ) -> Result<(), SendError> {
    self.send(IndexJob::Batch { files, progress }).await
  }

  /// Request the indexer to shutdown
  pub async fn shutdown(&self) -> Result<(), SendError> {
    self.send(IndexJob::Shutdown).await
  }
}

// ============================================================================
// Errors
// ============================================================================

/// Error when sending to an actor
#[derive(Debug, Clone, thiserror::Error)]
pub enum SendError {
  #[error("Actor has shut down")]
  ActorGone,
}
