//! Parser stage - chunks file content and determines embedding needs.

use std::{
  collections::HashMap,
  path::PathBuf,
  sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  },
};

use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, trace};

use super::DoneTracker;
use crate::{
  actor::message::{IndexProgress, PipelineContent, PipelineStage},
  context::files::{Chunk, FileMetadata, Indexer},
  db::ProjectDb,
};

/// Compute SHA-256 hash of content
fn compute_content_hash(content: &str) -> String {
  let mut hasher = Sha256::new();
  hasher.update(content.as_bytes());
  format!("{:x}", hasher.finalize())
}

/// Parsed chunks ready for embedding
#[derive(Debug)]
pub enum ParsedChunks {
  File {
    relative: String,
    chunks: Vec<Chunk>,
    existing_embeddings: HashMap<String, Vec<f32>>,
    needs_embedding: Vec<usize>,
    /// Character count of original content (for document metadata)
    char_count: Option<usize>,
    /// Content hash of original content (for document metadata)
    content_hash: Option<String>,
  },
  Done,
}

#[allow(clippy::too_many_arguments)]
/// Parser worker - chunks file content and determines embedding needs.
pub async fn parser_worker(
  worker_id: usize,
  root: PathBuf,
  mut indexer: Indexer,
  rx: Arc<tokio::sync::Mutex<mpsc::Receiver<PipelineContent>>>,
  tx: mpsc::Sender<ParsedChunks>,
  done_tx: mpsc::Sender<()>,
  db: Arc<ProjectDb>,
  cancel: CancellationToken,
  progress_tx: Option<mpsc::Sender<IndexProgress>>,
  processed_counter: Arc<AtomicUsize>,
  total_files: usize,
) {
  trace!(worker_id, "Parser worker starting");

  let mut local_processed = 0;

  loop {
    let msg = {
      let mut rx_guard = rx.lock().await;
      tokio::select! {
          biased;
          _ = cancel.cancelled() => {
              trace!(worker_id, local_processed, "Parser worker cancelled");
              break;
          }
          msg = rx_guard.recv() => msg
      }
    };

    match msg {
      Some(PipelineContent::File {
        relative,
        content,
        old_content,
      }) => {
        let path = root.join(&relative);

        // Use Indexer to scan file and get metadata
        let metadata = match indexer.scan_file(&path, &root) {
          Some(m) => m,
          None => {
            trace!(worker_id, file = %relative, "Unsupported file type");
            continue;
          }
        };

        // Use Indexer to chunk the content
        let chunks = match indexer.chunk_file(&content, &metadata, old_content.as_deref().map(|s| s.as_str())) {
          Ok(c) => c,
          Err(e) => {
            debug!(worker_id, file = %relative, error = %e, "Failed to chunk file");
            continue;
          }
        };

        if chunks.is_empty() {
          trace!(worker_id, file = %relative, "No chunks produced");
          continue;
        }

        // Query DB for existing embeddings
        let existing_embeddings = indexer
          .get_existing_embeddings(&db, &relative)
          .await
          .unwrap_or_default();

        // Determine which chunks need new embeddings
        let mut needs_embedding: Vec<usize> = Vec::new();
        let mut reusable: HashMap<String, Vec<f32>> = HashMap::new();

        for (idx, chunk) in chunks.iter().enumerate() {
          if let Some(key) = indexer.cache_key(chunk) {
            if let Some(vec) = existing_embeddings.get(&key) {
              reusable.insert(key, vec.clone());
            } else {
              needs_embedding.push(idx);
            }
          } else {
            needs_embedding.push(idx);
          }
        }

        // Compute document metadata for document files
        let (char_count, content_hash) = match &metadata {
          FileMetadata::Document { .. } => (Some(content.len()), Some(compute_content_hash(&content))),
          FileMetadata::Code { .. } => (None, None),
        };

        // Increment shared counter and send progress
        let global_processed = processed_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(ref ptx) = progress_tx {
          let progress =
            IndexProgress::new(PipelineStage::Parsing, global_processed, total_files).with_current_file(&relative);
          let _ = ptx.send(progress).await;
        }

        trace!(
            worker_id,
            file = %relative,
            total_chunks = chunks.len(),
            reused = reusable.len(),
            need_embedding = needs_embedding.len(),
            "Parsed file"
        );

        let msg = ParsedChunks::File {
          relative: relative.clone(),
          chunks,
          existing_embeddings: reusable,
          needs_embedding,
          char_count,
          content_hash,
        };

        if tx.send(msg).await.is_err() {
          trace!(worker_id, "Parser: downstream closed");
          break;
        }
        local_processed += 1;
      }
      Some(PipelineContent::Done) | None => {
        trace!(worker_id, local_processed, "Parser worker: input exhausted");
        break;
      }
    }
  }

  let _ = done_tx.send(()).await;
  trace!(worker_id, local_processed, "Parser worker finished");
}

/// Aggregates Done signals from parser workers
pub async fn parser_done_aggregator(
  worker_count: usize,
  mut done_rx: mpsc::Receiver<()>,
  tx: mpsc::Sender<ParsedChunks>,
) {
  let mut tracker = DoneTracker::new(worker_count);

  while let Some(()) = done_rx.recv().await {
    if tracker.record_done() {
      let _ = tx.send(ParsedChunks::Done).await;
      trace!(worker_count, "All parser workers finished, sent Done");
      break;
    }
  }
}
