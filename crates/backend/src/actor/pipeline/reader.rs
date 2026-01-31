//! Reader stage - reads file content from disk.

use std::sync::{
  Arc,
  atomic::{AtomicUsize, Ordering},
};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, trace};

use super::DoneTracker;
use crate::actor::message::{IndexProgress, PipelineContent, PipelineFile, PipelineStage};

/// Reader worker - reads file content from disk.
///
/// Multiple reader workers run in parallel (I/O-bound task).
/// Each worker pulls from a shared receiver and sends to the parser stage.
///
/// Failed reads are logged and skipped rather than failing the pipeline.
#[allow(clippy::too_many_arguments)]
pub async fn reader_worker(
  worker_id: usize,
  rx: Arc<tokio::sync::Mutex<mpsc::Receiver<PipelineFile>>>,
  tx: mpsc::Sender<PipelineContent>,
  done_tx: mpsc::Sender<()>,
  cancel: CancellationToken,
  progress_tx: Option<mpsc::Sender<IndexProgress>>,
  processed_counter: Arc<AtomicUsize>,
  total_files: usize,
) {
  trace!(worker_id, "Reader worker starting");
  let mut local_processed = 0;

  loop {
    // Get next file from shared receiver
    let msg = {
      let mut rx_guard = rx.lock().await;
      tokio::select! {
          biased;
          _ = cancel.cancelled() => {
              trace!(worker_id, local_processed, "Reader worker cancelled");
              break;
          }
          msg = rx_guard.recv() => msg
      }
    };

    match msg {
      Some(PipelineFile::File {
        path,
        relative,
        old_content,
      }) => {
        // Read file content
        match tokio::fs::read_to_string(&path).await {
          Ok(content) => {
            // Increment shared counter and send progress
            let global_processed = processed_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref ptx) = progress_tx {
              let progress =
                IndexProgress::new(PipelineStage::Reading, global_processed, total_files).with_current_file(&relative);
              let _ = ptx.send(progress).await;
            }

            let msg = match old_content {
              Some(old) => PipelineContent::file_with_old_content(relative, content, old),
              None => PipelineContent::file(relative, content),
            };

            if tx.send(msg).await.is_err() {
              trace!(worker_id, "Reader: downstream closed");
              break;
            }
            local_processed += 1;
          }
          Err(e) => {
            // Log and skip failed reads
            debug!(
                worker_id,
                path = %path.display(),
                error = %e,
                "Failed to read file, skipping"
            );
          }
        }
      }
      Some(PipelineFile::Done) | None => {
        trace!(worker_id, local_processed, "Reader worker: input exhausted");
        break;
      }
    }
  }

  // Signal this worker is done
  let _ = done_tx.send(()).await;
  trace!(worker_id, local_processed, "Reader worker finished");
}

/// Aggregates Done signals from reader workers and forwards to parser stage.
pub async fn reader_done_aggregator(
  worker_count: usize,
  mut done_rx: mpsc::Receiver<()>,
  tx: mpsc::Sender<PipelineContent>,
) {
  let mut tracker = DoneTracker::new(worker_count);

  while let Some(()) = done_rx.recv().await {
    if tracker.record_done() {
      let _ = tx.send(PipelineContent::Done).await;
      trace!(worker_count, "All reader workers finished, sent Done");
      break;
    }
  }
}
