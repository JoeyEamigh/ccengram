//! Scanner stage - enumerates files and sends them to the Reader stage.

use std::path::PathBuf;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::actor::message::{IndexProgress, PipelineFile, PipelineStage};

/// Scanner stage - enumerates files and sends them to the Reader stage.
///
/// For bulk indexing, this sends all files from a pre-computed list.
/// For watcher-triggered updates, files are injected directly into Reader.
///
/// # Cancellation
///
/// The scanner checks the cancellation token before each file send,
/// ensuring quick shutdown when requested.
pub async fn scanner_stage(
  root: PathBuf,
  files: Vec<PathBuf>,
  tx: mpsc::Sender<PipelineFile>,
  progress_tx: Option<mpsc::Sender<IndexProgress>>,
  cancel: CancellationToken,
) {
  let total = files.len();
  debug!(total = total, "Scanner stage starting");

  for (i, path) in files.into_iter().enumerate() {
    // Check for cancellation
    if cancel.is_cancelled() {
      debug!(processed = i, total = total, "Scanner cancelled");
      break;
    }

    // Compute relative path
    let relative = match path.strip_prefix(&root) {
      Ok(rel) => rel.to_string_lossy().to_string(),
      Err(_) => {
        warn!(path = %path.display(), "File not under root, skipping");
        continue;
      }
    };

    // Send progress update for every file (before moving relative)
    if let Some(ref ptx) = progress_tx {
      let progress = IndexProgress::new(PipelineStage::Scanning, i + 1, total).with_current_file(&relative);
      let _ = ptx.send(progress).await;
    }

    // Send to reader
    let msg = PipelineFile::file(path.clone(), relative);

    tokio::select! {
      biased;
      _ = cancel.cancelled() => {
        debug!(processed = i, total = total, "Scanner cancelled during send");
        break;
      }
      result = tx.send(msg) => {
        if result.is_err() {
          debug!(processed = i, "Scanner: downstream closed");
          break;
        }
      }
    }
  }

  // Signal completion
  let _ = tx.send(PipelineFile::Done).await;
  debug!(total = total, "Scanner stage complete");
}
