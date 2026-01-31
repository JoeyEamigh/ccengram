//! Streaming Pipeline for File Indexing
//!
//! A multi-stage pipeline with backpressure for efficient file indexing:
//!
//! ```text
//! Scanner → Reader → Parser → Embedder → Writer
//!   256      128      256       64       flush
//! ```
//!
//! Each stage has bounded channels. When downstream is full, upstream blocks,
//! naturally propagating backpressure through the pipeline.
//!
//! ## Unified Architecture
//!
//! The pipeline uses a single `Indexer` that handles both code and document files.
//! File type is detected automatically by extension.
//!
//! ## Watcher Integration
//!
//! The file watcher bypasses the Scanner and injects directly into the Reader
//! stage for low-latency incremental updates.

mod embedder;
mod parser;
mod reader;
mod scanner;
mod writer;

use std::{
  path::PathBuf,
  sync::{Arc, atomic::AtomicUsize},
};

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::debug;

use self::{
  embedder::embedder_stage,
  parser::{parser_done_aggregator, parser_worker},
  reader::{reader_done_aggregator, reader_worker},
  scanner::scanner_stage,
  writer::writer_stage,
};
pub use self::{
  embedder::{EmbedderConfig, ProcessedFile},
  writer::WriterConfig,
};
use crate::{
  actor::{
    indexer::PipelineConfig,
    message::{IndexProgress, PipelineStage},
  },
  context::files::Indexer,
  db::ProjectDb,
  embedding::{EmbeddingError, EmbeddingProvider},
};

#[allow(clippy::too_many_arguments)]
/// Run the indexing pipeline.
///
/// Creates all stages, connects them with channels, and runs until completion
/// or cancellation. Handles both code and document files automatically.
///
/// The `project_id` is used to update the indexed_files table for startup scan detection.
pub async fn run_pipeline(
  indexer: Indexer,
  root: PathBuf,
  files: Vec<PathBuf>,
  db: Arc<ProjectDb>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  config: PipelineConfig,
  progress_tx: Option<mpsc::Sender<IndexProgress>>,
  cancel: CancellationToken,
  project_id: Option<String>,
) -> Result<PipelineResult, PipelineError> {
  let file_count = files.len();
  debug!(file_count, "Starting indexing pipeline");

  if files.is_empty() {
    return Ok(PipelineResult::default());
  }

  // Create channels between stages
  let (scanner_tx, scanner_rx) = mpsc::channel(config.scanner_buffer);
  let (reader_tx, reader_rx) = mpsc::channel(config.reader_buffer);
  let (parser_tx, parser_rx) = mpsc::channel(config.parser_buffer);
  let (embedder_tx, embedder_rx) = mpsc::channel(config.embedder_buffer);

  // Wrap receivers in Arc<Mutex> for sharing among workers
  let scanner_rx = Arc::new(Mutex::new(scanner_rx));
  let reader_rx = Arc::new(Mutex::new(reader_rx));

  // Done signal channels for worker pools
  let (reader_done_tx, reader_done_rx) = mpsc::channel(config.reader_workers);
  let (parser_done_tx, parser_done_rx) = mpsc::channel::<()>(config.parser_workers);

  // Create child cancellation token for this pipeline
  let pipeline_cancel = cancel.child_token();

  // Spawn scanner stage (fast, no progress - embedder/writer report progress)
  let scanner_cancel = pipeline_cancel.clone();
  let scanner_root = root.clone();
  tokio::spawn(async move {
    scanner_stage(scanner_root, files, scanner_tx, None, scanner_cancel).await;
  });

  // Spawn reader workers with shared progress counter
  let reader_progress_counter = Arc::new(AtomicUsize::new(0));
  for worker_id in 0..config.reader_workers {
    let rx = scanner_rx.clone();
    let tx = reader_tx.clone();
    let done_tx = reader_done_tx.clone();
    let cancel = pipeline_cancel.clone();
    let ptx = progress_tx.clone();
    let counter = reader_progress_counter.clone();
    let total = file_count;
    tokio::spawn(async move {
      reader_worker(worker_id, rx, tx, done_tx, cancel, ptx, counter, total).await;
    });
  }
  drop(reader_done_tx);

  // Spawn reader done aggregator
  let reader_final_tx = reader_tx.clone();
  tokio::spawn(async move {
    reader_done_aggregator(config.reader_workers, reader_done_rx, reader_final_tx).await;
  });
  drop(reader_tx);

  // Spawn parser workers with shared progress counter
  let parser_progress_counter = Arc::new(AtomicUsize::new(0));
  for worker_id in 0..config.parser_workers {
    let rx = reader_rx.clone();
    let tx = parser_tx.clone();
    let done_tx = parser_done_tx.clone();
    let db = db.clone();
    let cancel = pipeline_cancel.clone();
    let root = root.clone();
    let worker_indexer = indexer.clone();
    let ptx = progress_tx.clone();
    let counter = parser_progress_counter.clone();
    let total = file_count;
    tokio::spawn(async move {
      parser_worker(
        worker_id,
        root,
        worker_indexer,
        rx,
        tx,
        done_tx,
        db,
        cancel,
        ptx,
        counter,
        total,
      )
      .await;
    });
  }
  drop(parser_done_tx);

  // Spawn parser done aggregator
  let parser_final_tx = parser_tx.clone();
  tokio::spawn(async move {
    parser_done_aggregator(config.parser_workers, parser_done_rx, parser_final_tx).await;
  });
  drop(parser_tx);

  // Spawn embedder stage with progress reporting
  let embedder_config = EmbedderConfig::from_pipeline_config(&config, db.vector_dim).with_total_files(file_count);
  let embedder_cancel = pipeline_cancel.clone();
  let embedder_indexer = indexer.clone();
  let embedder_progress = progress_tx.clone();
  tokio::spawn(async move {
    embedder_stage(
      embedder_indexer,
      parser_rx,
      embedder_tx,
      embedding_provider,
      embedder_config,
      embedder_progress,
      embedder_cancel,
    )
    .await;
  });

  // Run writer stage in the current task (blocks until complete)
  let writer_config = if let Some(ref pid) = project_id {
    WriterConfig::from_pipeline_config(&config)
      .with_project(root.clone(), pid.clone())
      .with_total_files(file_count)
  } else {
    WriterConfig::from_pipeline_config(&config).with_total_files(file_count)
  };
  let writer_stats = writer_stage(
    indexer,
    embedder_rx,
    db,
    writer_config,
    progress_tx.clone(),
    pipeline_cancel,
  )
  .await;

  debug!(
    file_count,
    chunks_indexed = writer_stats.chunks_written,
    "Pipeline complete"
  );

  // Send final progress with chunk count
  if let Some(tx) = progress_tx {
    let final_progress = IndexProgress::new(PipelineStage::Writing, file_count, file_count)
      .with_chunks_created(writer_stats.chunks_written);
    let _ = tx.send(final_progress).await;
  }

  Ok(PipelineResult {
    files_processed: file_count,
    chunks_indexed: writer_stats.chunks_written,
    errors: Vec::new(),
  })
}

/// Tracks "Done" signals across multiple workers
#[derive(Debug)]
pub struct DoneTracker {
  expected: usize,
  received: usize,
}

impl DoneTracker {
  pub fn new(worker_count: usize) -> Self {
    Self {
      expected: worker_count,
      received: 0,
    }
  }

  /// Record a Done signal. Returns true if this was the last one.
  pub fn record_done(&mut self) -> bool {
    self.received += 1;
    self.received >= self.expected
  }
}

/// Result of running the pipeline
#[derive(Debug, Default)]
pub struct PipelineResult {
  pub files_processed: usize,
  pub chunks_indexed: usize,
  pub errors: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Embedding error: {0}")]
  Embedding(#[from] EmbeddingError),
}
