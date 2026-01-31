//! IndexerActor - Unified code indexing actor
//!
//! This actor handles all file indexing operations:
//! - Single file indexing (from watcher or manual)
//! - Batch file indexing (startup scan, manual reindex) via streaming pipeline
//! - File deletion from index
//! - File rename (preserving embeddings)
//!
//! The actor owns an embedding provider and communicates with the database
//! to store parsed chunks and their embeddings.
//!
//! ## Streaming Pipeline
//!
//! Batch indexing uses a multi-stage streaming pipeline with backpressure:
//!
//! ```text
//! Scanner → Reader → Parser → Embedder → Writer
//!   256      128      256       64       flush
//! ```
//!
//! The pipeline is configured automatically based on batch size:
//! - **Bulk mode** (>100 files): Large buffers, longer timeouts, max throughput
//! - **Incremental mode** (≤100 files): Small buffers, short timeouts, low latency

use std::{
  path::{Path, PathBuf},
  sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  },
  time::Duration,
};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};

use super::{
  handle::IndexerHandle,
  message::{IndexJob, IndexProgress},
  pipeline::run_pipeline,
};
use crate::{
  context::files::{Chunk, Indexer},
  db::ProjectDb,
  domain::config::IndexConfig,
  embedding::EmbeddingProvider,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the IndexerActor
#[derive(Debug, Clone)]
pub struct IndexerConfig {
  /// Project root directory
  pub root: PathBuf,
  /// Index configuration from domain::config
  pub index: IndexConfig,
  /// Batch size for embedding requests (calculated from EmbeddingConfig)
  pub embedding_batch_size: usize,
  /// Context length for embedding validation/truncation (from EmbeddingConfig)
  pub embedding_context_length: usize,
  /// Log LanceDB cache stats after DB flushes (from DatabaseConfig)
  pub log_cache_stats: bool,
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Configuration for the streaming indexing pipeline
///
/// The pipeline consists of five stages with bounded channels:
/// ```text
/// Scanner → Reader → Parser → Embedder → Writer
///   256      128      256       64       flush
/// ```
///
/// Two presets are provided:
/// - `bulk()` - For startup scanning and full reindexing (large buffers, long timeouts)
/// - `incremental()` - For file watcher updates (small buffers, short timeouts)
#[derive(Debug, Clone)]
pub struct PipelineConfig {
  // ========================================================================
  // Channel Buffer Sizes
  // ========================================================================
  /// Scanner → Reader buffer size (file paths)
  pub scanner_buffer: usize,

  /// Reader → Parser buffer size (file contents)
  pub reader_buffer: usize,

  /// Parser → Embedder buffer size (parsed chunks)
  pub parser_buffer: usize,

  /// Embedder → Writer buffer size (embedded batches)
  pub embedder_buffer: usize,

  // ========================================================================
  // Embedding Batching
  // ========================================================================
  /// Maximum texts per embedding API call (provider-dependent, typically 64)
  pub embedding_batch_size: usize,

  /// Maximum time to wait before flushing an incomplete embedding batch
  pub embedding_batch_timeout: Duration,

  /// Context length for embedding validation/truncation (from EmbeddingConfig)
  pub embedding_context_length: usize,

  // ========================================================================
  // Database Flushing
  // ========================================================================
  /// Number of chunks to accumulate before DB flush
  pub db_flush_count: usize,

  /// Maximum time to wait before flushing to DB
  pub db_flush_timeout: Duration,

  // ========================================================================
  // Backpressure
  // ========================================================================
  /// Maximum pending embedding batches (backpressure limit)
  /// Prevents unbounded memory growth when embedding API is rate-limited.
  pub max_pending_batches: usize,

  /// Whether to flush batches on timeout (false in bulk mode for better batching)
  pub flush_on_timeout: bool,

  // ========================================================================
  // Worker Counts
  // ========================================================================
  /// Number of reader workers (I/O-bound, default: 8-16)
  pub reader_workers: usize,

  /// Number of parser workers (CPU-bound, default: num_cpus)
  pub parser_workers: usize,

  // ========================================================================
  // Debugging
  // ========================================================================
  /// Log LanceDB cache statistics after flushes (from DatabaseConfig)
  pub log_cache_stats: bool,
}

impl PipelineConfig {
  /// Create pipeline configuration from IndexConfig
  ///
  /// Uses the IndexConfig pipeline_* fields to configure the pipeline.
  /// The `is_bulk` flag controls whether to use bulk or incremental scaling.
  pub fn from_index_config(
    index: &IndexConfig,
    embedding_batch_size: usize,
    embedding_context_length: usize,
    is_bulk: bool,
  ) -> Self {
    if is_bulk {
      // Bulk mode: use full config values
      let parser_workers = if index.pipeline_parser_workers == 0 {
        num_cpus::get()
      } else {
        index.pipeline_parser_workers
      };

      Self {
        scanner_buffer: index.pipeline_scanner_buffer,
        reader_buffer: index.pipeline_reader_buffer,
        parser_buffer: index.pipeline_parser_buffer,
        embedder_buffer: index.pipeline_embedder_buffer,
        embedding_batch_size,
        embedding_batch_timeout: Duration::from_millis(index.pipeline_embedding_timeout_ms),
        embedding_context_length,
        db_flush_count: index.pipeline_db_flush_count,
        db_flush_timeout: Duration::from_millis(index.pipeline_db_flush_timeout_ms),
        max_pending_batches: index.pipeline_max_pending_batches,
        flush_on_timeout: false, // Bulk mode: only flush on size for better batching
        reader_workers: index.pipeline_reader_workers,
        parser_workers,
        log_cache_stats: false, // Set via with_log_cache_stats()
      }
    } else {
      // Incremental mode: scale down for low latency
      let parser_workers = if index.pipeline_parser_workers == 0 {
        4.min(num_cpus::get())
      } else {
        (index.pipeline_parser_workers / 4).max(2)
      };

      Self {
        scanner_buffer: (index.pipeline_scanner_buffer / 16).max(16),
        reader_buffer: (index.pipeline_reader_buffer / 16).max(8),
        parser_buffer: (index.pipeline_parser_buffer / 8).max(32),
        embedder_buffer: (index.pipeline_embedder_buffer / 8).max(8),
        embedding_batch_size: (embedding_batch_size / 4).max(8),
        // 200ms timeout allows small file batches to accumulate while still being responsive
        embedding_batch_timeout: Duration::from_millis(200),
        embedding_context_length,
        db_flush_count: (index.pipeline_db_flush_count / 10).max(50),
        db_flush_timeout: Duration::from_millis(100),
        max_pending_batches: (index.pipeline_max_pending_batches / 4).max(8),
        flush_on_timeout: true, // Incremental mode: flush on timeout for low latency
        reader_workers: (index.pipeline_reader_workers / 4).max(4),
        parser_workers,
        log_cache_stats: false, // Set via with_log_cache_stats()
      }
    }
  }

  /// Enable logging of LanceDB cache statistics after DB flushes
  pub fn with_log_cache_stats(mut self, enabled: bool) -> Self {
    self.log_cache_stats = enabled;
    self
  }

  /// Select configuration based on file count, using IndexConfig values
  pub fn auto_from_config(
    index: &IndexConfig,
    embedding_batch_size: usize,
    embedding_context_length: usize,
    file_count: usize,
  ) -> Self {
    let is_bulk = file_count > 100;
    Self::from_index_config(index, embedding_batch_size, embedding_context_length, is_bulk)
  }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during indexing
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
  #[error("Invalid path: {0}")]
  InvalidPath(PathBuf),
  #[error("Unsupported file type: {0}")]
  UnsupportedFile(PathBuf),
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Database error: {0}")]
  Database(#[from] crate::db::DbError),
  #[error("Embedding error: {0}")]
  Embedding(#[from] crate::embedding::EmbeddingError),
  #[error("Parse error: {0}")]
  Parse(String),
  #[error("Pipeline error: {0}")]
  Pipeline(#[from] super::pipeline::PipelineError),
  #[error("File indexing error: {0}")]
  FileIndex(#[from] crate::context::files::FileIndexError),
}

// ============================================================================
// IndexerActor
// ============================================================================

/// The indexer actor - handles all file indexing operations
///
/// This actor receives IndexJob messages and processes them asynchronously.
/// It owns the embedding provider and has access to the project database.
///
/// # Lifecycle
///
/// The actor runs in a loop until one of:
/// - The CancellationToken is triggered
/// - An IndexJob::Shutdown message is received
/// - The job channel is closed
pub struct IndexerActor {
  config: IndexerConfig,
  db: Arc<ProjectDb>,
  embedding: Arc<dyn EmbeddingProvider>,
  job_rx: mpsc::Receiver<IndexJob>,
  cancel: CancellationToken,
  /// Unified file indexer for code and documents
  indexer: Indexer,
  /// Shared counter for pending jobs (decremented after each job completes)
  pending: Arc<AtomicUsize>,
}

impl IndexerActor {
  /// Create a new IndexerActor
  ///
  /// The actor is not started until `run()` is called.
  pub fn new(
    config: IndexerConfig,
    db: Arc<ProjectDb>,
    embedding: Arc<dyn EmbeddingProvider>,
    job_rx: mpsc::Receiver<IndexJob>,
    cancel: CancellationToken,
    pending: Arc<AtomicUsize>,
  ) -> Self {
    // Generate a deterministic UUID from the project_id string using UUID v5
    let project_uuid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, db.project_id.as_str().as_bytes());
    Self {
      config,
      db,
      embedding,
      job_rx,
      cancel,
      indexer: Indexer::new(project_uuid),
      pending,
    }
  }

  /// Spawn the actor and return a handle for sending jobs
  ///
  /// This creates the message channel, spawns the actor task, and returns
  /// a handle that can be used to send IndexJob messages.
  pub fn spawn(
    config: IndexerConfig,
    db: Arc<ProjectDb>,
    embedding: Arc<dyn EmbeddingProvider>,
    cancel: CancellationToken,
  ) -> IndexerHandle {
    let (tx, rx) = mpsc::channel(256);
    let pending = Arc::new(AtomicUsize::new(0));
    let actor = Self::new(config, db, embedding, rx, cancel, pending.clone());
    tokio::spawn(actor.run());
    IndexerHandle::with_pending(tx, pending)
  }

  /// Main actor loop
  ///
  /// Processes IndexJob messages until shutdown is requested via:
  /// - CancellationToken being cancelled
  /// - IndexJob::Shutdown message
  /// - Job channel being closed
  ///
  /// File jobs are batched for efficient embedding API usage.
  pub async fn run(mut self) {
    info!(root = ?self.config.root, "IndexerActor started");

    // Batch collection for file jobs
    let mut file_batch: Vec<PathBuf> = Vec::new();
    let batch_size = self.config.index.watcher_batch_size;
    let batch_timeout = Duration::from_millis(self.config.index.watcher_batch_timeout_ms);
    let mut batch_timer = tokio::time::interval(batch_timeout);
    batch_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
      tokio::select! {
        // Check cancellation first (biased)
        biased;

        _ = self.cancel.cancelled() => {
          info!("IndexerActor shutting down (cancelled)");
          // Process any remaining batched files
          if !file_batch.is_empty() {
            self.flush_file_batch(&mut file_batch).await;
          }
          break;
        }

        // Batch timeout - flush accumulated files
        _ = batch_timer.tick() => {
          if !file_batch.is_empty() {
            self.flush_file_batch(&mut file_batch).await;
          }
        }

        job = self.job_rx.recv() => {
          match job {
            Some(IndexJob::Shutdown) => {
              info!("IndexerActor shutting down (requested)");
              // Process any remaining batched files
              if !file_batch.is_empty() {
                self.flush_file_batch(&mut file_batch).await;
              }
              break;
            }
            Some(IndexJob::File { path, old_content: _ }) => {
              // Accumulate file jobs for batching
              file_batch.push(path);

              // Flush if batch is full
              if file_batch.len() >= batch_size {
                self.flush_file_batch(&mut file_batch).await;
              }
            }
            Some(job) => {
              // Non-file jobs are handled immediately
              // First flush any pending file batch to maintain ordering
              if !file_batch.is_empty() {
                self.flush_file_batch(&mut file_batch).await;
              }
              if let Err(e) = self.handle_job(job).await {
                error!(error = %e, "IndexerActor job failed");
              }
              // Decrement pending count after job completes
              self.pending.fetch_sub(1, Ordering::Relaxed);
            }
            None => {
              info!("IndexerActor shutting down (channel closed)");
              // Process any remaining batched files
              if !file_batch.is_empty() {
                self.flush_file_batch(&mut file_batch).await;
              }
              break;
            }
          }
        }
      }
    }

    info!(root = ?self.config.root, "IndexerActor stopped");
  }

  /// Flush accumulated file batch through the pipeline
  async fn flush_file_batch(&mut self, batch: &mut Vec<PathBuf>) {
    let files: Vec<PathBuf> = std::mem::take(batch);
    let count = files.len();

    if count == 0 {
      return;
    }

    debug!(count, "Flushing file batch");

    // Use the batch pipeline for efficient processing
    if let Err(e) = self.batch_index(files, None).await {
      error!(error = %e, "Failed to index file batch");
    }

    // Decrement pending count for all files in batch
    self.pending.fetch_sub(count, Ordering::Relaxed);
  }

  /// Dispatch a job to the appropriate handler
  async fn handle_job(&mut self, job: IndexJob) -> Result<(), IndexError> {
    match job {
      IndexJob::File { path, old_content } => self.index_file(&path, old_content.as_deref()).await,
      IndexJob::Delete { path } => self.delete_file(&path).await,
      IndexJob::Rename { from, to } => self.rename_file(&from, &to).await,
      IndexJob::Batch { files, progress } => self.batch_index(files, progress).await,
      IndexJob::Shutdown => Ok(()), // Handled in main loop
    }
  }

  // ========================================================================
  // Job Handlers
  // ========================================================================

  /// Index a single file
  ///
  /// Reads the file content, parses it into chunks, generates embeddings,
  /// and stores everything in the database.
  async fn index_file(&mut self, path: &Path, old_content: Option<&str>) -> Result<(), IndexError> {
    let relative = path
      .strip_prefix(&self.config.root)
      .map_err(|_| IndexError::InvalidPath(path.to_path_buf()))?;

    debug!(file = %relative.display(), "Indexing file");

    // Read file content
    let content = tokio::fs::read_to_string(path).await?;

    // Use unified Indexer to scan and chunk
    let metadata = self
      .indexer
      .scan_file(path, &self.config.root)
      .ok_or_else(|| IndexError::UnsupportedFile(path.to_path_buf()))?;

    let chunks = self
      .indexer
      .chunk_file(&content, &metadata, old_content)
      .map_err(|e| IndexError::Parse(e.to_string()))?;

    if chunks.is_empty() {
      trace!(file = %relative.display(), "No chunks produced, skipping");
      return Ok(());
    }

    let relative_str = relative.to_string_lossy();

    // Generate embeddings
    let embeddings = self.embed_unified_chunks(&chunks).await?;

    // Prepare chunks with embeddings
    let chunks_with_embeddings: Vec<(Chunk, Vec<f32>)> = chunks.into_iter().zip(embeddings).collect();
    let chunk_count = chunks_with_embeddings.len();

    // Store via unified Indexer
    self
      .indexer
      .store_chunks(&self.db, &relative_str, chunks_with_embeddings)
      .await?;

    debug!(
        file = %relative.display(),
        chunks = chunk_count,
        "File indexed successfully"
    );

    Ok(())
  }

  /// Delete all chunks for a file from the index
  async fn delete_file(&self, path: &Path) -> Result<(), IndexError> {
    let relative = path
      .strip_prefix(&self.config.root)
      .map_err(|_| IndexError::InvalidPath(path.to_path_buf()))?;

    debug!(file = %relative.display(), "Deleting chunks for file");

    let relative_str = relative.to_string_lossy();

    // Delete code chunks
    self.db.delete_chunks_for_file(&relative_str).await?;

    // Delete document chunks and metadata (no-op for code files)
    self.db.delete_document_chunks_by_source(&relative_str).await.ok();
    self.db.delete_document_by_source(&relative_str).await.ok();

    // Delete indexed_files entry
    self
      .db
      .delete_indexed_file(self.db.project_id.as_str(), &relative_str)
      .await
      .ok();

    // Optimize indexes after delete to compact deleted rows
    // This ensures deleted content is immediately removed from vector search results
    if let Err(e) = self.db.optimize_indexes().await {
      warn!(error = %e, "Failed to optimize indexes after delete");
    }

    Ok(())
  }

  /// Rename a file in the index (preserves embeddings)
  ///
  /// This is more efficient than delete + re-index because it preserves
  /// existing embeddings and other computed data.
  async fn rename_file(&self, from: &Path, to: &Path) -> Result<(), IndexError> {
    let from_rel = from
      .strip_prefix(&self.config.root)
      .map_err(|_| IndexError::InvalidPath(from.to_path_buf()))?;
    let to_rel = to
      .strip_prefix(&self.config.root)
      .map_err(|_| IndexError::InvalidPath(to.to_path_buf()))?;

    debug!(
        from = %from_rel.display(),
        to = %to_rel.display(),
        "Renaming file in index"
    );

    let from_str = from_rel.to_string_lossy();
    let to_str = to_rel.to_string_lossy();

    // Use the unified indexer which handles both code and document files
    self.indexer.rename_file(&self.db, &from_str, &to_str).await?;

    Ok(())
  }

  /// Batch index multiple files with progress reporting
  ///
  /// Uses the streaming pipeline for efficient batch processing with:
  /// - Multi-stage processing with bounded channels
  /// - Backpressure propagation
  /// - Concurrent embedding batches
  /// - Batched database writes
  async fn batch_index(
    &mut self,
    files: Vec<PathBuf>,
    progress: Option<mpsc::Sender<IndexProgress>>,
  ) -> Result<(), IndexError> {
    let total = files.len();
    info!(total = total, "Starting batch indexing");

    if files.is_empty() {
      return Ok(());
    }

    // Always use the streaming pipeline (legacy path kept for potential debugging)
    self.batch_index_pipeline(files, progress).await
  }

  /// Batch index using the streaming pipeline
  ///
  /// This is the preferred approach - it provides:
  /// - Bounded memory usage through backpressure
  /// - No sync points (files stream through continuously)
  /// - Efficient batching of embeddings and DB writes
  async fn batch_index_pipeline(
    &self,
    files: Vec<PathBuf>,
    progress: Option<mpsc::Sender<IndexProgress>>,
  ) -> Result<(), IndexError> {
    let total = files.len();

    // Select pipeline configuration based on batch size, using IndexConfig values
    let config = PipelineConfig::auto_from_config(
      &self.config.index,
      self.config.embedding_batch_size,
      self.config.embedding_context_length,
      total,
    )
    .with_log_cache_stats(self.config.log_cache_stats);

    debug!(
      total = total,
      config = ?config,
      "Using streaming pipeline"
    );

    // Run the pipeline with unified Indexer
    let result = run_pipeline(
      self.indexer.clone(),
      self.config.root.clone(),
      files,
      self.db.clone(),
      self.embedding.clone(),
      config,
      progress,
      self.cancel.child_token(),
      Some(self.db.project_id.as_str().to_string()),
    )
    .await?;

    info!(
      files_processed = result.files_processed,
      chunks_indexed = result.chunks_indexed,
      errors = result.errors.len(),
      "Pipeline batch indexing complete"
    );

    // Log any non-fatal errors
    for error in &result.errors {
      warn!(error = %error, "Non-fatal indexing error");
    }

    Ok(())
  }

  // ========================================================================
  // Helper Methods
  // ========================================================================

  /// Generate embeddings for unified chunks
  ///
  /// Uses batch embedding for efficiency. Works with the unified Chunk type.
  /// Validates and truncates texts that exceed the embedding model's context limit.
  async fn embed_unified_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Vec<f32>>, IndexError> {
    use crate::embedding::validation::{TextValidationConfig, validate_and_truncate};

    let validation_config = TextValidationConfig::for_context_length(self.config.embedding_context_length);

    // Collect and validate texts for embedding
    let texts: Vec<String> = chunks
      .iter()
      .map(|c| {
        let text = self.indexer.prepare_embedding_text(c);
        let (validated, _) = validate_and_truncate(&text, &validation_config);
        validated
      })
      .collect();

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Batch embed in configured batch sizes
    let mut all_embeddings = Vec::with_capacity(chunks.len());

    for batch in text_refs.chunks(self.config.embedding_batch_size) {
      // Document mode - we're indexing, not searching
      let embeddings = self
        .embedding
        .embed_batch(batch, crate::embedding::EmbeddingMode::Document)
        .await?;
      all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
  }
}
