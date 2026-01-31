//! Embedder stage - generates embeddings with concurrent in-flight batches.

use std::{
  collections::HashMap,
  sync::Arc,
  time::{Duration, Instant},
};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, trace, warn};

use super::parser::ParsedChunks;
use crate::{
  actor::{
    indexer::PipelineConfig,
    message::{IndexProgress, PipelineStage},
  },
  context::files::{Chunk, Indexer},
  embedding::{EmbeddingError, EmbeddingMode, EmbeddingProvider, validation::TextValidationConfig},
};

/// Configuration for the embedder stage
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
  pub batch_size: usize,
  pub batch_timeout: Duration,
  pub vector_dim: usize,
  pub max_tokens: usize,
  pub total_files: usize,
  /// Maximum pending embedding batches (backpressure limit)
  pub max_pending_batches: usize,
  /// Whether to flush batches on timeout (false in bulk mode for better batching)
  pub flush_on_timeout: bool,
}

impl EmbedderConfig {
  pub fn from_pipeline_config(config: &PipelineConfig, vector_dim: usize) -> Self {
    Self {
      batch_size: config.embedding_batch_size,
      batch_timeout: config.embedding_batch_timeout,
      vector_dim,
      max_tokens: config.embedding_context_length,
      total_files: 0,
      max_pending_batches: config.max_pending_batches,
      flush_on_timeout: config.flush_on_timeout,
    }
  }

  pub fn with_total_files(mut self, total: usize) -> Self {
    self.total_files = total;
    self
  }
}

/// Embedded chunks ready for database insertion
#[derive(Debug)]
pub enum EmbeddedChunks {
  Batch { files: Vec<ProcessedFile> },
  Done,
}

/// A file with chunks and their embeddings
#[derive(Debug)]
pub struct ProcessedFile {
  pub relative: String,
  pub chunks_with_vectors: Vec<(Chunk, Vec<f32>)>,
  /// Character count of original content (for document metadata)
  pub char_count: Option<usize>,
  /// Content hash of original content (for document metadata)
  pub content_hash: Option<String>,
}

impl ProcessedFile {
  pub fn chunk_count(&self) -> usize {
    self.chunks_with_vectors.len()
  }
}

/// Pending batch waiting for embedding results
struct PendingBatch {
  files: Vec<PendingFile>,
  texts_to_embed: Vec<String>,
}

struct PendingFile {
  relative: String,
  chunks: Vec<Chunk>,
  existing_embeddings: HashMap<String, Vec<f32>>,
  needs_embedding: Vec<usize>,
  /// Character count of original content (for document metadata)
  char_count: Option<usize>,
  /// Content hash of original content (for document metadata)
  content_hash: Option<String>,
}

impl PendingBatch {
  fn new() -> Self {
    Self {
      files: Vec::new(),
      texts_to_embed: Vec::new(),
    }
  }

  fn add_file(&mut self, file: PendingFile, indexer: &Indexer, validation_config: &TextValidationConfig) {
    for &idx in &file.needs_embedding {
      if let Some(chunk) = file.chunks.get(idx) {
        let text = indexer.prepare_embedding_text(chunk);
        let (validated, _) = crate::embedding::validation::validate_and_truncate(&text, validation_config);
        self.texts_to_embed.push(validated);
      }
    }
    self.files.push(file);
  }

  fn text_count(&self) -> usize {
    self.texts_to_embed.len()
  }

  fn is_empty(&self) -> bool {
    self.files.is_empty()
  }

  fn finalize(self, embeddings: Vec<Vec<f32>>, fallback_dim: usize, indexer: &Indexer) -> Vec<ProcessedFile> {
    let mut embedding_iter = embeddings.into_iter();
    let mut processed_files = Vec::with_capacity(self.files.len());

    for file in self.files {
      let mut chunks_with_vectors: Vec<(Chunk, Vec<f32>)> = Vec::with_capacity(file.chunks.len());

      for (idx, chunk) in file.chunks.into_iter().enumerate() {
        let vector = if file.needs_embedding.contains(&idx) {
          embedding_iter.next().unwrap_or_else(|| vec![0.0; fallback_dim])
        } else if let Some(hash) = indexer.cache_key(&chunk) {
          file
            .existing_embeddings
            .get(&hash)
            .cloned()
            .unwrap_or_else(|| vec![0.0; fallback_dim])
        } else {
          vec![0.0; fallback_dim]
        };

        chunks_with_vectors.push((chunk, vector));
      }

      processed_files.push(ProcessedFile {
        relative: file.relative,
        chunks_with_vectors,
        char_count: file.char_count,
        content_hash: file.content_hash,
      });
    }

    processed_files
  }
}

struct EmbeddingBatchBuilder {
  current: PendingBatch,
  batch_size: usize,
  last_add: Instant,
  validation_config: TextValidationConfig,
}

impl EmbeddingBatchBuilder {
  fn new(batch_size: usize, validation_config: TextValidationConfig) -> Self {
    Self {
      current: PendingBatch::new(),
      batch_size,
      last_add: Instant::now(),
      validation_config,
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn add_file(
    &mut self,
    indexer: &Indexer,
    relative: String,
    chunks: Vec<Chunk>,
    existing_embeddings: HashMap<String, Vec<f32>>,
    needs_embedding: Vec<usize>,
    char_count: Option<usize>,
    content_hash: Option<String>,
  ) {
    let file = PendingFile {
      relative,
      chunks,
      existing_embeddings,
      needs_embedding,
      char_count,
      content_hash,
    };
    self.current.add_file(file, indexer, &self.validation_config);
    self.last_add = Instant::now();
  }

  fn should_flush_size(&self) -> bool {
    self.current.text_count() >= self.batch_size
  }

  fn should_flush_time(&self, timeout: Duration) -> bool {
    !self.current.is_empty() && self.last_add.elapsed() >= timeout
  }

  fn take(&mut self) -> PendingBatch {
    let batch = std::mem::replace(&mut self.current, PendingBatch::new());
    self.last_add = Instant::now();
    batch
  }

  fn is_empty(&self) -> bool {
    self.current.is_empty()
  }
}

type EmbeddingBatch = (u64, Result<Vec<Vec<f32>>, EmbeddingError>);

/// Embedder stage - generates embeddings with concurrent in-flight batches.
///
/// Implements backpressure via `max_pending_batches`: when the builder needs to flush
/// but pending is at capacity, we stop accepting new work from the parser until
/// results come back and free up slots.
pub async fn embedder_stage(
  indexer: Indexer,
  mut rx: mpsc::Receiver<ParsedChunks>,
  tx: mpsc::Sender<EmbeddedChunks>,
  provider: Arc<dyn EmbeddingProvider>,
  config: EmbedderConfig,
  progress_tx: Option<mpsc::Sender<IndexProgress>>,
  cancel: CancellationToken,
) {
  debug!(
    batch_size = config.batch_size,
    max_pending = config.max_pending_batches,
    flush_on_timeout = config.flush_on_timeout,
    "Embedder stage starting"
  );

  let validation_config = TextValidationConfig::for_context_length(config.max_tokens);
  let mut builder = EmbeddingBatchBuilder::new(config.batch_size, validation_config);
  let mut interval = tokio::time::interval(config.batch_timeout);
  let mut next_batch_id: u64 = 0;
  let mut files_embedded: usize = 0;
  let total_files = config.total_files;
  let max_pending = config.max_pending_batches;
  let flush_on_timeout = config.flush_on_timeout;

  let mut pending: HashMap<u64, PendingBatch> = HashMap::new();
  let (result_tx, mut result_rx) = mpsc::channel::<EmbeddingBatch>(config.batch_size * 4);

  loop {
    // Backpressure: don't accept new work if builder needs to flush but pending is full
    let builder_ready = builder.should_flush_size();
    let pending_full = pending.len() >= max_pending;
    let can_accept = !(builder_ready && pending_full);

    tokio::select! {
      biased;

      _ = cancel.cancelled() => {
        debug!("Embedder stage cancelled");
        break;
      }

      // Only receive from parser if we can accept more work
      msg = rx.recv(), if can_accept => {
        match msg {
          Some(ParsedChunks::File { relative, chunks, existing_embeddings, needs_embedding, char_count, content_hash }) => {
            builder.add_file(&indexer, relative, chunks, existing_embeddings, needs_embedding, char_count, content_hash);

            // Fire batch if ready and pending has room
            if builder.should_flush_size() && pending.len() < max_pending {
              fire_batch(&mut builder, &mut next_batch_id, &mut pending, &provider, &result_tx);
            }
          }
          Some(ParsedChunks::Done) | None => {
            if !builder.is_empty() {
              fire_batch(&mut builder, &mut next_batch_id, &mut pending, &provider, &result_tx);
            }

            // Drain remaining pending batches
            while !pending.is_empty() {
                if let Some((id, result)) = result_rx.recv().await {
                    let batch_files = handle_completed_batch(&indexer, id, result, &mut pending, &tx, config.vector_dim).await;
                    files_embedded += batch_files;

                    // Send progress update for batch completion
                    if let Some(ref ptx) = progress_tx {
                      let progress = IndexProgress::new(PipelineStage::Embedding, files_embedded, total_files);
                      let _ = ptx.send(progress).await;
                    }
                } else {
                    break;
                }
            }

            let _ = tx.send(EmbeddedChunks::Done).await;
            debug!("Embedder stage complete");
            return;
          }
        }
      }

      result = result_rx.recv() => {
        if let Some((batch_id, embeddings_result)) = result {
          let batch_files = handle_completed_batch(&indexer, batch_id, embeddings_result, &mut pending, &tx, config.vector_dim).await;
          files_embedded += batch_files;

          // Send progress update after each completed batch
          if let Some(ref ptx) = progress_tx {
            let progress = IndexProgress::new(PipelineStage::Embedding, files_embedded, total_files);
            let _ = ptx.send(progress).await;
          }

          // After freeing a slot, fire waiting batch if builder is ready
          if builder.should_flush_size() && pending.len() < max_pending {
            fire_batch(&mut builder, &mut next_batch_id, &mut pending, &provider, &result_tx);
          }
        }
      }

      _ = interval.tick(), if flush_on_timeout => {
        // Only timeout-flush in incremental mode (flush_on_timeout=true)
        if builder.should_flush_time(config.batch_timeout) && pending.len() < max_pending {
          fire_batch(&mut builder, &mut next_batch_id, &mut pending, &provider, &result_tx);
        }
      }
    }
  }
}

fn fire_batch(
  builder: &mut EmbeddingBatchBuilder,
  next_id: &mut u64,
  pending: &mut HashMap<u64, PendingBatch>,
  provider: &Arc<dyn EmbeddingProvider>,
  result_tx: &mpsc::Sender<EmbeddingBatch>,
) {
  let batch_id = *next_id;
  *next_id += 1;

  let batch = builder.take();
  let text_count = batch.text_count();

  if text_count == 0 {
    let result_tx = result_tx.clone();
    tokio::spawn(async move {
      let _ = result_tx.send((batch_id, Ok(Vec::new()))).await;
    });
    pending.insert(batch_id, batch);
    return;
  }

  let texts: Vec<String> = batch.texts_to_embed.clone();
  pending.insert(batch_id, batch);

  debug!(batch_id, text_count, "Firing embedding batch");

  let provider = provider.clone();
  let result_tx = result_tx.clone();
  tokio::spawn(async move {
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let result = provider.embed_batch(&text_refs, EmbeddingMode::Document).await;
    let _ = result_tx.send((batch_id, result)).await;
  });
}

async fn handle_completed_batch(
  indexer: &Indexer,
  batch_id: u64,
  result: Result<Vec<Vec<f32>>, EmbeddingError>,
  pending: &mut HashMap<u64, PendingBatch>,
  tx: &mpsc::Sender<EmbeddedChunks>,
  fallback_dim: usize,
) -> usize {
  let Some(batch) = pending.remove(&batch_id) else {
    warn!(batch_id, "Received result for unknown batch");
    return 0;
  };

  let embeddings = match result {
    Ok(e) => {
      trace!(batch_id, embeddings = e.len(), "Embedding batch succeeded");
      e
    }
    Err(e) => {
      warn!(batch_id, error = %e, "Embedding batch failed, using zero vectors");
      vec![vec![0.0f32; fallback_dim]; batch.text_count()]
    }
  };

  let files = batch.finalize(embeddings, fallback_dim, indexer);
  let file_count = files.len();
  let _ = tx.send(EmbeddedChunks::Batch { files }).await;
  file_count
}
