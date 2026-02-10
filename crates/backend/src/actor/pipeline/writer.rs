//! Writer stage - accumulates processed files and batch writes to DB.

use std::{
  path::{Path, PathBuf},
  sync::Arc,
  time::{Duration, Instant},
};

use chrono::Utc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, trace, warn};

#[cfg(feature = "statm")]
/// Get current process memory usage in MB from /proc/self/statm
/// Returns (rss_mb, virtual_mb) or None if unavailable
async fn get_memory_usage_mb() -> Option<(f64, f64)> {
  let content = tokio::fs::read_to_string("/proc/self/statm").await.ok()?;
  let parts: Vec<&str> = content.split_whitespace().collect();
  if parts.len() < 2 {
    return None;
  }
  let page_size = 4096_u64; // Standard Linux page size
  let virtual_pages: u64 = parts[0].parse().ok()?;
  let rss_pages: u64 = parts[1].parse().ok()?;
  let rss_mb = (rss_pages * page_size) as f64 / (1024.0 * 1024.0);
  let virtual_mb = (virtual_pages * page_size) as f64 / (1024.0 * 1024.0);
  Some((rss_mb, virtual_mb))
}

use super::{
  PipelineError,
  embedder::{EmbeddedChunks, ProcessedFile},
};
use crate::{
  actor::{
    indexer::PipelineConfig,
    message::{IndexProgress, PipelineStage},
  },
  context::files::{Chunk, Indexer},
  db::{IndexedFile, ProjectDb},
  domain::document::Document,
};

/// Configuration for the writer stage
#[derive(Debug, Clone)]
pub struct WriterConfig {
  pub flush_count: usize,
  pub flush_timeout: Duration,
  pub project_root: Option<PathBuf>,
  pub project_id: Option<String>,
  pub total_files: usize,
  pub log_cache_stats: bool,
}

impl WriterConfig {
  pub fn from_pipeline_config(config: &PipelineConfig) -> Self {
    Self {
      flush_count: config.db_flush_count,
      flush_timeout: config.db_flush_timeout,
      project_root: None,
      project_id: None,
      total_files: 0,
      log_cache_stats: config.log_cache_stats,
    }
  }

  pub fn with_project(mut self, root: PathBuf, project_id: String) -> Self {
    self.project_root = Some(root);
    self.project_id = Some(project_id);
    self
  }

  pub fn with_total_files(mut self, total: usize) -> Self {
    self.total_files = total;
    self
  }
}

struct WriteAccumulator {
  pending_files: Vec<ProcessedFile>,
  chunk_count: usize,
  last_activity: Instant,
}

impl WriteAccumulator {
  fn new() -> Self {
    Self {
      pending_files: Vec::new(),
      chunk_count: 0,
      last_activity: Instant::now(),
    }
  }

  fn add(&mut self, file: ProcessedFile) {
    let chunk_count = file.chunk_count();
    self.chunk_count += chunk_count;
    self.pending_files.push(file);
    self.last_activity = Instant::now();
  }

  fn should_flush_count(&self, threshold: usize) -> bool {
    self.chunk_count >= threshold
  }

  fn should_flush_time(&self, timeout: Duration) -> bool {
    !self.pending_files.is_empty() && self.last_activity.elapsed() >= timeout
  }

  fn take(&mut self) -> Vec<ProcessedFile> {
    self.chunk_count = 0;
    self.last_activity = Instant::now();
    std::mem::take(&mut self.pending_files)
  }

  fn is_empty(&self) -> bool {
    self.pending_files.is_empty()
  }
}

/// Stats returned by the writer stage
#[derive(Debug, Default)]
pub struct WriterStats {
  pub chunks_written: usize,
}

/// Writer stage - uses Indexer::store_chunks for DB writes.
pub async fn writer_stage(
  indexer: Indexer,
  mut rx: mpsc::Receiver<EmbeddedChunks>,
  db: Arc<ProjectDb>,
  config: WriterConfig,
  progress_tx: Option<mpsc::Sender<IndexProgress>>,
  cancel: CancellationToken,
) -> WriterStats {
  #[cfg(feature = "statm")]
  {
    let start_mem = get_memory_usage_mb().await;
    if let Some((rss, virt)) = start_mem {
      trace!(
        rss_mb = format!("{:.1}", rss),
        virt_mb = format!("{:.1}", virt),
        "[MEM] Writer stage starting - memory baseline"
      );
    }
  }
  debug!(
    flush_count = config.flush_count,
    flush_timeout_ms = config.flush_timeout.as_millis(),
    total_files = config.total_files,
    "Writer stage starting"
  );

  let mut accumulator = WriteAccumulator::new();
  let mut interval = tokio::time::interval(config.flush_timeout);
  let mut total_chunks_written = 0usize;
  let mut total_files_written = 0usize;
  let total_files = config.total_files;

  let project_root = config.project_root.as_ref();
  let project_id = config.project_id.as_deref();

  loop {
    tokio::select! {
      biased;

      _ = cancel.cancelled() => {
        debug!("Writer stage cancelled");
        if !accumulator.is_empty() {
          let files = accumulator.take();
          match flush_to_db(&indexer, &db, files, project_root, project_id).await {
            Ok((_, c)) => total_chunks_written += c,
            Err(e) => error!(error = %e, "Failed to flush on cancellation"),
          }
        }
        break;
      }

      msg = rx.recv() => {
        match msg {
          Some(EmbeddedChunks::Batch { files }) => {
              for file in files {
                accumulator.add(file);
              }

              if accumulator.should_flush_count(config.flush_count) {
                let files = accumulator.take();

                #[cfg(feature = "statm")]
                let pre_mem = get_memory_usage_mb().await;

                match flush_to_db(&indexer, &db, files, project_root, project_id).await {
                  Ok((f, c)) => {
                    total_chunks_written += c;
                    total_files_written += f;

                    #[cfg(feature = "statm")]
                    {
                      let post_mem = get_memory_usage_mb().await;
                      if let (Some((pre_rss, _)), Some((post_rss, _))) = (pre_mem, post_mem) {
                        let delta = post_rss - pre_rss;
                        trace!(
                          files = file_count,
                          chunks = chunk_count,
                          pre_rss_mb = format!("{:.1}", pre_rss),
                          post_rss_mb = format!("{:.1}", post_rss),
                          delta_mb = format!("{:+.1}", delta),
                          total_chunks = total_chunks_written,
                          "[MEM] Flush complete - memory"
                        );
                      }
                    }

                    trace!(chunks = c, total = total_chunks_written, "Flushed batch to DB");

                    // Log LanceDB cache stats if enabled
                    if config.log_cache_stats {
                      db.log_cache_stats().await;
                    }

                    // Send progress update after flush
                    if let Some(ref ptx) = progress_tx {
                      let progress = IndexProgress::new(PipelineStage::Writing, total_files_written, total_files)
                        .with_chunks_created(total_chunks_written);
                      let _ = ptx.send(progress).await;
                    }
                  }
                  Err(e) => error!(error = %e, "Failed to flush to DB"),
                }
              }
          }
          Some(EmbeddedChunks::Done) | None => {
            if !accumulator.is_empty() {
              let files = accumulator.take();
              match flush_to_db(&indexer, &db, files, project_root, project_id).await {
                Ok((f, c)) => {
                  total_chunks_written += c;
                  total_files_written += f;
                }
                Err(e) => error!(error = %e, "Failed to flush final batch to DB"),
              }
            }

            // Optimize indexes after all writes complete
            // This updates scalar indexes to include newly written data
            if let Err(e) = db.optimize_indexes().await {
              warn!(error = %e, "Failed to optimize indexes after indexing");
            }

            // Rebuild FTS indexes after bulk writes to ensure consistency
            if let Err(e) = db.rebuild_fts_indexes().await {
              warn!(error = %e, "Failed to rebuild FTS indexes after indexing");
            }

            #[cfg(feature = "statm")]
            {
              let end_mem = get_memory_usage_mb().await;
              if let (Some((start_rss, _)), Some((end_rss, end_virt))) = (start_mem, end_mem) {
                let delta = end_rss - start_rss;
                trace!(
                  start_rss_mb = format!("{:.1}", start_rss),
                  end_rss_mb = format!("{:.1}", end_rss),
                  end_virt_mb = format!("{:.1}", end_virt),
                  delta_mb = format!("{:+.1}", delta),
                  total_files = total_files_written,
                  total_chunks = total_chunks_written,
                  "[MEM] Writer stage complete - memory summary"
                );
              }
            }
            debug!(total_chunks_written, total_files_written, "Writer stage complete");
            return WriterStats {
              chunks_written: total_chunks_written,
            };
          }
        }
      }

      _ = interval.tick() => {
        if accumulator.should_flush_time(config.flush_timeout) {
          let files = accumulator.take();
          match flush_to_db(&indexer, &db, files, project_root, project_id).await {
            Ok((f, c)) => {
              total_chunks_written += c;
              total_files_written += f;
              trace!(chunks = c, "Timeout flush to DB");

              if let Some(ref ptx) = progress_tx {
                let progress = IndexProgress::new(PipelineStage::Writing, total_files_written, total_files)
                  .with_chunks_created(total_chunks_written);
                let _ = ptx.send(progress).await;
              }
            }
            Err(e) => error!(error = %e, "Failed to flush on timeout"),
          }
        }
      }
    }
  }

  WriterStats {
    chunks_written: total_chunks_written,
  }
}

#[tracing::instrument(level = "trace", skip_all)]
async fn flush_to_db(
  indexer: &Indexer,
  db: &ProjectDb,
  files: Vec<ProcessedFile>,
  project_root: Option<&PathBuf>,
  project_id: Option<&str>,
) -> Result<(usize, usize), PipelineError> {
  if files.is_empty() {
    return Ok((0, 0));
  }

  let total_files = files.len();
  let total_chunks: usize = files.iter().map(|f| f.chunks_with_vectors.len()).sum();

  #[cfg(feature = "statm")]
  let mem_entry = get_memory_usage_mb().await.map(|(r, _)| r).unwrap_or(0.0);

  // 1. Extract metadata BEFORE consuming files (avoids clone of chunks/vectors)
  let mut indexed_files: Vec<IndexedFile> = Vec::new();
  let mut doc_metadata: Vec<Document> = Vec::new();

  if let (Some(root), Some(pid)) = (project_root, project_id) {
    for file in &files {
      // Build indexed_files metadata
      if let Some(indexed) = build_indexed_file_metadata(&file.relative, &file.chunks_with_vectors, root, pid).await {
        indexed_files.push(indexed);
      }

      // Build document metadata
      let chunk_count = file.chunks_with_vectors.len();
      if let (Some(char_count), Some(content_hash)) = (file.char_count, file.content_hash.as_ref())
        && let Some((first_chunk, _)) = file.chunks_with_vectors.first()
        && let Chunk::Document(doc_chunk) = first_chunk
      {
        doc_metadata.push(Document {
          id: doc_chunk.document_id,
          project_id: doc_chunk.project_id,
          title: doc_chunk.title.clone(),
          source: file.relative.clone(),
          source_type: doc_chunk.source_type,
          content_hash: content_hash.clone(),
          char_count,
          chunk_count,
          full_content: None,
          created_at: Utc::now(),
          updated_at: Utc::now(),
        });
      }
    }
  }

  #[cfg(feature = "statm")]
  let mem_after_metadata = get_memory_usage_mb().await.map(|(r, _)| r).unwrap_or(0.0);

  // Run all table writes in parallel - they write to different tables:
  // - store_chunks_batch: code_chunks + documents tables
  // - save_indexed_files_batch: indexed_files table
  // - upsert_document_metadata_batch: document_metadata table

  let chunks_future = indexer.store_chunks_batch(db, files);

  let indexed_files_future = async {
    if !indexed_files.is_empty()
      && let Err(e) = db.save_indexed_files_batch(&indexed_files).await
    {
      warn!(error = %e, count = indexed_files.len(), "Failed to batch update indexed_files metadata");
    }
  };

  let doc_metadata_future = async {
    if !doc_metadata.is_empty() {
      if let Err(e) = db.upsert_document_metadata_batch(&doc_metadata).await {
        warn!(error = %e, count = doc_metadata.len(), "Failed to batch upsert document metadata");
      } else {
        trace!(count = doc_metadata.len(), "Batch created document metadata");
      }
    }
  };

  // Run all three in parallel
  let (chunks_result, _, _) = tokio::join!(chunks_future, indexed_files_future, doc_metadata_future);

  if let Err(e) = chunks_result {
    error!(error = %e, file_count = total_files, "Failed to batch store chunks");
    return Err(PipelineError::Io(std::io::Error::other(e.to_string())));
  }

  #[cfg(feature = "statm")]
  let mem_final = get_memory_usage_mb().await.map(|(r, _)| r).unwrap_or(0.0);

  #[cfg(feature = "statm")]
  trace!(
    files = total_files,
    chunks = total_chunks,
    entry_mb = format!("{:.1}", mem_entry),
    after_metadata_delta = format!("{:+.1}", mem_after_metadata - mem_entry),
    writes_delta = format!("{:+.1}", mem_final - mem_after_metadata),
    total_delta = format!("{:+.1}", mem_final - mem_entry),
    "[MEM] flush_to_db breakdown"
  );

  trace!(
    files = total_files,
    chunks = total_chunks,
    indexed_files = indexed_files.len(),
    doc_metadata = doc_metadata.len(),
    "Batch flushed to DB"
  );

  Ok((total_files, total_chunks))
}

#[tracing::instrument(level = "trace", skip_all)]
/// Build IndexedFile metadata for a file (for batch saving)
async fn build_indexed_file_metadata(
  file_path: &str,
  chunks_with_vectors: &[(Chunk, Vec<f32>)],
  project_root: &Path,
  project_id: &str,
) -> Option<IndexedFile> {
  let full_path = project_root.join(file_path);

  let metadata = match tokio::fs::metadata(&full_path).await {
    Ok(m) => m,
    Err(e) => {
      warn!(file_path = %file_path, error = %e, "Failed to get file metadata");
      return None;
    }
  };

  let mtime = metadata
    .modified()
    .ok()
    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
    .map(|d| d.as_secs() as i64)
    .unwrap_or(0);

  let file_size = metadata.len();

  // Get content hash from first chunk
  let content_hash = chunks_with_vectors
    .first()
    .map(|(chunk, _)| chunk.file_hash())
    .filter(|h| !h.is_empty())
    .map(|h| h.to_string())
    .unwrap_or_else(|| "unknown".to_string());

  Some(IndexedFile {
    file_path: file_path.to_string(),
    project_id: project_id.to_string(),
    mtime,
    content_hash,
    file_size,
    last_indexed_at: Utc::now().timestamp_millis(),
  })
}
