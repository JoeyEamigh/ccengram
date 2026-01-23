//! Code indexing and search tool methods

use super::ToolHandler;
use crate::router::{Request, Response};
use db::{CheckpointType, IndexCheckpoint};
use index::{Chunker, Scanner, compute_gitignore_hash};
use serde::Deserialize;
use std::path::PathBuf;
use tracing::{debug, warn};

impl ToolHandler {
  pub async fn code_search(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      query: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      language: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Build filter for language if provided
    let filter = args
      .language
      .as_ref()
      .map(|lang| format!("language = '{}'", lang.to_lowercase()));

    let limit = args.limit.unwrap_or(10);

    // Try vector search if embedding provider is available
    if let Some(query_vec) = self.get_embedding(&args.query).await {
      debug!("Using vector search for code query: {}", args.query);
      match db.search_code_chunks(&query_vec, limit, filter.as_deref()).await {
        Ok(results) => {
          let results: Vec<_> = results
            .into_iter()
            .map(|(chunk, distance)| {
              let similarity = 1.0 - distance.min(1.0);
              serde_json::json!({
                  "id": chunk.id.to_string(),
                  "file_path": chunk.file_path,
                  "content": chunk.content,
                  "language": format!("{:?}", chunk.language).to_lowercase(),
                  "chunk_type": format!("{:?}", chunk.chunk_type).to_lowercase(),
                  "symbols": chunk.symbols,
                  "start_line": chunk.start_line,
                  "end_line": chunk.end_line,
                  "similarity": similarity,
              })
            })
            .collect();

          return Response::success(request.id, serde_json::json!(results));
        }
        Err(e) => {
          warn!("Vector code search failed, falling back to text: {}", e);
        }
      }
    }

    // Fallback: text-based search
    debug!("Using text search for code query: {}", args.query);
    match db.list_code_chunks(filter.as_deref(), Some(limit * 10)).await {
      Ok(chunks) => {
        let query_lower = args.query.to_lowercase();
        let results: Vec<_> = chunks
          .into_iter()
          .filter(|c| {
            c.content.to_lowercase().contains(&query_lower)
              || c.symbols.iter().any(|s| s.to_lowercase().contains(&query_lower))
          })
          .take(limit)
          .map(|chunk| {
            serde_json::json!({
                "id": chunk.id.to_string(),
                "file_path": chunk.file_path,
                "content": chunk.content,
                "language": format!("{:?}", chunk.language).to_lowercase(),
                "chunk_type": format!("{:?}", chunk.chunk_type).to_lowercase(),
                "symbols": chunk.symbols,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Code search error: {}", e)),
    }
  }

  pub async fn code_index(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      force: Option<bool>,
      #[serde(default)]
      dry_run: Option<bool>,
      #[serde(default)]
      resume: Option<bool>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let force = args.force.unwrap_or(false);
    let dry_run = args.dry_run.unwrap_or(false);
    let resume = args.resume.unwrap_or(true); // Resume by default

    debug!(
      "Code index: path={:?}, force={}, dry_run={}, resume={}",
      project_path, force, dry_run, resume
    );

    let (info, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let project_id = info.id.as_str();

    // Load index config for this project
    let config = engram_core::Config::load_for_project(&project_path);

    // Scan the project directory with config
    let scanner = Scanner::new().with_max_file_size(config.index.max_file_size as u64);
    let scan_result = scanner.scan(&project_path, |progress| {
      debug!("Scanning: {} files, current: {:?}", progress.scanned, progress.path);
    });

    // Compute gitignore hash to detect config changes
    let current_gitignore_hash = Some(compute_gitignore_hash(&project_path));

    if dry_run {
      return Response::success(
        request.id,
        serde_json::json!({
            "status": "dry_run",
            "files_found": scan_result.files.len(),
            "skipped": scan_result.skipped_count,
            "total_bytes": scan_result.total_bytes,
            "scan_duration_ms": scan_result.scan_duration.as_millis(),
        }),
      );
    }

    // Check for existing checkpoint
    let mut checkpoint = if resume && !force {
      match db.get_checkpoint(project_id, CheckpointType::Code).await {
        Ok(Some(cp)) => {
          // Check if gitignore changed - if so, invalidate checkpoint
          if cp.gitignore_hash != current_gitignore_hash {
            debug!("Gitignore changed, starting fresh index");
            None
          } else if cp.is_complete {
            debug!("Previous indexing complete, starting fresh");
            None
          } else {
            debug!("Resuming from checkpoint: {}% complete", cp.progress_percent());
            Some(cp)
          }
        }
        Ok(None) => None,
        Err(e) => {
          warn!("Failed to get checkpoint: {}", e);
          None
        }
      }
    } else {
      None
    };

    // If force or no checkpoint, clear existing chunks and create new checkpoint
    if force || checkpoint.is_none() {
      if force {
        for file in &scan_result.files {
          if let Err(e) = db.delete_chunks_for_file(&file.relative_path).await {
            warn!("Failed to clear chunks for {}: {}", file.relative_path, e);
          }
        }
        // Clear any existing checkpoint
        let _ = db.clear_checkpoint(project_id, CheckpointType::Code).await;
      }

      // Create new checkpoint with all files
      let pending: Vec<String> = scan_result.files.iter().map(|f| f.relative_path.clone()).collect();
      let mut new_cp = IndexCheckpoint::new(project_id, CheckpointType::Code, pending);
      new_cp.gitignore_hash = current_gitignore_hash;
      if let Err(e) = db.save_checkpoint(&new_cp).await {
        warn!("Failed to save checkpoint: {}", e);
      }
      checkpoint = Some(new_cp);
    }

    // Safety: checkpoint is always set by this point - either from existing checkpoint
    // or from creation in the if block above
    let Some(mut checkpoint) = checkpoint else {
      return Response::error(request.id, -32603, "Internal error: checkpoint not initialized");
    };

    // Build a map of files to process for quick lookup
    let file_map: std::collections::HashMap<_, _> =
      scan_result.files.iter().map(|f| (f.relative_path.clone(), f)).collect();

    // Process only pending files
    let chunker = Chunker::default();
    let mut total_chunks = 0;
    let mut indexed_files = 0;
    let mut failed_files = Vec::new();
    let mut save_counter = 0;
    let mut bytes_processed: u64 = 0;

    // Clone pending files to avoid borrow issues
    let pending_to_process: Vec<String> = checkpoint.pending_files.clone();

    // Track indexing start time for performance metrics
    let index_start = std::time::Instant::now();

    for relative_path in &pending_to_process {
      let file = match file_map.get(relative_path) {
        Some(f) => *f,
        None => {
          // File no longer exists, mark as error
          checkpoint.mark_error(relative_path);
          continue;
        }
      };

      // Read file content
      let content = match std::fs::read_to_string(&file.path) {
        Ok(c) => c,
        Err(e) => {
          warn!("Failed to read {}: {}", relative_path, e);
          failed_files.push(relative_path.clone());
          checkpoint.mark_error(relative_path);
          save_counter += 1;
          continue;
        }
      };

      // Track bytes processed for metrics
      bytes_processed += file.size;

      // Chunk the file
      let chunks: Vec<_> = chunker.chunk(&content, relative_path, file.language, &file.checksum);

      // Generate embeddings in batch for better performance
      let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
      let embeddings = self.get_embeddings_batch(&texts).await;

      // Store chunks with their embeddings
      let mut file_success = true;
      for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
        let vector = embedding.unwrap_or_else(|| vec![0.0f32; db.vector_dim]);

        if let Err(e) = db.add_code_chunk(&chunk, Some(&vector)).await {
          warn!("Failed to store chunk for {}: {}", relative_path, e);
          file_success = false;
          break;
        }
        total_chunks += 1;
      }

      if file_success {
        checkpoint.mark_processed(relative_path);
        indexed_files += 1;
      } else {
        checkpoint.mark_error(relative_path);
        failed_files.push(relative_path.clone());
      }

      save_counter += 1;

      // Save checkpoint periodically (every 10 files)
      if save_counter >= 10 {
        if let Err(e) = db.save_checkpoint(&checkpoint).await {
          warn!("Failed to save checkpoint: {}", e);
        }
        save_counter = 0;
      }
    }

    // Mark complete and save final checkpoint
    checkpoint.mark_complete();
    if let Err(e) = db.save_checkpoint(&checkpoint).await {
      warn!("Failed to save final checkpoint: {}", e);
    }

    // Clear checkpoint on successful completion
    if failed_files.is_empty() {
      let _ = db.clear_checkpoint(project_id, CheckpointType::Code).await;
    }

    // Calculate performance metrics
    let index_duration = index_start.elapsed();
    let index_duration_ms = index_duration.as_millis() as u64;
    let files_per_second = if index_duration_ms > 0 && indexed_files > 0 {
      (indexed_files as f64 / index_duration_ms as f64) * 1000.0
    } else {
      0.0
    };
    let total_duration_ms = scan_result.scan_duration.as_millis() as u64 + index_duration_ms;

    Response::success(
      request.id,
      serde_json::json!({
          "status": "complete",
          "files_scanned": scan_result.files.len(),
          "files_indexed": indexed_files,
          "chunks_created": total_chunks,
          "failed_files": failed_files,
          "resumed_from_checkpoint": !pending_to_process.is_empty() && pending_to_process.len() < scan_result.files.len(),
          "scan_duration_ms": scan_result.scan_duration.as_millis(),
          "index_duration_ms": index_duration_ms,
          "total_duration_ms": total_duration_ms,
          "files_per_second": files_per_second,
          "bytes_processed": bytes_processed,
          "total_bytes": scan_result.total_bytes,
      }),
    )
  }

  /// List all code chunks for export
  pub async fn code_list(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    match db.list_code_chunks(None, args.limit).await {
      Ok(chunks) => {
        let results: Vec<_> = chunks
          .into_iter()
          .map(|chunk| {
            serde_json::json!({
                "id": chunk.id.to_string(),
                "file_path": chunk.file_path,
                "content": chunk.content,
                "language": format!("{:?}", chunk.language).to_lowercase(),
                "chunk_type": format!("{:?}", chunk.chunk_type).to_lowercase(),
                "symbols": chunk.symbols,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "file_hash": chunk.file_hash,
                "tokens_estimate": chunk.tokens_estimate,
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("List error: {}", e)),
    }
  }

  /// Import a single code chunk (used during index import)
  pub async fn code_import_chunk(&self, request: Request) -> Response {
    use engram_core::{ChunkType, CodeChunk, Language};

    #[derive(Deserialize)]
    struct ChunkData {
      file_path: String,
      content: String,
      language: String,
      chunk_type: String,
      symbols: Vec<String>,
      start_line: u32,
      end_line: u32,
      file_hash: String,
      #[serde(default)]
      tokens_estimate: Option<u32>,
    }

    #[derive(Deserialize)]
    struct Args {
      chunk: ChunkData,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Parse language from extension-like string
    let language = Language::from_extension(&args.chunk.language).unwrap_or(Language::Rust);

    // Parse chunk type
    let chunk_type = match args.chunk.chunk_type.as_str() {
      "function" => ChunkType::Function,
      "class" => ChunkType::Class,
      "module" => ChunkType::Module,
      "import" => ChunkType::Import,
      _ => ChunkType::Block,
    };

    let chunk = CodeChunk {
      id: uuid::Uuid::now_v7(),
      file_path: args.chunk.file_path,
      content: args.chunk.content.clone(),
      language,
      chunk_type,
      symbols: args.chunk.symbols,
      start_line: args.chunk.start_line,
      end_line: args.chunk.end_line,
      file_hash: args.chunk.file_hash,
      indexed_at: chrono::Utc::now(),
      tokens_estimate: args
        .chunk
        .tokens_estimate
        .unwrap_or((args.chunk.content.len() / 4) as u32),
    };

    // Generate embedding
    let vector = match self.get_embedding(&chunk.content).await {
      Some(v) => v,
      None => vec![0.0f32; db.vector_dim],
    };

    match db.add_code_chunk(&chunk, Some(&vector)).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "id": chunk.id.to_string(),
            "status": "imported"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Import failed: {}", e)),
    }
  }

  /// Get surrounding lines for a code chunk by reading from filesystem
  pub async fn code_context(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      chunk_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      lines_before: Option<usize>,
      #[serde(default)]
      lines_after: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Cap and default context lines
    let lines_before = args.lines_before.unwrap_or(20).min(500);
    let lines_after = args.lines_after.unwrap_or(20).min(500);

    // Look up chunk by ID or prefix
    let chunk = match db.get_code_chunk_by_id_or_prefix(&args.chunk_id).await {
      Ok(Some(c)) => c,
      Ok(None) => {
        return Response::error(request.id, -32000, &format!("Code chunk not found: {}", args.chunk_id));
      }
      Err(db::DbError::AmbiguousPrefix { prefix, count }) => {
        return Response::error(
          request.id,
          -32000,
          &format!(
            "Ambiguous prefix '{}' matches {} chunks. Use more characters.",
            prefix, count
          ),
        );
      }
      Err(db::DbError::InvalidInput(msg)) => {
        return Response::error(request.id, -32602, &msg);
      }
      Err(e) => {
        return Response::error(request.id, -32000, &format!("Database error: {}", e));
      }
    };

    // Construct the full file path
    let file_path = project_path.join(&chunk.file_path);

    // Read the file
    let file_content = match std::fs::read_to_string(&file_path) {
      Ok(content) => content,
      Err(e) => {
        // File not found or not readable - return chunk content as fallback
        warn!(
          "Could not read file {} for context: {}. Returning stored chunk content.",
          file_path.display(),
          e
        );
        return Response::success(
          request.id,
          serde_json::json!({
            "chunk_id": chunk.id.to_string(),
            "file_path": chunk.file_path,
            "language": format!("{:?}", chunk.language).to_lowercase(),
            "context": {
              "before": { "content": "", "start_line": 0, "end_line": 0 },
              "target": {
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line
              },
              "after": { "content": "", "start_line": 0, "end_line": 0 }
            },
            "total_file_lines": 0,
            "warning": format!("File not readable: {}", e)
          }),
        );
      }
    };

    let lines: Vec<&str> = file_content.lines().collect();
    let total_lines = lines.len();

    // Calculate line ranges (chunk lines are 1-indexed)
    let target_start = (chunk.start_line as usize).saturating_sub(1); // Convert to 0-indexed
    let target_end = (chunk.end_line as usize).min(total_lines); // Exclusive end

    let before_start = target_start.saturating_sub(lines_before);
    let after_end = (target_end + lines_after).min(total_lines);

    // Extract content for each section
    let before_content: String = lines[before_start..target_start].join("\n");
    let target_content: String = lines[target_start..target_end].join("\n");
    let after_content: String = lines[target_end..after_end].join("\n");

    Response::success(
      request.id,
      serde_json::json!({
        "chunk_id": chunk.id.to_string(),
        "file_path": chunk.file_path,
        "language": format!("{:?}", chunk.language).to_lowercase(),
        "context": {
          "before": {
            "content": before_content,
            "start_line": before_start + 1, // Convert back to 1-indexed
            "end_line": target_start        // Exclusive, so equals target_start
          },
          "target": {
            "content": target_content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line
          },
          "after": {
            "content": after_content,
            "start_line": target_end + 1,   // Convert back to 1-indexed
            "end_line": after_end           // This is the count
          }
        },
        "total_file_lines": total_lines
      }),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::super::create_test_handler;
  use crate::router::Request;

  #[tokio::test]
  async fn test_code_search_invalid_params() {
    let (_dir, handler) = create_test_handler();

    // Missing required 'query' param
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "code_search".to_string(),
      params: serde_json::json!({
          "language": "rust"
      }),
    };

    let response = handler.code_search(request).await;
    assert!(response.error.is_some());
  }
}
