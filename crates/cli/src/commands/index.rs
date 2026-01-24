//! Index commands for code and documents

use crate::IndexCommand;
use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::connect_or_start;
use ipc::{
  CodeImportChunkParams, CodeIndexParams, CodeListParams, CodeStatsParams, DocsIngestParams, Method,
  ProjectStatsParams, Request,
};
use std::io::{IsTerminal, Write};
use std::path::Path;
use tracing::{debug, error, warn};

/// Manage code and document index
pub async fn cmd_index(command: Option<IndexCommand>) -> Result<()> {
  match command {
    Some(IndexCommand::Code {
      force,
      stats,
      export,
      load,
    }) => cmd_index_code(force, stats, export.as_deref(), load.as_deref()).await,
    Some(IndexCommand::Docs {
      directory,
      force,
      stats,
    }) => cmd_index_docs_impl(directory.as_deref(), force, stats).await,
    Some(IndexCommand::File { path, title, force }) => cmd_index_file(&path, title.as_deref(), force).await,
    None => {
      // Default to code indexing with no flags
      cmd_index_code(false, false, None, None).await
    }
  }
}

/// Index a single file (auto-detects code vs document based on extension)
pub async fn cmd_index_file(path: &str, title: Option<&str>, _force: bool) -> Result<()> {
  use engram_core::Config;

  let file_path = std::path::Path::new(path);
  if !file_path.exists() {
    error!("File not found: {}", path);
    std::process::exit(1);
  }

  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let config = Config::load_for_project(&cwd);

  // Check if this is a document file based on extension
  let is_doc = file_path
    .extension()
    .and_then(|e| e.to_str())
    .is_some_and(|ext| config.docs.extensions.iter().any(|e| e == ext));

  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let abs_path = file_path.canonicalize().context("Failed to resolve path")?;

  if is_doc {
    // Index as document
    let doc_title = title.unwrap_or_else(|| abs_path.file_name().and_then(|n| n.to_str()).unwrap_or("Untitled"));

    let params = DocsIngestParams {
      cwd: Some(cwd.to_string_lossy().to_string()),
      directory: Some(abs_path.to_string_lossy().to_string()),
    };

    let request = Request {
      id: Some(1),
      method: Method::DocsIngest,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to index document")?;

    if let Some(err) = response.error {
      error!("Index error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      let chunks = result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Indexed document '{}' ({} chunks)", doc_title, chunks);
    }
  } else {
    // Index as code - trigger a targeted index of just this file
    let relative_path = abs_path
      .strip_prefix(&cwd)
      .map(|p| p.to_string_lossy().to_string())
      .unwrap_or_else(|_| abs_path.to_string_lossy().to_string());

    let params = CodeIndexParams {
      cwd: Some(cwd.to_string_lossy().to_string()),
      force: true,
      stream: false,
    };

    let request = Request {
      id: Some(1),
      method: Method::CodeIndex,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to index code file")?;

    if let Some(err) = response.error {
      error!("Index error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      let chunks = result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Indexed code file '{}' ({} chunks)", relative_path, chunks);
    }
  }

  Ok(())
}

/// Index documents from a directory (internal impl)
pub async fn cmd_index_docs_impl(directory: Option<&str>, force: bool, stats: bool) -> Result<()> {
  use engram_core::Config;

  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let config = Config::load_for_project(&cwd);

  // Determine the docs directory
  let docs_dir = if let Some(dir) = directory {
    if Path::new(dir).is_absolute() {
      std::path::PathBuf::from(dir)
    } else {
      cwd.join(dir)
    }
  } else if let Some(ref configured_dir) = config.docs.directory {
    cwd.join(configured_dir)
  } else {
    error!("No directory specified and docs.directory not configured");
    error!("Use --docs-dir <path> or set docs.directory in .claude/ccengram.toml");
    std::process::exit(1);
  };

  if !docs_dir.exists() {
    error!("Docs directory not found: {}", docs_dir.display());
    std::process::exit(1);
  }

  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  // Handle --stats
  if stats {
    let params = ProjectStatsParams {
      cwd: Some(cwd.to_string_lossy().to_string()),
    };

    let request = Request {
      id: Some(1),
      method: Method::ProjectStats,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to get stats")?;

    if let Some(err) = response.error {
      error!("Stats error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result
      && let Some(docs) = result.get("documents")
    {
      let total = docs.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
      let chunks = docs.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Document Statistics:");
      println!("  Total documents: {}", total);
      println!("  Total chunks: {}", chunks);
      println!(
        "  Configured directory: {}",
        config.docs.directory.as_deref().unwrap_or("(none)")
      );
      println!("  Extensions: {}", config.docs.extensions.join(", "));
    }
    return Ok(());
  }

  // Collect files to index
  let extensions: std::collections::HashSet<_> = config.docs.extensions.iter().map(|s| s.as_str()).collect();

  let mut files_to_index = Vec::new();

  fn collect_doc_files(
    dir: &Path,
    extensions: &std::collections::HashSet<&str>,
    max_size: usize,
    files: &mut Vec<std::path::PathBuf>,
  ) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
      let entry = entry?;
      let path = entry.path();

      if path.is_dir() {
        collect_doc_files(&path, extensions, max_size, files)?;
      } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
        && extensions.contains(ext)
        && let Ok(meta) = std::fs::metadata(&path)
        && meta.len() as usize <= max_size
      {
        files.push(path);
      }
    }
    Ok(())
  }

  collect_doc_files(&docs_dir, &extensions, config.docs.max_file_size, &mut files_to_index)
    .context("Failed to scan docs directory")?;

  if files_to_index.is_empty() {
    println!("No document files found in {}", docs_dir.display());
    println!("Looking for extensions: {}", config.docs.extensions.join(", "));
    return Ok(());
  }

  println!(
    "Found {} document files in {}",
    files_to_index.len(),
    docs_dir.display()
  );

  if force {
    println!("Force re-indexing all documents...");
  }

  let mut indexed = 0;
  let mut skipped = 0;
  let mut failed = 0;

  for file_path in &files_to_index {
    let abs_path = match file_path.canonicalize() {
      Ok(p) => p,
      Err(_) => {
        failed += 1;
        continue;
      }
    };

    let _doc_title = abs_path.file_name().and_then(|n| n.to_str()).unwrap_or("Untitled");

    let params = DocsIngestParams {
      cwd: Some(cwd.to_string_lossy().to_string()),
      directory: Some(abs_path.to_string_lossy().to_string()),
    };

    let request = Request {
      id: Some(1),
      method: Method::DocsIngest,
      params,
    };

    match client.request(to_daemon_request(request)).await {
      Ok(response) => {
        if let Some(err) = response.error {
          if err.message.contains("already indexed") && !force {
            skipped += 1;
          } else {
            warn!("Failed to index {}: {}", abs_path.display(), err.message);
            failed += 1;
          }
        } else {
          indexed += 1;
          debug!("Indexed: {}", abs_path.display());
        }
      }
      Err(e) => {
        warn!("Failed to index {}: {}", abs_path.display(), e);
        failed += 1;
      }
    }
  }

  println!("\nDocument indexing complete:");
  println!("  Indexed: {}", indexed);
  if skipped > 0 {
    println!("  Skipped (already indexed): {}", skipped);
  }
  if failed > 0 {
    println!("  Failed: {}", failed);
  }

  Ok(())
}

/// Index code files
pub async fn cmd_index_code(force: bool, stats: bool, export: Option<&str>, load: Option<&str>) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // Handle --stats
  if stats {
    let params = CodeStatsParams { cwd: Some(cwd.clone()) };

    let request = Request {
      id: Some(1),
      method: Method::CodeStats,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to get index stats")?;

    if let Some(err) = response.error {
      error!("Stats error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      println!("Code Index Statistics");
      println!("=====================");
      println!();

      if let Some(health) = result.get("index_health_score").and_then(|v| v.as_u64()) {
        println!("Health Score: {}%", health);
      }
      if let Some(files) = result.get("total_files").and_then(|v| v.as_u64()) {
        println!("Files Indexed: {}", files);
      }
      if let Some(chunks) = result.get("total_chunks").and_then(|v| v.as_u64()) {
        println!("Total Chunks: {}", chunks);
      }
      if let Some(tokens) = result.get("total_tokens_estimate").and_then(|v| v.as_u64()) {
        println!("Estimated Tokens: {}", tokens);
      }
      if let Some(lines) = result.get("total_lines").and_then(|v| v.as_u64()) {
        println!("Total Lines: {}", lines);
      }
      if let Some(avg) = result.get("average_chunks_per_file").and_then(|v| v.as_f64()) {
        println!("Avg Chunks/File: {:.1}", avg);
      }

      println!();
      println!("Language Breakdown:");
      if let Some(langs) = result.get("language_breakdown").and_then(|v| v.as_object()) {
        let mut sorted: Vec<_> = langs.iter().collect();
        sorted.sort_by(|a, b| b.1.as_u64().unwrap_or(0).cmp(&a.1.as_u64().unwrap_or(0)));
        for (lang, count) in sorted {
          println!("  {}: {}", lang, count);
        }
      }

      println!();
      println!("Chunk Type Breakdown:");
      if let Some(types) = result.get("chunk_type_breakdown").and_then(|v| v.as_object()) {
        let mut sorted: Vec<_> = types.iter().collect();
        sorted.sort_by(|a, b| b.1.as_u64().unwrap_or(0).cmp(&a.1.as_u64().unwrap_or(0)));
        for (ctype, count) in sorted {
          println!("  {}: {}", ctype, count);
        }
      }
    }
    return Ok(());
  }

  // Handle --export
  if let Some(output) = export {
    println!("Exporting code index...");

    let params = CodeListParams {
      cwd: Some(cwd.clone()),
      limit: None,
    };

    let request = Request {
      id: Some(1),
      method: Method::CodeList,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to export index")?;

    if let Some(err) = response.error {
      error!("Export error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      #[derive(serde::Serialize)]
      struct ExportData {
        version: &'static str,
        exported_at: String,
        chunks: serde_json::Value,
      }
      let export_data = ExportData {
        version: "1.0",
        exported_at: chrono::Utc::now().to_rfc3339(),
        chunks: result.clone(),
      };

      let json = serde_json::to_string_pretty(&export_data)?;
      std::fs::write(output, &json)?;

      if let Some(arr) = result.as_array() {
        println!("Exported {} code chunks to {}", arr.len(), output);
      } else {
        println!("Exported code index to {}", output);
      }
    }
    return Ok(());
  }

  // Handle --load
  if let Some(path) = load {
    let content = std::fs::read_to_string(path).context("Failed to read load file")?;
    let export_data: serde_json::Value = serde_json::from_str(&content).context("Invalid JSON in load file")?;

    let Some(chunks) = export_data.get("chunks").and_then(|v| v.as_array()) else {
      error!("Invalid export format: missing 'chunks' array");
      std::process::exit(1);
    };

    println!("Importing {} code chunks...", chunks.len());

    let mut imported = 0;
    for chunk in chunks {
      // Extract fields from chunk for CodeImportChunkParams
      let file_path = chunk
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
      let content = chunk.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
      let language = chunk.get("language").and_then(|v| v.as_str()).map(|s| s.to_string());

      let params = CodeImportChunkParams {
        cwd: Some(cwd.clone()),
        file_path,
        content,
        language,
      };

      let request = Request {
        id: Some(1),
        method: Method::CodeImportChunk,
        params,
      };

      match client.request(to_daemon_request(request)).await {
        Ok(resp) if resp.error.is_none() => imported += 1,
        Ok(resp) => {
          if let Some(err) = resp.error {
            error!("Failed to import chunk: {}", err.message);
          }
        }
        Err(e) => error!("Request failed: {}", e),
      }
    }

    println!("Imported {} of {} code chunks", imported, chunks.len());
    return Ok(());
  }

  // Default: run indexing with streaming progress
  let is_tty = std::io::stdout().is_terminal();

  // Use streaming for TTY, fall back to non-streaming otherwise
  if is_tty {
    println!("Indexing code in {}...", cwd);
    println!();

    let params = CodeIndexParams {
      cwd: Some(cwd.clone()),
      force,
      stream: true,
    };

    let request = Request {
      id: Some(1),
      method: Method::CodeIndex,
      params,
    };

    let mut stream = client
      .request_streaming(to_daemon_request(request))
      .await
      .context("Failed to start indexing")?;

    let mut last_progress_len = 0;
    let mut final_result = None;

    while let Some(response) = stream.recv().await {
      // Handle progress updates
      if let Some(progress) = &response.progress {
        // Clear the previous line
        if last_progress_len > 0 {
          print!("\r{}\r", " ".repeat(last_progress_len));
        }

        // Format progress message
        let msg = match progress.phase.as_str() {
          "scanning" => {
            let files = progress.processed_files.unwrap_or(0);
            let current = progress
              .current_file
              .as_ref()
              .map(|f| truncate_path(f, 40))
              .unwrap_or_default();
            if current.is_empty() {
              format!("Scanning... {} files found", files)
            } else {
              format!("Scanning... {} files found ({})", files, current)
            }
          }
          "indexing" => {
            let processed = progress.processed_files.unwrap_or(0);
            let total = progress.total_files.unwrap_or(0);
            let chunks = progress.chunks_created.unwrap_or(0);
            let percent = if total > 0 { (processed * 100) / total } else { 0 };
            let current = progress
              .current_file
              .as_ref()
              .map(|f| truncate_path(f, 30))
              .unwrap_or_default();

            // Progress bar
            let bar_width = 20;
            let filled = (bar_width * percent / 100) as usize;
            let empty = bar_width as usize - filled;
            let bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(empty));

            if current.is_empty() {
              format!("{} {}% ({}/{}) {} chunks", bar, percent, processed, total, chunks)
            } else {
              format!(
                "{} {}% ({}/{}) {} chunks - {}",
                bar, percent, processed, total, chunks, current
              )
            }
          }
          "complete" => {
            let files = progress.processed_files.unwrap_or(0);
            let chunks = progress.chunks_created.unwrap_or(0);
            format!("Complete: {} files, {} chunks", files, chunks)
          }
          _ => progress.message.clone().unwrap_or_default(),
        };

        print!("{}", msg);
        std::io::stdout().flush().ok();
        last_progress_len = msg.len();
      }

      // Check for final response
      if response.result.is_some() || response.error.is_some() {
        // Clear progress line before final output
        if last_progress_len > 0 {
          print!("\r{}\r", " ".repeat(last_progress_len));
        }
        final_result = Some(response);
        break;
      }
    }

    // Handle final response
    let Some(response) = final_result else {
      error!("Connection closed without response");
      std::process::exit(1);
    };

    if let Some(err) = response.error {
      error!("Index error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      print_index_result(&result);
    }
  } else {
    // Non-TTY: use simple non-streaming request
    println!("Indexing code in {}...", cwd);

    let params = CodeIndexParams {
      cwd: Some(cwd.clone()),
      force,
      stream: false,
    };

    let request = Request {
      id: Some(1),
      method: Method::CodeIndex,
      params,
    };

    let response = client
      .request(to_daemon_request(request))
      .await
      .context("Failed to index code")?;

    if let Some(err) = response.error {
      error!("Index error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      print_index_result(&result);
    }
  }

  Ok(())
}

/// Print the final index result
fn print_index_result(result: &serde_json::Value) {
  let files_scanned = result.get("files_scanned").and_then(|v| v.as_u64()).unwrap_or(0);
  let files_indexed = result.get("files_indexed").and_then(|v| v.as_u64()).unwrap_or(0);
  let chunks_created = result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0);
  let scan_duration_ms = result.get("scan_duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
  let index_duration_ms = result.get("index_duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
  let total_duration_ms = result.get("total_duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
  let files_per_second = result.get("files_per_second").and_then(|v| v.as_f64()).unwrap_or(0.0);
  let bytes_processed = result.get("bytes_processed").and_then(|v| v.as_u64()).unwrap_or(0);

  println!("Indexing complete:");
  println!("  Files scanned: {}", files_scanned);
  println!("  Files indexed: {}", files_indexed);
  println!("  Chunks created: {}", chunks_created);
  println!();
  println!("Performance:");
  println!("  Scan time:  {} ms", scan_duration_ms);
  println!("  Index time: {} ms", index_duration_ms);
  println!(
    "  Total time: {} ms ({:.1}s)",
    total_duration_ms,
    total_duration_ms as f64 / 1000.0
  );
  if files_per_second > 0.0 {
    println!("  Speed:      {:.1} files/second", files_per_second);
  }
  if bytes_processed > 0 {
    let kb = bytes_processed as f64 / 1024.0;
    let mb = kb / 1024.0;
    if mb >= 1.0 {
      println!("  Processed:  {:.1} MB", mb);
    } else {
      println!("  Processed:  {:.1} KB", kb);
    }
  }
}

/// Truncate a file path for display
fn truncate_path(path: &str, max_len: usize) -> String {
  if path.len() <= max_len {
    return path.to_string();
  }

  // Try to show just the filename
  if let Some(pos) = path.rfind('/') {
    let filename = &path[pos + 1..];
    if filename.len() <= max_len {
      return filename.to_string();
    }
    // Truncate filename with ellipsis
    return format!("...{}", &filename[filename.len().saturating_sub(max_len - 3)..]);
  }

  // Just truncate with ellipsis
  format!("...{}", &path[path.len().saturating_sub(max_len - 3)..])
}
