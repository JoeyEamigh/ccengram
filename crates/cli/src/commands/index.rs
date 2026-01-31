//! Index commands for code and documents

use std::{collections::HashMap, io::IsTerminal, path::Path};

use anyhow::{Context, Result};
use ccengram::ipc::{
  StreamUpdate,
  code::{CodeIndexParams, CodeIndexResult, CodeStatsParams},
  docs::{DocsIngestFullResult, DocsIngestParams},
  system::ProjectStatsParams,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing::error;

use crate::IndexCommand;

/// Manage code and document index
pub async fn cmd_index(command: Option<IndexCommand>) -> Result<()> {
  match command {
    Some(IndexCommand::Code { force, stats }) => cmd_index_code(force, stats).await,
    Some(IndexCommand::Docs {
      directory,
      force,
      stats,
    }) => cmd_index_docs_impl(directory.as_deref(), force, stats).await,
    Some(IndexCommand::File { path, title, force }) => cmd_index_file(&path, title.as_deref(), force).await,
    None => {
      // Default: index code, and also docs if docs.directory is configured
      cmd_index_all(false).await
    }
  }
}

/// Index both code and docs (if configured) with streaming progress
async fn cmd_index_all(force: bool) -> Result<()> {
  use ccengram::config::Config;

  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let config = Config::load_for_project(&cwd).await;
  let is_tty = std::io::stdout().is_terminal();

  let client = ccengram::Daemon::connect_or_start(cwd.clone())
    .await
    .context("Failed to connect to daemon")?;

  // Phase 1: Index code
  println!("Indexing code...");
  if is_tty {
    println!();
  }

  let code_params = CodeIndexParams { force, stream: true };

  let code_result = run_with_progress(&client, code_params, is_tty).await?;
  print_code_result(&code_result);

  // Phase 2: Index docs if configured
  if let Some(ref docs_dir) = config.docs.directory {
    let docs_path = cwd.join(docs_dir);
    if docs_path.exists() {
      println!();
      println!("Indexing documents from {}...", docs_dir);
      if is_tty {
        println!();
      }

      let docs_params = DocsIngestParams {
        directory: Some(docs_dir.clone()),
        file: None,
        stream: true,
      };

      let docs_result = run_with_progress(&client, docs_params, is_tty).await?;
      print_docs_result(&docs_result);
    }
  }

  Ok(())
}

/// Run a streaming request and show progress with per-stage bars
async fn run_with_progress<R>(
  client: &ccengram::ipc::Client,
  params: R,
  show_progress: bool,
) -> Result<R::Response, anyhow::Error>
where
  R: ccengram::ipc::IpcRequest + Send + 'static,
  R::Response: Send + 'static,
{
  let mut rx = client.call_streaming(params).await?;

  // Set up multi-progress for per-stage bars
  let mp = MultiProgress::new();
  let mut stage_bars: HashMap<String, ProgressBar> = HashMap::new();

  let stage_style = ProgressStyle::default_bar()
    .template("{prefix:>12.cyan} [{bar:30.cyan/blue}] {pos:>5}/{len:>5} {msg}")
    .unwrap()
    .progress_chars("=>-");

  while let Some(update) = rx.recv().await {
    match update {
      StreamUpdate::Progress {
        message: _,
        percent: _,
        stage,
      } => {
        if show_progress
          && let (Some(stage_name), Some(processed), Some(total)) = (&stage.stage, stage.processed, stage.total)
        {
          // Get or create progress bar for this stage
          let pb = stage_bars.entry(stage_name.clone()).or_insert_with(|| {
            let pb = mp.add(ProgressBar::new(total as u64));
            pb.set_style(stage_style.clone());
            pb.set_prefix(stage_name.clone());
            pb
          });

          // Update the bar
          pb.set_length(total as u64);
          pb.set_position(processed as u64);

          // Show current file (truncated if needed)
          if let Some(ref file) = stage.current_file {
            let file_display = if file.len() > 40 {
              format!("...{}", &file[file.len() - 37..])
            } else {
              file.clone()
            };
            pb.set_message(file_display);
          }

          // For writing stage, show chunks in message
          if stage_name == "writing"
            && let Some(chunks) = stage.chunks_created
          {
            pb.set_message(format!("{} chunks", chunks));
          }
        }
      }
      StreamUpdate::Done(result) => {
        // Finish all progress bars
        for (_, pb) in stage_bars {
          pb.finish_and_clear();
        }
        return result.map_err(|e| anyhow::anyhow!("{}", e));
      }
    }
  }

  Err(anyhow::anyhow!("Stream ended without result"))
}

/// Print code index result summary
fn print_code_result(result: &CodeIndexResult) {
  println!("Code indexing complete:");
  println!(
    "  Files: {} scanned, {} indexed",
    result.files_scanned, result.files_indexed
  );
  println!("  Chunks: {}", result.chunks_created);
  println!(
    "  Time: {:.1}s ({:.1} files/sec)",
    result.total_duration_ms as f64 / 1000.0,
    result.files_per_second
  );
}

/// Print docs ingest result summary
fn print_docs_result(result: &DocsIngestFullResult) {
  println!("Document indexing complete:");
  println!(
    "  Files: {} scanned, {} ingested",
    result.files_scanned, result.files_ingested
  );
  println!("  Chunks: {}", result.chunks_created);
  println!(
    "  Time: {:.1}s ({:.1} files/sec)",
    result.total_duration_ms as f64 / 1000.0,
    result.files_per_second
  );
}

/// Index a single file (auto-detects code vs document based on extension)
pub async fn cmd_index_file(path: &str, title: Option<&str>, _force: bool) -> Result<()> {
  use ccengram::config::Config;

  let file_path = std::path::Path::new(path);
  if !file_path.exists() {
    error!("File not found: {}", path);
    std::process::exit(1);
  }

  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let config = Config::load_for_project(&cwd).await;

  // Check if this is a document file based on extension
  let is_doc = file_path
    .extension()
    .and_then(|e| e.to_str())
    .is_some_and(|ext| config.docs.extensions.iter().any(|e| e == ext));

  let client = ccengram::Daemon::connect_or_start(cwd.clone())
    .await
    .context("Failed to connect to daemon")?;

  let abs_path = file_path.canonicalize().context("Failed to resolve path")?;

  if is_doc {
    // Index as document
    let doc_title = title.unwrap_or_else(|| abs_path.file_name().and_then(|n| n.to_str()).unwrap_or("Untitled"));

    let params = DocsIngestParams {
      directory: None,
      file: Some(abs_path.to_string_lossy().to_string()),
      stream: false,
    };

    match client.call(params).await {
      Ok(result) => {
        println!("Indexed document '{}' ({} chunks)", doc_title, result.chunks_created);
      }
      Err(e) => {
        error!("Index error: {}", e);
        std::process::exit(1);
      }
    }
  } else {
    // Index as code - trigger a targeted index of just this file
    let relative_path = abs_path
      .strip_prefix(&cwd)
      .map(|p| p.to_string_lossy().to_string())
      .unwrap_or_else(|_| abs_path.to_string_lossy().to_string());

    let params = CodeIndexParams {
      force: true,
      stream: true,
    };

    let is_tty = std::io::stdout().is_terminal();
    match run_with_progress(&client, params, is_tty).await {
      Ok(result) => {
        println!(
          "Indexed code file '{}' ({} chunks)",
          relative_path, result.chunks_created
        );
      }
      Err(e) => {
        error!("Index error: {}", e);
        std::process::exit(1);
      }
    }
  }

  Ok(())
}

/// Index documents from a directory (internal impl)
pub async fn cmd_index_docs_impl(directory: Option<&str>, _force: bool, stats: bool) -> Result<()> {
  use ccengram::config::Config;

  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let config = Config::load_for_project(&cwd).await;
  let is_tty = std::io::stdout().is_terminal();

  // Determine the docs directory (relative to project root)
  let docs_dir_str = if let Some(dir) = directory {
    Some(dir.to_string())
  } else {
    config.docs.directory.clone()
  };

  // Validate directory exists
  let docs_path = docs_dir_str
    .as_ref()
    .map(|d| {
      if Path::new(d).is_absolute() {
        std::path::PathBuf::from(d)
      } else {
        cwd.join(d)
      }
    })
    .unwrap_or_else(|| cwd.clone());

  if !docs_path.exists() {
    if docs_dir_str.is_some() {
      error!("Docs directory not found: {}", docs_path.display());
      std::process::exit(1);
    } else {
      error!("No directory specified and docs.directory not configured");
      error!("Use --docs-dir <path> or set docs.directory in .claude/ccengram.toml");
      std::process::exit(1);
    }
  }

  let client = ccengram::Daemon::connect_or_start(cwd.clone())
    .await
    .context("Failed to connect to daemon")?;

  // Handle --stats
  if stats {
    match client.call(ProjectStatsParams).await {
      Ok(stats) => {
        println!("Document Statistics:");
        println!("  Total documents: {}", stats.documents);
        println!(
          "  Configured directory: {}",
          config.docs.directory.as_deref().unwrap_or("(none)")
        );
        println!("  Extensions: {}", config.docs.extensions.join(", "));
      }
      Err(e) => {
        error!("Stats error: {}", e);
        std::process::exit(1);
      }
    }
    return Ok(());
  }

  println!("Indexing documents from {}...", docs_path.display());
  if is_tty {
    println!();
  }

  let params = DocsIngestParams {
    directory: docs_dir_str,
    file: None,
    stream: true,
  };

  match run_with_progress(&client, params, is_tty).await {
    Ok(result) => {
      print_docs_result(&result);
    }
    Err(e) => {
      error!("Index error: {}", e);
      std::process::exit(1);
    }
  }

  Ok(())
}

/// Index code files
pub async fn cmd_index_code(force: bool, stats: bool) -> Result<()> {
  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let client = ccengram::Daemon::connect_or_start(cwd.clone())
    .await
    .context("Failed to connect to daemon")?;

  // Handle --stats
  if stats {
    match client.call(CodeStatsParams).await {
      Ok(result) => {
        println!("Code Index Statistics");
        println!("=====================");
        println!();

        println!("Health Score: {}%", result.index_health_score);
        println!("Files Indexed: {}", result.total_files);
        println!("Total Chunks: {}", result.total_chunks);
        println!("Estimated Tokens: {}", result.total_tokens_estimate);
        println!("Total Lines: {}", result.total_lines);
        println!("Avg Chunks/File: {:.1}", result.average_chunks_per_file);

        println!();
        println!("Language Breakdown:");
        let mut sorted: Vec<_> = result.language_breakdown.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (lang, count) in sorted {
          println!("  {}: {}", lang, count);
        }

        println!();
        println!("Chunk Type Breakdown:");
        let mut sorted: Vec<_> = result.chunk_type_breakdown.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (ctype, count) in sorted {
          println!("  {}: {}", ctype, count);
        }
      }
      Err(e) => {
        error!("Stats error: {}", e);
        std::process::exit(1);
      }
    }
    return Ok(());
  }

  // Default: run indexing
  let is_tty = std::io::stdout().is_terminal();
  let cwd_str = cwd.to_string_lossy().to_string();

  println!("Indexing code in {}...", cwd_str);

  if is_tty {
    println!();
  }

  let params = CodeIndexParams { force, stream: true };

  match run_with_progress(&client, params, is_tty).await {
    Ok(result) => {
      print_code_result(&result);
    }
    Err(e) => {
      error!("Index error: {}", e);
      std::process::exit(1);
    }
  }

  Ok(())
}
