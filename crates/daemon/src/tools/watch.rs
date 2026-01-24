//! File watcher and code statistics tool methods

use super::ToolHandler;
use crate::router::{CodeStatsResult, Request, Response, WatchStartResult, WatchStatusResult, WatchStopResult};
use crate::startup_scan::StartupScanConfig;
use engram_core::{Config, ScanMode};
use serde::Deserialize;
use std::path::PathBuf;

impl ToolHandler {
  /// Start file watcher for a project
  pub async fn watch_start(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      /// Override startup_scan enabled setting
      #[serde(default)]
      startup_scan: Option<bool>,
      /// Override startup_scan_mode setting
      #[serde(default)]
      startup_scan_mode: Option<String>,
      /// Override startup_scan_blocking setting
      #[serde(default)]
      startup_scan_blocking: Option<bool>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Get or create project to get its ID
    let (info, _db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Build startup scan config with CLI overrides
    let config = Config::load_for_project(&project_path);
    let mut scan_config = StartupScanConfig::from_config(&config);

    // Apply CLI overrides
    if let Some(enabled) = args.startup_scan {
      scan_config.enabled = enabled;
    }
    if let Some(ref mode_str) = args.startup_scan_mode
      && let Ok(mode) = mode_str.parse::<ScanMode>()
    {
      scan_config.mode = mode;
    }
    if let Some(blocking) = args.startup_scan_blocking {
      scan_config.blocking = blocking;
    }

    // Start the watcher for this project (with embedding if available)
    if let Err(e) = self
      .registry
      .start_watcher_with_scan_config(
        info.id.as_str(),
        &project_path,
        self.embedding.clone(),
        Some(scan_config),
      )
      .await
    {
      return Response::error(request.id, -32000, &format!("Failed to start watcher: {}", e));
    }

    Response::success(
      request.id,
      WatchStartResult {
        status: "started".to_string(),
        path: project_path.to_string_lossy().to_string(),
        project_id: info.id.as_str().to_string(),
      },
    )
  }

  /// Stop file watcher for a project
  pub async fn watch_stop(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get project to get its ID
    let (info, _db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Stop the watcher
    if let Err(e) = self.registry.stop_watcher(info.id.as_str()).await {
      return Response::error(request.id, -32000, &format!("Failed to stop watcher: {}", e));
    }

    Response::success(
      request.id,
      WatchStopResult {
        status: "stopped".to_string(),
        path: project_path.to_string_lossy().to_string(),
        project_id: info.id.as_str().to_string(),
      },
    )
  }

  /// Get file watcher status
  pub async fn watch_status(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get project to get its ID
    let (info, _db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Get watcher status
    let status = self.registry.watcher_status(info.id.as_str()).await;

    // Get scan progress if scanning
    let scan_progress = if status.scanning {
      self.registry.scan_progress(info.id.as_str()).await
    } else {
      None
    };

    Response::success(
      request.id,
      WatchStatusResult {
        running: status.running,
        root: status.root.map(|p| p.to_string_lossy().to_string()),
        pending_changes: status.pending_changes,
        project_id: info.id.as_str().to_string(),
        scanning: status.scanning,
        scan_progress: scan_progress.map(|(p, t)| [p, t]),
      },
    )
  }

  /// Get code index statistics
  pub async fn code_stats(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get all chunks to compute statistics
    match db.list_code_chunks(None, None).await {
      Ok(chunks) => {
        use std::collections::HashMap;

        let mut language_counts: HashMap<String, usize> = HashMap::new();
        let mut chunk_type_counts: HashMap<String, usize> = HashMap::new();
        let mut files_indexed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut total_tokens: u64 = 0;
        let mut total_lines: u64 = 0;

        for chunk in &chunks {
          let lang = format!("{:?}", chunk.language).to_lowercase();
          *language_counts.entry(lang).or_default() += 1;

          let ctype = format!("{:?}", chunk.chunk_type).to_lowercase();
          *chunk_type_counts.entry(ctype).or_default() += 1;

          files_indexed.insert(chunk.file_path.clone());
          total_tokens += chunk.tokens_estimate as u64;
          total_lines += (chunk.end_line - chunk.start_line + 1) as u64;
        }

        let total_chunks = chunks.len();
        let total_files = files_indexed.len();
        let avg_chunks_per_file = if total_files > 0 {
          total_chunks as f32 / total_files as f32
        } else {
          0.0
        };

        // Compute health score (0-100)
        // Factors: coverage (has chunks), diversity (multiple languages), recent indexing
        let mut health_score: f32 = 0.0;
        if total_chunks > 0 {
          health_score += 40.0; // Base score for having any chunks
          if total_files > 0 {
            health_score += 20.0; // Has files indexed
          }
          if language_counts.len() > 1 {
            health_score += 10.0; // Multiple languages
          }
          if avg_chunks_per_file > 1.0 && avg_chunks_per_file < 50.0 {
            health_score += 20.0; // Reasonable chunk density
          }
          // Age-based scoring would require checking indexed_at times
          health_score += 10.0; // Assume recent for now
        }

        Response::success(
          request.id,
          CodeStatsResult {
            total_chunks,
            total_files,
            total_tokens_estimate: total_tokens,
            total_lines,
            average_chunks_per_file: avg_chunks_per_file,
            language_breakdown: language_counts,
            chunk_type_breakdown: chunk_type_counts,
            index_health_score: health_score.min(100.0).round() as u32,
          },
        )
      }
      Err(e) => Response::error(request.id, -32000, &format!("Stats error: {}", e)),
    }
  }
}
