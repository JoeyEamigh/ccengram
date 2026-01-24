//! Search commands for memories, code, and documents

use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::connect_or_start;
use ipc::{CodeSearchParams, DocsSearchParams, MemorySearchParams, Method, Request};
use tracing::error;

/// Format an ID for display
///
/// When `long` is false, shows only the first 8 characters with "..." suffix.
/// When `long` is true or ID is short, shows the full ID.
fn format_id(id: &str, long: bool) -> String {
  if long || id.len() <= 12 {
    id.to_string()
  } else {
    format!("{}...", &id[..8])
  }
}

/// Search memories
#[allow(clippy::too_many_arguments)]
pub async fn cmd_search(
  query: &str,
  limit: usize,
  project: Option<&str>,
  sector: Option<&str>,
  memory_type: Option<&str>,
  min_salience: Option<f32>,
  include_superseded: bool,
  scope: Option<&str>,
  json_output: bool,
  long_ids: bool,
) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = project
    .map(|p| p.to_string())
    .or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()))
    .unwrap_or_else(|| ".".to_string());

  let params = MemorySearchParams {
    query: query.to_string(),
    cwd: Some(cwd),
    sector: sector.map(|s| s.to_string()),
    memory_type: memory_type.map(|t| t.to_string()),
    min_salience,
    scope_path: scope.map(|s| s.to_string()),
    limit: Some(limit),
    include_superseded,
    ..Default::default()
  };

  let request = Request {
    id: Some(1),
    method: Method::MemorySearch,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to search memories")?;

  if let Some(err) = response.error {
    error!("Search error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(results) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&results)?);
      return Ok(());
    }

    let memories: Vec<serde_json::Value> = serde_json::from_value(results)?;

    if memories.is_empty() {
      println!("No memories found for: {}", query);
    } else {
      println!("Found {} memories:\n", memories.len());
      for (i, memory) in memories.iter().enumerate() {
        let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        println!(
          "{}. [{}] {}",
          i + 1,
          memory.get("sector").and_then(|v| v.as_str()).unwrap_or("unknown"),
          format_id(id, long_ids)
        );
        if let Some(content) = memory.get("content").and_then(|v| v.as_str()) {
          // Print first 200 chars
          let preview = if content.len() > 200 {
            format!("{}...", &content[..200])
          } else {
            content.to_string()
          };
          println!("   {}", preview.replace('\n', "\n   "));
        }
        if let Some(sim) = memory.get("similarity").and_then(|v| v.as_f64()) {
          println!("   Similarity: {:.2}", sim);
        }
        println!();
      }

      // Help message about prefix matching
      if !long_ids {
        println!("Tip: Use --long to show full IDs. Prefixes (8+ chars) work in commands.");
      }
    }
  }

  Ok(())
}

/// Search code
#[allow(clippy::too_many_arguments)]
pub async fn cmd_search_code(
  query: &str,
  limit: usize,
  project: Option<&str>,
  language: Option<&str>,
  chunk_type: Option<&str>,
  path: Option<&str>,
  symbol: Option<&str>,
  json_output: bool,
) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = project
    .map(|p| p.to_string())
    .or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()))
    .unwrap_or_else(|| ".".to_string());

  // Build file_pattern from optional filters
  let file_pattern = match (language, chunk_type, path, symbol) {
    (Some(lang), _, _, _) => Some(format!("*.{}", lang)),
    (_, _, Some(p), _) => Some(p.to_string()),
    _ => None,
  };

  let symbol_type = chunk_type.map(|ct| ct.to_string());

  // Note: The daemon may need to handle "symbol" separately if needed
  let _ = symbol; // Acknowledge unused for now

  let params = CodeSearchParams {
    query: query.to_string(),
    cwd: Some(cwd),
    limit: Some(limit),
    file_pattern,
    symbol_type,
  };

  let request = Request {
    id: Some(1),
    method: Method::CodeSearch,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to search code")?;

  if let Some(err) = response.error {
    error!("Code search error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(results) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&results)?);
      return Ok(());
    }

    let chunks: Vec<serde_json::Value> = serde_json::from_value(results)?;

    if chunks.is_empty() {
      println!("No code found for: {}", query);
    } else {
      println!("Found {} code chunks:\n", chunks.len());
      for (i, chunk) in chunks.iter().enumerate() {
        let file = chunk.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
        let start = chunk.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0);
        let end = chunk.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0);
        let lang = chunk.get("language").and_then(|v| v.as_str()).unwrap_or("?");

        println!("{}. {}:{}-{} [{}]", i + 1, file, start, end, lang);

        if let Some(symbols) = chunk.get("symbols").and_then(|v| v.as_array()) {
          let symbols: Vec<_> = symbols.iter().filter_map(|s| s.as_str()).collect();
          if !symbols.is_empty() {
            println!("   Symbols: {}", symbols.join(", "));
          }
        }

        if let Some(sim) = chunk.get("similarity").and_then(|v| v.as_f64()) {
          println!("   Similarity: {:.2}", sim);
        }
        println!();
      }
    }
  }

  Ok(())
}

/// Search documents
pub async fn cmd_search_docs(
  query: &str,
  limit: usize,
  project: Option<&str>,
  json_output: bool,
  long_ids: bool,
) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = project
    .map(|p| p.to_string())
    .or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()))
    .unwrap_or_else(|| ".".to_string());

  let params = DocsSearchParams {
    query: query.to_string(),
    cwd: Some(cwd),
    limit: Some(limit),
  };

  let request = Request {
    id: Some(1),
    method: Method::DocsSearch,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to search documents")?;

  if let Some(err) = response.error {
    error!("Document search error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(results) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&results)?);
      return Ok(());
    }

    let chunks: Vec<serde_json::Value> = serde_json::from_value(results)?;

    if chunks.is_empty() {
      println!("No documents found for: {}", query);
    } else {
      println!("Found {} document chunks:\n", chunks.len());
      for (i, chunk) in chunks.iter().enumerate() {
        let title = chunk.get("title").and_then(|v| v.as_str()).unwrap_or("Untitled");
        let doc_id = chunk.get("document_id").and_then(|v| v.as_str()).unwrap_or("?");

        println!("{}. {} [{}]", i + 1, title, format_id(doc_id, long_ids));

        if let Some(content) = chunk.get("content").and_then(|v| v.as_str()) {
          let preview = if content.len() > 200 {
            format!("{}...", &content[..200])
          } else {
            content.to_string()
          };
          println!("   {}", preview.replace('\n', "\n   "));
        }

        if let Some(sim) = chunk.get("similarity").and_then(|v| v.as_f64()) {
          println!("   Similarity: {:.2}", sim);
        }
        println!();
      }

      // Help message about prefix matching
      if !long_ids {
        println!("Tip: Use --long to show full IDs. Prefixes (8+ chars) work in commands.");
      }
    }
  }

  Ok(())
}
