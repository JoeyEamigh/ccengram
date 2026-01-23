//! Search commands for memories, code, and documents

use anyhow::{Context, Result};
use daemon::{Request, connect_or_start};
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

  let mut params = serde_json::json!({
      "query": query,
      "cwd": cwd,
      "limit": limit,
      "include_superseded": include_superseded,
  });

  if let Some(s) = sector {
    params["sector"] = serde_json::json!(s);
  }
  if let Some(t) = memory_type {
    params["type"] = serde_json::json!(t);
  }
  if let Some(sal) = min_salience {
    params["min_salience"] = serde_json::json!(sal);
  }
  if let Some(sc) = scope {
    params["scope_path"] = serde_json::json!(sc);
  }

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_search".to_string(),
    params,
  };

  let response = client.request(request).await.context("Failed to search memories")?;

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

  let mut params = serde_json::json!({
      "query": query,
      "cwd": cwd,
      "limit": limit,
  });

  if let Some(lang) = language {
    params["language"] = serde_json::json!(lang);
  }
  if let Some(ct) = chunk_type {
    params["chunk_type"] = serde_json::json!(ct);
  }
  if let Some(p) = path {
    params["file_path_prefix"] = serde_json::json!(p);
  }
  if let Some(s) = symbol {
    params["symbol"] = serde_json::json!(s);
  }

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_search".to_string(),
    params,
  };

  let response = client.request(request).await.context("Failed to search code")?;

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

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "docs_search".to_string(),
    params: serde_json::json!({
        "query": query,
        "cwd": cwd,
        "limit": limit,
    }),
  };

  let response = client.request(request).await.context("Failed to search documents")?;

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
