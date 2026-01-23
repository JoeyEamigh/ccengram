//! Context retrieval commands for code and document chunks

use anyhow::{Context, Result};
use daemon::{Request, connect_or_start};
use tracing::error;

/// Get context around a chunk (auto-detects code vs document)
pub async fn cmd_context(chunk_id: &str, before: Option<usize>, after: Option<usize>, json_output: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .ok()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|| ".".to_string());

  // Try code_context first
  let code_params = serde_json::json!({
    "chunk_id": chunk_id,
    "cwd": cwd,
    "lines_before": before,
    "lines_after": after,
  });

  let code_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_context".to_string(),
    params: code_params,
  };

  let code_response = client
    .request(code_request)
    .await
    .context("Failed to get code context")?;

  // Check if code_context succeeded
  if code_response.error.is_none()
    && let Some(result) = code_response.result
  {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
      print_code_context(&result)?;
    }
    return Ok(());
  }

  // If code_context failed, try doc_context
  let doc_params = serde_json::json!({
    "chunk_id": chunk_id,
    "cwd": cwd,
    "chunks_before": before.unwrap_or(1),
    "chunks_after": after.unwrap_or(1),
  });

  let doc_request = Request {
    id: Some(serde_json::json!(2)),
    method: "doc_context".to_string(),
    params: doc_params,
  };

  let doc_response = client
    .request(doc_request)
    .await
    .context("Failed to get document context")?;

  if let Some(err) = doc_response.error {
    // Both failed - show the original code error if we have it, otherwise the doc error
    let code_err = code_response
      .error
      .map(|e| e.message)
      .unwrap_or_else(|| "unknown error".to_string());
    error!(
      "Chunk not found in code or documents. Code error: {}. Doc error: {}",
      code_err, err.message
    );
    std::process::exit(1);
  }

  if let Some(result) = doc_response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
      print_doc_context(&result)?;
    }
  }

  Ok(())
}

/// Print code context in a readable format
fn print_code_context(result: &serde_json::Value) -> Result<()> {
  let file_path = result.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
  let language = result.get("language").and_then(|v| v.as_str()).unwrap_or("unknown");
  let total_lines = result.get("total_file_lines").and_then(|v| v.as_u64()).unwrap_or(0);

  println!("File: {} ({})", file_path, language);
  println!("Total lines: {}", total_lines);
  println!();

  if let Some(context) = result.get("context") {
    // Print before section
    if let Some(before) = context.get("before") {
      let start_line = before.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0);
      let content = before.get("content").and_then(|v| v.as_str()).unwrap_or("");
      if !content.is_empty() {
        println!("--- Before (line {}) ---", start_line);
        println!("{}", content);
        println!();
      }
    }

    // Print target section
    if let Some(target) = context.get("target") {
      let start_line = target.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0);
      let end_line = target.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0);
      let content = target.get("content").and_then(|v| v.as_str()).unwrap_or("");
      println!(">>> Target (lines {}-{}) <<<", start_line, end_line);
      println!("{}", content);
      println!();
    }

    // Print after section
    if let Some(after) = context.get("after") {
      let start_line = after.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0);
      let content = after.get("content").and_then(|v| v.as_str()).unwrap_or("");
      if !content.is_empty() {
        println!("--- After (line {}) ---", start_line);
        println!("{}", content);
      }
    }
  }

  Ok(())
}

/// Print document context in a readable format
fn print_doc_context(result: &serde_json::Value) -> Result<()> {
  let title = result.get("title").and_then(|v| v.as_str()).unwrap_or("?");
  let source = result.get("source").and_then(|v| v.as_str()).unwrap_or("?");
  let total_chunks = result.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(0);

  println!("Document: {}", title);
  println!("Source: {}", source);
  println!("Total chunks: {}", total_chunks);
  println!();

  if let Some(context) = result.get("context") {
    // Print before chunks
    if let Some(before) = context.get("before").and_then(|v| v.as_array()) {
      for chunk in before {
        let index = chunk.get("chunk_index").and_then(|v| v.as_u64()).unwrap_or(0);
        let content = chunk.get("content").and_then(|v| v.as_str()).unwrap_or("");
        println!("--- Chunk {} ---", index);
        println!("{}", content);
        println!();
      }
    }

    // Print target chunk
    if let Some(target) = context.get("target") {
      let index = target.get("chunk_index").and_then(|v| v.as_u64()).unwrap_or(0);
      let content = target.get("content").and_then(|v| v.as_str()).unwrap_or("");
      println!(">>> Chunk {} (target) <<<", index);
      println!("{}", content);
      println!();
    }

    // Print after chunks
    if let Some(after) = context.get("after").and_then(|v| v.as_array()) {
      for chunk in after {
        let index = chunk.get("chunk_index").and_then(|v| v.as_u64()).unwrap_or(0);
        let content = chunk.get("content").and_then(|v| v.as_str()).unwrap_or("");
        println!("--- Chunk {} ---", index);
        println!("{}", content);
        println!();
      }
    }
  }

  Ok(())
}
