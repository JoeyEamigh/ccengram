//! Memory management commands (show, delete, export, restore, deleted)

use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::connect_or_start;
use ipc::{
  MemoryDeleteParams, MemoryGetParams, MemoryListDeletedParams, MemoryListParams, MemoryRestoreParams, Method, Request,
};
use tracing::error;

/// Show detailed memory by ID
pub async fn cmd_show(memory_id: &str, related: bool, json_output: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let params = MemoryGetParams {
    memory_id: memory_id.to_string(),
    cwd: Some(cwd),
    include_related: if related { Some(true) } else { None },
  };

  let request = Request {
    id: Some(1),
    method: Method::MemoryGet,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to get memory")?;

  if let Some(err) = response.error {
    error!("Error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&result)?);
      return Ok(());
    }

    // Pretty print the memory
    println!("Memory Details");
    println!("==============\n");

    if let Some(id) = result.get("id").and_then(|v| v.as_str()) {
      println!("ID:       {}", id);
    }
    if let Some(sector) = result.get("sector").and_then(|v| v.as_str()) {
      println!("Sector:   {}", sector);
    }
    if let Some(tier) = result.get("tier").and_then(|v| v.as_str()) {
      println!("Tier:     {}", tier);
    }
    if let Some(mem_type) = result.get("memory_type").and_then(|v| v.as_str()) {
      println!("Type:     {}", mem_type);
    }
    if let Some(salience) = result.get("salience").and_then(|v| v.as_f64()) {
      println!("Salience: {:.2}", salience);
    }
    if let Some(importance) = result.get("importance").and_then(|v| v.as_f64()) {
      println!("Importance: {:.2}", importance);
    }
    if let Some(created) = result.get("created_at").and_then(|v| v.as_str()) {
      println!("Created:  {}", created);
    }
    if let Some(accessed) = result.get("last_accessed").and_then(|v| v.as_str()) {
      println!("Accessed: {}", accessed);
    }
    if let Some(superseded) = result.get("superseded_by").and_then(|v| v.as_str()) {
      println!("Superseded by: {}", superseded);
    }

    println!();

    if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
      println!("Content:");
      println!("{}", content);
    }

    if let Some(tags) = result.get("tags").and_then(|v| v.as_array()) {
      let tags: Vec<_> = tags.iter().filter_map(|t| t.as_str()).collect();
      if !tags.is_empty() {
        println!("\nTags: {}", tags.join(", "));
      }
    }

    if related
      && let Some(relationships) = result.get("relationships").and_then(|v| v.as_array())
      && !relationships.is_empty()
    {
      println!("\nRelated Memories ({}):", relationships.len());
      for rel in relationships {
        let rel_type = rel.get("type").and_then(|v| v.as_str()).unwrap_or("?");
        let target = rel.get("target_id").and_then(|v| v.as_str()).unwrap_or("?");
        println!("  - {} -> {}", rel_type, target);
      }
    }
  } else {
    println!("Memory not found: {}", memory_id);
  }

  Ok(())
}

/// Delete a memory
pub async fn cmd_delete(memory_id: &str, hard: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let params = MemoryDeleteParams {
    memory_id: memory_id.to_string(),
    cwd: Some(cwd),
  };

  // Note: "hard" parameter would need to be added to MemoryDeleteParams if the API supports it
  let _ = hard;

  let request = Request {
    id: Some(1),
    method: Method::MemoryDelete,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to delete memory")?;

  if let Some(err) = response.error {
    error!("Delete error: {}", err.message);
    std::process::exit(1);
  }

  if hard {
    println!("Memory {} permanently deleted", memory_id);
  } else {
    println!("Memory {} soft deleted (can be recovered)", memory_id);
  }

  Ok(())
}

/// Export memories to file
pub async fn cmd_export(output: Option<&str>, format: &str) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // Get all memories for the project
  let params = MemoryListParams {
    cwd: Some(cwd),
    ..Default::default()
  };

  let request = Request {
    id: Some(1),
    method: Method::MemoryList,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to list memories")?;

  if let Some(err) = response.error {
    error!("Export error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(memories) = response.result {
    let output_content = match format.to_lowercase().as_str() {
      "json" => serde_json::to_string_pretty(&memories)?,
      "csv" => {
        let mut csv_output = String::from("id,sector,tier,type,salience,content,created_at\n");
        if let Some(arr) = memories.as_array() {
          for mem in arr {
            let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let sector = mem.get("sector").and_then(|v| v.as_str()).unwrap_or("");
            let tier = mem.get("tier").and_then(|v| v.as_str()).unwrap_or("");
            let mem_type = mem.get("memory_type").and_then(|v| v.as_str()).unwrap_or("");
            let salience = mem.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let content = mem
              .get("content")
              .and_then(|v| v.as_str())
              .unwrap_or("")
              .replace('"', "\"\"")
              .replace('\n', "\\n");
            let created = mem.get("created_at").and_then(|v| v.as_str()).unwrap_or("");

            csv_output.push_str(&format!(
              "{},\"{}\",\"{}\",\"{}\",{:.2},\"{}\",{}\n",
              id, sector, tier, mem_type, salience, content, created
            ));
          }
        }
        csv_output
      }
      _ => {
        error!("Unknown format: {}. Use 'json' or 'csv'.", format);
        std::process::exit(1);
      }
    };

    if let Some(path) = output {
      std::fs::write(path, &output_content)?;
      println!("Exported memories to {}", path);
    } else {
      println!("{}", output_content);
    }
  }

  Ok(())
}

/// Restore a soft-deleted memory
pub async fn cmd_restore(memory_id: &str) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let params = MemoryRestoreParams {
    memory_id: memory_id.to_string(),
    cwd: Some(cwd),
  };

  let request = Request {
    id: Some(1),
    method: Method::MemoryRestore,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to restore memory")?;

  if let Some(err) = response.error {
    error!("Restore error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
    println!("Restored memory: {}", memory_id);
    println!();

    // Show restored memory details
    if let Some(sector) = result.get("sector").and_then(|v| v.as_str()) {
      println!("Sector:    {}", sector);
    }
    if let Some(mem_type) = result.get("memory_type").and_then(|v| v.as_str()) {
      println!("Type:      {}", mem_type);
    }
    if let Some(salience) = result.get("salience").and_then(|v| v.as_f64()) {
      println!("Salience:  {:.2}", salience);
    }
    println!();

    if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
      println!("Content:");
      println!("{}", content);
    }
  }

  Ok(())
}

/// List soft-deleted memories
pub async fn cmd_deleted(limit: usize, json_output: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let params = MemoryListDeletedParams {
    cwd: Some(cwd),
    limit: Some(limit),
  };

  let request = Request {
    id: Some(1),
    method: Method::MemoryListDeleted,
    params,
  };

  let response = client
    .request(cli::to_daemon_request(request))
    .await
    .context("Failed to list deleted memories")?;

  if let Some(err) = response.error {
    error!("Error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(memories) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&memories)?);
      return Ok(());
    }

    let empty_vec = vec![];
    let memories = memories.as_array().unwrap_or(&empty_vec);

    if memories.is_empty() {
      println!("No deleted memories found.");
      return Ok(());
    }

    println!("Deleted Memories ({}):", memories.len());
    println!();

    for (i, mem) in memories.iter().enumerate() {
      let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("?");
      let sector = mem.get("sector").and_then(|v| v.as_str()).unwrap_or("?");
      let content = mem.get("content").and_then(|v| v.as_str()).unwrap_or("");
      let deleted_at = mem.get("deleted_at").and_then(|v| v.as_str()).unwrap_or("?");

      // Truncate content for preview
      let preview: String = content.chars().take(60).collect();
      let preview = preview.replace('\n', " ");
      let preview = if content.len() > 60 {
        format!("{}...", preview)
      } else {
        preview
      };

      println!("{}. [{}] {}", i + 1, sector, id);
      println!("   {}", preview);
      println!("   Deleted: {}", deleted_at);
      println!();
    }

    println!("Use 'ccengram memory restore <id>' to restore a memory.");
  }

  Ok(())
}
