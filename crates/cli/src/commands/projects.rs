//! Project management commands (list, show, clean)

use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::connect_or_start;
use ipc::{Method, ProjectCleanParams, ProjectInfoParams, ProjectsCleanAllParams, ProjectsListParams, Request};
use std::io::Write;
use tracing::error;

/// List all indexed projects
pub async fn cmd_projects_list(json_output: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let request = Request {
    id: Some(1),
    method: Method::ProjectsList,
    params: ProjectsListParams,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to list projects")?;

  if let Some(err) = response.error {
    error!("Error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(projects) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&projects)?);
      return Ok(());
    }

    let empty_vec = vec![];
    let projects = projects.as_array().unwrap_or(&empty_vec);

    if projects.is_empty() {
      println!("No projects indexed.");
      return Ok(());
    }

    println!("Indexed Projects ({})", projects.len());
    println!("==================\n");

    for project in projects {
      let id = project.get("id").and_then(|v| v.as_str()).unwrap_or("?");
      let path = project.get("path").and_then(|v| v.as_str()).unwrap_or("?");
      let name = project.get("name").and_then(|v| v.as_str()).unwrap_or("?");

      // Truncate ID for display
      let short_id = if id.len() > 8 { &id[..8] } else { id };

      println!("{} [{}]", name, short_id);
      println!("  Path: {}", path);

      if let Some(memory_count) = project.get("memory_count").and_then(|v| v.as_u64()) {
        print!("  Memories: {}", memory_count);
      }
      if let Some(code_count) = project.get("code_chunk_count").and_then(|v| v.as_u64()) {
        print!("  | Code chunks: {}", code_count);
      }
      if let Some(doc_count) = project.get("document_count").and_then(|v| v.as_u64()) {
        print!("  | Documents: {}", doc_count);
      }
      println!();

      if let Some(last_active) = project.get("last_active").and_then(|v| v.as_str()) {
        println!("  Last active: {}", last_active);
      }

      println!();
    }
  }

  Ok(())
}

/// Show details for a specific project
pub async fn cmd_projects_show(project: &str, json_output: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let params = ProjectInfoParams {
    cwd: None,
    path: Some(project.to_string()),
  };

  let request = Request {
    id: Some(1),
    method: Method::ProjectInfo,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to get project info")?;

  if let Some(err) = response.error {
    error!("Error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(info) = response.result {
    if json_output {
      println!("{}", serde_json::to_string_pretty(&info)?);
      return Ok(());
    }

    println!("Project Details");
    println!("===============\n");

    if let Some(id) = info.get("id").and_then(|v| v.as_str()) {
      println!("ID:           {}", id);
    }
    if let Some(path) = info.get("path").and_then(|v| v.as_str()) {
      println!("Path:         {}", path);
    }
    if let Some(name) = info.get("name").and_then(|v| v.as_str()) {
      println!("Name:         {}", name);
    }
    if let Some(created) = info.get("created_at").and_then(|v| v.as_str()) {
      println!("Created:      {}", created);
    }
    if let Some(last_active) = info.get("last_active").and_then(|v| v.as_str()) {
      println!("Last Active:  {}", last_active);
    }

    println!();
    println!("Statistics:");

    if let Some(count) = info.get("memory_count").and_then(|v| v.as_u64()) {
      println!("  Memories:     {}", count);
    }
    if let Some(count) = info.get("code_chunk_count").and_then(|v| v.as_u64()) {
      println!("  Code Chunks:  {}", count);
    }
    if let Some(count) = info.get("document_count").and_then(|v| v.as_u64()) {
      println!("  Documents:    {}", count);
    }
    if let Some(count) = info.get("entity_count").and_then(|v| v.as_u64()) {
      println!("  Entities:     {}", count);
    }
    if let Some(count) = info.get("session_count").and_then(|v| v.as_u64()) {
      println!("  Sessions:     {}", count);
    }

    if let Some(db_path) = info.get("db_path").and_then(|v| v.as_str()) {
      println!();
      println!("Database Path: {}", db_path);
    }
  } else {
    println!("Project not found: {}", project);
  }

  Ok(())
}

/// Remove a project's data
pub async fn cmd_projects_clean(project: &str, force: bool) -> Result<()> {
  if !force {
    print!("Remove all data for project '{}'? [y/N] ", project);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    if !input.trim().eq_ignore_ascii_case("y") {
      println!("Cancelled.");
      return Ok(());
    }
  }

  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let params = ProjectCleanParams {
    path: project.to_string(),
  };

  let request = Request {
    id: Some(1),
    method: Method::ProjectClean,
    params,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to clean project")?;

  if let Some(err) = response.error {
    error!("Error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
    let path = result.get("path").and_then(|v| v.as_str()).unwrap_or("?");
    println!("Removed project: {}", path);

    if let Some(memories) = result.get("memories_deleted").and_then(|v| v.as_u64()) {
      println!("  Memories deleted: {}", memories);
    }
    if let Some(chunks) = result.get("code_chunks_deleted").and_then(|v| v.as_u64()) {
      println!("  Code chunks deleted: {}", chunks);
    }
    if let Some(docs) = result.get("documents_deleted").and_then(|v| v.as_u64()) {
      println!("  Documents deleted: {}", docs);
    }
  }

  Ok(())
}

/// Remove all project data
pub async fn cmd_projects_clean_all(force: bool) -> Result<()> {
  if !force {
    print!("Remove ALL project data? This cannot be undone! [y/N] ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    if !input.trim().eq_ignore_ascii_case("y") {
      println!("Cancelled.");
      return Ok(());
    }
  }

  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let request = Request {
    id: Some(1),
    method: Method::ProjectsCleanAll,
    params: ProjectsCleanAllParams,
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to clean all projects")?;

  if let Some(err) = response.error {
    error!("Error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
    let count = result.get("projects_removed").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("Removed {} projects", count);
  }

  Ok(())
}
