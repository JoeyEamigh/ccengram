//! Administrative commands (stats, health, archive, config)

use anyhow::{Context, Result};
use ccengram::ipc::{
  memory::{MemoryDeleteParams, MemoryListParams},
  system::{HealthCheckParams, MetricsParams, PingParams, ProjectStatsParams, StatusParams},
};
use tracing::error;

/// Show statistics
pub async fn cmd_stats() -> Result<()> {
  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let client = ccengram::Daemon::connect_or_start(cwd)
    .await
    .context("Failed to connect to daemon")?;

  // Get daemon metrics (includes status info plus more)
  let metrics = client.call(MetricsParams).await.context("Failed to get metrics")?;

  println!("CCEngram Statistics");
  println!("===================\n");

  // Print daemon metrics
  println!("--- Daemon ---");
  println!("Version:        {}", metrics.daemon.version);
  println!(
    "Mode:           {}",
    if metrics.daemon.foreground {
      "foreground"
    } else {
      "background"
    }
  );
  println!("Uptime:         {}", format_duration(metrics.daemon.uptime_seconds));
  println!("Idle:           {}", format_duration(metrics.daemon.idle_seconds));

  // Requests
  println!(
    "Requests:       {} total ({:.2}/s)",
    metrics.requests.total, metrics.requests.per_second
  );

  // Sessions
  if metrics.sessions.active > 0 {
    println!("Sessions:       {} active", metrics.sessions.active);
    for id in metrics.sessions.ids.iter().take(5) {
      println!("                - {}", id);
    }
    if metrics.sessions.ids.len() > 5 {
      println!("                ... and {} more", metrics.sessions.ids.len() - 5);
    }
  } else {
    println!("Sessions:       none active");
  }

  // Projects
  println!("Projects:       {} loaded", metrics.projects.count);
  for name in metrics.projects.names.iter().take(5) {
    println!("                - {}", name);
  }
  if metrics.projects.names.len() > 5 {
    println!("                ... and {} more", metrics.projects.names.len() - 5);
  }

  // Embedding provider
  if let Some(ref embedding) = metrics.embedding {
    println!("\n--- Embedding Provider ---");
    println!("Provider:       {}", embedding.name);
    println!("Model:          {}", embedding.model);
    println!("Dimensions:     {}", embedding.dimensions);
  }

  // Memory usage
  println!("\n--- Memory ---");
  if let Some(rss_kb) = metrics.memory.rss_kb {
    println!("RSS:            {}", format_memory(rss_kb));
  } else {
    println!("RSS:            (unavailable)");
  }

  // Get project-specific stats for current directory
  let stats = client
    .call(ProjectStatsParams)
    .await
    .context("Failed to get project stats")?;

  println!("\n--- Project Statistics ---");
  println!("Project ID:     {}", stats.project_id);
  println!("Path:           {}", stats.path);
  println!("Total memories: {}", stats.memories);
  println!("Code chunks:    {}", stats.code_chunks);
  println!("Documents:      {}", stats.documents);
  println!("Sessions:       {}", stats.sessions);

  Ok(())
}

/// Health check
pub async fn cmd_health() -> Result<()> {
  let socket_path = ccengram::dirs::default_socket_path();
  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

  println!("CCEngram Health Check");
  println!("=====================\n");

  // Try to connect (auto-starting if needed)
  let client = match ccengram::Daemon::connect_or_start(cwd).await {
    Ok(c) => c,
    Err(e) => {
      println!("Daemon:     NOT RUNNING");
      println!("Socket:     {:?}", socket_path);
      println!("Error:      {}", e);
      println!("\nFailed to auto-start daemon. Check logs for details.");
      std::process::exit(1);
    }
  };

  // Ping test
  let ping_ok = client.call(PingParams).await.is_ok();

  if ping_ok {
    println!("Daemon:     HEALTHY");
  } else {
    println!("Daemon:     UNHEALTHY (ping failed)");
    std::process::exit(1);
  }
  println!("Socket:     {:?}", socket_path);

  // Get daemon status info
  if let Ok(status) = client.call(StatusParams).await {
    println!("\n--- Daemon Status ---");
    println!("Version:    {}", status.version);
    println!("Sessions:   {} active", status.active_sessions);
    println!("Idle:       {}", format_duration(status.idle_seconds));
    println!("Uptime:     {}", format_duration(status.uptime_seconds));
    println!(
      "Auto-shutdown: {}",
      if status.auto_shutdown {
        "enabled"
      } else {
        "disabled (foreground mode)"
      }
    );
  }

  // Get comprehensive health status
  if let Ok(health) = client.call(HealthCheckParams).await {
    println!(
      "\nOverall Health: {}",
      if health.healthy { "HEALTHY" } else { "UNHEALTHY" }
    );

    for check in &health.checks {
      println!("\n--- {} ---", check.name);
      println!("Status:     {}", check.status.to_uppercase());
      if let Some(ref msg) = check.message {
        println!("Message:    {}", msg);
      }
    }
  }

  Ok(())
}

/// Archive old low-salience memories
pub async fn cmd_archive(before: Option<&str>, threshold: f32, dry_run: bool) -> Result<()> {
  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let client = ccengram::Daemon::connect_or_start(cwd)
    .await
    .context("Failed to connect to daemon")?;

  // First, get all memories to find archival candidates
  let memories = client
    .call(MemoryListParams::default())
    .await
    .context("Failed to list memories")?;

  // Parse the before date if provided
  let before_date: Option<chrono::NaiveDateTime> = before.and_then(|s| {
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
      .ok()
      .and_then(|d| d.and_hms_opt(0, 0, 0))
  });

  let mut candidates: Vec<(String, f32, String)> = Vec::new();

  for mem in &memories {
    let salience = mem.salience;

    // Skip if above threshold
    if salience >= threshold {
      continue;
    }

    // Check date if specified
    if let Some(cutoff) = before_date
      && let Ok(mem_date) = chrono::DateTime::parse_from_rfc3339(&mem.created_at)
    {
      let mem_naive: chrono::NaiveDateTime = mem_date.naive_utc();
      if mem_naive >= cutoff {
        continue;
      }
    }

    let summary = if mem.content.len() > 60 {
      format!("{}...", &mem.content[..60])
    } else {
      mem.content.clone()
    };

    candidates.push((mem.id.clone(), salience, summary));
  }

  if candidates.is_empty() {
    println!(
      "No memories match archival criteria (salience < {}{})",
      threshold,
      before.map(|d| format!(", before {}", d)).unwrap_or_default()
    );
    return Ok(());
  }

  println!("Found {} memories to archive:", candidates.len());
  println!();

  for (id, salience, summary) in &candidates {
    let short_id = if id.len() > 8 { &id[..8] } else { id };
    println!("  [{:.2}] {}... - {}", salience, short_id, summary.replace('\n', " "));
  }
  println!();
  println!("Note: You can use the ID prefix (8+ characters) to reference memories.");
  println!();

  if dry_run {
    println!("Dry run - no changes made");
    return Ok(());
  }

  // Archive (soft delete) each memory
  let mut archived = 0;
  for (id, _, _) in candidates {
    match client.call(MemoryDeleteParams { memory_id: id.clone() }).await {
      Ok(_) => archived += 1,
      Err(e) => error!("Failed to archive memory {}: {}", id, e),
    }
  }

  println!("Archived {} memories (soft deleted)", archived);
  Ok(())
}

/// Show current effective configuration
pub async fn cmd_config_show() -> Result<()> {
  use ccengram::config::Config;

  let cwd = std::env::current_dir()?;
  let config = Config::load_for_project(&cwd).await;

  // Check which config file is being used
  let project_config = Config::project_config_path(&cwd);
  let user_config = Config::user_config_path();

  println!("Effective configuration for: {:?}", cwd);
  println!();

  if project_config.exists() {
    println!("Using project config: {:?}", project_config);
  } else if let Some(ref user_path) = user_config {
    if user_path.exists() {
      println!("Using user config: {:?}", user_path);
    } else {
      println!("Using default configuration (no config file found)");
    }
  } else {
    println!("Using default configuration");
  }
  println!();

  // Show config as TOML
  let toml_str = toml::to_string_pretty(&config)?;
  println!("{}", toml_str);

  Ok(())
}

/// Initialize project configuration file
pub async fn cmd_config_init(preset: &str) -> Result<()> {
  use ccengram::config::{Config, ToolPreset};

  let cwd = std::env::current_dir()?;
  let config_path = Config::project_config_path(&cwd);

  if config_path.exists() {
    error!("Config file already exists: {:?}", config_path);
    println!("Delete it first if you want to regenerate");
    std::process::exit(1);
  }

  // Parse preset
  let tool_preset = match preset.to_lowercase().as_str() {
    "minimal" => ToolPreset::Minimal,
    "standard" => ToolPreset::Standard,
    "full" => ToolPreset::Full,
    _ => {
      error!("Invalid preset: {}. Use minimal, standard, or full", preset);
      std::process::exit(1);
    }
  };

  // Create .claude directory if needed
  if let Some(parent) = config_path.parent() {
    std::fs::create_dir_all(parent)?;
  }

  // Generate and write config (project template excludes daemon-level sections)
  let template = Config::generate_project_template(tool_preset);
  std::fs::write(&config_path, &template)?;

  println!("Created project config: {:?}", config_path);
  println!();
  println!("Note: Daemon-level settings (embedding, auto-shutdown) should be");
  println!("configured in ~/.config/ccengram/config.toml instead.");
  println!();
  println!("Tool preset: {}", preset);
  println!("Edit the file to customize settings.");

  Ok(())
}

/// Reset user configuration to defaults
pub async fn cmd_config_reset() -> Result<()> {
  use ccengram::config::{Config, ToolPreset};

  if let Some(user_config_path) = Config::user_config_path() {
    if let Some(parent) = user_config_path.parent() {
      std::fs::create_dir_all(parent)?;
    }
    let template = Config::generate_template(ToolPreset::Standard);
    std::fs::write(&user_config_path, &template)?;
    println!("Reset user config to defaults: {:?}", user_config_path);
  } else {
    error!("Could not determine user config path");
    std::process::exit(1);
  }

  Ok(())
}

/// Format duration in human-readable form
fn format_duration(seconds: u64) -> String {
  if seconds < 60 {
    format!("{} seconds", seconds)
  } else if seconds < 3600 {
    let mins = seconds / 60;
    let secs = seconds % 60;
    if secs > 0 {
      format!("{} min {} sec", mins, secs)
    } else {
      format!("{} minutes", mins)
    }
  } else {
    let hours = seconds / 3600;
    let mins = (seconds % 3600) / 60;
    if mins > 0 {
      format!("{} hr {} min", hours, mins)
    } else {
      format!("{} hours", hours)
    }
  }
}

/// Format memory size in human-readable form
fn format_memory(kb: u64) -> String {
  if kb < 1024 {
    format!("{} KB", kb)
  } else if kb < 1024 * 1024 {
    format!("{:.1} MB", kb as f64 / 1024.0)
  } else {
    format!("{:.2} GB", kb as f64 / (1024.0 * 1024.0))
  }
}
