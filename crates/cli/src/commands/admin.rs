//! Administrative commands (stats, health, archive, config)

use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::{connect_or_start, default_socket_path};
use ipc::{
  HealthCheckParams, MemoryDeleteParams, MemoryListParams, Method, MetricsParams, PingParams, ProjectStatsParams,
  Request, StatusParams,
};
use tracing::error;

/// Show statistics
pub async fn cmd_stats() -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  // Get daemon metrics (includes status info plus more)
  let request = Request {
    id: Some(1),
    method: Method::Metrics,
    params: MetricsParams,
  };

  let metrics_response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to get metrics")?;

  println!("CCEngram Statistics");
  println!("===================\n");

  // Print daemon metrics
  if let Some(metrics) = metrics_response.result {
    println!("--- Daemon ---");
    if let Some(daemon) = metrics.get("daemon") {
      if let Some(version) = daemon.get("version").and_then(|v| v.as_str()) {
        println!("Version:        {}", version);
      }
      let foreground = daemon.get("foreground").and_then(|v| v.as_bool()).unwrap_or(false);
      println!(
        "Mode:           {}",
        if foreground { "foreground" } else { "background" }
      );
      if let Some(uptime_secs) = daemon.get("uptime_seconds").and_then(|v| v.as_u64()) {
        println!("Uptime:         {}", format_duration(uptime_secs));
      }
      if let Some(idle_secs) = daemon.get("idle_seconds").and_then(|v| v.as_u64()) {
        println!("Idle:           {}", format_duration(idle_secs));
      }
    }

    // Requests
    if let Some(requests) = metrics.get("requests") {
      let total = requests.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
      let per_sec = requests.get("per_second").and_then(|v| v.as_f64()).unwrap_or(0.0);
      println!("Requests:       {} total ({:.2}/s)", total, per_sec);
    }

    // Sessions
    if let Some(sessions) = metrics.get("sessions") {
      let active = sessions.get("active").and_then(|v| v.as_u64()).unwrap_or(0);
      if active > 0 {
        println!("Sessions:       {} active", active);
        if let Some(ids) = sessions.get("ids").and_then(|v| v.as_array()) {
          for id in ids.iter().take(5) {
            if let Some(s) = id.as_str() {
              println!("                - {}", s);
            }
          }
          if ids.len() > 5 {
            println!("                ... and {} more", ids.len() - 5);
          }
        }
      } else {
        println!("Sessions:       none active");
      }
    }

    // Projects
    if let Some(projects) = metrics.get("projects") {
      let count = projects.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Projects:       {} loaded", count);
      if let Some(names) = projects.get("names").and_then(|v| v.as_array()) {
        for name in names.iter().take(5) {
          if let Some(s) = name.as_str() {
            println!("                - {}", s);
          }
        }
        if names.len() > 5 {
          println!("                ... and {} more", names.len() - 5);
        }
      }
    }

    // Embedding provider
    if let Some(embedding) = metrics.get("embedding")
      && !embedding.is_null()
    {
      println!("\n--- Embedding Provider ---");
      if let Some(name) = embedding.get("name").and_then(|v| v.as_str()) {
        println!("Provider:       {}", name);
      }
      if let Some(model) = embedding.get("model").and_then(|v| v.as_str()) {
        println!("Model:          {}", model);
      }
      if let Some(dims) = embedding.get("dimensions").and_then(|v| v.as_u64()) {
        println!("Dimensions:     {}", dims);
      }
    }

    // Memory usage
    if let Some(memory) = metrics.get("memory")
      && let Some(rss_kb) = memory.get("rss_kb").and_then(|v| v.as_u64())
    {
      println!("\n--- Memory ---");
      println!("RSS:            {}", format_memory(rss_kb));
    }
  }

  // Get project-specific stats for current directory
  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let stats_request = Request {
    id: Some(2),
    method: Method::ProjectStats,
    params: ProjectStatsParams { cwd: Some(cwd) },
  };

  let stats_response = client
    .request(to_daemon_request(stats_request))
    .await
    .context("Failed to get project stats")?;

  if let Some(err) = stats_response.error {
    error!("Stats error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(stats) = stats_response.result {
    println!("\n--- Memory Statistics ---");

    if let Some(memories) = stats.get("memories") {
      let total = memories.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Total memories: {}", total);

      // By sector
      if let Some(by_sector) = memories.get("by_sector").and_then(|v| v.as_object()) {
        println!("\nBy sector:");
        for (sector, count) in by_sector {
          let cnt = count.as_u64().unwrap_or(0);
          if cnt > 0 {
            println!("  {}: {}", sector, cnt);
          }
        }
      }

      // By tier
      if let Some(by_tier) = memories.get("by_tier").and_then(|v| v.as_object()) {
        println!("\nBy tier:");
        for (tier, count) in by_tier {
          let cnt = count.as_u64().unwrap_or(0);
          if cnt > 0 {
            println!("  {}: {}", tier, cnt);
          }
        }
      }

      // Salience distribution
      if let Some(by_salience) = memories.get("by_salience") {
        println!("\nSalience distribution:");
        let high = by_salience.get("high").and_then(|v| v.as_u64()).unwrap_or(0);
        let medium = by_salience.get("medium").and_then(|v| v.as_u64()).unwrap_or(0);
        let low = by_salience.get("low").and_then(|v| v.as_u64()).unwrap_or(0);
        let very_low = by_salience.get("very_low").and_then(|v| v.as_u64()).unwrap_or(0);
        println!("  high (>=0.7): {}", high);
        println!("  medium (0.4-0.7): {}", medium);
        println!("  low (0.2-0.4): {}", low);
        println!("  very_low (<0.2): {}", very_low);
      }

      let superseded = memories.get("superseded_count").and_then(|v| v.as_u64()).unwrap_or(0);
      if superseded > 0 {
        println!("\nSuperseded memories: {}", superseded);
      }
    }

    println!("\n--- Code Index Statistics ---");

    if let Some(code) = stats.get("code") {
      let total_chunks = code.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
      let total_files = code.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Total chunks: {}", total_chunks);
      println!("Total files: {}", total_files);

      // By language
      if let Some(by_language) = code.get("by_language").and_then(|v| v.as_object())
        && !by_language.is_empty()
      {
        println!("\nBy language:");
        let mut langs: Vec<_> = by_language.iter().collect();
        langs.sort_by(|a, b| b.1.as_u64().unwrap_or(0).cmp(&a.1.as_u64().unwrap_or(0)));
        for (lang, count) in langs.iter().take(10) {
          let cnt = count.as_u64().unwrap_or(0);
          if cnt > 0 {
            println!("  {}: {}", lang, cnt);
          }
        }
      }

      // Recent activity
      if let Some(recent) = code.get("recent_indexed").and_then(|v| v.as_array())
        && !recent.is_empty()
      {
        println!("\nRecent index activity:");
        for item in recent.iter().take(5) {
          let file_path = item.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
          let chunks = item.get("chunks").and_then(|v| v.as_u64()).unwrap_or(0);
          println!("  {} ({} chunks)", file_path, chunks);
        }
      }
    }

    println!("\n--- Document Statistics ---");

    if let Some(docs) = stats.get("documents") {
      let total = docs.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
      let total_chunks = docs.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Total documents: {}", total);
      println!("Total document chunks: {}", total_chunks);
    }

    println!("\n--- Entity Statistics ---");

    if let Some(entities) = stats.get("entities") {
      let total = entities.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
      println!("Total entities: {}", total);

      if let Some(by_type) = entities.get("by_type").and_then(|v| v.as_object())
        && !by_type.is_empty()
      {
        println!("\nBy type:");
        for (entity_type, count) in by_type {
          let cnt = count.as_u64().unwrap_or(0);
          if cnt > 0 {
            println!("  {}: {}", entity_type, cnt);
          }
        }
      }
    }
  }

  Ok(())
}

/// Health check
pub async fn cmd_health() -> Result<()> {
  let socket_path = default_socket_path();

  println!("CCEngram Health Check");
  println!("=====================\n");

  // Try to connect (auto-starting if needed)
  let mut client = match connect_or_start().await {
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
  let ping_request = Request {
    id: Some(1),
    method: Method::Ping,
    params: PingParams,
  };

  let ping_ok = match client.request(to_daemon_request(ping_request)).await {
    Ok(response) => response.error.is_none(),
    Err(_) => false,
  };

  if ping_ok {
    println!("Daemon:     HEALTHY");
  } else {
    println!("Daemon:     UNHEALTHY (ping failed)");
    std::process::exit(1);
  }
  println!("Socket:     {:?}", socket_path);

  // Get daemon status info
  let status_request = Request {
    id: Some(1),
    method: Method::Status,
    params: StatusParams::default(),
  };

  if let Ok(status_response) = client.request(to_daemon_request(status_request)).await
    && let Some(status) = status_response.result
  {
    println!("\n--- Daemon Status ---");
    if let Some(version) = status.get("version").and_then(|v| v.as_str()) {
      println!("Version:    {}", version);
    }
    if let Some(sessions) = status.get("active_sessions").and_then(|v| v.as_u64()) {
      println!("Sessions:   {} active", sessions);
    }
    if let Some(idle_secs) = status.get("idle_seconds").and_then(|v| v.as_u64()) {
      if idle_secs < 60 {
        println!("Idle:       {} seconds", idle_secs);
      } else if idle_secs < 3600 {
        println!("Idle:       {} minutes", idle_secs / 60);
      } else {
        println!("Idle:       {} hours", idle_secs / 3600);
      }
    }
    if let Some(uptime_secs) = status.get("uptime_seconds").and_then(|v| v.as_u64()) {
      if uptime_secs < 60 {
        println!("Uptime:     {} seconds", uptime_secs);
      } else if uptime_secs < 3600 {
        println!("Uptime:     {} minutes", uptime_secs / 60);
      } else {
        println!("Uptime:     {} hours", uptime_secs / 3600);
      }
    }
    let auto_shutdown = status.get("auto_shutdown").and_then(|v| v.as_bool()).unwrap_or(true);
    println!(
      "Auto-shutdown: {}",
      if auto_shutdown {
        "enabled"
      } else {
        "disabled (foreground mode)"
      }
    );
  }

  // Get comprehensive health status
  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let health_request = Request {
    id: Some(2),
    method: Method::HealthCheck,
    params: HealthCheckParams,
  };

  if let Ok(response) = client.request(to_daemon_request(health_request)).await
    && let Some(health) = response.result
  {
    // Database status
    println!("\n--- Database ---");
    if let Some(db) = health.get("database") {
      let status = db.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
      if status == "healthy" {
        println!("Status:     HEALTHY");
      } else {
        let error = db.get("error").and_then(|v| v.as_str()).unwrap_or("unknown error");
        println!("Status:     ERROR ({})", error);
      }
    }

    // Ollama status
    println!("\n--- Ollama ---");
    if let Some(ollama) = health.get("ollama") {
      let available = ollama.get("available").and_then(|v| v.as_bool()).unwrap_or(false);
      if available {
        println!("Status:     AVAILABLE");
        let models_count = ollama.get("models_count").and_then(|v| v.as_u64()).unwrap_or(0);
        println!("Models:     {} available", models_count);
        let configured = ollama
          .get("configured_model")
          .and_then(|v| v.as_str())
          .unwrap_or("unknown");
        let model_available = ollama
          .get("configured_model_available")
          .and_then(|v| v.as_bool())
          .unwrap_or(false);
        println!(
          "Configured: {} ({})",
          configured,
          if model_available { "available" } else { "NOT FOUND" }
        );
      } else {
        println!("Status:     NOT AVAILABLE");
        println!("            Make sure Ollama is running: ollama serve");
      }
    }

    // Embedding status
    println!("\n--- Embedding Service ---");
    if let Some(embed) = health.get("embedding") {
      let configured = embed.get("configured").and_then(|v| v.as_bool()).unwrap_or(false);
      if configured {
        let provider = embed.get("provider").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = embed.get("model").and_then(|v| v.as_str()).unwrap_or("unknown");
        let dimensions = embed.get("dimensions").and_then(|v| v.as_u64()).unwrap_or(0);
        let available = embed.get("available").and_then(|v| v.as_bool()).unwrap_or(false);

        println!("Provider:   {}", provider);
        println!("Model:      {}", model);
        println!("Dimensions: {}", dimensions);
        println!("Status:     {}", if available { "AVAILABLE" } else { "NOT AVAILABLE" });
      } else {
        println!("Status:     NOT CONFIGURED");
      }
    }
  }

  // Use cwd for health check context
  let _ = cwd;

  Ok(())
}

/// Archive old low-salience memories
pub async fn cmd_archive(before: Option<&str>, threshold: f32, dry_run: bool) -> Result<()> {
  let mut client = connect_or_start().await.context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // First, get all memories to find archival candidates
  let request = Request {
    id: Some(1),
    method: Method::MemoryList,
    params: MemoryListParams {
      cwd: Some(cwd.clone()),
      ..Default::default()
    },
  };

  let response = client
    .request(to_daemon_request(request))
    .await
    .context("Failed to list memories")?;

  if let Some(err) = response.error {
    error!("Archive error: {}", err.message);
    std::process::exit(1);
  }

  let Some(memories) = response.result else {
    println!("No memories found");
    return Ok(());
  };

  let Some(arr) = memories.as_array() else {
    println!("No memories found");
    return Ok(());
  };

  // Parse the before date if provided
  let before_date: Option<chrono::NaiveDateTime> = before.and_then(|s| {
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
      .ok()
      .and_then(|d| d.and_hms_opt(0, 0, 0))
  });

  let mut candidates: Vec<(String, f32, String)> = Vec::new();

  for mem in arr {
    let salience = mem.get("salience").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    // Skip if above threshold
    if salience >= threshold {
      continue;
    }

    // Check date if specified
    if let Some(cutoff) = before_date
      && let Some(created) = mem.get("created_at").and_then(|v| v.as_str())
      && let Ok(mem_date) = chrono::DateTime::parse_from_rfc3339(created)
    {
      let mem_naive: chrono::NaiveDateTime = mem_date.naive_utc();
      if mem_naive >= cutoff {
        continue;
      }
    }

    let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let content = mem.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let summary = if content.len() > 60 {
      format!("{}...", &content[..60])
    } else {
      content
    };

    candidates.push((id, salience, summary));
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
    let request = Request {
      id: Some(1),
      method: Method::MemoryDelete,
      params: MemoryDeleteParams {
        memory_id: id.clone(),
        cwd: Some(cwd.clone()),
      },
    };

    match client.request(to_daemon_request(request)).await {
      Ok(resp) if resp.error.is_none() => archived += 1,
      _ => error!("Failed to archive memory {}", id),
    }
  }

  println!("Archived {} memories (soft deleted)", archived);
  Ok(())
}

/// Show current effective configuration
pub async fn cmd_config_show() -> Result<()> {
  use engram_core::Config;

  let cwd = std::env::current_dir()?;
  let config = Config::load_for_project(&cwd);

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
  use engram_core::{Config, ToolPreset};

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

  // Generate and write config
  let template = Config::generate_template(tool_preset);
  std::fs::write(&config_path, &template)?;

  println!("Created project config: {:?}", config_path);
  println!();
  println!("Tool preset: {}", preset);
  println!("Edit the file to customize settings.");

  Ok(())
}

/// Reset user configuration to defaults
pub async fn cmd_config_reset() -> Result<()> {
  use engram_core::{Config, ToolPreset};

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
