use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use daemon::{
  Client, Daemon, DaemonConfig, HookEvent, HookHandler, ProjectRegistry, Request, default_socket_path, is_running,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tracing::{error, info};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser)]
#[command(name = "ccengram")]
#[command(about = "Intelligent memory and code search for Claude Code")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Start the daemon
  Daemon {
    #[arg(long)]
    foreground: bool,
  },
  /// MCP server (for Claude Code integration)
  Mcp,
  /// Handle hook event
  Hook { name: String },
  /// Search memories
  Search {
    query: String,
    #[arg(short, long, default_value = "10")]
    limit: usize,
    /// Project path (default: current directory)
    #[arg(short, long)]
    project: Option<String>,
    /// Filter by sector (episodic, semantic, procedural, emotional, reflective)
    #[arg(long)]
    sector: Option<String>,
    /// Filter by memory type (preference, codebase, decision, gotcha, pattern)
    #[arg(long, name = "type")]
    memory_type: Option<String>,
    /// Minimum salience threshold (0.0-1.0)
    #[arg(long)]
    min_salience: Option<f32>,
    /// Include superseded memories
    #[arg(long)]
    include_superseded: bool,
    /// Filter by scope path prefix
    #[arg(long)]
    scope: Option<String>,
    /// Use semantic (vector) search only (default, this flag is for compatibility)
    #[arg(long)]
    semantic: bool,
    /// Use keyword search only (not supported - emits warning)
    #[arg(long)]
    keywords: bool,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Search code (semantic search over indexed code)
  SearchCode {
    query: String,
    #[arg(short, long, default_value = "10")]
    limit: usize,
    /// Project path (default: current directory)
    #[arg(short, long)]
    project: Option<String>,
    /// Filter by programming language (rust, python, typescript, etc.)
    #[arg(long)]
    language: Option<String>,
    /// Filter by chunk type (function, class, module, block, import)
    #[arg(long, name = "type")]
    chunk_type: Option<String>,
    /// Filter by file path prefix
    #[arg(long)]
    path: Option<String>,
    /// Filter by symbol name
    #[arg(long)]
    symbol: Option<String>,
    /// Use semantic (vector) search only (default, this flag is for compatibility)
    #[arg(long)]
    semantic: bool,
    /// Use keyword search only (not supported - emits warning)
    #[arg(long)]
    keywords: bool,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Search documents (semantic search over ingested docs)
  SearchDocs {
    query: String,
    #[arg(short, long, default_value = "10")]
    limit: usize,
    /// Project path (default: current directory)
    #[arg(short, long)]
    project: Option<String>,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Show detailed memory by ID
  Show {
    /// Memory ID to show
    memory_id: String,
    /// Include related memories
    #[arg(long)]
    related: bool,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Delete a memory
  Delete {
    /// Memory ID to delete
    memory_id: String,
    /// Permanently delete (hard delete)
    #[arg(long)]
    hard: bool,
  },
  /// Export memories to file
  Export {
    /// Output file path
    #[arg(short, long)]
    output: Option<String>,
    /// Output format (json or csv)
    #[arg(short, long, default_value = "json")]
    format: String,
  },
  /// Import a document for searchable reference
  Import {
    /// File path to import
    path: String,
    /// Document title (optional)
    #[arg(short, long)]
    title: Option<String>,
  },
  /// Manage code index
  Index {
    /// Force re-index all files
    #[arg(long)]
    force: bool,
    /// Show index statistics
    #[arg(long)]
    stats: bool,
    /// Export index to file
    #[arg(long, value_name = "FILE")]
    export: Option<String>,
    /// Import index from file
    #[arg(long, value_name = "FILE")]
    import: Option<String>,
  },
  /// Watch for file changes and update index
  Watch {
    /// Stop any running watcher
    #[arg(long)]
    stop: bool,
    /// Check watcher status
    #[arg(long)]
    status: bool,
  },
  /// Archive old low-salience memories
  Archive {
    /// Archive memories older than this date (YYYY-MM-DD)
    #[arg(long)]
    before: Option<String>,
    /// Minimum salience threshold (archive below this)
    #[arg(long, default_value = "0.1")]
    threshold: f32,
    /// Preview what would be archived without making changes
    #[arg(long)]
    dry_run: bool,
  },
  /// Show statistics
  Stats,
  /// Health check
  Health,
  /// Show or set configuration
  Config {
    /// Configuration key to get/set
    key: Option<String>,
    /// Value to set (if provided with key)
    value: Option<String>,
    /// Reset all configuration to defaults
    #[arg(long)]
    reset: bool,
    /// List all configuration keys
    #[arg(long)]
    list: bool,
    /// Initialize project config file (.claude/ccengram.toml)
    #[arg(long)]
    init: bool,
    /// Tool preset for init: minimal, standard (default), or full
    #[arg(long, default_value = "standard")]
    preset: String,
    /// Show current effective config
    #[arg(long)]
    show: bool,
  },
  /// Check for updates or update to latest version
  Update {
    /// Only check for updates without installing
    #[arg(long)]
    check: bool,
    /// Specific version to update to
    #[arg(long)]
    version: Option<String>,
  },
  /// Migrate embeddings to new dimensions/model
  Migrate {
    /// Preview what would be migrated without making changes
    #[arg(long)]
    dry_run: bool,
    /// Force re-embed even if dimensions match
    #[arg(long)]
    force: bool,
  },
  /// Generate a MemExplore subagent for Claude Code
  Agent {
    /// Output path (default: .claude/agents/MemExplore.md)
    #[arg(long)]
    output: Option<String>,
    /// Overwrite existing file
    #[arg(long)]
    force: bool,
  },
  /// Launch interactive TUI
  Tui {
    /// Project path (default: current directory)
    #[arg(short, long)]
    project: Option<PathBuf>,
  },
}

/// Get the CCEngram data directory (respects env vars)
fn data_dir() -> PathBuf {
  daemon::default_data_dir()
}

/// Get the log file path
#[allow(dead_code)]
fn log_file_path() -> PathBuf {
  data_dir().join("ccengram.log")
}

/// Initialize logging for CLI commands (console only)
fn init_cli_logging() {
  tracing_subscriber::fmt()
    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
    .init();
}

/// Initialize logging for daemon with file appender
/// Returns the guard that must be kept alive for the duration of the program
fn init_daemon_logging() -> Option<WorkerGuard> {
  let log_dir = data_dir();
  if std::fs::create_dir_all(&log_dir).is_err() {
    // Fall back to console-only logging
    init_cli_logging();
    return None;
  }

  // Create a rolling file appender (daily rotation)
  let file_appender = tracing_appender::rolling::daily(&log_dir, "ccengram.log");
  let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

  // Create layers for both console and file
  let env_filter = tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into());

  let console_layer = tracing_subscriber::fmt::layer().with_target(true).with_ansi(true);

  let file_layer = tracing_subscriber::fmt::layer()
    .with_target(true)
    .with_ansi(false)
    .with_writer(file_writer);

  tracing_subscriber::registry()
    .with(env_filter)
    .with(console_layer)
    .with(file_layer)
    .init();

  Some(guard)
}

#[tokio::main]
async fn main() -> Result<()> {
  let cli = Cli::parse();

  // Use file logging for daemon, console-only for other commands
  let _guard = match &cli.command {
    Commands::Daemon { .. } => init_daemon_logging(),
    _ => {
      init_cli_logging();
      None
    }
  };

  match cli.command {
    Commands::Daemon { foreground: _ } => cmd_daemon().await,
    Commands::Mcp => cmd_mcp().await,
    Commands::Hook { name } => cmd_hook(&name).await,
    Commands::Search {
      query,
      limit,
      project,
      sector,
      memory_type,
      min_salience,
      include_superseded,
      scope,
      semantic: _,
      keywords,
      json,
    } => {
      if keywords {
        eprintln!("Warning: --keywords mode is not supported in this version. Using semantic search.");
      }
      cmd_search(
        &query,
        limit,
        project.as_deref(),
        sector.as_deref(),
        memory_type.as_deref(),
        min_salience,
        include_superseded,
        scope.as_deref(),
        json,
      )
      .await
    }
    Commands::SearchCode {
      query,
      limit,
      project,
      language,
      chunk_type,
      path,
      symbol,
      semantic: _,
      keywords,
      json,
    } => {
      if keywords {
        eprintln!("Warning: --keywords mode is not supported in this version. Using semantic search.");
      }
      cmd_search_code(
        &query,
        limit,
        project.as_deref(),
        language.as_deref(),
        chunk_type.as_deref(),
        path.as_deref(),
        symbol.as_deref(),
        json,
      )
      .await
    }
    Commands::SearchDocs {
      query,
      limit,
      project,
      json,
    } => cmd_search_docs(&query, limit, project.as_deref(), json).await,
    Commands::Show {
      memory_id,
      related,
      json,
    } => cmd_show(&memory_id, related, json).await,
    Commands::Delete { memory_id, hard } => cmd_delete(&memory_id, hard).await,
    Commands::Export { output, format } => cmd_export(output.as_deref(), &format).await,
    Commands::Import { path, title } => cmd_import(&path, title.as_deref()).await,
    Commands::Index {
      force,
      stats,
      export,
      import,
    } => cmd_index(force, stats, export.as_deref(), import.as_deref()).await,
    Commands::Watch { stop, status } => cmd_watch(stop, status).await,
    Commands::Archive {
      before,
      threshold,
      dry_run,
    } => cmd_archive(before.as_deref(), threshold, dry_run).await,
    Commands::Stats => cmd_stats().await,
    Commands::Health => cmd_health().await,
    Commands::Config {
      key,
      value,
      reset,
      list,
      init,
      preset,
      show,
    } => cmd_config(key, value, reset, list, init, &preset, show).await,
    Commands::Update { check, version } => cmd_update(check, version).await,
    Commands::Migrate { dry_run, force } => cmd_migrate(dry_run, force).await,
    Commands::Agent { output, force } => cmd_agent(output.as_deref(), force).await,
    Commands::Tui { project } => cmd_tui(project).await,
  }
}

/// Start the daemon
async fn cmd_daemon() -> Result<()> {
  let config = DaemonConfig::default();
  let mut daemon = Daemon::new(config);

  info!("Starting CCEngram daemon");
  daemon.run().await.context("Failed to run daemon")?;

  Ok(())
}

/// MCP stdio server - implements the Model Context Protocol for Claude Code
async fn cmd_mcp() -> Result<()> {
  use serde::{Deserialize, Serialize};

  #[derive(Debug, Deserialize)]
  struct McpRequest {
    #[serde(rename = "jsonrpc")]
    _jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
  }

  #[derive(Debug, Serialize)]
  struct McpResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
  }

  #[derive(Debug, Serialize)]
  struct McpError {
    code: i32,
    message: String,
  }

  fn mcp_success(id: Option<serde_json::Value>, result: serde_json::Value) -> McpResponse {
    McpResponse {
      jsonrpc: "2.0",
      id,
      result: Some(result),
      error: None,
    }
  }

  fn mcp_error(id: Option<serde_json::Value>, code: i32, message: &str) -> McpResponse {
    McpResponse {
      jsonrpc: "2.0",
      id,
      result: None,
      error: Some(McpError {
        code,
        message: message.to_string(),
      }),
    }
  }

  // Tool definitions are loaded from cli::tools and filtered based on config

  let socket_path = default_socket_path();

  // Use async IO for proper non-blocking behavior with MCP
  let stdin = tokio::io::stdin();
  let mut stdout = tokio::io::stdout();
  let reader = tokio::io::BufReader::new(stdin);
  let mut lines = reader.lines();

  // Process MCP requests
  while let Some(line) = lines.next_line().await.context("Failed to read line from stdin")? {
    if line.trim().is_empty() {
      continue;
    }

    let mcp_request: McpRequest = match serde_json::from_str(&line) {
      Ok(r) => r,
      Err(e) => {
        let response = mcp_error(None, -32700, &format!("Parse error: {}", e));
        let out = serde_json::to_string(&response)? + "\n";
        stdout.write_all(out.as_bytes()).await?;
        stdout.flush().await?;
        continue;
      }
    };

    let response = match mcp_request.method.as_str() {
      // MCP protocol methods
      "initialize" => mcp_success(
        mcp_request.id,
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "ccengram",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
      ),
      "notifications/initialized" => {
        // No response needed for notification
        continue;
      }
      "tools/list" => mcp_success(
        mcp_request.id,
        serde_json::json!({
            "tools": cli::get_tool_definitions_for_cwd()
        }),
      ),
      "tools/call" => {
        // Extract tool name and arguments
        let tool_name = mcp_request.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = mcp_request
          .params
          .get("arguments")
          .cloned()
          .unwrap_or(serde_json::json!({}));

        // Check if daemon is running
        if !is_running(&socket_path) {
          mcp_error(
            mcp_request.id,
            -32000,
            "CCEngram daemon is not running. Start it with: ccengram daemon",
          )
        } else {
          // Add cwd to arguments for project context
          let mut args = arguments;
          if let Some(obj) = args.as_object_mut()
            && !obj.contains_key("cwd")
            && let Ok(cwd) = std::env::current_dir()
          {
            obj.insert("cwd".to_string(), serde_json::json!(cwd.to_string_lossy()));
          }

          // Forward to daemon
          match Client::connect_to(&socket_path).await {
            Ok(mut client) => {
              let request = Request {
                id: Some(serde_json::json!(1)),
                method: tool_name.to_string(),
                params: args,
              };

              match client.request(request).await {
                Ok(daemon_response) => {
                  if let Some(err) = daemon_response.error {
                    // Return error as text content (MCP style)
                    mcp_success(
                      mcp_request.id,
                      serde_json::json!({
                          "content": [{
                              "type": "text",
                              "text": format!("Error: {}", err.message)
                          }],
                          "isError": true
                      }),
                    )
                  } else if let Some(result) = daemon_response.result {
                    // Format result as text content
                    let text = serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());
                    mcp_success(
                      mcp_request.id,
                      serde_json::json!({
                          "content": [{
                              "type": "text",
                              "text": text
                          }]
                      }),
                    )
                  } else {
                    mcp_success(
                      mcp_request.id,
                      serde_json::json!({
                          "content": [{
                              "type": "text",
                              "text": "Success"
                          }]
                      }),
                    )
                  }
                }
                Err(e) => mcp_error(mcp_request.id, -32000, &format!("Daemon error: {}", e)),
              }
            }
            Err(e) => mcp_error(mcp_request.id, -32000, &format!("Connection error: {}", e)),
          }
        }
      }
      // Unknown method
      _ => mcp_error(
        mcp_request.id,
        -32601,
        &format!("Method not found: {}", mcp_request.method),
      ),
    };

    let out = serde_json::to_string(&response)? + "\n";
    stdout.write_all(out.as_bytes()).await?;
    stdout.flush().await?;
  }

  Ok(())
}

/// Handle a hook event
async fn cmd_hook(name: &str) -> Result<()> {
  // Parse hook event name
  let event: HookEvent = name.parse().map_err(|e| anyhow::anyhow!("Unknown hook: {}", e))?;

  // Read input from stdin
  let input = daemon::hooks::read_hook_input().context("Failed to read hook input")?;

  // Try to connect to running daemon first
  let socket_path = default_socket_path();
  if is_running(&socket_path) {
    let mut client = Client::connect_to(&socket_path)
      .await
      .context("Failed to connect to daemon")?;

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "hook".to_string(),
      params: serde_json::json!({
          "event": name,
          "params": input,
      }),
    };

    let response = client.request(request).await.context("Failed to send hook to daemon")?;

    if let Some(err) = response.error {
      error!("Hook error: {}", err.message);
    }
  } else {
    // Handle hook directly (stateless mode)
    let registry = Arc::new(ProjectRegistry::new());
    let handler = HookHandler::new(registry);

    match handler.handle(event, input).await {
      Ok(result) => {
        println!("{}", serde_json::to_string(&result)?);
      }
      Err(e) => {
        error!("Hook error: {}", e);
        std::process::exit(1);
      }
    }
  }

  Ok(())
}

/// Search memories
#[allow(clippy::too_many_arguments)]
async fn cmd_search(
  query: &str,
  limit: usize,
  project: Option<&str>,
  sector: Option<&str>,
  memory_type: Option<&str>,
  min_salience: Option<f32>,
  include_superseded: bool,
  scope: Option<&str>,
  json_output: bool,
) -> Result<()> {
  let socket_path = default_socket_path();

  // Ensure daemon is running
  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

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
        println!(
          "{}. [{}] {}",
          i + 1,
          memory.get("sector").and_then(|v| v.as_str()).unwrap_or("unknown"),
          memory.get("id").and_then(|v| v.as_str()).unwrap_or("?")
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
    }
  }

  Ok(())
}

/// Search code
#[allow(clippy::too_many_arguments)]
async fn cmd_search_code(
  query: &str,
  limit: usize,
  project: Option<&str>,
  language: Option<&str>,
  chunk_type: Option<&str>,
  path: Option<&str>,
  symbol: Option<&str>,
  json_output: bool,
) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

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
async fn cmd_search_docs(query: &str, limit: usize, project: Option<&str>, json_output: bool) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

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

        println!("{}. {} [{}]", i + 1, title, &doc_id[..8.min(doc_id.len())]);

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
    }
  }

  Ok(())
}

/// Show detailed memory by ID
async fn cmd_show(memory_id: &str, related: bool, json_output: bool) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_get".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "cwd": cwd,
        "include_related": related,
    }),
  };

  let response = client.request(request).await.context("Failed to get memory")?;

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
async fn cmd_delete(memory_id: &str, hard: bool) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_delete".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "cwd": cwd,
        "hard": hard,
    }),
  };

  let response = client.request(request).await.context("Failed to delete memory")?;

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
async fn cmd_export(output: Option<&str>, format: &str) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // Get all memories for the project
  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_list".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
    }),
  };

  let response = client.request(request).await.context("Failed to list memories")?;

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

/// Import a document
async fn cmd_import(path: &str, title: Option<&str>) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  // Verify file exists
  let file_path = std::path::Path::new(path);
  if !file_path.exists() {
    error!("File not found: {}", path);
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let abs_path = file_path.canonicalize().context("Failed to resolve path")?;
  let doc_title = title.unwrap_or_else(|| abs_path.file_name().and_then(|n| n.to_str()).unwrap_or("Untitled"));

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "docs_ingest".to_string(),
    params: serde_json::json!({
        "path": abs_path.to_string_lossy(),
        "title": doc_title,
        "cwd": cwd,
    }),
  };

  let response = client.request(request).await.context("Failed to import document")?;

  if let Some(err) = response.error {
    error!("Import error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
    let chunks = result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("Imported '{}' ({} chunks created)", doc_title, chunks);
  }

  Ok(())
}

/// Manage code index (index, stats, export, import)
async fn cmd_index(force: bool, stats: bool, export: Option<&str>, import: Option<&str>) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // Handle --stats
  if stats {
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "code_stats".to_string(),
      params: serde_json::json!({ "cwd": cwd }),
    };

    let response = client.request(request).await.context("Failed to get index stats")?;

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

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "code_list".to_string(),
      params: serde_json::json!({ "cwd": cwd }),
    };

    let response = client.request(request).await.context("Failed to export index")?;

    if let Some(err) = response.error {
      error!("Export error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      let export_data = serde_json::json!({
          "version": "1.0",
          "exported_at": chrono::Utc::now().to_rfc3339(),
          "chunks": result,
      });

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

  // Handle --import
  if let Some(path) = import {
    let content = std::fs::read_to_string(path).context("Failed to read import file")?;
    let export_data: serde_json::Value = serde_json::from_str(&content).context("Invalid JSON in import file")?;

    let Some(chunks) = export_data.get("chunks").and_then(|v| v.as_array()) else {
      error!("Invalid export format: missing 'chunks' array");
      std::process::exit(1);
    };

    println!("Importing {} code chunks...", chunks.len());

    let mut imported = 0;
    for chunk in chunks {
      let request = Request {
        id: Some(serde_json::json!(1)),
        method: "code_import_chunk".to_string(),
        params: serde_json::json!({
            "cwd": cwd,
            "chunk": chunk,
        }),
      };

      match client.request(request).await {
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

  // Default: run indexing
  println!("Indexing code in {}...", cwd);

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "force": force,
    }),
  };

  let response = client.request(request).await.context("Failed to index code")?;

  if let Some(err) = response.error {
    error!("Index error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
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

  Ok(())
}

/// Show statistics
async fn cmd_stats() -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  // Get daemon status
  let status_request = Request {
    id: Some(serde_json::json!(1)),
    method: "status".to_string(),
    params: serde_json::json!({}),
  };

  let status_response = client.request(status_request).await.context("Failed to get status")?;

  println!("CCEngram Statistics");
  println!("===================\n");

  // Print daemon status
  if let Some(result) = status_response.result {
    if let Some(version) = result.get("version").and_then(|v| v.as_str()) {
      println!("Version: {}", version);
    }
    if let Some(status) = result.get("status").and_then(|v| v.as_str()) {
      println!("Status: {}", status);
    }
    if let Some(projects) = result.get("projects").and_then(|v| v.as_u64()) {
      println!("Active projects: {}", projects);
    }
  }

  // Get project-specific stats for current directory
  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let stats_request = Request {
    id: Some(serde_json::json!(2)),
    method: "project_stats".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };

  let stats_response = client
    .request(stats_request)
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
async fn cmd_health() -> Result<()> {
  let socket_path = default_socket_path();

  println!("CCEngram Health Check");
  println!("=====================\n");

  // Check if daemon is running
  if !is_running(&socket_path) {
    println!("Daemon:     NOT RUNNING");
    println!("Socket:     {:?}", socket_path);
    println!("\nStart the daemon with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  // Ping test
  let ping_request = Request {
    id: Some(serde_json::json!(1)),
    method: "ping".to_string(),
    params: serde_json::json!({}),
  };

  let ping_ok = match client.request(ping_request).await {
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

  // Get comprehensive health status
  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  let health_request = Request {
    id: Some(serde_json::json!(2)),
    method: "health_check".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };

  if let Ok(response) = client.request(health_request).await
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

  Ok(())
}

/// Archive old low-salience memories
async fn cmd_archive(before: Option<&str>, threshold: f32, dry_run: bool) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // First, get all memories to find archival candidates
  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_list".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
    }),
  };

  let response = client.request(request).await.context("Failed to list memories")?;

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
    println!("  [{:.2}] {} - {}", salience, &id[..8], summary.replace('\n', " "));
  }
  println!();

  if dry_run {
    println!("Dry run - no changes made");
    return Ok(());
  }

  // Archive (soft delete) each memory
  let mut archived = 0;
  for (id, _, _) in candidates {
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_delete".to_string(),
      params: serde_json::json!({
          "memory_id": id,
          "cwd": cwd,
          "hard": false,
      }),
    };

    match client.request(request).await {
      Ok(resp) if resp.error.is_none() => archived += 1,
      _ => error!("Failed to archive memory {}", id),
    }
  }

  println!("Archived {} memories (soft deleted)", archived);
  Ok(())
}

/// Show or modify configuration
async fn cmd_config(
  key: Option<String>,
  value: Option<String>,
  reset: bool,
  list: bool,
  init: bool,
  preset: &str,
  show: bool,
) -> Result<()> {
  use engram_core::{Config, ToolPreset};

  // Handle --init: create project-relative config
  if init {
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
    return Ok(());
  }

  // Handle --show: display effective config
  if show {
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
    return Ok(());
  }

  // Handle --list: show available tool presets
  if list {
    println!("Tool presets:");
    println!();
    println!("  minimal   - memory_search, code_search, docs_search");
    println!("  standard  - memory_search, memory_add, memory_reinforce, memory_deemphasize,");
    println!("              code_search, docs_search, memory_timeline, entity_top, project_stats");
    println!("  full      - all {} tools", engram_core::ALL_TOOLS.len());
    println!();
    println!("Configuration sections: tools, embedding, decay, search, index");
    println!();
    println!("Usage:");
    println!("  ccengram config --init                  # Create project config with standard preset");
    println!("  ccengram config --init --preset minimal # Create with minimal preset");
    println!("  ccengram config --show                  # Show effective config");
    return Ok(());
  }

  // Handle --reset: reset user config to defaults
  if reset {
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
    return Ok(());
  }

  // Legacy key/value handling
  match (key, value) {
    (Some(_k), Some(_v)) => {
      println!("Direct key/value setting is deprecated.");
      println!("Use 'ccengram config --init' to create a config file, then edit it directly.");
    }
    (Some(k), None) => {
      // Get a value from current config
      let cwd = std::env::current_dir()?;
      let config = Config::load_for_project(&cwd);
      let toml_str = toml::to_string_pretty(&config)?;

      // Simple grep-style search
      for line in toml_str.lines() {
        if line.contains(&k) {
          println!("{}", line);
        }
      }
    }
    (None, _) => {
      // Show help
      println!("CCEngram Configuration");
      println!();
      println!("Usage:");
      println!("  ccengram config --init                  # Create project config");
      println!("  ccengram config --init --preset minimal # Create with minimal preset");
      println!("  ccengram config --show                  # Show effective config");
      println!("  ccengram config --list                  # Show available presets");
      println!("  ccengram config --reset                 # Reset user config");
      println!();
      println!("Config locations:");
      println!("  Project: .claude/ccengram.toml");
      if let Some(user_path) = Config::user_config_path() {
        println!("  User:    {:?}", user_path);
      }
    }
  }

  Ok(())
}

/// Watch for file changes
async fn cmd_watch(stop: bool, status: bool) -> Result<()> {
  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  if stop {
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "watch_stop".to_string(),
      params: serde_json::json!({ "cwd": cwd }),
    };

    let response = client.request(request).await.context("Failed to stop watcher")?;

    if let Some(err) = response.error {
      error!("Stop error: {}", err.message);
      std::process::exit(1);
    }

    println!("File watcher stopped");
    return Ok(());
  }

  if status {
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "watch_status".to_string(),
      params: serde_json::json!({ "cwd": cwd }),
    };

    let response = client.request(request).await.context("Failed to get watcher status")?;

    if let Some(err) = response.error {
      error!("Status error: {}", err.message);
      std::process::exit(1);
    }

    if let Some(result) = response.result {
      let is_running = result.get("running").and_then(|v| v.as_bool()).unwrap_or(false);
      println!("Watcher Status: {}", if is_running { "RUNNING" } else { "STOPPED" });

      if is_running {
        if let Some(paths) = result.get("watched_paths").and_then(|v| v.as_u64()) {
          println!("Watched Paths: {}", paths);
        }
        if let Some(changes) = result.get("pending_changes").and_then(|v| v.as_u64()) {
          println!("Pending Changes: {}", changes);
        }
      }
    }
    return Ok(());
  }

  // Start watching
  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "watch_start".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };

  let response = client.request(request).await.context("Failed to start watcher")?;

  if let Some(err) = response.error {
    error!("Watch error: {}", err.message);
    std::process::exit(1);
  }

  println!("File watcher started for {}", cwd);
  println!("Press Ctrl+C to stop watching");

  // Keep the CLI alive until interrupted
  tokio::signal::ctrl_c().await?;

  // Send stop command on exit
  let stop_request = Request {
    id: Some(serde_json::json!(1)),
    method: "watch_stop".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let _ = client.request(stop_request).await;

  println!("\nWatcher stopped");
  Ok(())
}

/// Check for updates or update to latest version
async fn cmd_update(check_only: bool, target_version: Option<String>) -> Result<()> {
  const GITHUB_REPO: &str = "joey-goodjob/ccengram";
  const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

  println!("CCEngram v{}", CURRENT_VERSION);
  println!();

  // Fetch latest release info from GitHub API
  let client = reqwest::Client::builder().user_agent("ccengram-updater").build()?;

  let releases_url = format!("https://api.github.com/repos/{}/releases", GITHUB_REPO);

  let response = client
    .get(&releases_url)
    .send()
    .await
    .context("Failed to fetch releases from GitHub")?;

  if !response.status().is_success() {
    anyhow::bail!("Failed to fetch releases: HTTP {}", response.status());
  }

  #[derive(serde::Deserialize)]
  struct Release {
    tag_name: String,
    name: String,
    html_url: String,
    assets: Vec<Asset>,
    prerelease: bool,
    draft: bool,
  }

  #[derive(serde::Deserialize)]
  struct Asset {
    name: String,
    browser_download_url: String,
  }

  let releases: Vec<Release> = response.json().await?;

  // Filter out prereleases and drafts
  let stable_releases: Vec<_> = releases.iter().filter(|r| !r.prerelease && !r.draft).collect();

  if stable_releases.is_empty() {
    println!("No releases found");
    return Ok(());
  }

  // Find target version or use latest
  let target = if let Some(ref ver) = target_version {
    stable_releases
      .iter()
      .find(|r| r.tag_name.trim_start_matches('v') == ver.trim_start_matches('v'))
      .copied()
      .ok_or_else(|| anyhow::anyhow!("Version {} not found", ver))?
  } else {
    stable_releases[0]
  };

  let target_ver = target.tag_name.trim_start_matches('v');

  // Compare versions
  let current_parts: Vec<u32> = CURRENT_VERSION.split('.').filter_map(|p| p.parse().ok()).collect();
  let target_parts: Vec<u32> = target_ver.split('.').filter_map(|p| p.parse().ok()).collect();

  let needs_update = target_parts
    .iter()
    .zip(current_parts.iter().chain(std::iter::repeat(&0)))
    .any(|(t, c)| t > c)
    || target_parts.len() > current_parts.len();

  if !needs_update {
    println!("✓ You are running the latest version (v{})", CURRENT_VERSION);
    return Ok(());
  }

  println!("New version available: v{} -> v{}", CURRENT_VERSION, target_ver);
  println!("  Release: {}", target.name);
  println!("  URL: {}", target.html_url);
  println!();

  if check_only {
    println!("Run 'ccengram update' to install the update");
    return Ok(());
  }

  // Determine platform-specific asset name
  let os = std::env::consts::OS;
  let arch = std::env::consts::ARCH;

  let platform = match (os, arch) {
    ("linux", "x86_64") => "linux-x86_64",
    ("linux", "aarch64") => "linux-aarch64",
    ("macos", "x86_64") => "darwin-x86_64",
    ("macos", "aarch64") => "darwin-aarch64",
    ("windows", "x86_64") => "windows-x86_64.exe",
    _ => {
      println!("Unsupported platform: {} {}", os, arch);
      println!("Please download manually from: {}", target.html_url);
      return Ok(());
    }
  };

  let asset_name = format!("ccengram-{}", platform);
  let asset = target
    .assets
    .iter()
    .find(|a| a.name.starts_with(&asset_name))
    .ok_or_else(|| {
      anyhow::anyhow!(
        "No binary found for platform {}. Available: {:?}",
        platform,
        target.assets.iter().map(|a| &a.name).collect::<Vec<_>>()
      )
    })?;

  println!("Downloading: {}", asset.name);

  // Download the binary
  let download_response = client.get(&asset.browser_download_url).send().await?;

  if !download_response.status().is_success() {
    anyhow::bail!("Failed to download: HTTP {}", download_response.status());
  }

  let bytes = download_response.bytes().await?;

  // Get current executable path
  let current_exe = std::env::current_exe().context("Failed to get current executable path")?;
  let backup_path = current_exe.with_extension("bak");

  // Backup current binary
  println!("Backing up current binary...");
  std::fs::rename(&current_exe, &backup_path).context("Failed to backup current binary")?;

  // Write new binary
  println!("Installing new version...");
  std::fs::write(&current_exe, &bytes).context("Failed to write new binary")?;

  // Set executable permissions on Unix
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(&current_exe, perms)?;
  }

  println!();
  println!("✓ Successfully updated to v{}", target_ver);
  println!();
  println!("Backup saved to: {:?}", backup_path);

  Ok(())
}

/// Migrate embeddings to new dimensions/model
async fn cmd_migrate(dry_run: bool, force: bool) -> Result<()> {
  use engram_core::Config;

  let socket_path = default_socket_path();

  if !is_running(&socket_path) {
    error!("Daemon is not running. Start it with: ccengram daemon");
    std::process::exit(1);
  }

  let mut client = Client::connect_to(&socket_path)
    .await
    .context("Failed to connect to daemon")?;

  let cwd = std::env::current_dir()
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_else(|_| ".".to_string());

  // Load config to show what we're migrating to
  let config = Config::load_for_project(&std::path::PathBuf::from(&cwd));

  println!("Embedding Migration");
  println!("===================\n");
  println!("Target configuration:");
  println!("  Provider:   {:?}", config.embedding.provider);
  println!("  Model:      {}", config.embedding.model);
  println!("  Dimensions: {}", config.embedding.dimensions);
  println!();

  if dry_run {
    println!("DRY RUN - no changes will be made\n");
  }

  // Get current stats
  let stats_request = Request {
    id: Some(serde_json::json!(1)),
    method: "project_stats".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };

  let stats_response = client.request(stats_request).await.context("Failed to get stats")?;

  let mut memory_count = 0u64;
  let mut code_count = 0u64;
  let mut doc_count = 0u64;

  if let Some(stats) = stats_response.result {
    if let Some(memories) = stats.get("memories") {
      memory_count = memories.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    }
    if let Some(code) = stats.get("code") {
      code_count = code.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
    }
    if let Some(docs) = stats.get("documents") {
      doc_count = docs.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
    }
  }

  println!("Items to migrate:");
  println!("  Memories:       {}", memory_count);
  println!("  Code chunks:    {}", code_count);
  println!("  Document chunks: {}", doc_count);
  println!();

  let total = memory_count + code_count + doc_count;
  if total == 0 {
    println!("Nothing to migrate.");
    return Ok(());
  }

  if dry_run {
    println!("Would migrate {} total items.", total);
    return Ok(());
  }

  // Call the migration tool
  println!("Starting migration...\n");

  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "migrate_embedding".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "force": force,
    }),
  };

  let response = client.request(request).await.context("Failed to migrate embeddings")?;

  if let Some(err) = response.error {
    error!("Migration error: {}", err.message);
    std::process::exit(1);
  }

  if let Some(result) = response.result {
    let migrated = result.get("migrated_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let skipped = result.get("skipped_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let errors = result.get("error_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let duration_ms = result.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);

    println!("Migration complete:");
    println!("  Migrated: {}", migrated);
    println!("  Skipped:  {}", skipped);
    println!("  Errors:   {}", errors);
    println!("  Duration: {:.1}s", duration_ms as f64 / 1000.0);
  }

  Ok(())
}

/// Generate a MemExplore subagent for Claude Code
async fn cmd_agent(output: Option<&str>, force: bool) -> Result<()> {
  let cwd = std::env::current_dir()?;
  let default_path = cwd.join(".claude").join("agents").join("MemExplore.md");
  let output_path = output.map(std::path::PathBuf::from).unwrap_or(default_path);

  // Check if file exists
  if output_path.exists() && !force {
    error!("Agent file already exists: {:?}", output_path);
    println!("Use --force to overwrite");
    std::process::exit(1);
  }

  // Create parent directories
  if let Some(parent) = output_path.parent() {
    std::fs::create_dir_all(parent)?;
  }

  // Generate agent content
  let agent_content = generate_memexplore_agent();

  std::fs::write(&output_path, &agent_content)?;

  println!("Generated MemExplore agent: {:?}", output_path);
  println!();
  println!("This agent has access to CCEngram memory tools for codebase exploration.");
  println!("Claude Code will automatically use it when the description matches your task.");

  Ok(())
}

/// Launch interactive TUI
async fn cmd_tui(project: Option<PathBuf>) -> Result<()> {
  let path = project.unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
  tui::run(path).await
}

/// Generate the MemExplore agent markdown content
fn generate_memexplore_agent() -> String {
  r#"---
name: MemExplore
description: "Use when exploring the codebase, or when you need code, preference, or history questions answered. (use this over Explore agent because it has memory access)"
tools: Glob, Grep, Read, WebFetch, TodoWrite, WebSearch, mcp__plugin__memory_search, mcp__plugin__code_search, mcp__plugin__docs_search, mcp__plugin__memory_timeline, mcp__plugin__entity_top
model: haiku
color: green
---
You are a file search and memory specialist for Claude Code, Anthropic's official CLI for Claude. You excel at thoroughly navigating and exploring codebases while leveraging persistent memory to provide context-aware answers.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search, analyze, and recall information. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents
- Searching project memories for preferences, decisions, and history
- Finding relevant code using semantic search
- Recalling past context and patterns from memory

=== MEMORY TOOLS ===
You have access to CCEngram memory tools:
- memory_search: Search memories by semantic similarity for preferences, decisions, gotchas, patterns
- code_search: Semantic search over indexed code chunks with file paths and line numbers
- docs_search: Search ingested documents and references
- memory_timeline: Get chronological context around a memory
- entity_top: Get top mentioned entities (people, technologies, concepts)

Use these tools PROACTIVELY to:
- Check for relevant past decisions before answering questions
- Look up user preferences and coding style
- Find related code patterns that were previously discussed
- Recall gotchas and issues encountered before

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use Read when you know the specific file path you need to read
- Use memory_search FIRST when the question involves preferences, history, or past decisions
- Use code_search when looking for implementations or code patterns
- NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files
- Check memory FIRST before doing extensive file searches - the answer may already be known

Complete the user's search request efficiently and report your findings clearly.
"#.to_string()
}
