//! CCEngram CLI - Intelligent memory and code search for Claude Code

use std::{io, path::PathBuf};

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;

mod commands;
mod format;
mod logging;
mod mcp;
mod tools;
mod tui;

#[cfg(all(unix, feature = "jemalloc-pprof"))]
use commands::cmd_pprof;
use commands::{
  cmd_agent, cmd_archive, cmd_config_init, cmd_config_reset, cmd_config_show, cmd_context, cmd_daemon, cmd_delete,
  cmd_deleted, cmd_health, cmd_hook, cmd_index, cmd_logs, cmd_logs_list, cmd_projects_clean, cmd_projects_clean_all,
  cmd_projects_list, cmd_projects_show, cmd_restore, cmd_search, cmd_search_code, cmd_search_docs, cmd_show, cmd_stats,
  cmd_tui, cmd_update, cmd_watch,
};
use logging::{init_cli_logging, init_daemon_logging_with_config};
use mcp::cmd_mcp;

#[derive(Parser)]
#[command(name = "ccengram")]
#[command(about = "Intelligent memory and code search for Claude Code")]
#[command(after_help = "\
QUICK START:
  ccengram daemon                 # Start daemon (required)
  ccengram config init            # Initialize project config
  ccengram index code             # Index codebase
  ccengram search memories \"q\"    # Search memories

COMMON WORKFLOWS:
  ccengram watch                  # Auto-index on file changes
  ccengram tui                    # Interactive terminal UI
  ccengram health                 # Check system status")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

/// Subcommands for `ccengram index`
#[derive(Subcommand)]
pub enum IndexCommand {
  /// Index code files in the project
  Code {
    /// Force re-index all files
    #[arg(long)]
    force: bool,
    /// Show index statistics
    #[arg(long)]
    stats: bool,
  },
  /// Index documents from a directory
  Docs {
    /// Directory to index (default: configured docs.directories)
    #[arg(short, long)]
    directory: Option<String>,
    /// Force re-index all documents
    #[arg(long)]
    force: bool,
    /// Show document index statistics
    #[arg(long)]
    stats: bool,
  },
  /// Index a single file (auto-detects code vs document)
  File {
    /// File path to index
    path: String,
    /// Document title (optional, for documents only)
    #[arg(short, long)]
    title: Option<String>,
    /// Force re-index even if unchanged
    #[arg(long)]
    force: bool,
  },
}

/// Subcommands for `ccengram search`
#[derive(Subcommand)]
pub enum SearchCommand {
  /// Search memories (default if no subcommand)
  #[command(
    alias = "mem",
    after_help = "\
NOTE:
  IDs are shown as 8-character prefixes by default. Use --long to show full IDs.
  You can use these prefixes directly in commands like 'memory show <prefix>'."
  )]
  Memories {
    /// Search query
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
    /// Output as JSON
    #[arg(long)]
    json: bool,
    /// Show full IDs instead of truncated prefixes
    #[arg(long)]
    long: bool,
  },
  /// Search indexed code
  Code {
    /// Search query
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
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Search indexed documents
  Docs {
    /// Search query
    query: String,
    #[arg(short, long, default_value = "10")]
    limit: usize,
    /// Project path (default: current directory)
    #[arg(short, long)]
    project: Option<String>,
    /// Output as JSON
    #[arg(long)]
    json: bool,
    /// Show full IDs instead of truncated prefixes
    #[arg(long)]
    long: bool,
  },
}

/// Subcommands for `ccengram memory`
#[derive(Subcommand)]
pub enum MemoryCommand {
  /// Show detailed memory by ID
  Show {
    /// Memory ID to show
    id: String,
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
    id: String,
    /// Permanently delete (hard delete)
    #[arg(long)]
    hard: bool,
  },
  /// Archive old low-salience memories
  #[command(
    long_about = "Archive old, low-salience memories by soft-deleting them.\n\n\
    Archived memories are hidden from search but can be restored. Use --dry-run \
    to preview what would be archived before committing.",
    after_help = "\
EXAMPLES:
  ccengram memory archive --dry-run           # Preview what would be archived
  ccengram memory archive --threshold 0.2     # Archive memories with salience < 0.2
  ccengram memory archive --before 2024-01-01 # Archive old memories"
  )]
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
  /// Restore a soft-deleted memory
  Restore {
    /// Memory ID to restore
    id: String,
  },
  /// List soft-deleted memories
  Deleted {
    /// Maximum number of memories to show
    #[arg(short, long, default_value = "20")]
    limit: usize,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
}

/// Subcommands for `ccengram config`
#[derive(Subcommand)]
pub enum ConfigCommand {
  /// Show current effective configuration
  #[command(long_about = "Show the current effective configuration.\n\n\
    Displays which config file is being used and its contents as TOML.")]
  Show,

  /// Initialize project config file (.claude/ccengram.toml)
  #[command(long_about = "Initialize a project-specific configuration file.\n\n\
    Creates .claude/ccengram.toml with the specified tool preset.")]
  Init {
    /// Tool preset: minimal, standard, or full
    #[arg(long, default_value = "minimal", value_parser = ["minimal", "standard", "full"])]
    preset: String,
  },

  /// Reset user configuration to defaults
  #[command(long_about = "Reset the user-level configuration file to defaults.\n\n\
    This affects ~/.config/ccengram/config.toml, not project configs.")]
  Reset,
}

/// Subcommands for `ccengram projects`
#[derive(Subcommand)]
pub enum ProjectsCommand {
  /// List all indexed projects
  List {
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Show details for a specific project
  Show {
    /// Project ID or path
    project: String,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Remove a project's data
  Clean {
    /// Project ID or path
    project: String,
    /// Skip confirmation prompt
    #[arg(long)]
    force: bool,
  },
  /// Remove all project data
  CleanAll {
    /// Skip confirmation prompt
    #[arg(long)]
    force: bool,
  },
}

#[derive(Subcommand)]
enum Commands {
  /// Start the daemon
  #[command(long_about = "Start the CCEngram daemon.\n\n\
    By default, runs in background mode with auto-shutdown enabled.\n\
    Use --foreground for persistent daemon that logs to console.\n\
    Use --background when auto-starting from CLI commands.")]
  Daemon {
    /// Stop the running daemon
    #[arg(long, conflicts_with_all = ["foreground", "background"])]
    stop: bool,
    /// Run in foreground (disables auto-shutdown, logs to console)
    #[arg(long, conflicts_with = "background")]
    foreground: bool,
    /// Run in background (enables auto-shutdown, used by auto-start)
    #[arg(long, conflicts_with = "foreground")]
    background: bool,
    /// Embedding provider override (ollama or openrouter)
    #[arg(long, hide = true)]
    embedding_provider: Option<String>,
    /// OpenRouter API key override
    #[arg(long, hide = true)]
    openrouter_api_key: Option<String>,
  },
  /// MCP server (for Claude Code integration)
  Mcp,
  /// Handle hook event
  Hook {
    /// Hook name to handle
    name: String,
  },
  /// Search memories, code, or documents
  #[command(after_help = "\
EXAMPLES:
  ccengram search memories \"authentication flow\"
  ccengram search memories \"user preferences\" --sector semantic
  ccengram search code \"error handling\" --language rust
  ccengram search code \"database\" --type function
  ccengram search docs \"API reference\"")]
  Search {
    #[command(subcommand)]
    command: SearchCommand,
  },
  /// Manage memories (show, delete, export, archive)
  #[command(after_help = "\
NOTE:
  Memories are created automatically via hooks during Claude Code sessions.
  Use 'ccengram search memories' to find memories by content.")]
  Memory {
    #[command(subcommand)]
    command: MemoryCommand,
  },
  /// Manage code and document index
  #[command(after_help = "\
WORKFLOW:
  Indexing runs automatically via 'ccengram watch', or manually:

  ccengram index code             # Index source files
  ccengram index code --force     # Re-index everything
  ccengram index docs             # Index documentation
  ccengram index file <path>      # Index a single file

SUPPORTED LANGUAGES:
  Rust, Python, TypeScript, JavaScript, Go, and more via tree-sitter")]
  Index {
    #[command(subcommand)]
    command: Option<IndexCommand>,
  },
  /// Manage configuration
  #[command(after_help = "\
PRESETS:
  minimal   - explore, context (2 tools; recommended)
  standard  - Above + memory_add, memory_reinforce, memory_deemphasize,
              code_index, code_stats, watch_start, watch_stop,
              watch_status, project_stats (11 tools)
  full      - All 34 available tools

CONFIG LOCATIONS:
  Project: .claude/ccengram.toml
  User:    ~/.config/ccengram/config.toml")]
  Config {
    #[command(subcommand)]
    command: ConfigCommand,
  },
  /// Watch for file changes and update index
  Watch {
    /// Stop any running watcher
    #[arg(long)]
    stop: bool,
    /// Check watcher status
    #[arg(long)]
    status: bool,
    /// Skip startup scan (don't reconcile with filesystem on start)
    #[arg(long)]
    no_startup_scan: bool,
    /// Startup scan mode: deleted_only, deleted_and_new, full (default: from config)
    #[arg(long)]
    startup_scan_mode: Option<String>,
    /// Wait for startup scan to complete before watching
    #[arg(long)]
    startup_scan_sync: bool,
  },
  /// Get surrounding context for a code or document chunk
  #[command(after_help = "\
EXAMPLES:
  ccengram context 019abc                    # Get context (auto-detects type)
  ccengram context 019abc --before 30        # Get 30 lines/chunks before
  ccengram context 019abc --json             # Output as JSON

USAGE:
  Use chunk IDs from 'search code' or 'search docs' results.
  For code chunks: --before/--after specify lines (default: 20, max: 500)
  For doc chunks: --before/--after specify chunks (default: 1, max: 10)")]
  Context {
    /// Chunk ID from search results (8+ character prefix works)
    chunk_id: String,
    /// Lines/chunks to include before (code: 20, docs: 1)
    #[arg(short, long)]
    before: Option<usize>,
    /// Lines/chunks to include after (code: 20, docs: 1)
    #[arg(short, long)]
    after: Option<usize>,
    /// Output as JSON
    #[arg(long)]
    json: bool,
  },
  /// Show statistics
  Stats,
  /// Health check
  Health,
  /// Check for updates or update to latest version
  Update {
    /// Only check for updates without installing
    #[arg(long)]
    check: bool,
    /// Specific version to update to
    #[arg(long)]
    version: Option<String>,
  },
  /// Generate a SemExplore subagent for Claude Code
  Agent {
    /// Output path (default: .claude/agents/SemExplore.md)
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
  /// Manage indexed projects
  #[command(after_help = "\
EXAMPLES:
  ccengram projects list                  # List all indexed projects
  ccengram projects show /path/to/project # Show project details
  ccengram projects clean /path/to/project # Remove project data")]
  Projects {
    #[command(subcommand)]
    command: ProjectsCommand,
  },
  /// View daemon logs
  #[command(after_help = "\
EXAMPLES:
  ccengram logs                    # Show last 50 lines
  ccengram logs -f                 # Follow logs in real-time
  ccengram logs -n 100             # Show last 100 lines
  ccengram logs --level error      # Filter by log level
  ccengram logs --open             # Open log directory")]
  Logs {
    /// Follow log output in real-time (like tail -f)
    #[arg(short, long)]
    follow: bool,
    /// Number of lines to show (default: 50)
    #[arg(short = 'n', long, default_value = "50")]
    lines: usize,
    /// Show logs from a specific date (YYYY-MM-DD)
    #[arg(long)]
    date: Option<String>,
    /// Filter logs by level (error, warn, info, debug, trace)
    #[arg(long)]
    level: Option<String>,
    /// Open log directory in file manager
    #[arg(long)]
    open: bool,
    /// List available log files
    #[arg(long)]
    list: bool,
  },
  /// Generate shell completions
  #[command(after_help = "\
EXAMPLES:
  ccengram completions bash > ~/.local/share/bash-completion/completions/ccengram
  ccengram completions zsh > ~/.zfunc/_ccengram
  ccengram completions fish > ~/.config/fish/completions/ccengram.fish
  ccengram completions powershell >> $PROFILE

INSTALLATION:
  Bash:
    Add to ~/.bashrc:
      source <(ccengram completions bash)
    Or generate once:
      ccengram completions bash > ~/.local/share/bash-completion/completions/ccengram

  Zsh:
    Add to ~/.zshrc (before compinit):
      fpath=(~/.zfunc $fpath)
    Then generate:
      ccengram completions zsh > ~/.zfunc/_ccengram

  Fish:
    ccengram completions fish > ~/.config/fish/completions/ccengram.fish

  PowerShell:
    ccengram completions powershell >> $PROFILE")]
  Completions {
    /// Shell to generate completions for
    #[arg(value_enum)]
    shell: Shell,
  },

  /// Download heap profile from daemon (requires jemalloc-pprof feature)
  #[cfg(all(unix, feature = "jemalloc-pprof"))]
  #[command(after_help = "\
EXAMPLES:
  ccengram pprof                      # Download to heap_<timestamp>.pb.gz
  ccengram pprof -o profile.pb.gz    # Download to specific file

USAGE:
  The daemon must be running with the jemalloc-pprof feature enabled.
  View the profile with: go tool pprof <file>")]
  Pprof {
    /// Output file path (default: heap_<timestamp>.pb.gz in current directory)
    #[arg(short, long)]
    output: Option<String>,
  },
}

#[tokio::main]
async fn main() -> Result<()> {
  let cli = Cli::parse();

  // Use file logging for daemon (background mode), console-only for other commands
  let _guard = match &cli.command {
    Commands::Daemon { foreground, .. } => init_daemon_logging_with_config(*foreground).await,
    _ => {
      init_cli_logging();
      None
    }
  };

  match cli.command {
    Commands::Daemon {
      stop,
      foreground,
      background,
      embedding_provider,
      openrouter_api_key,
    } => cmd_daemon(stop, foreground, background, embedding_provider, openrouter_api_key).await,
    Commands::Mcp => cmd_mcp().await,
    Commands::Hook { name } => cmd_hook(&name).await,

    // Search subcommands
    Commands::Search { command } => match command {
      SearchCommand::Memories {
        query,
        limit,
        project,
        sector,
        memory_type,
        min_salience,
        include_superseded,
        scope,
        json,
        long,
      } => {
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
          long,
        )
        .await
      }
      SearchCommand::Code {
        query,
        limit,
        project,
        language,
        chunk_type,
        path,
        symbol,
        json,
      } => {
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
      SearchCommand::Docs {
        query,
        limit,
        project,
        json,
        long,
      } => cmd_search_docs(&query, limit, project.as_deref(), json, long).await,
    },

    // Memory subcommands
    Commands::Memory { command } => match command {
      MemoryCommand::Show { id, related, json } => cmd_show(&id, related, json).await,
      MemoryCommand::Delete { id, hard } => cmd_delete(&id, hard).await,
      MemoryCommand::Archive {
        before,
        threshold,
        dry_run,
      } => cmd_archive(before.as_deref(), threshold, dry_run).await,
      MemoryCommand::Restore { id } => cmd_restore(&id).await,
      MemoryCommand::Deleted { limit, json } => cmd_deleted(limit, json).await,
    },

    Commands::Index { command } => cmd_index(command).await,

    // Config subcommands
    Commands::Config { command } => match command {
      ConfigCommand::Show => cmd_config_show().await,
      ConfigCommand::Init { preset } => cmd_config_init(&preset).await,
      ConfigCommand::Reset => cmd_config_reset().await,
    },

    Commands::Watch {
      stop,
      status,
      no_startup_scan,
      startup_scan_mode,
      startup_scan_sync,
    } => cmd_watch(stop, status, no_startup_scan, startup_scan_mode, startup_scan_sync).await,
    Commands::Context {
      chunk_id,
      before,
      after,
      json,
    } => cmd_context(&chunk_id, before, after, json).await,
    Commands::Stats => cmd_stats().await,
    Commands::Health => cmd_health().await,
    Commands::Update { check, version } => cmd_update(check, version).await,
    Commands::Agent { output, force } => cmd_agent(output.as_deref(), force).await,
    Commands::Tui { project } => cmd_tui(project).await,

    // Projects subcommands
    Commands::Projects { command } => match command {
      ProjectsCommand::List { json } => cmd_projects_list(json).await,
      ProjectsCommand::Show { project, json } => cmd_projects_show(&project, json).await,
      ProjectsCommand::Clean { project, force } => cmd_projects_clean(&project, force).await,
      ProjectsCommand::CleanAll { force } => cmd_projects_clean_all(force).await,
    },

    // Logs command
    Commands::Logs {
      follow,
      lines,
      date,
      level,
      open,
      list,
    } => {
      if list {
        cmd_logs_list()
      } else {
        cmd_logs(follow, lines, date.as_deref(), level.as_deref(), open)
      }
    }

    // Completions command
    Commands::Completions { shell } => {
      print_completions(shell);
      Ok(())
    }

    // Pprof command (unix + jemalloc-pprof feature only)
    #[cfg(all(unix, feature = "jemalloc-pprof"))]
    Commands::Pprof { output } => cmd_pprof(output.as_deref()).await,
  }
}

/// Print shell completions to stdout
fn print_completions(shell: Shell) {
  clap_complete::generate(shell, &mut Cli::command(), "ccengram", &mut io::stdout());
}
