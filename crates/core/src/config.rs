//! Configuration system for CCEngram with per-project overrides.
//!
//! Config priority: project-relative (.claude/ccengram.toml) > user (~/.config/ccengram/config.toml)

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

// ============================================================================
// Tool Configuration
// ============================================================================

/// All available MCP tools (excluding internal-only tools like hook, ping, status)
pub const ALL_TOOLS: &[&str] = &[
  // Unified exploration tools (new)
  "explore",
  "context",
  // Memory tools
  "memory_search",
  "memory_get",
  "memory_list",
  "memory_add",
  "memory_reinforce",
  "memory_deemphasize",
  "memory_delete",
  "memory_supersede",
  "memory_timeline",
  "memory_related",
  // Code tools
  "code_search",
  "code_context",
  "code_index",
  "code_list",
  "code_import_chunk",
  "code_stats",
  "code_memories",
  "code_callers",
  "code_callees",
  "code_related",
  "code_context_full",
  // Watch tools
  "watch_start",
  "watch_stop",
  "watch_status",
  // Document tools
  "docs_search",
  "doc_context",
  "docs_ingest",
  // Entity tools
  "entity_list",
  "entity_get",
  "entity_top",
  // Relationship tools
  "relationship_add",
  "relationship_list",
  "relationship_delete",
  "relationship_related",
  // Statistics
  "project_stats",
  "health_check",
];

/// Internal tools that are always available but not exposed in tool lists
pub const INTERNAL_TOOLS: &[&str] = &["hook", "ping", "status"];

/// Minimal preset: streamlined exploration tools (2 tools)
/// This is the recommended preset for most users.
pub const PRESET_MINIMAL: &[&str] = &[
  "explore",
  "context",
];

/// Standard preset: exploration + management + diagnostics (11 tools)
pub const PRESET_STANDARD: &[&str] = &[
  // Exploration tools
  "explore",
  "context",
  // Memory management (for manual curation)
  "memory_add",
  "memory_reinforce",
  "memory_deemphasize",
  // Code maintenance
  "code_index",
  "code_stats",
  // Watch management
  "watch_start",
  "watch_stop",
  "watch_status",
  // Diagnostics
  "project_stats",
];

/// Tool preset options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolPreset {
  Minimal,
  #[default]
  Standard,
  Full,
}

impl ToolPreset {
  /// Get the tools for this preset
  pub fn tools(&self) -> Vec<&'static str> {
    match self {
      ToolPreset::Minimal => PRESET_MINIMAL.to_vec(),
      ToolPreset::Standard => PRESET_STANDARD.to_vec(),
      ToolPreset::Full => ALL_TOOLS.to_vec(),
    }
  }
}

// ============================================================================
// Embedding Configuration
// ============================================================================

/// Embedding provider options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
  #[default]
  Ollama,
  OpenRouter,
}

/// Embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
  /// Which embedding provider to use
  pub provider: EmbeddingProvider,

  /// Model name (e.g., "qwen3-embedding", "openai/text-embedding-3-small")
  pub model: String,

  /// Embedding dimensions (e.g., 4096, 1536, 4096)
  pub dimensions: usize,

  /// Ollama server URL (only used when provider = ollama)
  pub ollama_url: String,

  /// OpenRouter API key (only used when provider = openrouter)
  /// If not set, reads from OPENROUTER_API_KEY env var
  #[serde(skip_serializing_if = "Option::is_none")]
  pub openrouter_api_key: Option<String>,

  /// Context length for batch size calculation (default: 32768)
  /// Should match OLLAMA_CONTEXT_LENGTH environment variable if set
  /// Lower VRAM requires smaller context_length:
  ///   24 GB -> 32768, 12 GB -> 16384, 8 GB -> 8192, 6 GB -> 4096
  pub context_length: usize,

  /// Maximum batch size for embedding requests
  /// Auto-calculated from context_length if not set: min(context_length / 512, 64)
  /// Set explicitly to override auto-calculation
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_batch_size: Option<usize>,
}

impl Default for EmbeddingConfig {
  fn default() -> Self {
    Self {
      provider: EmbeddingProvider::Ollama,
      model: "qwen3-embedding".to_string(),
      dimensions: 4096,
      ollama_url: "http://localhost:11434".to_string(),
      openrouter_api_key: None,
      context_length: 32768,
      max_batch_size: None, // Auto-calculated
    }
  }
}

// ============================================================================
// Decay Configuration
// ============================================================================

/// Decay and memory lifecycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecayConfig {
  /// How often to run decay in hours (default: 60)
  pub decay_interval_hours: u64,

  /// Minimum salience threshold (default: 0.05)
  pub min_salience: f64,

  /// Salience threshold below which memories are archived (default: 0.1)
  pub archive_threshold: f64,

  /// Maximum days without access before forced decay (default: 90)
  pub max_idle_days: i64,

  /// Session cleanup interval in hours (default: 6)
  pub session_cleanup_hours: u64,

  /// Maximum session age in hours before cleanup (default: 6)
  pub max_session_age_hours: u64,
}

impl Default for DecayConfig {
  fn default() -> Self {
    Self {
      decay_interval_hours: 60,
      min_salience: 0.05,
      archive_threshold: 0.1,
      max_idle_days: 90,
      session_cleanup_hours: 6,
      max_session_age_hours: 6,
    }
  }
}

// ============================================================================
// Search Configuration
// ============================================================================

/// Search defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
  /// Default number of results (default: 10)
  pub default_limit: usize,

  /// Whether to include superseded memories by default (default: false)
  pub include_superseded: bool,

  /// Semantic weight in ranking (default: 0.5)
  pub semantic_weight: f64,

  /// Salience weight in ranking (default: 0.3)
  pub salience_weight: f64,

  /// Recency weight in ranking (default: 0.2)
  pub recency_weight: f64,

  // ---- Explore tool settings ----

  /// Default expand_top for explore tool - how many top results include full context (default: 3)
  pub explore_expand_top: usize,

  /// Default limit for explore tool - max results per scope (default: 10)
  pub explore_limit: usize,

  /// Default depth for context tool - items per section like callers, callees (default: 5)
  pub context_depth: usize,

  /// Max items in batch context call (default: 5)
  pub context_max_batch: usize,

  /// Max suggestions to generate from explore results (default: 5)
  pub explore_max_suggestions: usize,
}

impl Default for SearchConfig {
  fn default() -> Self {
    Self {
      default_limit: 10,
      include_superseded: false,
      semantic_weight: 0.5,
      salience_weight: 0.3,
      recency_weight: 0.2,
      explore_expand_top: 3,
      explore_limit: 10,
      context_depth: 5,
      context_max_batch: 5,
      explore_max_suggestions: 5,
    }
  }
}

// ============================================================================
// Indexing Configuration
// ============================================================================

/// Code indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
  /// Checkpoint save interval in seconds (default: 30)
  pub checkpoint_interval_secs: u64,

  /// File watcher debounce in milliseconds (default: 500)
  pub watcher_debounce_ms: u64,

  /// Maximum file size to index in bytes (default: 1MB)
  pub max_file_size: usize,

  /// Maximum chunk size in characters (default: 2000)
  pub max_chunk_chars: usize,

  /// Number of files to process in parallel during indexing (default: 4)
  /// Higher values use more memory but may be faster on SSDs.
  /// Reduce if experiencing memory pressure.
  pub parallel_files: usize,
}

impl Default for IndexConfig {
  fn default() -> Self {
    Self {
      checkpoint_interval_secs: 30,
      watcher_debounce_ms: 500,
      max_file_size: 1024 * 1024, // 1MB
      max_chunk_chars: 2000,
      parallel_files: 4,
    }
  }
}

// ============================================================================
// Daemon Configuration
// ============================================================================

/// Daemon lifecycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DaemonConfig {
  /// Idle timeout in seconds before auto-shutdown (0 = shutdown immediately when last session ends)
  /// Default: 300 (5 minutes)
  #[serde(default = "default_idle_timeout_secs")]
  pub idle_timeout_secs: u64,

  /// Session timeout in seconds - sessions without activity are considered dead
  /// Default: 1800 (30 minutes)
  #[serde(default = "default_session_timeout_secs")]
  pub session_timeout_secs: u64,

  /// Log level: "off", "error", "warn", "info", "debug", "trace"
  /// Default: "info"
  #[serde(default = "default_log_level")]
  pub log_level: String,

  /// Log file rotation: "daily", "hourly", "never"
  /// Default: "daily"
  #[serde(default = "default_log_rotation")]
  pub log_rotation: String,

  /// Maximum log file age in days (0 = keep forever)
  /// Default: 7
  #[serde(default = "default_log_retention_days")]
  pub log_retention_days: u64,
}

fn default_idle_timeout_secs() -> u64 {
  300
} // 5 minutes
fn default_session_timeout_secs() -> u64 {
  1800
} // 30 minutes
fn default_log_level() -> String {
  "info".to_string()
}
fn default_log_rotation() -> String {
  "daily".to_string()
}
fn default_log_retention_days() -> u64 {
  7
}

impl Default for DaemonConfig {
  fn default() -> Self {
    Self {
      idle_timeout_secs: default_idle_timeout_secs(),
      session_timeout_secs: default_session_timeout_secs(),
      log_level: default_log_level(),
      log_rotation: default_log_rotation(),
      log_retention_days: default_log_retention_days(),
    }
  }
}

// ============================================================================
// Hooks Configuration
// ============================================================================

/// Hook behavior configuration for automatic memory capture
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HooksConfig {
  /// Enable automatic memory capture from hooks (default: true)
  /// When false, hooks will still run but won't create memories automatically.
  /// Manual memory creation via memory_add tool is still available.
  pub enabled: bool,

  /// Enable LLM-based memory extraction (default: true)
  /// When false, uses basic summary extraction without LLM inference.
  /// This uses your Claude Code subscription.
  pub llm_extraction: bool,

  /// Enable background extraction for PreCompact/Stop hooks (default: true)
  /// When true, extraction runs asynchronously without blocking hook responses.
  /// It is not recommended to disable this unless debugging.
  pub background_extraction: bool,

  /// Enable tool observation memories (default: true)
  /// Tool observations are episodic memories capturing individual tool uses.
  pub tool_observations: bool,

  /// Enable high-priority signal detection (default: true)
  /// When true, user prompts are scanned for corrections/preferences for immediate extraction.
  pub high_priority_signals: bool,
}

impl Default for HooksConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      llm_extraction: true,
      background_extraction: true,
      tool_observations: true,
      high_priority_signals: true,
    }
  }
}

// ============================================================================
// Workspace Configuration
// ============================================================================

/// Workspace aliasing configuration for sharing memory across directories.
///
/// This is useful for:
/// - Git worktrees (auto-detected, but can be overridden)
/// - Multiple clones of the same repository
/// - Monorepo subdirectories that should share context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkspaceConfig {
  /// Alias this project to share memory with another project path.
  ///
  /// When set, this project will use the same database as the aliased path.
  /// The alias path should be the canonical path to the main repository.
  ///
  /// Example: alias = "/home/user/main-repo"
  #[serde(skip_serializing_if = "Option::is_none")]
  pub alias: Option<String>,

  /// Disable automatic worktree detection.
  ///
  /// By default, git worktrees are automatically detected and aliased to
  /// their main repository. Set this to true to treat worktrees as separate
  /// projects.
  #[serde(default)]
  pub disable_worktree_detection: bool,
}

// ============================================================================
// Documents Configuration
// ============================================================================

/// Document indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DocsConfig {
  /// Directory to watch for documents (relative to project root)
  /// When set, the file watcher will auto-index files in this directory
  #[serde(skip_serializing_if = "Option::is_none")]
  pub directory: Option<String>,

  /// File extensions to treat as documents (default: md, txt, rst, adoc)
  pub extensions: Vec<String>,

  /// Maximum document file size in bytes (default: 5MB)
  pub max_file_size: usize,
}

impl Default for DocsConfig {
  fn default() -> Self {
    Self {
      directory: None,
      extensions: vec![
        "md".to_string(),
        "txt".to_string(),
        "rst".to_string(),
        "adoc".to_string(),
        "org".to_string(),
      ],
      max_file_size: 5 * 1024 * 1024, // 5MB
    }
  }
}

// ============================================================================
// Main Configuration
// ============================================================================

/// CCEngram configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
  /// Tool filtering settings
  #[serde(default)]
  pub tools: ToolConfig,

  /// Embedding provider settings
  #[serde(default)]
  pub embedding: EmbeddingConfig,

  /// Decay and memory lifecycle settings
  #[serde(default)]
  pub decay: DecayConfig,

  /// Search defaults
  #[serde(default)]
  pub search: SearchConfig,

  /// Indexing settings
  #[serde(default)]
  pub index: IndexConfig,

  /// Document indexing settings
  #[serde(default)]
  pub docs: DocsConfig,

  /// Daemon lifecycle settings
  #[serde(default)]
  pub daemon: DaemonConfig,

  /// Workspace aliasing settings
  #[serde(default)]
  pub workspace: WorkspaceConfig,

  /// Hook behavior settings
  #[serde(default)]
  pub hooks: HooksConfig,
}

/// Tool filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ToolConfig {
  /// Tool preset to use (minimal, standard, full)
  pub preset: ToolPreset,

  /// Explicit list of enabled tools (overrides preset if set)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub enabled: Option<Vec<String>>,

  /// Tools to disable (applied after preset/enabled)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub disabled: Option<Vec<String>>,
}

impl Config {
  /// Get the final set of enabled tools after applying all rules
  pub fn enabled_tool_set(&self) -> HashSet<String> {
    // Start with preset or explicit list
    let base_tools: HashSet<String> = if let Some(ref enabled) = self.tools.enabled {
      enabled.iter().cloned().collect()
    } else {
      self.tools.preset.tools().into_iter().map(String::from).collect()
    };

    // Apply disabled filter
    if let Some(ref disabled) = self.tools.disabled {
      let disabled_set: HashSet<_> = disabled.iter().cloned().collect();
      base_tools.difference(&disabled_set).cloned().collect()
    } else {
      base_tools
    }
  }

  /// Check if a tool is enabled
  pub fn is_tool_enabled(&self, tool: &str) -> bool {
    // Internal tools are always enabled
    if INTERNAL_TOOLS.contains(&tool) {
      return true;
    }
    self.enabled_tool_set().contains(tool)
  }

  /// Load config for a project, with fallback to user config
  pub fn load_for_project(project_path: &Path) -> Self {
    // Try project-relative first
    let project_config = project_path.join(".claude").join("ccengram.toml");
    if project_config.exists()
      && let Ok(content) = std::fs::read_to_string(&project_config)
      && let Ok(config) = toml::from_str(&content)
    {
      return config;
    }

    // Fall back to user config
    if let Some(user_config_path) = Self::user_config_path()
      && user_config_path.exists()
      && let Ok(content) = std::fs::read_to_string(&user_config_path)
      && let Ok(config) = toml::from_str(&content)
    {
      return config;
    }

    // Default
    Self::default()
  }

  /// Get the user-level config path
  pub fn user_config_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CONFIG_DIR") {
      return Some(PathBuf::from(path).join("config.toml"));
    }

    if let Ok(path) = std::env::var("XDG_CONFIG_HOME") {
      return Some(PathBuf::from(path).join("ccengram").join("config.toml"));
    }

    dirs::config_dir().map(|p: PathBuf| p.join("ccengram").join("config.toml"))
  }

  /// Get the project-relative config path
  pub fn project_config_path(project_path: &Path) -> PathBuf {
    project_path.join(".claude").join("ccengram.toml")
  }

  /// Check if embedding dimensions have changed from stored dimensions
  pub fn needs_reembedding(&self, stored_dimensions: usize) -> bool {
    self.embedding.dimensions != stored_dimensions
  }

  /// Generate a default config file as a string
  pub fn generate_template(preset: ToolPreset) -> String {
    let preset_name = match preset {
      ToolPreset::Minimal => "minimal",
      ToolPreset::Standard => "standard",
      ToolPreset::Full => "full",
    };

    format!(
      r#"# CCEngram Configuration
# Place in .claude/ccengram.toml (project) or ~/.config/ccengram/config.toml (user)

# ============================================================================
# Tool Filtering
# ============================================================================

[tools]
# Preset: minimal, standard, or full
#   minimal  = explore, context (2 tools - recommended for exploration)
#   standard = explore, context, memory management, code maintenance, diagnostics (11 tools)
#   full     = all {tool_count} tools including legacy search tools
preset = "{preset_name}"

# Override preset with explicit tool list (uncomment to use):
# enabled = [
#     "memory_search",
#     "memory_add",
#     "code_search",
# ]

# Disable specific tools (applied after preset/enabled):
# disabled = ["memory_delete", "memory_supersede"]

# ============================================================================
# Embedding Provider
# ============================================================================

[embedding]
# Provider: ollama (local) or openrouter (cloud)
provider = "ollama"

# Model name
model = "qwen3-embedding"

# Embedding dimensions (must match model output)
# WARNING: Changing dimensions requires re-embedding all data!
dimensions = 4096

# Ollama server URL (for ollama provider)
ollama_url = "http://localhost:11434"

# OpenRouter API key (for openrouter provider)
# Can also be set via OPENROUTER_API_KEY env var
# openrouter_api_key = "sk-or-..."

# Context length for batch size calculation
# Should match your OLLAMA_CONTEXT_LENGTH environment variable
# Lower VRAM requires smaller context_length:
#   24 GB VRAM -> 32768 (default)
#   12 GB VRAM -> 16384
#   8 GB VRAM  -> 8192
#   6 GB VRAM  -> 4096
context_length = 32768

# Maximum batch size (auto-calculated if not set)
# Formula: min(context_length / 512, 64)
# Set explicitly to override auto-calculation
# max_batch_size = 64

# ============================================================================
# Decay & Memory Lifecycle
# ============================================================================

[decay]
# How often to run decay (hours)
decay_interval_hours = 60

# Minimum salience before archival consideration
min_salience = 0.05

# Threshold below which memories are archived
archive_threshold = 0.1

# Days without access before forced consideration
max_idle_days = 90

# Session cleanup interval (hours)
session_cleanup_hours = 6

# ============================================================================
# Search Defaults
# ============================================================================

[search]
# Default number of results
default_limit = 10

# Include superseded memories by default
include_superseded = false

# Ranking weights (should sum to 1.0)
semantic_weight = 0.5
salience_weight = 0.3
recency_weight = 0.2

# ---- Explore tool settings ----

# How many top results include full context (callers, callees, memories)
# Higher = fewer tool calls needed, but larger responses
explore_expand_top = 3

# Max results per scope in explore
explore_limit = 10

# Items per section in context (callers, callees, siblings, memories)
context_depth = 5

# Max IDs in a single batch context call
context_max_batch = 5

# Max suggestions to generate from explore results
explore_max_suggestions = 5

# ============================================================================
# Code Indexing
# ============================================================================

[index]
# Checkpoint save interval (seconds)
checkpoint_interval_secs = 30

# File watcher debounce (milliseconds)
watcher_debounce_ms = 500

# Maximum file size to index (bytes)
max_file_size = 1048576  # 1MB

# Maximum chunk size (characters)
max_chunk_chars = 2000

# Number of files to process in parallel (default: 4)
# Higher values use more memory but may be faster on SSDs
# Reduce if experiencing memory pressure
parallel_files = 4

# ============================================================================
# Document Indexing
# ============================================================================

[docs]
# Directory to watch for documents (relative to project root)
# When set, file watcher will auto-index files in this directory
# directory = "docs"

# File extensions to treat as documents
extensions = ["md", "txt", "rst", "adoc", "org"]

# Maximum document file size (bytes)
max_file_size = 5242880  # 5MB

# ============================================================================
# Daemon Lifecycle
# ============================================================================

[daemon]
# Idle timeout before auto-shutdown (seconds). 0 = shutdown immediately when
# last Claude Code session ends. Default: 300 (5 minutes)
idle_timeout_secs = 300

# Session timeout (seconds). Sessions without activity for this duration are
# considered dead and removed. Default: 1800 (30 minutes)
session_timeout_secs = 1800

# Log level: off, error, warn, info, debug, trace
# Default: info
log_level = "info"

# Log rotation: daily, hourly, never
# Default: daily
log_rotation = "daily"

# Log retention in days (0 = keep forever)
# Default: 7
log_retention_days = 7

# ============================================================================
# Workspace Aliasing
# ============================================================================

[workspace]
# Alias this project to share memory with another project.
# Useful for multiple clones of the same repo or custom workspace groupings.
# Git worktrees are automatically detected and don't need this setting.
# alias = "/home/user/main-repo"

# Disable automatic worktree detection (default: false)
# Set to true to treat git worktrees as separate projects.
# disable_worktree_detection = false

# ============================================================================
# Hook Behavior
# ============================================================================

[hooks]
# Enable automatic memory capture from hooks (default: true)
# When false, hooks still run but don't create memories automatically.
# Manual memory creation via memory_add tool is still available.
enabled = true

# Enable LLM-based memory extraction (default: true)
# When false, uses basic summary extraction without LLM inference.
# This uses your Claude Code subscription.
llm_extraction = true

# Enable background extraction for PreCompact/Stop hooks (default: true)
# When true, extraction runs asynchronously without blocking hook responses.
# It is not recommended to disable this unless debugging.
background_extraction = true

# Enable tool observation memories (default: true)
# Creates episodic memories for individual tool uses (the "tool trail").
tool_observations = true

# Enable high-priority signal detection (default: true)
# Scans user prompts for corrections/preferences for immediate extraction.
high_priority_signals = true
"#,
      tool_count = ALL_TOOLS.len(),
      preset_name = preset_name
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_preset_minimal() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Minimal,
        ..Default::default()
      },
      ..Default::default()
    };
    let tools = config.enabled_tool_set();
    assert_eq!(tools.len(), 2);
    assert!(tools.contains("explore"));
    assert!(tools.contains("context"));
  }

  #[test]
  fn test_preset_standard() {
    let config = Config::default();
    let tools = config.enabled_tool_set();
    assert_eq!(tools.len(), 11);
    assert!(tools.contains("explore"));
    assert!(tools.contains("context"));
    assert!(tools.contains("memory_add"));
    assert!(tools.contains("memory_reinforce"));
    assert!(tools.contains("memory_deemphasize"));
    assert!(tools.contains("code_index"));
    assert!(tools.contains("code_stats"));
    assert!(tools.contains("watch_start"));
    assert!(tools.contains("watch_stop"));
    assert!(tools.contains("watch_status"));
    assert!(tools.contains("project_stats"));
    assert!(!tools.contains("memory_delete")); // Not in standard
  }

  #[test]
  fn test_preset_full() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Full,
        ..Default::default()
      },
      ..Default::default()
    };
    let tools = config.enabled_tool_set();
    assert_eq!(tools.len(), ALL_TOOLS.len());
  }

  #[test]
  fn test_enabled_tools_overrides_preset() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Full,
        enabled: Some(vec!["memory_search".to_string()]),
        disabled: None,
      },
      ..Default::default()
    };
    let tools = config.enabled_tool_set();
    assert_eq!(tools.len(), 1);
    assert!(tools.contains("memory_search"));
  }

  #[test]
  fn test_disabled_tools() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Standard,
        enabled: None,
        disabled: Some(vec!["memory_add".to_string()]),
      },
      ..Default::default()
    };
    let tools = config.enabled_tool_set();
    assert!(!tools.contains("memory_add"));
    assert!(tools.contains("explore"));
  }

  #[test]
  fn test_search_config_explore_defaults() {
    let config = SearchConfig::default();
    assert_eq!(config.explore_expand_top, 3);
    assert_eq!(config.explore_limit, 10);
    assert_eq!(config.context_depth, 5);
    assert_eq!(config.context_max_batch, 5);
    assert_eq!(config.explore_max_suggestions, 5);
  }

  #[test]
  fn test_internal_tools_always_enabled() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Minimal,
        ..Default::default()
      },
      ..Default::default()
    };
    assert!(config.is_tool_enabled("hook"));
    assert!(config.is_tool_enabled("ping"));
    assert!(config.is_tool_enabled("status"));
  }

  #[test]
  fn test_load_project_config() {
    let temp = TempDir::new().unwrap();
    let claude_dir = temp.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();

    let config_content = r#"
[tools]
preset = "minimal"

[embedding]
dimensions = 4096
"#;
    std::fs::write(claude_dir.join("ccengram.toml"), config_content).unwrap();

    let config = Config::load_for_project(temp.path());
    assert_eq!(config.tools.preset, ToolPreset::Minimal);
    assert_eq!(config.embedding.dimensions, 4096);
  }

  #[test]
  fn test_load_default_when_no_config() {
    let temp = TempDir::new().unwrap();
    let config = Config::load_for_project(temp.path());
    assert_eq!(config.tools.preset, ToolPreset::Standard);
    assert_eq!(config.embedding.dimensions, 4096);
  }

  #[test]
  fn test_generate_template() {
    let template = Config::generate_template(ToolPreset::Standard);
    assert!(template.contains("preset = \"standard\""));
    assert!(template.contains("[embedding]"));
    assert!(template.contains("[decay]"));
    assert!(template.contains("[search]"));
    assert!(template.contains("[index]"));
  }

  #[test]
  fn test_toml_roundtrip() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Minimal,
        enabled: None,
        disabled: Some(vec!["memory_delete".to_string()]),
      },
      embedding: EmbeddingConfig {
        provider: EmbeddingProvider::OpenRouter,
        model: "custom-model".to_string(),
        dimensions: 1536,
        ..Default::default()
      },
      ..Default::default()
    };

    let toml_str = toml::to_string_pretty(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(parsed.tools.preset, ToolPreset::Minimal);
    assert_eq!(parsed.tools.disabled, Some(vec!["memory_delete".to_string()]));
    assert_eq!(parsed.embedding.provider, EmbeddingProvider::OpenRouter);
    assert_eq!(parsed.embedding.dimensions, 1536);
  }

  #[test]
  fn test_needs_reembedding() {
    let config = Config {
      embedding: EmbeddingConfig {
        dimensions: 1536,
        ..Default::default()
      },
      ..Default::default()
    };

    assert!(config.needs_reembedding(4096)); // Different dimensions
    assert!(!config.needs_reembedding(1536)); // Same dimensions
  }

  #[test]
  fn test_embedding_defaults() {
    let config = EmbeddingConfig::default();
    assert_eq!(config.provider, EmbeddingProvider::Ollama);
    assert_eq!(config.model, "qwen3-embedding");
    assert_eq!(config.dimensions, 4096);
    assert_eq!(config.ollama_url, "http://localhost:11434");
    assert_eq!(config.context_length, 32768);
    assert!(config.max_batch_size.is_none());
  }

  #[test]
  fn test_embedding_context_length_parsing() {
    let toml_content = r#"
[embedding]
context_length = 8192
max_batch_size = 16
"#;
    let config: Config = toml::from_str(toml_content).unwrap();
    assert_eq!(config.embedding.context_length, 8192);
    assert_eq!(config.embedding.max_batch_size, Some(16));
  }

  #[test]
  fn test_decay_defaults() {
    let config = DecayConfig::default();
    assert_eq!(config.decay_interval_hours, 60);
    assert_eq!(config.min_salience, 0.05);
    assert_eq!(config.archive_threshold, 0.1);
    assert_eq!(config.max_idle_days, 90);
  }

  #[test]
  fn test_daemon_defaults() {
    let config = DaemonConfig::default();
    assert_eq!(config.idle_timeout_secs, 300); // 5 minutes
    assert_eq!(config.session_timeout_secs, 1800); // 30 minutes
    assert_eq!(config.log_level, "info");
    assert_eq!(config.log_rotation, "daily");
    assert_eq!(config.log_retention_days, 7);
  }

  #[test]
  fn test_daemon_config_in_template() {
    let template = Config::generate_template(ToolPreset::Standard);
    assert!(template.contains("[daemon]"));
    assert!(template.contains("idle_timeout_secs"));
    assert!(template.contains("session_timeout_secs"));
    assert!(template.contains("log_level"));
  }

  #[test]
  fn test_daemon_config_roundtrip() {
    let config = Config {
      daemon: DaemonConfig {
        idle_timeout_secs: 600,
        session_timeout_secs: 3600,
        log_level: "debug".to_string(),
        log_rotation: "hourly".to_string(),
        log_retention_days: 14,
      },
      ..Default::default()
    };

    let toml_str = toml::to_string_pretty(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(parsed.daemon.idle_timeout_secs, 600);
    assert_eq!(parsed.daemon.session_timeout_secs, 3600);
    assert_eq!(parsed.daemon.log_level, "debug");
    assert_eq!(parsed.daemon.log_rotation, "hourly");
    assert_eq!(parsed.daemon.log_retention_days, 14);
  }

  #[test]
  fn test_workspace_defaults() {
    let config = WorkspaceConfig::default();
    assert!(config.alias.is_none());
    assert!(!config.disable_worktree_detection);
  }

  #[test]
  fn test_workspace_config_in_template() {
    let template = Config::generate_template(ToolPreset::Standard);
    assert!(template.contains("[workspace]"));
    assert!(template.contains("alias"));
    assert!(template.contains("disable_worktree_detection"));
  }

  #[test]
  fn test_workspace_config_roundtrip() {
    let config = Config {
      workspace: WorkspaceConfig {
        alias: Some("/home/user/main-repo".to_string()),
        disable_worktree_detection: true,
      },
      ..Default::default()
    };

    let toml_str = toml::to_string_pretty(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(parsed.workspace.alias, Some("/home/user/main-repo".to_string()));
    assert!(parsed.workspace.disable_worktree_detection);
  }

  #[test]
  fn test_workspace_config_parsing() {
    let toml_content = r#"
[workspace]
alias = "/path/to/main/repo"
disable_worktree_detection = false
"#;
    let config: Config = toml::from_str(toml_content).unwrap();
    assert_eq!(config.workspace.alias, Some("/path/to/main/repo".to_string()));
    assert!(!config.workspace.disable_worktree_detection);
  }

  #[test]
  fn test_workspace_config_optional_fields() {
    // Workspace section can be completely empty
    let toml_content = r#"
[tools]
preset = "minimal"
"#;
    let config: Config = toml::from_str(toml_content).unwrap();
    assert!(config.workspace.alias.is_none());
    assert!(!config.workspace.disable_worktree_detection);
  }

  #[test]
  fn test_hooks_defaults() {
    let config = HooksConfig::default();
    assert!(config.enabled);
    assert!(config.llm_extraction);
    assert!(config.background_extraction);
    assert!(config.tool_observations);
    assert!(config.high_priority_signals);
  }

  #[test]
  fn test_hooks_config_in_template() {
    let template = Config::generate_template(ToolPreset::Standard);
    assert!(template.contains("[hooks]"));
    assert!(template.contains("enabled = true"));
    assert!(template.contains("llm_extraction = true"));
    assert!(template.contains("background_extraction"));
    assert!(template.contains("tool_observations"));
    assert!(template.contains("high_priority_signals"));
  }

  #[test]
  fn test_hooks_config_roundtrip() {
    let config = Config {
      hooks: HooksConfig {
        enabled: false,
        llm_extraction: false,
        background_extraction: false,
        tool_observations: false,
        high_priority_signals: false,
      },
      ..Default::default()
    };

    let toml_str = toml::to_string_pretty(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert!(!parsed.hooks.enabled);
    assert!(!parsed.hooks.llm_extraction);
    assert!(!parsed.hooks.background_extraction);
    assert!(!parsed.hooks.tool_observations);
    assert!(!parsed.hooks.high_priority_signals);
  }

  #[test]
  fn test_hooks_config_parsing() {
    let toml_content = r#"
[hooks]
enabled = false
llm_extraction = false
"#;
    let config: Config = toml::from_str(toml_content).unwrap();
    assert!(!config.hooks.enabled);
    assert!(!config.hooks.llm_extraction);
    // Other fields should default to true
    assert!(config.hooks.background_extraction);
    assert!(config.hooks.tool_observations);
    assert!(config.hooks.high_priority_signals);
  }

  #[test]
  fn test_hooks_config_optional() {
    // Hooks section can be completely omitted (uses defaults)
    let toml_content = r#"
[tools]
preset = "minimal"
"#;
    let config: Config = toml::from_str(toml_content).unwrap();
    assert!(config.hooks.enabled);
    assert!(config.hooks.llm_extraction);
  }
}
