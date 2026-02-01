//! Configuration system for CCEngram with per-project overrides.
//!
//! Config priority: project-relative (.claude/ccengram.toml) > user (~/.config/ccengram/config.toml)

use std::{
  collections::HashSet,
  path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

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
pub const PRESET_MINIMAL: &[&str] = &["explore", "context"];

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

/// Characters per token estimate (for LLM token counting)
pub const CHARS_PER_TOKEN: usize = 4;
/// Safety margin for per-text truncation. Embedding models typically handle ~8K tokens per text,
/// but context_length defaults to 32K (for batch sizing). 0.125 gives us ~16K chars max per text.
pub const TOKEN_SAFETY_MARGIN: f32 = 0.5;

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
  Ollama,
  #[default]
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

  /// Task instruction to prepend to queries for retrieval.
  ///
  /// Some embedding models (like qwen3-embedding) produce better retrieval
  /// results when queries are prefixed with a task instruction.
  ///
  /// When set, queries are formatted as:
  ///   `Instruct: {query_instruction}\nQuery:{query}`
  ///
  /// When None or empty, queries are embedded as-is.
  /// Documents are always embedded without instruction regardless of this setting.
  ///
  /// Default: A code search instruction for qwen3-embedding.
  /// Set to empty string "" to disable instruction prefixing.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub query_instruction: Option<String>,
}

/// Default query instruction for qwen3-embedding.
/// Set query_instruction to empty string "" or None to disable.
pub const DEFAULT_QUERY_INSTRUCTION: &str =
  "Given a code search query, retrieve relevant code snippets and documentation that match the query";

impl Default for EmbeddingConfig {
  fn default() -> Self {
    Self {
      provider: EmbeddingProvider::OpenRouter,
      model: "qwen/qwen3-embedding-8b".to_string(),
      dimensions: 4096,
      ollama_url: "http://localhost:11434".to_string(),
      openrouter_api_key: None,
      context_length: 32768,
      max_batch_size: None, // Auto-calculated from context_length
      query_instruction: Some(DEFAULT_QUERY_INSTRUCTION.to_string()),
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

  // ---- Query embedding cache settings ----
  /// Embedding cache size for query embeddings (default: 1000)
  #[serde(default = "default_embedding_cache_size")]
  pub embedding_cache_size: u64,

  /// Embedding cache TTL in seconds (default: 300)
  #[serde(default = "default_embedding_cache_ttl_secs")]
  pub embedding_cache_ttl_secs: u64,
}

fn default_embedding_cache_size() -> u64 {
  1000
}
fn default_embedding_cache_ttl_secs() -> u64 {
  300
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
      embedding_cache_size: default_embedding_cache_size(),
      embedding_cache_ttl_secs: default_embedding_cache_ttl_secs(),
    }
  }
}

// ============================================================================
// Indexing Configuration
// ============================================================================

/// Startup scan mode determines what changes to detect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanMode {
  /// Only detect and remove deleted files (fastest)
  DeletedOnly,
  /// Detect deleted and new files
  DeletedAndNew,
  /// Full reconciliation: deleted, new, and modified
  #[default]
  Full,
}

impl std::str::FromStr for ScanMode {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "deleted_only" | "deletedonly" => Ok(ScanMode::DeletedOnly),
      "deleted_and_new" | "deletedandnew" => Ok(ScanMode::DeletedAndNew),
      "full" => Ok(ScanMode::Full),
      _ => Err(format!("Invalid scan mode: {}", s)),
    }
  }
}

/// Code indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
  /// Checkpoint save interval in seconds (default: 30)
  pub checkpoint_interval_secs: u64,

  /// File watcher debounce in milliseconds (default: 1000)
  pub watcher_debounce_ms: u64,

  /// Maximum file size to index in bytes (default: 1MB)
  pub max_file_size: usize,

  /// Maximum chunk size in characters (default: 2000)
  pub max_chunk_chars: usize,

  /// Number of files to process in parallel during indexing (default: 32)
  /// Higher values improve throughput by parallelizing file preparation
  /// and DB queries. Reduce if experiencing memory pressure.
  pub parallel_files: usize,

  /// Number of files per indexing batch (default: 512)
  /// Larger batches allow more concurrent embedding API requests,
  /// improving throughput with cloud providers like OpenRouter.
  /// Reduce if experiencing memory pressure on very large repos.
  pub index_batch_size: usize,

  // ---- Startup Scan Settings ----
  /// Enable startup scan when watcher starts (default: true)
  /// The scan reconciles the database with filesystem state to detect
  /// files that were added, modified, or deleted while the watcher was stopped.
  pub startup_scan: bool,

  /// Startup scan mode (default: full)
  /// - deleted_only: Only detect and remove deleted files (fastest)
  /// - deleted_and_new: Detect deleted and new files
  /// - full: Full reconciliation including modified file detection
  pub startup_scan_mode: ScanMode,

  /// Block watcher startup until scan completes (default: false)
  /// When false, watcher starts immediately and scan runs in background.
  /// When true, watcher waits for scan to complete before processing events.
  pub startup_scan_blocking: bool,

  /// Timeout for startup scan in seconds (default: 300)
  /// Set to 0 for no timeout.
  pub startup_scan_timeout_secs: u64,

  // ---- Watcher Settings ----
  /// Watcher poll interval in seconds (default: 2)
  #[serde(default = "default_watcher_poll_secs")]
  pub watcher_poll_secs: u64,

  /// Maximum files to cache content for watcher (default: 1000)
  #[serde(default = "default_content_cache_size")]
  pub content_cache_size: usize,

  /// Maximum file size to cache in bytes (default: 512KB)
  #[serde(default = "default_max_cached_file_size")]
  pub max_cached_file_size: usize,

  /// Number of files to batch from watcher before processing (default: 32)
  /// Larger batches improve embedding efficiency but increase latency.
  #[serde(default = "default_watcher_batch_size")]
  pub watcher_batch_size: usize,

  /// Timeout in ms to flush watcher batch even if not full (default: 100)
  /// Balances latency vs batching efficiency for incremental updates.
  #[serde(default = "default_watcher_batch_timeout_ms")]
  pub watcher_batch_timeout_ms: u64,

  // ---- Pipeline Tuning (advanced) ----
  /// Scanner buffer size for bulk indexing (default: 256)
  #[serde(default = "default_pipeline_scanner_buffer")]
  pub pipeline_scanner_buffer: usize,

  /// Reader buffer size (default: 128)
  #[serde(default = "default_pipeline_reader_buffer")]
  pub pipeline_reader_buffer: usize,

  /// Parser buffer size (default: 256)
  #[serde(default = "default_pipeline_parser_buffer")]
  pub pipeline_parser_buffer: usize,

  /// Embedder buffer size (default: 64)
  #[serde(default = "default_pipeline_embedder_buffer")]
  pub pipeline_embedder_buffer: usize,

  /// Embedding batch timeout in ms (default: 50)
  #[serde(default = "default_pipeline_embedding_timeout_ms")]
  pub pipeline_embedding_timeout_ms: u64,

  /// Chunks before DB flush (default: 500)
  #[serde(default = "default_pipeline_db_flush_count")]
  pub pipeline_db_flush_count: usize,

  /// DB flush timeout in ms (default: 1000)
  #[serde(default = "default_pipeline_db_flush_timeout_ms")]
  pub pipeline_db_flush_timeout_ms: u64,

  /// Maximum pending embedding batches (default: 64)
  /// Limits memory usage when embedding API is slow/rate-limited.
  /// With batch_size=512 and ~2KB/chunk, 64 batches = ~65MB worst case.
  #[serde(default = "default_pipeline_max_pending_batches")]
  pub pipeline_max_pending_batches: usize,

  /// Reader workers (default: 16)
  #[serde(default = "default_pipeline_reader_workers")]
  pub pipeline_reader_workers: usize,

  /// Parser workers, 0 = num_cpus (default: 0)
  #[serde(default = "default_pipeline_parser_workers")]
  pub pipeline_parser_workers: usize,
}

fn default_watcher_poll_secs() -> u64 {
  2
}
fn default_content_cache_size() -> usize {
  1000
}
fn default_max_cached_file_size() -> usize {
  512 * 1024 // 512KB
}
fn default_watcher_batch_size() -> usize {
  32
}
fn default_watcher_batch_timeout_ms() -> u64 {
  100
}
fn default_pipeline_scanner_buffer() -> usize {
  256
}
fn default_pipeline_reader_buffer() -> usize {
  128
}
fn default_pipeline_parser_buffer() -> usize {
  256
}
fn default_pipeline_embedder_buffer() -> usize {
  1024
}
fn default_pipeline_embedding_timeout_ms() -> u64 {
  250
}
fn default_pipeline_db_flush_count() -> usize {
  500
}
fn default_pipeline_db_flush_timeout_ms() -> u64 {
  1000
}
fn default_pipeline_max_pending_batches() -> usize {
  64
}
fn default_pipeline_reader_workers() -> usize {
  16
}
fn default_pipeline_parser_workers() -> usize {
  0 // 0 means use num_cpus
}

impl Default for IndexConfig {
  fn default() -> Self {
    Self {
      checkpoint_interval_secs: 30,
      watcher_debounce_ms: 1000,
      max_file_size: 1024 * 1024, // 1MB
      max_chunk_chars: 2000,
      parallel_files: 32,
      index_batch_size: 512,
      startup_scan: true,
      startup_scan_mode: ScanMode::Full,
      startup_scan_blocking: false,
      startup_scan_timeout_secs: 300,
      watcher_poll_secs: default_watcher_poll_secs(),
      content_cache_size: default_content_cache_size(),
      max_cached_file_size: default_max_cached_file_size(),
      watcher_batch_size: default_watcher_batch_size(),
      watcher_batch_timeout_ms: default_watcher_batch_timeout_ms(),
      pipeline_scanner_buffer: default_pipeline_scanner_buffer(),
      pipeline_reader_buffer: default_pipeline_reader_buffer(),
      pipeline_parser_buffer: default_pipeline_parser_buffer(),
      pipeline_embedder_buffer: default_pipeline_embedder_buffer(),
      pipeline_embedding_timeout_ms: default_pipeline_embedding_timeout_ms(),
      pipeline_db_flush_count: default_pipeline_db_flush_count(),
      pipeline_db_flush_timeout_ms: default_pipeline_db_flush_timeout_ms(),
      pipeline_max_pending_batches: default_pipeline_max_pending_batches(),
      pipeline_reader_workers: default_pipeline_reader_workers(),
      pipeline_parser_workers: default_pipeline_parser_workers(),
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
  /// Default: 300 (5 minutes)
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

  /// Idle check interval in seconds (default: 30)
  /// How often the scheduler checks if the daemon should shutdown due to inactivity.
  #[serde(default = "default_idle_check_interval_secs")]
  pub idle_check_interval_secs: u64,
}

fn default_idle_timeout_secs() -> u64 {
  300
} // 5 minutes
fn default_session_timeout_secs() -> u64 {
  300
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
fn default_idle_check_interval_secs() -> u64 {
  30
}

impl Default for DaemonConfig {
  fn default() -> Self {
    Self {
      idle_timeout_secs: default_idle_timeout_secs(),
      session_timeout_secs: default_session_timeout_secs(),
      log_level: default_log_level(),
      log_rotation: default_log_rotation(),
      log_retention_days: default_log_retention_days(),
      idle_check_interval_secs: default_idle_check_interval_secs(),
    }
  }
}

// ============================================================================
// Database Configuration
// ============================================================================

/// LanceDB cache and connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
  /// Index cache size in MB (default: 256)
  /// Controls memory used for vector/scalar index data.
  /// LanceDB default is 6 GB per table - we share across all tables.
  pub index_cache_mb: usize,

  /// Metadata cache size in MB (default: 64)
  /// Controls memory used for dataset manifests, schemas, statistics.
  /// LanceDB default is 1 GB per table - we share across all tables.
  pub metadata_cache_mb: usize,

  /// Log cache stats periodically during indexing (default: false)
  /// Useful for debugging memory usage.
  pub log_cache_stats: bool,
}

impl Default for DatabaseConfig {
  fn default() -> Self {
    Self {
      index_cache_mb: 256,
      metadata_cache_mb: 64,
      log_cache_stats: false,
    }
  }
}

// ============================================================================
// Daemon-Level Settings (for passing to ProjectActors)
// ============================================================================

/// Daemon-level settings that ProjectActors need access to.
///
/// These settings are read from the global config at daemon startup and should
/// NOT be overridden by project-level configs. They are passed from the daemon
/// through ProjectRouter to each ProjectActor.
///
/// This separation ensures that:
/// - Embedding settings are consistent across all projects (shared provider)
/// - Database cache settings are daemon-wide
///
/// Note: Hook settings are NOT included here - they are project-specific
/// and loaded via Config::load_for_project() which merges global + project configs.
#[derive(Debug, Clone)]
pub struct DaemonSettings {
  /// Maximum batch size for embedding requests (from embedding.max_batch_size)
  pub embedding_batch_size: Option<usize>,
  /// Context length for batch sizing (from embedding.context_length)
  pub embedding_context_length: usize,
  /// Whether to log cache stats during indexing (from database.log_cache_stats)
  pub log_cache_stats: bool,
}

impl DaemonSettings {
  /// Create DaemonSettings from a global Config
  pub fn from_config(config: &Config) -> Self {
    Self {
      embedding_batch_size: config.embedding.max_batch_size,
      embedding_context_length: config.embedding.context_length,
      log_cache_stats: config.database.log_cache_stats,
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
      enabled: false,
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

  /// Database cache settings
  #[serde(default)]
  pub database: DatabaseConfig,
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

  pub async fn load_global() -> Self {
    if let Some(user_config_path) = Self::user_config_path()
      && user_config_path.exists()
      && let Ok(content) = tokio::fs::read_to_string(&user_config_path).await
      && let Ok(config) = toml::from_str(&content)
    {
      config
    } else {
      Self::default()
    }
  }

  /// Load config for a project, merging with global config.
  ///
  /// The merge strategy is:
  /// 1. Load global config as the base
  /// 2. Load project config and overlay it on top
  /// 3. Only fields actually present in the project config override global values
  ///
  /// This allows sparse project configs that only override specific settings.
  pub async fn load_for_project(project_path: &Path) -> Self {
    let project_config_path = Self::project_config_path(project_path);

    // If no project config, just use global
    if !project_config_path.exists() {
      return Self::load_global().await;
    }

    // Load project config content
    let Ok(project_content) = tokio::fs::read_to_string(&project_config_path).await else {
      return Self::load_global().await;
    };

    // Trim whitespace from project config before parsing
    let project_content = project_content.trim();

    // Parse project config as TOML value for merging
    let Ok(project_toml) = toml::from_str::<toml::Value>(project_content) else {
      return Self::load_global().await;
    };

    // Load global config and serialize to TOML value
    let global_config = Self::load_global().await;
    let Ok(global_str) = toml::to_string(&global_config) else {
      return toml::from_str(project_content).unwrap_or(global_config);
    };
    let Ok(global_toml) = toml::from_str::<toml::Value>(&global_str) else {
      return toml::from_str(project_content).unwrap_or(global_config);
    };

    // Merge project on top of global
    let merged_toml = Self::merge_toml(global_toml, project_toml);

    // Deserialize merged config
    merged_toml.try_into().unwrap_or(global_config)
  }

  /// Recursively merge two TOML values, with `overlay` taking precedence.
  ///
  /// For tables, this merges keys recursively. For other types, overlay replaces base.
  fn merge_toml(base: toml::Value, overlay: toml::Value) -> toml::Value {
    match (base, overlay) {
      // Both are tables: merge recursively
      (toml::Value::Table(mut base_table), toml::Value::Table(overlay_table)) => {
        for (key, overlay_value) in overlay_table {
          let merged_value = if let Some(base_value) = base_table.remove(&key) {
            Self::merge_toml(base_value, overlay_value)
          } else {
            overlay_value
          };
          base_table.insert(key, merged_value);
        }
        toml::Value::Table(base_table)
      }
      // Overlay is not a table (or base isn't): overlay wins
      (_, overlay) => overlay,
    }
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

  /// Generate a project-level config file (excludes daemon-only sections)
  ///
  /// Project configs should NOT include `[embedding]`, `[daemon]`, or `[database]`
  /// because these are only read at daemon startup and shared across all projects.
  /// `[hooks]` IS supported at project level for per-project memory capture settings.
  pub fn generate_project_template(preset: ToolPreset) -> String {
    let preset_name = match preset {
      ToolPreset::Minimal => "minimal",
      ToolPreset::Standard => "standard",
      ToolPreset::Full => "full",
    };

    format!(
      r#"# CCEngram Project Configuration
# Place in .claude/ccengram.toml
#
# NOTE: This is a project-level config. The following are daemon-level
# and should be configured in ~/.config/ccengram/config.toml instead:
#   [embedding]  - Embedding provider (shared across all projects)
#   [daemon]     - Daemon lifecycle settings
#   [database]   - Database cache settings
#   decay.decay_interval_hours, decay.session_cleanup_hours, decay.max_session_age_hours

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
# Decay & Memory Lifecycle (project-level settings only)
# ============================================================================
# NOTE: Timing settings (decay_interval_hours, session_cleanup_hours,
# max_session_age_hours) are daemon-level and configured in global config.

[decay]
# Salience threshold below which memories are archived
archive_threshold = 0.1

# Days without access before forced decay consideration
max_idle_days = 90

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

# ---- Query embedding cache settings ----

# Embedding cache size for query embeddings
embedding_cache_size = 1000

# Embedding cache TTL in seconds
embedding_cache_ttl_secs = 300

# ============================================================================
# Code Indexing
# ============================================================================

[index]
# Checkpoint save interval (seconds)
checkpoint_interval_secs = 30

# File watcher debounce (milliseconds)
watcher_debounce_ms = 1000

# Maximum file size to index (bytes)
max_file_size = 1048576  # 1MB

# Maximum chunk size (characters)
max_chunk_chars = 2000

# Number of files to process in parallel (default: 32)
# Higher values improve throughput by parallelizing file preparation
# Reduce if experiencing memory pressure
parallel_files = 32

# Number of files per indexing batch (default: 512)
# Larger batches allow more concurrent embedding API requests
index_batch_size = 512

# ---- Startup Scan Settings ----

# Enable startup scan when watcher starts (default: true)
# Reconciles database with filesystem state to detect files added/modified/deleted
# while the watcher was stopped. Only runs if project was previously indexed.
startup_scan = true

# Startup scan mode (default: full)
#   deleted_only   = Only detect and remove deleted files (fastest)
#   deleted_and_new = Detect deleted and new files
#   full           = Full reconciliation including modified file detection
startup_scan_mode = "full"

# Block watcher startup until scan completes (default: false)
# When false, watcher starts immediately and scan runs in background.
# When true, watcher waits for scan to complete before processing events.
# Note: Searches will block until scan completes regardless of this setting.
startup_scan_blocking = false

# Timeout for startup scan in seconds (default: 300)
# Set to 0 for no timeout.
startup_scan_timeout_secs = 300

# ---- Watcher Settings ----

# Watcher poll interval in seconds
watcher_poll_secs = 2

# Maximum files to cache content for watcher
content_cache_size = 1000

# Maximum file size to cache in bytes
max_cached_file_size = 524288  # 512KB

# Number of files to batch before processing (default: 32)
# Larger batches improve embedding API efficiency but increase latency.
watcher_batch_size = 32

# Timeout to flush watcher batch even if not full, in milliseconds (default: 100)
# Balances latency vs batching efficiency for incremental updates.
watcher_batch_timeout_ms = 100

# ---- Pipeline Tuning (advanced) ----

# Scanner buffer size for bulk indexing
pipeline_scanner_buffer = 256

# Reader buffer size
pipeline_reader_buffer = 128

# Parser buffer size
pipeline_parser_buffer = 256

# Embedder buffer size
pipeline_embedder_buffer = 1024

# Embedding batch timeout in ms
pipeline_embedding_timeout_ms = 250

# Chunks before DB flush
pipeline_db_flush_count = 500

# DB flush timeout in ms
pipeline_db_flush_timeout_ms = 1000

# Maximum pending embedding batches
pipeline_max_pending_batches = 64

# Reader workers
pipeline_reader_workers = 16

# Parser workers (0 = num_cpus)
pipeline_parser_workers = 0

# ============================================================================
# Document Indexing
# ============================================================================

[docs]
# Directory to watch for documents (relative to project root)
# When set, file watcher will auto-index files in this directory
directory = "docs"

# File extensions to treat as documents
extensions = ["md", "txt", "rst", "adoc", "org"]

# Maximum document file size (bytes)
max_file_size = 5242880  # 5MB

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
# Hook Behavior (Automatic Memory Creation)
# ============================================================================

[hooks]
# Enable automatic memory capture from hooks (default: false)
# When true, memories are automatically created from your Claude Code sessions.
# When false, hooks still run but don't create memories automatically.
# Manual memory creation via memory_add tool is still available.
enabled = false

# Enable LLM-based memory extraction (default: true)
# When false, uses basic summary extraction without LLM inference.
# This uses your Claude Code subscription.
llm_extraction = true

# Enable background extraction for PreCompact/Stop hooks (default: true)
# When true, extraction runs asynchronously without blocking hook responses.
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

  /// Generate a full user-level config file (includes all sections)
  ///
  /// User configs include daemon-level sections like `[embedding]`, `[daemon]`,
  /// and `[hooks]` which are only read at daemon startup.
  pub fn generate_template(preset: ToolPreset) -> String {
    let preset_name = match preset {
      ToolPreset::Minimal => "minimal",
      ToolPreset::Standard => "standard",
      ToolPreset::Full => "full",
    };

    format!(
      r#"# CCEngram User Configuration
# Place in ~/.config/ccengram/config.toml
#
# This file includes all settings. For project-specific overrides,
# create .claude/ccengram.toml with only the project-level sections.

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
provider = "openrouter"

# Model name
model = "qwen/qwen3-embedding-8b" # would be "qwen3-embedding" for ollama

# Embedding dimensions (must match model output)
# WARNING: Changing dimensions requires re-embedding all data!
dimensions = 4096

# Ollama server URL (for ollama provider)
# ollama_url = "http://localhost:11434"

# OpenRouter API key (for openrouter provider)
# Can also be set via OPENROUTER_API_KEY env var
# openrouter_api_key = "sk-or-..."

# Context length for batch size calculation
# for OpenRouter, set to the context length of your model
# for Ollama, this should match your OLLAMA_CONTEXT_LENGTH environment variable
# Lower VRAM requires smaller context_length:
#   24 GB VRAM -> 32768 (default)
#   12 GB VRAM -> 16384
#   8 GB VRAM  -> 8192
#   6 GB VRAM  -> 4096
context_length = 32768

# Maximum batch size (default: auto-calculated from context_length, or 512 for openrouter)
# max_batch_size = 512

# Query instruction for retrieval-optimized embeddings.
# Some models (like qwen3-embedding) produce better results when queries are
# prefixed with a task instruction. Documents are always embedded as-is.
#
# Default: A code search instruction for qwen3-embedding.
# Set to empty string "" to disable instruction prefixing.
query_instruction = "Given a code search query, retrieve relevant code snippets and documentation that match the query"

# ============================================================================
# Decay & Memory Lifecycle
# ============================================================================

[decay]
# How often to run decay (hours)
decay_interval_hours = 60

# Threshold below which memories are archived
archive_threshold = 0.1

# Days without access before forced consideration
max_idle_days = 90

# Session cleanup interval (hours)
session_cleanup_hours = 6

# Maximum session age before cleanup (hours)
max_session_age_hours = 6

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

# ---- Query embedding cache settings ----

# Embedding cache size for query embeddings
embedding_cache_size = 1000

# Embedding cache TTL in seconds
embedding_cache_ttl_secs = 300

# ============================================================================
# Code Indexing
# ============================================================================

[index]
# Checkpoint save interval (seconds)
checkpoint_interval_secs = 30

# File watcher debounce (milliseconds)
watcher_debounce_ms = 1000

# Maximum file size to index (bytes)
max_file_size = 1048576  # 1MB

# Maximum chunk size (characters)
max_chunk_chars = 2000

# Number of files to process in parallel (default: 32)
# Higher values improve throughput by parallelizing file preparation
# Reduce if experiencing memory pressure
parallel_files = 32

# Number of files per indexing batch (default: 512)
# Larger batches allow more concurrent embedding API requests
index_batch_size = 512

# ---- Startup Scan Settings ----

# Enable startup scan when watcher starts (default: true)
# Reconciles database with filesystem state to detect files added/modified/deleted
# while the watcher was stopped. Only runs if project was previously indexed.
startup_scan = true

# Startup scan mode (default: full)
#   deleted_only   = Only detect and remove deleted files (fastest)
#   deleted_and_new = Detect deleted and new files
#   full           = Full reconciliation including modified file detection
startup_scan_mode = "full"

# Block watcher startup until scan completes (default: false)
# When false, watcher starts immediately and scan runs in background.
# When true, watcher waits for scan to complete before processing events.
# Note: Searches will block until scan completes regardless of this setting.
startup_scan_blocking = false

# Timeout for startup scan in seconds (default: 300)
# Set to 0 for no timeout.
startup_scan_timeout_secs = 300

# ---- Watcher Settings ----

# Watcher poll interval in seconds
watcher_poll_secs = 2

# Maximum files to cache content for watcher
content_cache_size = 1000

# Maximum file size to cache in bytes
max_cached_file_size = 524288  # 512KB

# Number of files to batch before processing (default: 32)
# Larger batches improve embedding API efficiency but increase latency.
watcher_batch_size = 32

# Timeout to flush watcher batch even if not full, in milliseconds (default: 100)
# Balances latency vs batching efficiency for incremental updates.
watcher_batch_timeout_ms = 100

# ---- Pipeline Tuning (advanced) ----

# Scanner buffer size for bulk indexing
pipeline_scanner_buffer = 256

# Reader buffer size
pipeline_reader_buffer = 128

# Parser buffer size
pipeline_parser_buffer = 256

# Embedder buffer size
pipeline_embedder_buffer = 1024

# Embedding batch timeout in ms
pipeline_embedding_timeout_ms = 250

# Chunks before DB flush
pipeline_db_flush_count = 500

# DB flush timeout in ms
pipeline_db_flush_timeout_ms = 1000

# Maximum pending embedding batches
pipeline_max_pending_batches = 64

# Reader workers
pipeline_reader_workers = 16

# Parser workers (0 = num_cpus)
pipeline_parser_workers = 0

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
# considered dead and removed. Default: 300 (5 minutes)
session_timeout_secs = 300

# Log level: off, error, warn, info, debug, trace
# Default: info
log_level = "info"

# Log rotation: daily, hourly, never
# Default: daily
log_rotation = "daily"

# Log retention in days (0 = keep forever)
# Default: 7
log_retention_days = 7

# Idle check interval in seconds (default: 30)
# How often the scheduler checks if the daemon should shutdown due to inactivity.
idle_check_interval_secs = 30

# ============================================================================
# Database Cache Settings
# ============================================================================

[database]
# Index cache size in MB (default: 256)
# Controls memory used for vector/scalar index data.
# LanceDB default is 6 GB per table - we share a single cache across all tables.
# Increase for better query performance, decrease for lower memory usage.
index_cache_mb = 256

# Metadata cache size in MB (default: 64)
# Controls memory used for dataset manifests, schemas, statistics.
# LanceDB default is 1 GB per table - we share a single cache across all tables.
metadata_cache_mb = 64

# Log cache stats periodically during indexing (default: false)
# Useful for debugging memory usage and cache hit rates.
log_cache_stats = false
"#,
      tool_count = ALL_TOOLS.len(),
      preset_name = preset_name
    )
  }
}

#[cfg(test)]
mod tests {
  use tempfile::TempDir;

  use super::*;

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

  #[tokio::test]
  async fn test_load_project_config() {
    let temp = TempDir::new().unwrap();
    let claude_dir = temp.path().join(".claude");
    tokio::fs::create_dir_all(&claude_dir).await.unwrap();

    let config_content = r#"[tools]
preset = "minimal"

[embedding]
dimensions = 4096
"#;
    let config_path = claude_dir.join("ccengram.toml");
    tokio::fs::write(&config_path, config_content).await.unwrap();

    // Verify file was written
    assert!(config_path.exists(), "Config file should exist at {:?}", config_path);

    let config = Config::load_for_project(temp.path()).await;
    assert_eq!(config.tools.preset, ToolPreset::Minimal, "preset should be minimal");
    assert_eq!(config.embedding.dimensions, 4096, "dimensions should be 4096");
  }

  #[tokio::test]
  async fn test_load_default_when_no_config() {
    let temp = TempDir::new().unwrap();
    let config = Config::load_for_project(temp.path()).await;
    assert_eq!(config.tools.preset, ToolPreset::Standard);
    assert_eq!(config.embedding.dimensions, 4096);
  }

  #[tokio::test]
  async fn test_project_config_merges_with_global() {
    let temp = TempDir::new().unwrap();
    let claude_dir = temp.path().join(".claude");
    tokio::fs::create_dir_all(&claude_dir).await.unwrap();

    // Project config only sets tools.preset and search.default_limit
    // Other values should come from global/defaults
    let config_content = r#"[tools]
preset = "minimal"

[search]
default_limit = 25
"#;
    tokio::fs::write(claude_dir.join("ccengram.toml"), config_content)
      .await
      .unwrap();

    let config = Config::load_for_project(temp.path()).await;

    // Project overrides should be applied
    assert_eq!(
      config.tools.preset,
      ToolPreset::Minimal,
      "tools.preset should be overridden"
    );
    assert_eq!(
      config.search.default_limit, 25,
      "search.default_limit should be overridden"
    );

    // Values not in project config should come from global/defaults
    assert_eq!(
      config.embedding.dimensions, 4096,
      "embedding.dimensions should use default"
    );
    assert_eq!(
      config.search.semantic_weight, 0.5,
      "search.semantic_weight should use default"
    );
    assert_eq!(
      config.index.max_file_size,
      1024 * 1024,
      "index.max_file_size should use default"
    );
    assert_eq!(
      config.decay.archive_threshold, 0.1,
      "decay.archive_threshold should use default"
    );
  }

  #[test]
  fn test_merge_toml_tables() {
    use toml::Value;

    let base: Value = toml::from_str(
      r#"
[section]
a = 1
b = 2
"#,
    )
    .unwrap();

    let overlay: Value = toml::from_str(
      r#"
[section]
b = 99
c = 3
"#,
    )
    .unwrap();

    let merged = Config::merge_toml(base, overlay);
    let table = merged.as_table().unwrap();
    let section = table.get("section").unwrap().as_table().unwrap();

    assert_eq!(section.get("a").unwrap().as_integer(), Some(1), "base value preserved");
    assert_eq!(section.get("b").unwrap().as_integer(), Some(99), "overlay value wins");
    assert_eq!(section.get("c").unwrap().as_integer(), Some(3), "overlay adds new key");
  }

  #[test]
  fn test_config_to_toml_value() {
    // Ensure Config can be serialized to toml::Value
    let config = Config::default();
    let toml_value = toml::Value::try_from(&config).expect("Config should serialize to toml::Value");
    assert!(toml_value.is_table(), "Config should serialize to a table");

    // Verify we can merge with it
    let overlay: toml::Value = toml::from_str(
      r#"[tools]
preset = "minimal"
"#,
    )
    .unwrap();

    let merged = Config::merge_toml(toml_value, overlay);

    // Verify the merge worked
    let merged_config: Config = merged.try_into().expect("Merged should deserialize");
    assert_eq!(
      merged_config.tools.preset,
      ToolPreset::Minimal,
      "preset should be overridden"
    );
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
        idle_check_interval_secs: 60,
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
  fn test_workspace_config_in_template() {
    let template = Config::generate_project_template(ToolPreset::Standard);
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
  fn test_hooks_config_roundtrip() {
    let config = Config {
      hooks: HooksConfig {
        enabled: true,
        llm_extraction: false,
        background_extraction: false,
        tool_observations: false,
        high_priority_signals: false,
      },
      ..Default::default()
    };

    let toml_str = toml::to_string_pretty(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert!(parsed.hooks.enabled);
    assert!(!parsed.hooks.llm_extraction);
    assert!(!parsed.hooks.background_extraction);
    assert!(!parsed.hooks.tool_observations);
    assert!(!parsed.hooks.high_priority_signals);
  }

  #[test]
  fn test_hooks_config_parsing() {
    let toml_content = r#"
[hooks]
enabled = true
llm_extraction = false
"#;
    let config: Config = toml::from_str(toml_content).unwrap();
    assert!(config.hooks.enabled);
    assert!(!config.hooks.llm_extraction);
    // Other fields should default to true
    assert!(config.hooks.background_extraction);
    assert!(config.hooks.tool_observations);
    assert!(config.hooks.high_priority_signals);
  }

  #[test]
  fn test_project_config_overrides_global_hooks_enabled() {
    // Regression test: project config should override global config
    // Bug: when global had hooks.enabled=false and project had hooks.enabled=true,
    // the global value was incorrectly winning.

    let global_content = r#"
[hooks]
enabled = false
"#;

    let project_content = r#"
[hooks]
enabled = true
"#;

    let global_toml: toml::Value = toml::from_str(global_content).unwrap();
    let project_toml: toml::Value = toml::from_str(project_content).unwrap();

    let merged = Config::merge_toml(global_toml, project_toml);
    let merged_config: Config = merged.try_into().unwrap();

    assert!(
      merged_config.hooks.enabled,
      "project config hooks.enabled=true should override global hooks.enabled=false"
    );
  }

  #[test]
  fn test_project_config_overrides_global_with_full_config() {
    // Test with a more realistic scenario: global config has many fields set,
    // project config only overrides one field in a section.

    // Simulate serialized global config (as if loaded from file and re-serialized)
    let global_config = Config {
      hooks: HooksConfig {
        enabled: false, // Global disables hooks
        llm_extraction: true,
        background_extraction: true,
        tool_observations: true,
        high_priority_signals: true,
      },
      ..Default::default()
    };
    let global_str = toml::to_string(&global_config).unwrap();
    let global_toml: toml::Value = toml::from_str(&global_str).unwrap();

    // Project config only sets hooks.enabled = true (sparse config)
    let project_content = r#"
[hooks]
enabled = true
"#;
    let project_toml: toml::Value = toml::from_str(project_content).unwrap();

    let merged = Config::merge_toml(global_toml, project_toml);
    let merged_config: Config = merged.try_into().unwrap();

    assert!(
      merged_config.hooks.enabled,
      "project hooks.enabled=true should override global hooks.enabled=false"
    );
    // Other hooks fields should retain global values
    assert!(merged_config.hooks.llm_extraction);
    assert!(merged_config.hooks.background_extraction);
  }

  #[tokio::test]
  async fn test_load_for_project_overrides_global_hooks() {
    // End-to-end test: project config file should override global config file
    use tempfile::TempDir;

    // Create a temp directory to act as both config home and project
    let temp = TempDir::new().unwrap();
    let config_dir = temp.path().join("config");
    let project_dir = temp.path().join("project");
    let claude_dir = project_dir.join(".claude");

    tokio::fs::create_dir_all(&config_dir).await.unwrap();
    tokio::fs::create_dir_all(&claude_dir).await.unwrap();

    // Global config: hooks disabled
    let global_config = r#"
[hooks]
enabled = false
"#;
    tokio::fs::write(config_dir.join("config.toml"), global_config)
      .await
      .unwrap();

    // Project config: hooks enabled
    let project_config = r#"
[hooks]
enabled = true
"#;
    tokio::fs::write(claude_dir.join("ccengram.toml"), project_config)
      .await
      .unwrap();

    // Set CONFIG_DIR env var to use our temp config
    // SAFETY: This test runs single-threaded and we clean up immediately after
    unsafe {
      std::env::set_var("CONFIG_DIR", &config_dir);
    }

    let config = Config::load_for_project(&project_dir).await;

    // Clean up env var
    unsafe {
      std::env::remove_var("CONFIG_DIR");
    }

    assert!(
      config.hooks.enabled,
      "project hooks.enabled=true should override global hooks.enabled=false"
    );
  }
}
