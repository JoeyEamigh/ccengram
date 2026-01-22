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
  // Code tools
  "code_search",
  "code_index",
  "code_list",
  "code_import_chunk",
  "code_stats",
  // Watch tools
  "watch_start",
  "watch_stop",
  "watch_status",
  // Document tools
  "docs_search",
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

/// Minimal preset: just search tools
pub const PRESET_MINIMAL: &[&str] = &["memory_search", "code_search", "docs_search"];

/// Standard preset: recommended daily driver set
pub const PRESET_STANDARD: &[&str] = &[
  "memory_search",
  "memory_add",
  "memory_reinforce",
  "memory_deemphasize",
  "code_search",
  "docs_search",
  "memory_timeline",
  "entity_top",
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

  /// Embedding dimensions (e.g., 4096, 1536, 768)
  pub dimensions: usize,

  /// Ollama server URL (only used when provider = ollama)
  pub ollama_url: String,

  /// OpenRouter API key (only used when provider = openrouter)
  /// If not set, reads from OPENROUTER_API_KEY env var
  #[serde(skip_serializing_if = "Option::is_none")]
  pub openrouter_api_key: Option<String>,
}

impl Default for EmbeddingConfig {
  fn default() -> Self {
    Self {
      provider: EmbeddingProvider::Ollama,
      model: "qwen3-embedding".to_string(),
      dimensions: 4096,
      ollama_url: "http://localhost:11434".to_string(),
      openrouter_api_key: None,
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
}

impl Default for SearchConfig {
  fn default() -> Self {
    Self {
      default_limit: 10,
      include_superseded: false,
      semantic_weight: 0.5,
      salience_weight: 0.3,
      recency_weight: 0.2,
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
}

impl Default for IndexConfig {
  fn default() -> Self {
    Self {
      checkpoint_interval_secs: 30,
      watcher_debounce_ms: 500,
      max_file_size: 1024 * 1024, // 1MB
      max_chunk_chars: 2000,
    }
  }
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
#   minimal  = memory_search, code_search, docs_search
#   standard = memory_search, memory_add, memory_reinforce, memory_deemphasize,
#              code_search, docs_search, memory_timeline, entity_top, project_stats
#   full     = all {tool_count} tools
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
    assert_eq!(tools.len(), 3);
    assert!(tools.contains("memory_search"));
    assert!(tools.contains("code_search"));
    assert!(tools.contains("docs_search"));
  }

  #[test]
  fn test_preset_standard() {
    let config = Config::default();
    let tools = config.enabled_tool_set();
    assert_eq!(tools.len(), 9);
    assert!(tools.contains("memory_search"));
    assert!(tools.contains("memory_add"));
    assert!(tools.contains("code_search"));
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
    assert!(tools.contains("memory_search"));
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
dimensions = 768
"#;
    std::fs::write(claude_dir.join("ccengram.toml"), config_content).unwrap();

    let config = Config::load_for_project(temp.path());
    assert_eq!(config.tools.preset, ToolPreset::Minimal);
    assert_eq!(config.embedding.dimensions, 768);
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
  }

  #[test]
  fn test_decay_defaults() {
    let config = DecayConfig::default();
    assert_eq!(config.decay_interval_hours, 60);
    assert_eq!(config.min_salience, 0.05);
    assert_eq!(config.archive_threshold, 0.1);
    assert_eq!(config.max_idle_days, 90);
  }
}
