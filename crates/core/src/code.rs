use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Characters per token estimate (for LLM token counting)
pub const CHARS_PER_TOKEN: usize = 4;

/// Compute a content hash for differential re-indexing
///
/// Uses SHA-256 truncated to 16 hex chars for compact storage
/// while still having negligible collision probability.
pub fn compute_content_hash(content: &str) -> String {
  let mut hasher = Sha256::new();
  hasher.update(content.as_bytes());
  let result = hasher.finalize();
  // Take first 8 bytes (16 hex chars) for compact storage
  format!("{:016x}", u64::from_be_bytes(result[0..8].try_into().unwrap()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
  pub id: Uuid,
  pub file_path: String,
  pub content: String,
  pub language: Language,
  pub chunk_type: ChunkType,
  pub symbols: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
  pub file_hash: String,
  pub indexed_at: DateTime<Utc>,
  /// Estimated token count (content.len() / CHARS_PER_TOKEN)
  pub tokens_estimate: u32,
  /// Import paths referenced by this chunk
  /// e.g., ["std::collections::HashMap", "crate::db::ProjectDb"]
  #[serde(default)]
  pub imports: Vec<String>,
  /// Function/method calls made within this chunk
  /// e.g., ["process", "HashMap::new", "db.query"]
  #[serde(default)]
  pub calls: Vec<String>,

  // === Definition metadata for AST-level chunking ===
  /// The kind of definition this chunk represents
  /// e.g., "function", "struct", "impl", "trait", "class", "method"
  #[serde(default)]
  pub definition_kind: Option<String>,

  /// The primary symbol name for this definition
  /// e.g., "calculate_total", "UserService", "impl Display for User"
  #[serde(default)]
  pub definition_name: Option<String>,

  /// Visibility modifier
  /// e.g., "pub", "pub(crate)", "pub(super)", "private"
  #[serde(default)]
  pub visibility: Option<String>,

  /// Full signature for display (function signature, struct definition line)
  /// e.g., "pub fn calculate_total(items: Vec<Item>) -> f64"
  #[serde(default)]
  pub signature: Option<String>,

  /// Extracted documentation comments (/// or /** */ style)
  #[serde(default)]
  pub docstring: Option<String>,

  /// Parent definition name for nested items (methods inside impl/class)
  /// e.g., for method `save` in `impl UserRepo`, this would be "UserRepo"
  #[serde(default)]
  pub parent_definition: Option<String>,

  /// Enriched text representation used for embedding
  /// Contains structured metadata + code for better semantic search
  #[serde(default)]
  pub embedding_text: Option<String>,

  /// Hash of the content for detecting unchanged chunks during re-indexing
  /// Used to skip re-embedding when only file position changes
  #[serde(default)]
  pub content_hash: Option<String>,

  // === Pre-computed relationship counts for fast hint computation ===
  /// Number of chunks that call symbols defined in this chunk
  /// Pre-computed during indexing to avoid expensive LIKE queries
  #[serde(default)]
  pub caller_count: u32,

  /// Number of unique symbols this chunk calls
  /// Pre-computed during indexing to avoid expensive LIKE queries
  #[serde(default)]
  pub callee_count: u32,
}

impl CodeChunk {
  /// Estimate token count from content length
  pub fn estimate_tokens(content: &str) -> u32 {
    (content.len() / CHARS_PER_TOKEN) as u32
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
  TypeScript,
  JavaScript,
  Tsx,
  Jsx,
  Html,
  Css,
  Scss,
  Sass,
  Less,
  Rust,
  Python,
  Go,
  Java,
  Kotlin,
  Scala,
  CSharp,
  Cpp,
  C,
  Swift,
  Ruby,
  Php,
  Lua,
  Elixir,
  Haskell,
  Ocaml,
  Clojure,
  Zig,
  Nim,
  Json,
  Yaml,
  Toml,
  Xml,
  Markdown,
  Shell,
  Sql,
  Dockerfile,
  GraphQL,
  Proto,
}

impl Language {
  pub fn from_extension(ext: &str) -> Option<Self> {
    match ext.to_lowercase().as_str() {
      "ts" | "mts" => Some(Language::TypeScript),
      "js" | "mjs" | "cjs" => Some(Language::JavaScript),
      "tsx" => Some(Language::Tsx),
      "jsx" => Some(Language::Jsx),
      "html" | "htm" => Some(Language::Html),
      "css" => Some(Language::Css),
      "scss" => Some(Language::Scss),
      "sass" => Some(Language::Sass),
      "less" => Some(Language::Less),
      "rs" => Some(Language::Rust),
      "py" | "pyi" | "pyw" => Some(Language::Python),
      "go" => Some(Language::Go),
      "java" => Some(Language::Java),
      "kt" | "kts" => Some(Language::Kotlin),
      "scala" | "sc" => Some(Language::Scala),
      "cs" => Some(Language::CSharp),
      "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "h" => Some(Language::Cpp),
      "c" => Some(Language::C),
      "swift" => Some(Language::Swift),
      "rb" | "rake" => Some(Language::Ruby),
      "php" => Some(Language::Php),
      "lua" => Some(Language::Lua),
      "ex" | "exs" => Some(Language::Elixir),
      "hs" => Some(Language::Haskell),
      "ml" | "mli" => Some(Language::Ocaml),
      "clj" | "cljs" | "cljc" | "edn" => Some(Language::Clojure),
      "zig" => Some(Language::Zig),
      "nim" => Some(Language::Nim),
      "json" | "jsonc" => Some(Language::Json),
      "yaml" | "yml" => Some(Language::Yaml),
      "toml" => Some(Language::Toml),
      "xml" | "xsd" | "xsl" | "svg" => Some(Language::Xml),
      "md" | "markdown" => Some(Language::Markdown),
      "sh" | "bash" | "zsh" | "fish" => Some(Language::Shell),
      "sql" => Some(Language::Sql),
      "dockerfile" => Some(Language::Dockerfile),
      "graphql" | "gql" => Some(Language::GraphQL),
      "proto" => Some(Language::Proto),
      _ => None,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkType {
  Function,
  Class,
  Module,
  Block,
  Import,
}
