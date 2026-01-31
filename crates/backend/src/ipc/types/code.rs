//! Code IPC types - requests, responses, and conversions
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::memory::MemoryItem;
use crate::domain::code::CodeChunk;

// ============================================================================
// Request types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum CodeRequest {
  Search(CodeSearchParams),
  Context(CodeContextParams),
  Index(CodeIndexParams),
  List(CodeListParams),
  Stats(CodeStatsParams),
  Memories(CodeMemoriesParams),
  Callers(CodeCallersParams),
  Callees(CodeCalleesParams),
  Related(CodeRelatedParams),
  ContextFull(CodeContextFullParams),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeSearchParams {
  pub query: String,
  pub limit: Option<usize>,
  pub file_pattern: Option<String>,
  pub symbol_type: Option<String>,

  // === Metadata filters (Phase 3) ===
  /// Filter by programming language (e.g., "rust", "typescript", "python").
  /// Matches the lowercase language name stored in the database.
  pub language: Option<String>,

  /// Filter by visibility (e.g., ["pub", "pub(crate)"] for Rust).
  /// Pass multiple values to match any of them.
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub visibility: Vec<String>,

  /// Filter by chunk type (e.g., ["function", "class"]).
  /// Valid types: function, class, module, block, import.
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub chunk_type: Vec<String>,

  /// Minimum caller count filter. Only returns code that is called
  /// by at least this many other code chunks (indicates importance/centrality).
  pub min_caller_count: Option<u32>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextParams {
  pub chunk_id: String,
  pub before: Option<usize>,
  pub after: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeIndexParams {
  #[serde(default)]
  pub force: bool,
  #[serde(default)]
  pub stream: bool,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeListParams {
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeStatsParams;

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCallersParams {
  pub chunk_id: String,
  pub limit: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleesParams {
  pub chunk_id: String,
  pub limit: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeMemoriesParams {
  pub chunk_id: String,
  pub limit: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeRelatedParams {
  pub chunk_id: String,
  pub limit: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeContextFullParams {
  pub chunk_id: String,
  pub depth: Option<usize>,
}

// ============================================================================
// Response types
// ============================================================================

#[allow(clippy::large_enum_variant)]
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum CodeResponse {
  Search(CodeSearchResult),
  Context(CodeContextResponse),
  Index(CodeIndexResult),
  List(Vec<CodeItem>),
  ImportChunk(CodeImportChunkResult),
  Stats(CodeStatsResult),
  Memories(CodeMemoriesResponse),
  Callers(CodeCallersResponse),
  Callees(CodeCalleesResponse),
  Related(CodeRelatedResponse),
  ContextFull(CodeContextFullResponse),
}

/// Unified code chunk item - consolidates CodeChunkItem, CodeChunkDetail, CodeListItem
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeItem {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub start_line: u32,
  pub end_line: u32,

  // Optional fields based on context
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub chunk_type: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub symbol_name: Option<String>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub symbols: Vec<String>,

  // Semantic context fields
  /// The kind of definition (e.g., "function", "struct", "impl", "trait", "class", "method")
  #[serde(skip_serializing_if = "Option::is_none")]
  pub definition_kind: Option<String>,
  /// Visibility modifier (e.g., "pub", "pub(crate)", "private")
  #[serde(skip_serializing_if = "Option::is_none")]
  pub visibility: Option<String>,
  /// Full signature for display (e.g., "pub fn calculate_total(items: Vec<Item>) -> f64")
  #[serde(skip_serializing_if = "Option::is_none")]
  pub signature: Option<String>,
  /// Extracted documentation comments
  #[serde(skip_serializing_if = "Option::is_none")]
  pub docstring: Option<String>,
  /// Parent definition name for nested items (e.g., "UserRepo" for method in impl UserRepo)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub parent_definition: Option<String>,

  // Search-specific
  #[serde(skip_serializing_if = "Option::is_none")]
  pub similarity: Option<f32>,
  /// Raw confidence score based on vector distance (0.0-1.0).
  /// Higher = more confident match. Derived from: 1.0 - min(distance, 1.0).
  /// Unlike `similarity` (which is the weighted rank score), this is the pure
  /// semantic match confidence from the embedding model.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub confidence: Option<f32>,

  // Detail-specific
  #[serde(skip_serializing_if = "Option::is_none")]
  pub file_hash: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tokens_estimate: Option<u32>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub imports: Vec<String>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub calls: Vec<String>,

  // Relationship hints
  #[serde(skip_serializing_if = "Option::is_none")]
  pub caller_count: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callee_count: Option<u32>,
}

/// Search quality information based on distance scores.
///
/// Provides insight into the quality of search results, helping users
/// understand whether their query matched well or if they should refine it.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuality {
  /// Best (lowest) distance among all results. Lower = better match.
  /// Distance 0.0 = identical, 1.0+ = very different.
  pub best_distance: f32,
  /// Whether the search results may be low quality (best_distance > threshold).
  /// When true, users should consider refining their query.
  pub low_confidence: bool,
  /// Average confidence of the top results (1.0 - avg_distance).
  pub avg_confidence: f32,
  /// Number of high-confidence results (distance < 0.3).
  pub high_confidence_count: usize,
  /// Suggested action when results are low confidence, or None if confident.
  /// Examples: "Try more specific terms", "Check for typos"
  #[serde(skip_serializing_if = "Option::is_none")]
  pub suggested_action: Option<String>,
}

impl SearchQuality {
  /// Create SearchQuality from a list of distances (sorted ascending).
  pub fn from_distances(distances: &[f32]) -> Self {
    if distances.is_empty() {
      return Self {
        best_distance: 1.0,
        low_confidence: true,
        avg_confidence: 0.0,
        high_confidence_count: 0,
        suggested_action: Some("No results found. Try different search terms.".to_string()),
      };
    }

    let best_distance = distances.first().copied().unwrap_or(1.0);
    // Low confidence if best match has distance > 0.5
    let low_confidence = best_distance > 0.5;

    // Calculate average confidence of top 5 results
    let top_n = distances.iter().take(5);
    let count = top_n.clone().count() as f32;
    let avg_distance: f32 = top_n.sum::<f32>() / count.max(1.0);
    let avg_confidence = 1.0 - avg_distance.min(1.0);

    // Count high-confidence results (distance < 0.3)
    let high_confidence_count = distances.iter().filter(|&&d| d < 0.3).count();

    // Generate suggested action for low confidence searches
    let suggested_action = if best_distance > 0.7 {
      Some("Results may not be relevant. Try more specific terms or check spelling.".to_string())
    } else if best_distance > 0.5 {
      Some("Results have moderate confidence. Consider refining your query.".to_string())
    } else {
      None
    };

    Self {
      best_distance,
      low_confidence,
      avg_confidence,
      high_confidence_count,
      suggested_action,
    }
  }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSearchResult {
  pub query: String,
  pub chunks: Vec<CodeItem>,
  /// Search quality metadata. When `low_confidence` is true, consider
  /// refining the query for better results.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub search_quality: Option<SearchQuality>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextResponse {
  pub chunk_id: String,
  pub file_path: String,
  pub language: String,
  pub context: CodeContextSections,
  pub total_file_lines: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub warning: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextSections {
  pub before: CodeContextSection,
  pub target: CodeContextSection,
  pub after: CodeContextSection,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextSection {
  pub content: String,
  pub start_line: usize,
  pub end_line: usize,
}

/// Code index result (full)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexResult {
  pub status: String,
  pub files_scanned: usize,
  pub files_indexed: usize,
  pub chunks_created: usize,
  pub failed_files: usize,
  pub resumed_from_checkpoint: bool,
  pub scan_duration_ms: u64,
  pub index_duration_ms: u64,
  pub total_duration_ms: u64,
  pub files_per_second: f64,
  pub bytes_processed: u64,
  pub total_bytes: u64,
}

/// Code index dry run response
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexDryRunResult {
  pub status: String,
  pub files_found: usize,
  pub skipped: usize,
  pub total_bytes: u64,
  pub scan_duration_ms: u64,
}

/// Code index file info
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexFileInfo {
  pub path: String,
  pub language: String,
  pub chunks: usize,
  pub skipped: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reason: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeImportChunkResult {
  pub chunk_id: String,
  pub message: String,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStatsResult {
  pub total_chunks: usize,
  pub total_files: usize,
  pub total_tokens_estimate: u64,
  pub total_lines: u64,
  pub average_chunks_per_file: f32,
  pub language_breakdown: HashMap<String, usize>,
  pub chunk_type_breakdown: HashMap<String, usize>,
  pub index_health_score: u32,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMemoriesResponse {
  pub file_path: String,
  pub memories: Vec<MemoryItem>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCallersResponse {
  pub symbol: String,
  pub callers: Vec<CodeItem>,
  pub count: usize,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleesResponse {
  pub chunk_id: String,
  pub calls: Vec<String>,
  pub callees: Vec<CodeCalleeItem>,
  pub unresolved: Vec<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleeItem {
  pub call: String,
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub end_line: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelatedResponse {
  pub chunk_id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub related: Vec<CodeRelatedItem>,
  pub count: usize,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelatedItem {
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
  pub language: String,
  pub chunk_type: String,
  pub score: f32,
  pub relationship: String,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextFullResponse {
  pub chunk: CodeItem,
  pub callers: Vec<CodeItem>,
  pub callees: Vec<CodeCalleeItem>,
  pub unresolved_calls: Vec<String>,
  pub same_file: Vec<CodeItem>,
  pub memories: Vec<MemoryItem>,
  pub documentation: Vec<CodeFullDoc>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFullDoc {
  pub id: String,
  pub title: String,
  pub content: String,
  pub similarity: f32,
}

// ============================================================================
// Conversions from domain types
// ============================================================================

/// Options for code chunk conversion
#[derive(Debug, Clone, Copy, Default)]
pub struct CodeConvertOptions {
  pub include_search_scores: bool,
  pub include_details: bool,
  pub include_relationships: bool,
  /// Weighted rank score (combination of semantic + symbol + importance)
  pub similarity: Option<f32>,
  /// Raw confidence from vector distance (1.0 - distance)
  pub confidence: Option<f32>,
}

impl CodeConvertOptions {
  /// Create options for search results with both rank score and confidence.
  ///
  /// # Arguments
  /// * `similarity` - The weighted rank score (combined signals)
  /// * `confidence` - Raw confidence from vector distance (1.0 - distance)
  pub fn for_search_with_confidence(similarity: f32, confidence: f32) -> Self {
    Self {
      include_search_scores: true,
      include_details: false,
      include_relationships: false,
      similarity: Some(similarity),
      confidence: Some(confidence),
    }
  }

  pub fn for_search(similarity: f32) -> Self {
    Self {
      include_search_scores: true,
      include_details: false,
      include_relationships: false,
      similarity: Some(similarity),
      confidence: None,
    }
  }

  pub fn for_detail() -> Self {
    Self {
      include_search_scores: false,
      include_details: true,
      include_relationships: true,
      similarity: None,
      confidence: None,
    }
  }

  pub fn for_list() -> Self {
    Self {
      include_search_scores: false,
      include_details: true,
      include_relationships: false,
      similarity: None,
      confidence: None,
    }
  }

  pub fn for_caller() -> Self {
    Self {
      include_search_scores: false,
      include_details: false,
      include_relationships: false,
      similarity: None,
      confidence: None,
    }
  }
}

impl From<&CodeChunk> for CodeItem {
  fn from(c: &CodeChunk) -> Self {
    Self::from_chunk(c, CodeConvertOptions::default())
  }
}

impl CodeItem {
  /// Convert from domain CodeChunk with options
  pub fn from_chunk(c: &CodeChunk, opts: CodeConvertOptions) -> Self {
    Self {
      id: c.id.to_string(),
      file_path: c.file_path.clone(),
      content: c.content.clone(),
      start_line: c.start_line,
      end_line: c.end_line,
      language: Some(format!("{:?}", c.language).to_lowercase()),
      chunk_type: Some(format!("{:?}", c.chunk_type).to_lowercase()),
      symbol_name: c.definition_name.clone(),
      symbols: c.symbols.clone(),
      // Semantic context - always include when available
      definition_kind: c.definition_kind.clone(),
      visibility: c.visibility.clone(),
      signature: c.signature.clone(),
      docstring: c.docstring.clone(),
      parent_definition: c.parent_definition.clone(),
      similarity: opts.similarity,
      confidence: opts.confidence,
      file_hash: if opts.include_details {
        Some(c.file_hash.clone())
      } else {
        None
      },
      tokens_estimate: if opts.include_details {
        Some(c.tokens_estimate)
      } else {
        None
      },
      imports: if opts.include_details {
        c.imports.clone()
      } else {
        Vec::new()
      },
      calls: if opts.include_details {
        c.calls.clone()
      } else {
        Vec::new()
      },
      caller_count: if opts.include_relationships {
        Some(c.caller_count)
      } else {
        None
      },
      callee_count: if opts.include_relationships {
        Some(c.callee_count)
      } else {
        None
      },
    }
  }

  /// Convert for search results with both rank score and confidence.
  ///
  /// # Arguments
  /// * `similarity` - The weighted rank score (combined signals)
  /// * `confidence` - Raw confidence from vector distance (1.0 - distance)
  pub fn from_search_with_confidence(c: &CodeChunk, similarity: f32, confidence: f32) -> Self {
    Self::from_chunk(
      c,
      CodeConvertOptions::for_search_with_confidence(similarity, confidence),
    )
  }

  /// Convert for search results (backwards compatible, no confidence)
  pub fn from_search(c: &CodeChunk, similarity: f32) -> Self {
    Self::from_chunk(c, CodeConvertOptions::for_search(similarity))
  }

  /// Convert for detail view
  pub fn from_detail(c: &CodeChunk) -> Self {
    Self::from_chunk(c, CodeConvertOptions::for_detail())
  }

  /// Convert for list view
  pub fn from_list(c: &CodeChunk) -> Self {
    Self::from_chunk(c, CodeConvertOptions::for_list())
  }

  /// Convert for caller/callee view (minimal)
  pub fn from_caller(c: &CodeChunk) -> Self {
    Self::from_chunk(c, CodeConvertOptions::for_caller())
  }
}

impl From<&CodeChunk> for CodeCalleeItem {
  fn from(c: &CodeChunk) -> Self {
    Self {
      call: c.definition_name.clone().unwrap_or_default(),
      id: c.id.to_string(),
      file_path: c.file_path.clone(),
      symbols: c.symbols.clone(),
      start_line: c.start_line,
      end_line: Some(c.end_line),
      language: Some(format!("{:?}", c.language).to_lowercase()),
    }
  }
}

impl CodeCalleeItem {
  pub fn from_chunk_with_call(c: &CodeChunk, call: &str) -> Self {
    Self {
      call: call.to_string(),
      id: c.id.to_string(),
      file_path: c.file_path.clone(),
      symbols: c.symbols.clone(),
      start_line: c.start_line,
      end_line: Some(c.end_line),
      language: Some(format!("{:?}", c.language).to_lowercase()),
    }
  }
}

impl From<&CodeChunk> for CodeRelatedItem {
  fn from(c: &CodeChunk) -> Self {
    Self {
      id: c.id.to_string(),
      file_path: c.file_path.clone(),
      symbols: c.symbols.clone(),
      start_line: c.start_line,
      end_line: c.end_line,
      language: format!("{:?}", c.language).to_lowercase(),
      chunk_type: format!("{:?}", c.chunk_type).to_lowercase(),
      score: 0.0,
      relationship: String::new(),
    }
  }
}

impl CodeRelatedItem {
  pub fn from_chunk_with_score(c: &CodeChunk, score: f32, relationship: &str) -> Self {
    Self {
      id: c.id.to_string(),
      file_path: c.file_path.clone(),
      symbols: c.symbols.clone(),
      start_line: c.start_line,
      end_line: c.end_line,
      language: format!("{:?}", c.language).to_lowercase(),
      chunk_type: format!("{:?}", c.chunk_type).to_lowercase(),
      score,
      relationship: relationship.to_string(),
    }
  }
}

// ============================================================================
// IpcRequest implementations
// ============================================================================

use crate::{
  impl_ipc_request,
  ipc::{RequestData, ResponseData},
};

impl_ipc_request!(
  CodeSearchParams => CodeSearchResult,
  ResponseData::Code(CodeResponse::Search(v)) => v,
  v => RequestData::Code(CodeRequest::Search(v)),
  v => ResponseData::Code(CodeResponse::Search(v))
);
impl_ipc_request!(
  CodeContextParams => CodeContextResponse,
  ResponseData::Code(CodeResponse::Context(v)) => v,
  v => RequestData::Code(CodeRequest::Context(v)),
  v => ResponseData::Code(CodeResponse::Context(v))
);
impl_ipc_request!(
  CodeIndexParams => CodeIndexResult,
  ResponseData::Code(CodeResponse::Index(v)) => v,
  v => RequestData::Code(CodeRequest::Index(v)),
  v => ResponseData::Code(CodeResponse::Index(v))
);
impl_ipc_request!(
  CodeListParams => Vec<CodeItem>,
  ResponseData::Code(CodeResponse::List(v)) => v,
  v => RequestData::Code(CodeRequest::List(v)),
  v => ResponseData::Code(CodeResponse::List(v))
);
impl_ipc_request!(
  CodeStatsParams => CodeStatsResult,
  ResponseData::Code(CodeResponse::Stats(v)) => v,
  v => RequestData::Code(CodeRequest::Stats(v)),
  v => ResponseData::Code(CodeResponse::Stats(v))
);
impl_ipc_request!(
  CodeMemoriesParams => CodeMemoriesResponse,
  ResponseData::Code(CodeResponse::Memories(v)) => v,
  v => RequestData::Code(CodeRequest::Memories(v)),
  v => ResponseData::Code(CodeResponse::Memories(v))
);
impl_ipc_request!(
  CodeCallersParams => CodeCallersResponse,
  ResponseData::Code(CodeResponse::Callers(v)) => v,
  v => RequestData::Code(CodeRequest::Callers(v)),
  v => ResponseData::Code(CodeResponse::Callers(v))
);
impl_ipc_request!(
  CodeCalleesParams => CodeCalleesResponse,
  ResponseData::Code(CodeResponse::Callees(v)) => v,
  v => RequestData::Code(CodeRequest::Callees(v)),
  v => ResponseData::Code(CodeResponse::Callees(v))
);
impl_ipc_request!(
  CodeRelatedParams => CodeRelatedResponse,
  ResponseData::Code(CodeResponse::Related(v)) => v,
  v => RequestData::Code(CodeRequest::Related(v)),
  v => ResponseData::Code(CodeResponse::Related(v))
);
impl_ipc_request!(
  CodeContextFullParams => CodeContextFullResponse,
  ResponseData::Code(CodeResponse::ContextFull(v)) => v,
  v => RequestData::Code(CodeRequest::ContextFull(v)),
  v => ResponseData::Code(CodeResponse::ContextFull(v))
);
