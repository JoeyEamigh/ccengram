//! Type definitions for the explore service.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{db::ProjectDb, embedding::EmbeddingProvider};

// ============================================================================
// Search Types
// ============================================================================

/// Scope for explore search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExploreScope {
  Code,
  Memory,
  Docs,
  #[default]
  All,
}

impl ExploreScope {
  /// Parse a scope string, returning None for invalid values.
  pub fn from_str(s: &str) -> Option<Self> {
    match s.to_lowercase().as_str() {
      "code" => Some(ExploreScope::Code),
      "memory" => Some(ExploreScope::Memory),
      "docs" => Some(ExploreScope::Docs),
      "all" => Some(ExploreScope::All),
      _ => None,
    }
  }
}

/// Navigation hints for an explore result
#[derive(Debug, Clone, Serialize, Default)]
pub struct ExploreHints {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callers: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callees: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub siblings: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub related_memories: Option<usize>,
  /// Related code count for memory results (Phase 4: cross-domain search)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub related_code: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub timeline_depth: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub total_chunks: Option<usize>,
}

/// Caller/callee info for expanded context
#[derive(Debug, Clone, Serialize)]
pub struct CallInfo {
  pub id: String,
  pub file: String,
  pub lines: (u32, u32),
  pub preview: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub symbols: Option<Vec<String>>,
}

/// Sibling symbol info
#[derive(Debug, Clone, Serialize)]
pub struct SiblingInfo {
  pub symbol: String,
  pub kind: String,
  pub line: u32,
}

/// Related memory info for expanded context
#[derive(Debug, Clone, Serialize)]
pub struct RelatedMemoryInfo {
  pub id: String,
  pub content: String,
  #[serde(rename = "type")]
  pub memory_type: String,
  pub sector: String,
}

/// Related code info for memory context (Phase 4: cross-domain search)
#[derive(Debug, Clone, Serialize)]
pub struct RelatedCodeInfo {
  pub id: String,
  pub file: String,
  pub lines: (u32, u32),
  pub preview: String,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub symbols: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
}

/// Expanded context for a result (included when rank <= expand_top)
#[derive(Debug, Clone, Serialize)]
pub struct ExpandedContext {
  pub content: String,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub callers: Vec<CallInfo>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub callees: Vec<CallInfo>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub siblings: Vec<SiblingInfo>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub memories: Vec<RelatedMemoryInfo>,
}

/// A single explore result
#[derive(Debug, Clone, Serialize)]
pub struct ExploreResult {
  pub id: String,
  #[serde(rename = "type")]
  pub result_type: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub file: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub lines: Option<(u32, u32)>,
  pub preview: String,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub symbols: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
  pub hints: ExploreHints,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context: Option<ExpandedContext>,
  pub score: f32,

  // === Semantic metadata for relevance evaluation ===
  /// Definition kind (function, class, struct, method, etc.)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub definition_kind: Option<String>,
  /// Full signature for quick relevance evaluation
  #[serde(skip_serializing_if = "Option::is_none")]
  pub signature: Option<String>,
  /// Docstring/documentation (truncated)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub docstring: Option<String>,
  /// Parent definition (for methods: the class/impl they belong to)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub parent: Option<String>,
  /// Key imports used by this chunk
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub imports: Vec<String>,
  /// Key function calls made by this chunk
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub calls: Vec<String>,
}

/// Full explore response
#[derive(Debug, Clone, Serialize)]
pub struct ExploreResponse {
  pub results: Vec<ExploreResult>,
  pub counts: HashMap<String, usize>,
  pub suggestions: Vec<String>,
}

// ============================================================================
// Context Types
// ============================================================================

/// Context for a code chunk
#[derive(Debug, Clone, Serialize)]
pub struct CodeContext {
  pub id: String,
  pub file: String,
  pub content: String,
  pub language: String,
  pub lines: (u32, u32),
  pub symbols: Vec<String>,
  pub imports: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub signature: Option<String>,
  pub callers: Vec<CallInfo>,
  pub callees: Vec<CallInfo>,
  pub siblings: Vec<SiblingInfo>,
  pub memories: Vec<RelatedMemoryInfo>,
}

/// Context for a memory
#[derive(Debug, Clone, Serialize)]
pub struct MemoryContext {
  pub id: String,
  pub content: String,
  pub sector: String,
  #[serde(rename = "type")]
  pub memory_type: String,
  pub salience: f32,
  pub created_at: String,
  pub timeline: TimelineContext,
  pub related: Vec<RelatedMemoryInfo>,
  /// Related code found via cross-domain vector search (Phase 4)
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub related_code: Vec<RelatedCodeInfo>,
}

/// Timeline context around a memory
#[derive(Debug, Clone, Serialize)]
pub struct TimelineContext {
  pub before: Vec<TimelineEntry>,
  pub after: Vec<TimelineEntry>,
}

/// A timeline entry
#[derive(Debug, Clone, Serialize)]
pub struct TimelineEntry {
  pub id: String,
  pub content: String,
  #[serde(rename = "type")]
  pub memory_type: String,
  pub created_at: String,
}

/// Context for a document chunk
#[derive(Debug, Clone, Serialize)]
pub struct DocContext {
  pub id: String,
  pub title: String,
  pub content: String,
  pub source: String,
  pub chunk_index: usize,
  pub total_chunks: usize,
  pub before: Vec<DocChunkEntry>,
  pub after: Vec<DocChunkEntry>,
}

/// A document chunk entry
#[derive(Debug, Clone, Serialize)]
pub struct DocChunkEntry {
  pub chunk_index: usize,
  pub content: String,
}

/// Full context response
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContextResponse {
  #[serde(rename = "code")]
  Code { items: Vec<CodeContext> },
  #[serde(rename = "memory")]
  Memory { items: Vec<MemoryContext> },
  #[serde(rename = "doc")]
  Doc { items: Vec<DocContext> },
  #[serde(rename = "mixed")]
  Mixed {
    code: Vec<CodeContext>,
    memories: Vec<MemoryContext>,
    docs: Vec<DocContext>,
  },
}

/// Helper enum for context results
pub enum ContextResult {
  Code(CodeContext),
  Memory(MemoryContext),
  Doc(DocContext),
}

// ============================================================================
// Service Context
// ============================================================================

/// Context for explore service operations.
pub struct ExploreContext<'a> {
  /// Project database connection
  pub db: &'a ProjectDb,
  /// Optional embedding provider for vector search
  pub embedding: &'a dyn EmbeddingProvider,
}

impl<'a> ExploreContext<'a> {
  /// Create a new explore context
  pub fn new(db: &'a ProjectDb, embedding: &'a dyn EmbeddingProvider) -> Self {
    Self { db, embedding }
  }
}

// ============================================================================
// Search Parameters
// ============================================================================

/// Parameters for explore search.
#[derive(Debug, Clone)]
pub struct SearchParams {
  /// The search query
  pub query: String,
  /// Search scope (code, memory, docs, all)
  pub scope: ExploreScope,
  /// Number of top results to expand with full context
  pub expand_top: usize,
  /// Maximum number of results per domain
  pub limit: usize,
  /// Context depth for expanded results
  pub depth: usize,
  /// Maximum number of suggestions to generate
  pub max_suggestions: usize,
}

impl Default for SearchParams {
  fn default() -> Self {
    Self {
      query: String::new(),
      scope: ExploreScope::All,
      expand_top: 3,
      limit: 10,
      depth: 5,
      max_suggestions: 5,
    }
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_explore_scope_parse() {
    let scope: ExploreScope = serde_json::from_str(r#""code""#).unwrap();
    assert_eq!(scope, ExploreScope::Code);

    let scope: ExploreScope = serde_json::from_str(r#""all""#).unwrap();
    assert_eq!(scope, ExploreScope::All);

    let scope: ExploreScope = serde_json::from_str(r#""memory""#).unwrap();
    assert_eq!(scope, ExploreScope::Memory);

    let scope: ExploreScope = serde_json::from_str(r#""docs""#).unwrap();
    assert_eq!(scope, ExploreScope::Docs);
  }

  #[test]
  fn test_explore_scope_from_str() {
    assert_eq!(ExploreScope::from_str("code"), Some(ExploreScope::Code));
    assert_eq!(ExploreScope::from_str("memory"), Some(ExploreScope::Memory));
    assert_eq!(ExploreScope::from_str("docs"), Some(ExploreScope::Docs));
    assert_eq!(ExploreScope::from_str("all"), Some(ExploreScope::All));
    assert_eq!(ExploreScope::from_str("invalid"), None);
  }

  #[test]
  fn test_explore_hints_serialization() {
    let hints = ExploreHints {
      callers: Some(5),
      callees: Some(3),
      siblings: Some(2),
      related_memories: Some(1),
      related_code: None,
      timeline_depth: None,
      total_chunks: None,
    };

    let json = serde_json::to_value(&hints).unwrap();
    assert_eq!(json["callers"], 5);
    assert_eq!(json["callees"], 3);
    assert_eq!(json["siblings"], 2);
    assert_eq!(json["related_memories"], 1);
    // None values should be skipped
    assert!(json.get("timeline_depth").is_none());
    assert!(json.get("total_chunks").is_none());
    assert!(json.get("related_code").is_none());
  }

  #[test]
  fn test_explore_hints_empty() {
    let hints = ExploreHints::default();

    let json = serde_json::to_value(&hints).unwrap();
    // All fields should be skipped when None
    assert!(json.as_object().unwrap().is_empty());
  }

  #[test]
  fn test_explore_result_serialization() {
    let result = ExploreResult {
      id: "abc123".to_string(),
      result_type: "code".to_string(),
      file: Some("src/main.rs".to_string()),
      lines: Some((10, 20)),
      preview: "fn main() {}".to_string(),
      symbols: vec!["main".to_string()],
      language: Some("rust".to_string()),
      hints: ExploreHints {
        callers: Some(5),
        ..Default::default()
      },
      context: None,
      score: 0.95,
      definition_kind: Some("function".to_string()),
      signature: Some("fn main()".to_string()),
      docstring: Some("Entry point".to_string()),
      parent: None,
      imports: vec![],
      calls: vec!["println".to_string()],
    };

    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["id"], "abc123");
    assert_eq!(json["type"], "code");
    assert_eq!(json["file"], "src/main.rs");
    assert_eq!(json["lines"][0], 10);
    assert_eq!(json["lines"][1], 20);
    assert_eq!(json["preview"], "fn main() {}");
    assert_eq!(json["symbols"][0], "main");
    assert_eq!(json["language"], "rust");
    assert_eq!(json["definition_kind"], "function");
    assert_eq!(json["signature"], "fn main()");
  }

  #[test]
  fn test_explore_result_no_file() {
    let result = ExploreResult {
      id: "mem123".to_string(),
      result_type: "memory".to_string(),
      file: None,
      lines: None,
      preview: "Some memory content".to_string(),
      symbols: vec![],
      language: None,
      hints: ExploreHints {
        related_memories: Some(3),
        timeline_depth: Some(5),
        ..Default::default()
      },
      context: None,
      score: 0.8,
      // Not applicable to memories
      definition_kind: None,
      signature: None,
      docstring: None,
      parent: None,
      imports: vec![],
      calls: vec![],
    };

    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["type"], "memory");
    // Optional fields should be skipped when None
    assert!(json.get("file").is_none());
    assert!(json.get("lines").is_none());
    assert!(json.get("language").is_none());
    // New semantic fields should also be skipped when empty
    assert!(json.get("definition_kind").is_none());
    assert!(json.get("signature").is_none());
  }

  #[test]
  fn test_explore_response_serialization() {
    let response = ExploreResponse {
      results: vec![ExploreResult {
        id: "test".to_string(),
        result_type: "code".to_string(),
        file: Some("test.rs".to_string()),
        lines: Some((1, 10)),
        preview: "test preview".to_string(),
        symbols: vec!["test_fn".to_string()],
        language: Some("rust".to_string()),
        hints: ExploreHints {
          callers: Some(0),
          callees: Some(0),
          siblings: Some(0),
          related_memories: Some(0),
          ..Default::default()
        },
        context: None,
        score: 1.0,
        definition_kind: None,
        signature: None,
        docstring: None,
        parent: None,
        imports: vec![],
        calls: vec![],
      }],
      counts: {
        let mut m = HashMap::new();
        m.insert("code".to_string(), 1);
        m.insert("memory".to_string(), 0);
        m
      },
      suggestions: vec!["related_query".to_string()],
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["results"].as_array().unwrap().len(), 1);
    assert_eq!(json["counts"]["code"], 1);
    assert_eq!(json["counts"]["memory"], 0);
    assert_eq!(json["suggestions"][0], "related_query");
  }

  #[test]
  fn test_context_response_code() {
    let response = ContextResponse::Code {
      items: vec![CodeContext {
        id: "code1".to_string(),
        file: "test.rs".to_string(),
        content: "fn test() {}".to_string(),
        language: "rust".to_string(),
        lines: (1, 3),
        symbols: vec!["test".to_string()],
        imports: vec![],
        signature: Some("fn test()".to_string()),
        callers: vec![],
        callees: vec![],
        siblings: vec![],
        memories: vec![],
      }],
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["type"], "code");
    assert_eq!(json["items"].as_array().unwrap().len(), 1);
  }

  #[test]
  fn test_context_response_memory() {
    let response = ContextResponse::Memory {
      items: vec![MemoryContext {
        id: "mem1".to_string(),
        content: "Memory content".to_string(),
        sector: "semantic".to_string(),
        memory_type: "observation".to_string(),
        salience: 0.75,
        created_at: "2024-01-01T00:00:00Z".to_string(),
        timeline: TimelineContext {
          before: vec![],
          after: vec![],
        },
        related: vec![],
        related_code: vec![],
      }],
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["type"], "memory");
    assert_eq!(json["items"][0]["salience"], 0.75);
    // related_code should be skipped when empty
    assert!(json["items"][0].get("related_code").is_none());
  }

  #[test]
  fn test_context_response_mixed() {
    let response = ContextResponse::Mixed {
      code: vec![],
      memories: vec![],
      docs: vec![],
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["type"], "mixed");
    assert!(json["code"].is_array());
    assert!(json["memories"].is_array());
    assert!(json["docs"].is_array());
  }
}
