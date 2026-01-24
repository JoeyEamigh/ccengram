//! Unified exploration tools: `explore` and `context`.
//!
//! These tools provide a streamlined interface for codebase exploration,
//! replacing the need for multiple separate search/context tools.

use super::ToolHandler;
use super::format::{format_context_response, format_explore_response};
use super::suggestions::{extract_content_words, generate_suggestions};
use crate::router::{Request, Response};
use engram_core::{CodeChunk, Config, DocumentChunk, Memory, MemoryId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// Types for explore tool
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

/// Navigation hints for an explore result
#[derive(Debug, Clone, Serialize)]
pub struct ExploreHints {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callers: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callees: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub siblings: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub related_memories: Option<usize>,
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
}

/// Full explore response
#[derive(Debug, Clone, Serialize)]
pub struct ExploreResponse {
  pub results: Vec<ExploreResult>,
  pub counts: HashMap<String, usize>,
  pub suggestions: Vec<String>,
}

// ============================================================================
// Types for context tool
// ============================================================================

/// Type of item being contextualized
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ContextType {
  Code,
  Memory,
  Doc,
}

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

// ============================================================================
// Implementation
// ============================================================================

impl ToolHandler {
  /// Explore tool: unified search across code, memories, and documents.
  pub async fn explore(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      query: String,
      #[serde(default)]
      scope: Option<String>,
      #[serde(default)]
      expand_top: Option<usize>,
      #[serde(default)]
      limit: Option<usize>,
      #[serde(default)]
      cwd: Option<String>,
      /// Output format: "json" (default) or "text" (human-readable)
      #[serde(default)]
      format: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    if args.query.trim().is_empty() {
      return Response::error(request.id, -32602, "Query cannot be empty");
    }

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Load config for defaults
    let config = Config::load_for_project(&project_path);
    let expand_top = args.expand_top.unwrap_or(config.search.explore_expand_top);
    let limit = args.limit.unwrap_or(config.search.explore_limit);
    let depth = config.search.context_depth;
    let max_suggestions = config.search.explore_max_suggestions;

    // Parse scope
    let scope = match args.scope.as_deref() {
      Some("code") => ExploreScope::Code,
      Some("memory") => ExploreScope::Memory,
      Some("docs") => ExploreScope::Docs,
      Some("all") | None => ExploreScope::All,
      Some(other) => {
        return Response::error(
          request.id,
          -32602,
          &format!("Invalid scope: '{}'. Use 'code', 'memory', 'docs', or 'all'", other),
        );
      }
    };

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Get query embedding
    let query_embedding = self.get_embedding(&args.query).await;

    let mut all_results: Vec<ExploreResult> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut result_symbols: Vec<String> = Vec::new();
    let mut result_content: Vec<String> = Vec::new();

    // Determine which scopes to search
    let search_code = scope == ExploreScope::Code || scope == ExploreScope::All;
    let search_memory = scope == ExploreScope::Memory || scope == ExploreScope::All;
    let search_docs = scope == ExploreScope::Docs || scope == ExploreScope::All;

    // Run all applicable searches in parallel
    let (code_results, memory_results, doc_results) = tokio::join!(
      // Code search
      async {
        if !search_code {
          return Vec::new();
        }
        if let Some(ref embedding) = query_embedding {
          db.search_code_chunks(embedding, limit, None).await.unwrap_or_default()
        } else {
          // Fallback to text search
          db.list_code_chunks(None, Some(limit * 10))
            .await
            .map(|chunks| {
              let query_lower = args.query.to_lowercase();
              chunks
                .into_iter()
                .filter(|c| {
                  c.content.to_lowercase().contains(&query_lower)
                    || c.symbols.iter().any(|s| s.to_lowercase().contains(&query_lower))
                })
                .take(limit)
                .map(|c| (c, 0.5f32))
                .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        }
      },
      // Memory search
      async {
        if !search_memory {
          return Vec::new();
        }
        if let Some(ref embedding) = query_embedding {
          db.search_memories(embedding, limit, None).await.unwrap_or_default()
        } else {
          // Fallback to text search
          db.list_memories(Some("is_deleted = false"), Some(limit * 10))
            .await
            .map(|memories| {
              let query_lower = args.query.to_lowercase();
              memories
                .into_iter()
                .filter(|m| m.content.to_lowercase().contains(&query_lower))
                .take(limit)
                .map(|m| (m, 0.5f32))
                .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        }
      },
      // Document search
      async {
        if !search_docs {
          return Vec::new();
        }
        if let Some(ref embedding) = query_embedding {
          db.search_documents(embedding, limit, None).await.unwrap_or_default()
        } else {
          // Fallback to text search
          db.list_document_chunks(None, Some(limit * 10))
            .await
            .map(|chunks| {
              let query_lower = args.query.to_lowercase();
              chunks
                .into_iter()
                .filter(|c| {
                  c.content.to_lowercase().contains(&query_lower) || c.title.to_lowercase().contains(&query_lower)
                })
                .take(limit)
                .map(|c| (c, 0.5f32))
                .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        }
      }
    );

    // Process code results
    if search_code {
      counts.insert("code".to_string(), code_results.len());

      for (chunk, distance) in code_results {
        let similarity: f32 = 1.0 - distance.min(1.0);
        result_symbols.extend(chunk.symbols.clone());
        result_content.push(chunk.content.clone());

        // Compute hints
        let hints = self.compute_code_hints(&db, &chunk).await;

        all_results.push(ExploreResult {
          id: chunk.id.to_string(),
          result_type: "code".to_string(),
          file: Some(chunk.file_path.clone()),
          lines: Some((chunk.start_line, chunk.end_line)),
          preview: truncate_preview(&chunk.content, 200),
          symbols: chunk.symbols.clone(),
          language: Some(format!("{:?}", chunk.language).to_lowercase()),
          hints,
          context: None,
          score: similarity,
        });
      }
    }

    // Process memory results
    if search_memory {
      counts.insert("memory".to_string(), memory_results.len());

      for (memory, distance) in memory_results {
        let similarity: f32 = 1.0 - distance.min(1.0);
        result_content.push(memory.content.clone());

        // Compute hints
        let hints = self.compute_memory_hints(&db, &memory).await;

        all_results.push(ExploreResult {
          id: memory.id.to_string(),
          result_type: "memory".to_string(),
          file: None,
          lines: None,
          preview: truncate_preview(&memory.content, 200),
          symbols: vec![],
          language: None,
          hints,
          context: None,
          score: similarity * memory.salience, // Weight by salience
        });
      }
    }

    // Process document results
    if search_docs {
      counts.insert("docs".to_string(), doc_results.len());

      for (chunk, distance) in doc_results {
        let similarity: f32 = 1.0 - distance.min(1.0);
        result_content.push(chunk.content.clone());

        let hints = ExploreHints {
          callers: None,
          callees: None,
          siblings: None,
          related_memories: None,
          timeline_depth: None,
          total_chunks: Some(chunk.total_chunks),
        };

        all_results.push(ExploreResult {
          id: chunk.id.to_string(),
          result_type: "doc".to_string(),
          file: Some(chunk.source.clone()),
          lines: None,
          preview: truncate_preview(&chunk.content, 200),
          symbols: vec![chunk.title.clone()],
          language: None,
          hints,
          context: None,
          score: similarity,
        });
      }
    }

    // Sort all results by score
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Expand top N results
    for (i, result) in all_results.iter_mut().enumerate() {
      if i >= expand_top {
        break;
      }

      match result.result_type.as_str() {
        "code" => {
          if let Some(expanded) = self.expand_code_result(&db, &result.id, depth).await {
            result.context = Some(expanded);
          }
        }
        "memory" => {
          // Memory expansion is handled differently - include in context tool
        }
        "doc" => {
          // Doc expansion is handled differently - include in context tool
        }
        _ => {}
      }
    }

    // Generate suggestions
    let content_words = extract_content_words(&result_content.join(" "), 20);
    let suggestions = generate_suggestions(&args.query, &result_symbols, &content_words, max_suggestions);

    let response = ExploreResponse {
      results: all_results,
      counts,
      suggestions,
    };

    // Return text or JSON based on format parameter
    let use_text = args.format.as_deref() == Some("text");
    if use_text {
      Response::success(
        request.id,
        serde_json::Value::String(format_explore_response(&response)),
      )
    } else {
      Response::success(request.id, serde_json::to_value(response).unwrap_or_default())
    }
  }

  /// Context tool: get comprehensive context for any explore result.
  pub async fn context(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      id: Option<String>,
      #[serde(default)]
      ids: Option<Vec<String>>,
      #[serde(default)]
      depth: Option<usize>,
      #[serde(default)]
      cwd: Option<String>,
      /// Output format: "json" (default) or "text" (human-readable)
      #[serde(default)]
      format: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    // Must provide either id or ids
    let ids: Vec<String> = if let Some(id) = args.id {
      vec![id]
    } else if let Some(ids) = args.ids {
      ids
    } else {
      return Response::error(request.id, -32602, "Must provide 'id' or 'ids' parameter");
    };

    if ids.is_empty() {
      return Response::error(request.id, -32602, "Must provide at least one ID");
    }

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Load config for defaults
    let config = Config::load_for_project(&project_path);
    let max_batch = config.search.context_max_batch;
    let depth = args.depth.unwrap_or(config.search.context_depth);

    if ids.len() > max_batch {
      return Response::error(
        request.id,
        -32602,
        &format!(
          "Too many IDs (max {}). Use multiple calls for larger batches.",
          max_batch
        ),
      );
    }

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let mut code_contexts: Vec<CodeContext> = Vec::new();
    let mut memory_contexts: Vec<MemoryContext> = Vec::new();
    let mut doc_contexts: Vec<DocContext> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for id in &ids {
      // Try to detect type and fetch context
      match self.fetch_context(&db, id, depth).await {
        Ok(ContextResult::Code(ctx)) => code_contexts.push(ctx),
        Ok(ContextResult::Memory(ctx)) => memory_contexts.push(ctx),
        Ok(ContextResult::Doc(ctx)) => doc_contexts.push(ctx),
        Err(e) => errors.push(format!("{}: {}", id, e)),
      }
    }

    if !errors.is_empty() && code_contexts.is_empty() && memory_contexts.is_empty() && doc_contexts.is_empty() {
      return Response::error(request.id, -32000, &errors.join("; "));
    }

    // Build response based on what we found
    let response = if !code_contexts.is_empty() && memory_contexts.is_empty() && doc_contexts.is_empty() {
      ContextResponse::Code { items: code_contexts }
    } else if code_contexts.is_empty() && !memory_contexts.is_empty() && doc_contexts.is_empty() {
      ContextResponse::Memory { items: memory_contexts }
    } else if code_contexts.is_empty() && memory_contexts.is_empty() && !doc_contexts.is_empty() {
      ContextResponse::Doc { items: doc_contexts }
    } else {
      ContextResponse::Mixed {
        code: code_contexts,
        memories: memory_contexts,
        docs: doc_contexts,
      }
    };

    // Return text or JSON based on format parameter
    let use_text = args.format.as_deref() == Some("text");
    if use_text {
      Response::success(
        request.id,
        serde_json::Value::String(format_context_response(&response)),
      )
    } else {
      Response::success(request.id, serde_json::to_value(response).unwrap_or_default())
    }
  }

  // ========================================================================
  // Helper methods
  // ========================================================================

  /// Compute navigation hints for a code chunk.
  async fn compute_code_hints(&self, db: &db::ProjectDb, chunk: &CodeChunk) -> ExploreHints {
    // Count callers
    let callers = self.count_callers(db, chunk).await;

    // Count callees (already in the chunk)
    let callees = chunk.calls.len();

    // Count siblings (other chunks in same file)
    let siblings = db
      .list_code_chunks(
        Some(&format!("file_path = '{}'", chunk.file_path.replace('\'', "''"))),
        None,
      )
      .await
      .map(|chunks| chunks.len().saturating_sub(1))
      .unwrap_or(0);

    // Count related memories
    let related_memories = self.count_related_memories(db, chunk).await;

    ExploreHints {
      callers: Some(callers),
      callees: Some(callees),
      siblings: Some(siblings),
      related_memories: Some(related_memories),
      timeline_depth: None,
      total_chunks: None,
    }
  }

  /// Count callers for a code chunk.
  ///
  /// Uses the pre-computed caller_count field when available, falling back
  /// to the code_references table for efficient indexed lookups.
  async fn count_callers(&self, db: &db::ProjectDb, chunk: &CodeChunk) -> usize {
    // Use pre-computed count if available
    if chunk.caller_count > 0 {
      return chunk.caller_count as usize;
    }

    // Fall back to code_references table lookup
    db.count_callers_for_symbols(&chunk.symbols).await.unwrap_or(0)
  }

  /// Count related memories for a code chunk.
  async fn count_related_memories(&self, db: &db::ProjectDb, chunk: &CodeChunk) -> usize {
    // Search for memories mentioning the file or symbols
    let mut count = 0;

    // Check file path mentions
    let file_name = std::path::Path::new(&chunk.file_path)
      .file_name()
      .map(|s| s.to_string_lossy().to_string())
      .unwrap_or_default();

    if !file_name.is_empty()
      && let Ok(memories) = db
        .list_memories(
          Some(&format!(
            "is_deleted = false AND content LIKE '%{}%'",
            file_name.replace('\'', "''")
          )),
          Some(10),
        )
        .await
    {
      count += memories.len();
    }

    // Check symbol mentions
    for symbol in &chunk.symbols {
      if let Ok(memories) = db
        .list_memories(
          Some(&format!(
            "is_deleted = false AND content LIKE '%{}%'",
            symbol.replace('\'', "''")
          )),
          Some(10),
        )
        .await
      {
        count += memories.len();
      }
    }

    count
  }

  /// Compute navigation hints for a memory.
  async fn compute_memory_hints(&self, db: &db::ProjectDb, memory: &Memory) -> ExploreHints {
    // Count related memories
    let related = db.get_all_relationships(&memory.id).await.map(|r| r.len()).unwrap_or(0);

    // Check timeline depth
    let timeline_depth = 5; // Default

    ExploreHints {
      callers: None,
      callees: None,
      siblings: None,
      related_memories: Some(related),
      timeline_depth: Some(timeline_depth),
      total_chunks: None,
    }
  }

  /// Expand a code result with full context.
  async fn expand_code_result(&self, db: &db::ProjectDb, chunk_id: &str, depth: usize) -> Option<ExpandedContext> {
    // Look up the chunk
    let chunk = match db.get_code_chunk_by_id_or_prefix(chunk_id).await {
      Ok(Some(c)) => c,
      _ => return None,
    };

    let content = chunk.content.clone();

    // Fetch all context in parallel for better performance
    let (callers, callees, siblings, memories) = tokio::join!(
      self.get_callers(db, &chunk, depth),
      self.get_callees(db, &chunk, depth),
      self.get_siblings(db, &chunk, depth),
      self.get_related_memories(db, &chunk, depth)
    );

    Some(ExpandedContext {
      content,
      callers,
      callees,
      siblings,
      memories,
    })
  }

  /// Get callers for a code chunk.
  ///
  /// Uses the code_references table for efficient indexed lookups instead of
  /// LIKE queries on JSON columns.
  async fn get_callers(&self, db: &db::ProjectDb, chunk: &CodeChunk, limit: usize) -> Vec<CallInfo> {
    let mut callers = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    // Get caller chunk IDs from references table
    if let Ok(caller_refs) = db.get_callers_for_symbols(&chunk.symbols, limit * 2).await {
      for (source_chunk_id, _target_symbol) in caller_refs {
        // Skip self-references and duplicates
        if source_chunk_id == chunk.id.to_string() || seen_ids.contains(&source_chunk_id) {
          continue;
        }
        seen_ids.insert(source_chunk_id.clone());

        // Look up the caller chunk to get full details
        if let Ok(Some(caller)) = db.get_code_chunk_by_id_or_prefix(&source_chunk_id).await {
          callers.push(CallInfo {
            id: caller.id.to_string(),
            file: caller.file_path.clone(),
            lines: (caller.start_line, caller.end_line),
            preview: truncate_preview(&caller.content, 100),
            symbols: Some(caller.symbols.clone()),
          });

          if callers.len() >= limit {
            break;
          }
        }
      }
    }

    callers
  }

  /// Get callees for a code chunk.
  ///
  /// Uses the code_references table when possible, falling back to symbol
  /// lookup for definitions. The references table stores the target_chunk_id
  /// when resolved.
  async fn get_callees(&self, db: &db::ProjectDb, chunk: &CodeChunk, limit: usize) -> Vec<CallInfo> {
    let mut callees = Vec::new();
    let mut seen_symbols = std::collections::HashSet::new();

    // Get callees from references table
    if let Ok(callee_refs) = db.get_callees_for_chunk(&chunk.id.to_string(), limit * 2).await {
      for (target_symbol, target_chunk_id) in callee_refs {
        // Skip duplicates
        if seen_symbols.contains(&target_symbol) {
          continue;
        }
        seen_symbols.insert(target_symbol.clone());

        // If we have a resolved target_chunk_id, use it directly
        if let Some(ref chunk_id) = target_chunk_id
          && let Ok(Some(callee)) = db.get_code_chunk_by_id_or_prefix(chunk_id).await
        {
          if callee.id != chunk.id {
            callees.push(CallInfo {
              id: callee.id.to_string(),
              file: callee.file_path.clone(),
              lines: (callee.start_line, callee.end_line),
              preview: truncate_preview(&callee.content, 100),
              symbols: Some(callee.symbols.clone()),
            });
          }
        } else {
          // Fall back to symbol lookup (still uses LIKE, but only for unresolved refs)
          let filter = format!("symbols LIKE '%\"{}%'", target_symbol.replace('\'', "''"));
          if let Ok(chunks) = db.list_code_chunks(Some(&filter), Some(1)).await
            && let Some(callee) = chunks.into_iter().next()
            && callee.id != chunk.id
            && callee.symbols.iter().any(|s| s == &target_symbol)
          {
            callees.push(CallInfo {
              id: callee.id.to_string(),
              file: callee.file_path.clone(),
              lines: (callee.start_line, callee.end_line),
              preview: truncate_preview(&callee.content, 100),
              symbols: Some(callee.symbols.clone()),
            });
          }
        }

        if callees.len() >= limit {
          break;
        }
      }
    }

    callees
  }

  /// Get sibling symbols in the same file.
  async fn get_siblings(&self, db: &db::ProjectDb, chunk: &CodeChunk, limit: usize) -> Vec<SiblingInfo> {
    let mut siblings = Vec::new();

    if let Ok(chunks) = db
      .list_code_chunks(
        Some(&format!("file_path = '{}'", chunk.file_path.replace('\'', "''"))),
        None,
      )
      .await
    {
      for sibling in chunks {
        if sibling.id != chunk.id {
          for symbol in &sibling.symbols {
            siblings.push(SiblingInfo {
              symbol: symbol.clone(),
              kind: format!("{:?}", sibling.chunk_type).to_lowercase(),
              line: sibling.start_line,
            });
          }
        }

        if siblings.len() >= limit {
          break;
        }
      }
    }

    siblings.truncate(limit);
    siblings
  }

  /// Get related memories for a code chunk.
  async fn get_related_memories(&self, db: &db::ProjectDb, chunk: &CodeChunk, limit: usize) -> Vec<RelatedMemoryInfo> {
    let mut memories = Vec::new();

    // Search by file name
    let file_name = std::path::Path::new(&chunk.file_path)
      .file_name()
      .map(|s| s.to_string_lossy().to_string())
      .unwrap_or_default();

    if !file_name.is_empty()
      && let Ok(found) = db
        .list_memories(
          Some(&format!(
            "is_deleted = false AND content LIKE '%{}%'",
            file_name.replace('\'', "''")
          )),
          Some(limit),
        )
        .await
    {
      for memory in found {
        memories.push(RelatedMemoryInfo {
          id: memory.id.to_string(),
          content: truncate_preview(&memory.content, 150),
          memory_type: format!("{:?}", memory.memory_type).to_lowercase(),
          sector: format!("{:?}", memory.sector).to_lowercase(),
        });
      }
    }

    // Search by symbol names
    for symbol in &chunk.symbols {
      if memories.len() >= limit {
        break;
      }

      if let Ok(found) = db
        .list_memories(
          Some(&format!(
            "is_deleted = false AND content LIKE '%{}%'",
            symbol.replace('\'', "''")
          )),
          Some(limit - memories.len()),
        )
        .await
      {
        for memory in found {
          if !memories.iter().any(|m| m.id == memory.id.to_string()) {
            memories.push(RelatedMemoryInfo {
              id: memory.id.to_string(),
              content: truncate_preview(&memory.content, 150),
              memory_type: format!("{:?}", memory.memory_type).to_lowercase(),
              sector: format!("{:?}", memory.sector).to_lowercase(),
            });
          }
        }
      }
    }

    memories.truncate(limit);
    memories
  }

  /// Fetch full context for an ID (auto-detects type).
  async fn fetch_context(&self, db: &db::ProjectDb, id: &str, depth: usize) -> Result<ContextResult, String> {
    // Validate ID length for prefix matching
    if id.len() < 6 {
      return Err("ID must be at least 6 characters".to_string());
    }

    // Try code chunk first
    match db.get_code_chunk_by_id_or_prefix(id).await {
      Ok(Some(chunk)) => {
        return Ok(ContextResult::Code(self.build_code_context(db, chunk, depth).await));
      }
      Err(db::DbError::AmbiguousPrefix { prefix, count }) => {
        return Err(format!(
          "Ambiguous prefix '{}' matches {} items. Use more characters.",
          prefix, count
        ));
      }
      _ => {}
    }

    // Try memory
    if let Ok(memory_id) = id.parse::<MemoryId>()
      && let Ok(Some(memory)) = db.get_memory(&memory_id).await
    {
      return Ok(ContextResult::Memory(
        self.build_memory_context(db, memory, depth).await,
      ));
    }

    // Try document chunk
    match db.get_document_chunk_by_id_or_prefix(id).await {
      Ok(Some(chunk)) => {
        return Ok(ContextResult::Doc(self.build_doc_context(db, chunk, depth).await));
      }
      Err(db::DbError::AmbiguousPrefix { prefix, count }) => {
        return Err(format!(
          "Ambiguous prefix '{}' matches {} items. Use more characters.",
          prefix, count
        ));
      }
      _ => {}
    }

    Err(format!("Item not found: {}", id))
  }

  /// Build full code context.
  async fn build_code_context(&self, db: &db::ProjectDb, chunk: CodeChunk, depth: usize) -> CodeContext {
    // Fetch all context in parallel for better performance
    let (callers, callees, siblings, memories) = tokio::join!(
      self.get_callers(db, &chunk, depth),
      self.get_callees(db, &chunk, depth),
      self.get_siblings(db, &chunk, depth),
      self.get_related_memories(db, &chunk, depth)
    );

    // Extract signature (first line for functions)
    let signature = if chunk.chunk_type == engram_core::ChunkType::Function {
      chunk.content.lines().next().map(String::from)
    } else {
      None
    };

    CodeContext {
      id: chunk.id.to_string(),
      file: chunk.file_path,
      content: chunk.content,
      language: format!("{:?}", chunk.language).to_lowercase(),
      lines: (chunk.start_line, chunk.end_line),
      symbols: chunk.symbols,
      imports: chunk.imports,
      signature,
      callers,
      callees,
      siblings,
      memories,
    }
  }

  /// Build full memory context.
  async fn build_memory_context(&self, db: &db::ProjectDb, memory: Memory, depth: usize) -> MemoryContext {
    // Fetch timeline and related in parallel
    let (timeline, related) = tokio::join!(
      self.get_memory_timeline(db, &memory, depth),
      self.get_related_memories_for_memory(db, &memory, depth)
    );

    MemoryContext {
      id: memory.id.to_string(),
      content: memory.content,
      sector: format!("{:?}", memory.sector).to_lowercase(),
      memory_type: format!("{:?}", memory.memory_type).to_lowercase(),
      salience: memory.salience,
      created_at: memory.created_at.to_rfc3339(),
      timeline,
      related,
    }
  }

  /// Get timeline around a memory.
  async fn get_memory_timeline(&self, db: &db::ProjectDb, memory: &Memory, depth: usize) -> TimelineContext {
    let mut before = Vec::new();
    let mut after = Vec::new();

    // Get memories before this one
    let before_filter = format!(
      "is_deleted = false AND created_at < '{}' ORDER BY created_at DESC",
      memory.created_at.to_rfc3339()
    );
    if let Ok(memories) = db.list_memories(Some(&before_filter), Some(depth)).await {
      for m in memories {
        before.push(TimelineEntry {
          id: m.id.to_string(),
          content: truncate_preview(&m.content, 100),
          memory_type: format!("{:?}", m.memory_type).to_lowercase(),
          created_at: m.created_at.to_rfc3339(),
        });
      }
    }

    // Get memories after this one
    let after_filter = format!(
      "is_deleted = false AND created_at > '{}' ORDER BY created_at ASC",
      memory.created_at.to_rfc3339()
    );
    if let Ok(memories) = db.list_memories(Some(&after_filter), Some(depth)).await {
      for m in memories {
        after.push(TimelineEntry {
          id: m.id.to_string(),
          content: truncate_preview(&m.content, 100),
          memory_type: format!("{:?}", m.memory_type).to_lowercase(),
          created_at: m.created_at.to_rfc3339(),
        });
      }
    }

    TimelineContext { before, after }
  }

  /// Get related memories for a memory via relationships.
  async fn get_related_memories_for_memory(
    &self,
    db: &db::ProjectDb,
    memory: &Memory,
    limit: usize,
  ) -> Vec<RelatedMemoryInfo> {
    let mut related = Vec::new();

    if let Ok(relationships) = db.get_all_relationships(&memory.id).await {
      for rel in relationships.into_iter().take(limit) {
        let other_id = if rel.from_memory_id == memory.id {
          rel.to_memory_id
        } else {
          rel.from_memory_id
        };

        if let Ok(Some(other)) = db.get_memory(&other_id).await {
          related.push(RelatedMemoryInfo {
            id: other.id.to_string(),
            content: truncate_preview(&other.content, 100),
            memory_type: format!("{:?}", other.memory_type).to_lowercase(),
            sector: format!("{:?}", other.sector).to_lowercase(),
          });
        }
      }
    }

    related
  }

  /// Build full document context.
  async fn build_doc_context(&self, db: &db::ProjectDb, chunk: DocumentChunk, depth: usize) -> DocContext {
    let mut before = Vec::new();
    let mut after = Vec::new();

    // Get adjacent chunks
    if let Ok(chunks) = db
      .get_adjacent_document_chunks(&chunk.document_id, chunk.chunk_index, depth, depth)
      .await
    {
      for adj in chunks {
        if adj.chunk_index < chunk.chunk_index {
          before.push(DocChunkEntry {
            chunk_index: adj.chunk_index,
            content: adj.content,
          });
        } else if adj.chunk_index > chunk.chunk_index {
          after.push(DocChunkEntry {
            chunk_index: adj.chunk_index,
            content: adj.content,
          });
        }
      }
    }

    DocContext {
      id: chunk.id.to_string(),
      title: chunk.title,
      content: chunk.content,
      source: chunk.source,
      chunk_index: chunk.chunk_index,
      total_chunks: chunk.total_chunks,
      before,
      after,
    }
  }
}

/// Helper enum for context results
enum ContextResult {
  Code(CodeContext),
  Memory(MemoryContext),
  Doc(DocContext),
}

/// Truncate content to a preview length.
fn truncate_preview(content: &str, max_len: usize) -> String {
  let content = content.trim();
  if content.len() <= max_len {
    content.to_string()
  } else {
    format!("{}...", &content[..max_len.saturating_sub(3)])
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::collections::HashMap;

  #[test]
  fn test_truncate_preview() {
    assert_eq!(truncate_preview("short", 10), "short");
    assert_eq!(truncate_preview("this is a longer string", 10), "this is...");
  }

  #[test]
  fn test_truncate_preview_exact_length() {
    assert_eq!(truncate_preview("exactly10!", 10), "exactly10!");
  }

  #[test]
  fn test_truncate_preview_whitespace() {
    assert_eq!(truncate_preview("  trimmed  ", 20), "trimmed");
  }

  #[test]
  fn test_truncate_preview_empty() {
    assert_eq!(truncate_preview("", 10), "");
  }

  #[test]
  fn test_explore_scope_parse() {
    let scope: ExploreScope = serde_json::from_str(r#""code""#).unwrap();
    assert_eq!(scope, ExploreScope::Code);

    let scope: ExploreScope = serde_json::from_str(r#""all""#).unwrap();
    assert_eq!(scope, ExploreScope::All);
  }

  #[test]
  fn test_explore_scope_parse_all_variants() {
    let scope: ExploreScope = serde_json::from_str(r#""memory""#).unwrap();
    assert_eq!(scope, ExploreScope::Memory);

    let scope: ExploreScope = serde_json::from_str(r#""docs""#).unwrap();
    assert_eq!(scope, ExploreScope::Docs);
  }

  #[test]
  fn test_explore_scope_default() {
    let scope = ExploreScope::default();
    assert_eq!(scope, ExploreScope::All);
  }

  #[test]
  fn test_explore_hints_serialization() {
    let hints = ExploreHints {
      callers: Some(5),
      callees: Some(3),
      siblings: Some(2),
      related_memories: Some(1),
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
  }

  #[test]
  fn test_explore_hints_empty() {
    let hints = ExploreHints {
      callers: None,
      callees: None,
      siblings: None,
      related_memories: None,
      timeline_depth: None,
      total_chunks: None,
    };

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
        callees: None,
        siblings: None,
        related_memories: None,
        timeline_depth: None,
        total_chunks: None,
      },
      context: None,
      score: 0.95,
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
    // Use approximate comparison for floats
    let score = json["score"].as_f64().unwrap();
    assert!((score - 0.95).abs() < 0.001, "score should be approximately 0.95");
    // Context should be skipped when None
    assert!(json.get("context").is_none());
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
        callers: None,
        callees: None,
        siblings: None,
        related_memories: Some(3),
        timeline_depth: Some(5),
        total_chunks: None,
      },
      context: None,
      score: 0.8,
    };

    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["type"], "memory");
    // Optional fields should be skipped when None
    assert!(json.get("file").is_none());
    assert!(json.get("lines").is_none());
    assert!(json.get("language").is_none());
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
          timeline_depth: None,
          total_chunks: None,
        },
        context: None,
        score: 1.0,
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
  fn test_call_info_serialization() {
    let info = CallInfo {
      id: "caller123".to_string(),
      file: "caller.rs".to_string(),
      lines: (5, 15),
      preview: "fn caller() { callee() }".to_string(),
      symbols: Some(vec!["caller".to_string()]),
    };

    let json = serde_json::to_value(&info).unwrap();
    assert_eq!(json["id"], "caller123");
    assert_eq!(json["file"], "caller.rs");
    assert_eq!(json["lines"][0], 5);
    assert_eq!(json["lines"][1], 15);
    assert_eq!(json["symbols"][0], "caller");
  }

  #[test]
  fn test_sibling_info_serialization() {
    let info = SiblingInfo {
      symbol: "sibling_fn".to_string(),
      kind: "function".to_string(),
      line: 42,
    };

    let json = serde_json::to_value(&info).unwrap();
    assert_eq!(json["symbol"], "sibling_fn");
    assert_eq!(json["kind"], "function");
    assert_eq!(json["line"], 42);
  }

  #[test]
  fn test_related_memory_info_serialization() {
    let info = RelatedMemoryInfo {
      id: "mem456".to_string(),
      content: "Related memory content".to_string(),
      memory_type: "decision".to_string(),
      sector: "semantic".to_string(),
    };

    let json = serde_json::to_value(&info).unwrap();
    assert_eq!(json["id"], "mem456");
    assert_eq!(json["content"], "Related memory content");
    assert_eq!(json["type"], "decision");
    assert_eq!(json["sector"], "semantic");
  }

  #[test]
  fn test_expanded_context_serialization() {
    let ctx = ExpandedContext {
      content: "Full content here".to_string(),
      callers: vec![CallInfo {
        id: "c1".to_string(),
        file: "file.rs".to_string(),
        lines: (1, 5),
        preview: "preview".to_string(),
        symbols: None,
      }],
      callees: vec![],
      siblings: vec![],
      memories: vec![],
    };

    let json = serde_json::to_value(&ctx).unwrap();
    assert_eq!(json["content"], "Full content here");
    assert_eq!(json["callers"].as_array().unwrap().len(), 1);
    // Empty arrays should be skipped
    assert!(json.get("callees").is_none());
    assert!(json.get("siblings").is_none());
    assert!(json.get("memories").is_none());
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
      }],
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["type"], "memory");
    assert_eq!(json["items"][0]["salience"], 0.75);
  }

  #[test]
  fn test_context_response_doc() {
    let response = ContextResponse::Doc {
      items: vec![DocContext {
        id: "doc1".to_string(),
        title: "Document Title".to_string(),
        content: "Document content".to_string(),
        source: "doc.md".to_string(),
        chunk_index: 0,
        total_chunks: 3,
        before: vec![],
        after: vec![],
      }],
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["type"], "doc");
    assert_eq!(json["items"][0]["title"], "Document Title");
    assert_eq!(json["items"][0]["total_chunks"], 3);
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

  #[test]
  fn test_timeline_context_serialization() {
    let timeline = TimelineContext {
      before: vec![TimelineEntry {
        id: "before1".to_string(),
        content: "Earlier memory".to_string(),
        memory_type: "observation".to_string(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
      }],
      after: vec![TimelineEntry {
        id: "after1".to_string(),
        content: "Later memory".to_string(),
        memory_type: "decision".to_string(),
        created_at: "2024-01-02T00:00:00Z".to_string(),
      }],
    };

    let json = serde_json::to_value(&timeline).unwrap();
    assert_eq!(json["before"].as_array().unwrap().len(), 1);
    assert_eq!(json["after"].as_array().unwrap().len(), 1);
    assert_eq!(json["before"][0]["id"], "before1");
    assert_eq!(json["after"][0]["id"], "after1");
  }

  #[test]
  fn test_doc_chunk_entry_serialization() {
    let entry = DocChunkEntry {
      chunk_index: 2,
      content: "Chunk content here".to_string(),
    };

    let json = serde_json::to_value(&entry).unwrap();
    assert_eq!(json["chunk_index"], 2);
    assert_eq!(json["content"], "Chunk content here");
  }
}
