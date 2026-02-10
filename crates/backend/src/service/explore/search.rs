//! Search functionality for the explore service.
//!
//! This module provides the core search implementation with parallel execution
//! across code, memories, and documents.

use std::collections::HashMap;

use tracing::{debug, warn};

use super::{
  types::{ExpandedContext, ExploreContext, ExploreHints, ExploreResponse, ExploreResult, SearchParams},
  util::{semantic_code_preview, truncate_preview},
};
use crate::{
  db::ProjectDb,
  domain::{code::CodeChunk, document::DocumentChunk, memory::Memory},
  rerank::{RerankCandidate, RerankRequest, RerankerProvider},
  service::util::{ServiceError, fusion},
};

// ============================================================================
// Core Search Implementation
// ============================================================================

/// Minimum similarity score threshold for results.
/// Results below this threshold are filtered out as noise.
const MIN_SCORE_THRESHOLD: f32 = 0.15;

/// Unified search across code, memories, and documents.
///
/// Executes searches in parallel using `tokio::join!` for performance.
///
/// # Arguments
/// * `ctx` - Explore context with database and embedding provider
/// * `params` - Search parameters
///
/// # Returns
/// * `Ok(ExploreResponse)` - Unified search results with suggestions
/// * `Err(ServiceError)` - If search fails
pub async fn search(ctx: &ExploreContext<'_>, params: &SearchParams) -> Result<ExploreResponse, ServiceError> {
  if params.query.trim().is_empty() {
    return Err(ServiceError::validation("Query cannot be empty"));
  }

  let query_embedding = get_embedding(ctx, &params.query).await?;

  let mut all_results: Vec<ExploreResult> = Vec::new();
  let mut counts: HashMap<String, usize> = HashMap::new();

  // Determine which scopes to search
  let search_code = params.scope.includes_code();
  let search_memory = params.scope.includes_memory();
  let search_docs = params.scope.includes_docs();

  let fts_enabled = ctx.search_config.is_some_and(|c| c.fts_enabled);
  let rrf_k = ctx.search_config.map_or(60, |c| c.rrf_k);
  let oversample = if fts_enabled { 50 } else { params.limit };

  // Phase 1: Run all domain searches in parallel (vector + FTS fusion, no reranking yet)
  let (code_results, memory_results, doc_results) = tokio::join!(
    search_code_domain(
      ctx.db,
      &query_embedding,
      &params.query,
      oversample,
      search_code,
      fts_enabled,
      rrf_k
    ),
    search_memory_domain(
      ctx.db,
      &query_embedding,
      &params.query,
      oversample,
      search_memory,
      fts_enabled,
      rrf_k
    ),
    search_docs_domain(
      ctx.db,
      &query_embedding,
      &params.query,
      oversample,
      search_docs,
      fts_enabled,
      rrf_k
    ),
  );

  // Phase 2: Cross-domain reranking on the combined corpus
  let (code_results, memory_results, doc_results) = if let Some(reranker) = ctx.reranker {
    let rerank_candidates = ctx.search_config.map_or(30, |c| c.rerank_candidates);
    rerank_cross_domain(
      code_results,
      memory_results,
      doc_results,
      &params.query,
      rerank_candidates,
      reranker,
    )
    .await
  } else {
    (code_results, memory_results, doc_results)
  };

  // Phase 3: Process results into ExploreResult structs
  if search_code {
    counts.insert("code".to_string(), code_results.len());

    for (chunk, score) in code_results {
      let hints = compute_code_hints(ctx.db, &chunk).await;
      let preview = semantic_code_preview(&chunk, 300);

      let docstring = chunk.docstring.as_ref().map(|d| {
        d.lines()
          .filter(|l| !l.trim().is_empty())
          .take(2)
          .collect::<Vec<_>>()
          .join(" ")
      });

      let imports: Vec<String> = chunk.imports.iter().take(5).cloned().collect();
      let calls: Vec<String> = chunk.calls.iter().take(5).cloned().collect();

      all_results.push(ExploreResult {
        id: chunk.id.to_string(),
        result_type: "code".to_string(),
        file: Some(chunk.file_path.clone()),
        lines: Some((chunk.start_line, chunk.end_line)),
        preview,
        symbols: chunk.symbols.clone(),
        language: Some(format!("{:?}", chunk.language).to_lowercase()),
        hints,
        context: None,
        score,
        definition_kind: chunk.definition_kind.clone(),
        signature: chunk.signature.clone(),
        docstring,
        parent: chunk.parent_definition.clone(),
        imports,
        calls,
      });
    }
  }

  if search_memory {
    counts.insert("memory".to_string(), memory_results.len());

    for (memory, score) in memory_results {
      let hints = compute_memory_hints(ctx.db, &memory).await;

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
        score: score * memory.salience,
        definition_kind: None,
        signature: None,
        docstring: None,
        parent: None,
        imports: vec![],
        calls: vec![],
      });
    }
  }

  if search_docs {
    counts.insert("docs".to_string(), doc_results.len());

    for (chunk, score) in doc_results {
      let hints = ExploreHints {
        total_chunks: Some(chunk.total_chunks),
        related_code: None,
        ..Default::default()
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
        score,
        definition_kind: None,
        signature: None,
        docstring: None,
        parent: None,
        imports: vec![],
        calls: vec![],
      });
    }
  }

  // Sort all results by score and filter out low-score noise
  all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
  all_results.retain(|r| r.score >= MIN_SCORE_THRESHOLD);

  // Expand top N results
  for (i, result) in all_results.iter_mut().enumerate() {
    if i >= params.expand_top {
      break;
    }

    if result.result_type == "code"
      && let Some(expanded) = expand_code_result(ctx.db, &result.id, params.depth).await
    {
      result.context = Some(expanded);
    }
  }

  Ok(ExploreResponse {
    results: all_results,
    counts,
  })
}

/// Get an embedding for the given text, if a provider is available
async fn get_embedding(ctx: &ExploreContext<'_>, text: &str) -> Result<Vec<f32>, ServiceError> {
  // Query mode - this is used for explore search queries
  Ok(
    ctx
      .embedding
      .embed(text, crate::embedding::EmbeddingMode::Query)
      .await?,
  )
}

// ============================================================================
// Domain Search Helpers
// ============================================================================

/// Search code chunks with hybrid FTS + vector search and RRF fusion.
///
/// Returns `(CodeChunk, score)` where score is a similarity (higher = better).
#[allow(clippy::too_many_arguments)]
async fn search_code_domain(
  db: &ProjectDb,
  embedding: &[f32],
  query: &str,
  limit: usize,
  enabled: bool,
  fts_enabled: bool,
  rrf_k: u32,
) -> Vec<(CodeChunk, f32)> {
  if !enabled {
    return Vec::new();
  }

  if fts_enabled {
    let (vector_results, fts_results) = tokio::join!(
      db.search_code_chunks(embedding, limit, None),
      db.fts_search_code_chunks(query, limit, None),
    );

    let vector_results = vector_results.unwrap_or_default();
    let fts_results = fts_results.unwrap_or_else(|e| {
      warn!(error = %e, "FTS code search failed in explore, falling back to vector-only");
      Vec::new()
    });

    debug!(
      vector_count = vector_results.len(),
      fts_count = fts_results.len(),
      "Explore hybrid code retrieval complete"
    );

    fuse_rrf(vector_results, fts_results, rrf_k)
  } else {
    db.search_code_chunks(embedding, limit, None)
      .await
      .unwrap_or_default()
      .into_iter()
      .map(|(chunk, dist)| (chunk, 1.0 - dist.min(1.0)))
      .collect()
  }
}

/// Search memories with hybrid FTS + vector search and RRF fusion.
///
/// Returns `(Memory, score)` where score is a similarity (higher = better).
#[allow(clippy::too_many_arguments)]
async fn search_memory_domain(
  db: &ProjectDb,
  embedding: &[f32],
  query: &str,
  limit: usize,
  enabled: bool,
  fts_enabled: bool,
  rrf_k: u32,
) -> Vec<(Memory, f32)> {
  if !enabled {
    return Vec::new();
  }

  let deleted_filter = Some("is_deleted = false");

  if fts_enabled {
    let (vector_results, fts_results) = tokio::join!(
      db.search_memories(embedding, limit, deleted_filter),
      db.fts_search_memories(query, limit, deleted_filter),
    );

    let vector_results = vector_results.unwrap_or_default();
    let fts_results = fts_results.unwrap_or_else(|e| {
      warn!(error = %e, "FTS memory search failed in explore, falling back to vector-only");
      Vec::new()
    });

    debug!(
      vector_count = vector_results.len(),
      fts_count = fts_results.len(),
      "Explore hybrid memory retrieval complete"
    );

    fuse_rrf(vector_results, fts_results, rrf_k)
  } else {
    crate::service::memory::search::search_by_embedding(db, embedding, limit, None)
      .await
      .unwrap_or_default()
      .into_iter()
      .map(|(mem, dist)| (mem, 1.0 - dist.min(1.0)))
      .collect()
  }
}

/// Search documents with hybrid FTS + vector search and RRF fusion.
///
/// Returns `(DocumentChunk, score)` where score is a similarity (higher = better).
#[allow(clippy::too_many_arguments)]
async fn search_docs_domain(
  db: &ProjectDb,
  embedding: &[f32],
  query: &str,
  limit: usize,
  enabled: bool,
  fts_enabled: bool,
  rrf_k: u32,
) -> Vec<(DocumentChunk, f32)> {
  if !enabled {
    return Vec::new();
  }

  if fts_enabled {
    let (vector_results, fts_results) = tokio::join!(
      db.search_documents(embedding, limit, None),
      db.fts_search_documents(query, limit, None),
    );

    let vector_results = vector_results.unwrap_or_default();
    let fts_results = fts_results.unwrap_or_else(|e| {
      warn!(error = %e, "FTS document search failed in explore, falling back to vector-only");
      Vec::new()
    });

    debug!(
      vector_count = vector_results.len(),
      fts_count = fts_results.len(),
      "Explore hybrid document retrieval complete"
    );

    fuse_rrf(vector_results, fts_results, rrf_k)
  } else {
    db.search_documents(embedding, limit, None)
      .await
      .unwrap_or_default()
      .into_iter()
      .map(|(doc, dist)| (doc, 1.0 - dist.min(1.0)))
      .collect()
  }
}

// ============================================================================
// Hybrid Search Helpers
// ============================================================================

/// Fuse vector + FTS results with RRF into a single scored list.
///
/// Returns items with RRF scores (higher = better).
fn fuse_rrf<T: Clone>(vector_results: Vec<(T, f32)>, fts_results: Vec<(T, f32)>, rrf_k: u32) -> Vec<(T, f32)> {
  let mut item_map: HashMap<String, T> = HashMap::new();
  let mut vector_ids: Vec<String> = Vec::with_capacity(vector_results.len());
  let mut fts_ids: Vec<String> = Vec::with_capacity(fts_results.len());

  for (i, (item, _)) in vector_results.iter().enumerate() {
    let key = format!("v{i}");
    item_map.insert(key.clone(), item.clone());
    vector_ids.push(key);
  }
  for (i, (item, _)) in fts_results.iter().enumerate() {
    let key = format!("f{i}");
    item_map.insert(key.clone(), item.clone());
    fts_ids.push(key);
  }

  let fused = fusion::reciprocal_rank_fusion(&[vector_ids, fts_ids], rrf_k);

  fused
    .into_iter()
    .filter_map(|(id, score)| item_map.remove(&id).map(|item| (item, score)))
    .collect()
}

/// Cross-domain reranking: merge all domain results into a single pool,
/// rerank once with the cross-encoder, then split back by domain.
async fn rerank_cross_domain(
  code_results: Vec<(CodeChunk, f32)>,
  memory_results: Vec<(Memory, f32)>,
  doc_results: Vec<(DocumentChunk, f32)>,
  query: &str,
  max_candidates: usize,
  reranker: &dyn RerankerProvider,
) -> (Vec<(CodeChunk, f32)>, Vec<(Memory, f32)>, Vec<(DocumentChunk, f32)>) {
  if !reranker.is_available() {
    warn!(
      provider = reranker.name(),
      "Reranker not available for explore, skipping cross-domain reranking"
    );
    return (code_results, memory_results, doc_results);
  }

  // Build a unified candidate list with domain-prefixed keys
  let mut candidates: Vec<(String, f32)> = Vec::new();
  let mut texts: HashMap<String, String> = HashMap::new();

  for (i, (chunk, score)) in code_results.iter().enumerate() {
    let key = format!("c{i}");
    let text = chunk
      .embedding_text
      .as_deref()
      .unwrap_or(&chunk.content)
      .chars()
      .take(4000)
      .collect();
    candidates.push((key.clone(), *score));
    texts.insert(key, text);
  }
  for (i, (mem, score)) in memory_results.iter().enumerate() {
    let key = format!("m{i}");
    let text = mem.content.chars().take(4000).collect();
    candidates.push((key.clone(), *score));
    texts.insert(key, text);
  }
  for (i, (doc, score)) in doc_results.iter().enumerate() {
    let key = format!("d{i}");
    let text = doc.content.chars().take(4000).collect();
    candidates.push((key.clone(), *score));
    texts.insert(key, text);
  }

  // Sort by score descending and take top N for reranking
  candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
  candidates.truncate(max_candidates);

  let max = reranker.max_candidates();
  let rerank_items: Vec<RerankCandidate> = candidates
    .iter()
    .take(max)
    .filter_map(|(key, _)| {
      texts.get(key).map(|text| RerankCandidate {
        id: key.clone(),
        text: text.clone(),
      })
    })
    .collect();

  if rerank_items.is_empty() {
    return (code_results, memory_results, doc_results);
  }

  let request = RerankRequest {
    query: query.to_string(),
    instruction: None,
    candidates: rerank_items,
    top_n: None,
  };

  let reranked = match reranker.rerank(request).await {
    Ok(response) => {
      debug!(
        duration_ms = response.duration_ms,
        results = response.results.len(),
        "Explore cross-domain reranking complete"
      );
      fusion::blend_scores(&candidates, &response.results)
    }
    Err(e) => {
      warn!(error = %e, "Explore cross-domain reranking failed, using original scores");
      return (code_results, memory_results, doc_results);
    }
  };

  // Build score lookup from reranked results
  let score_map: HashMap<String, f32> = reranked.into_iter().collect();

  // Apply reranked scores back to each domain
  let code_results: Vec<(CodeChunk, f32)> = code_results
    .into_iter()
    .enumerate()
    .map(|(i, (chunk, original))| {
      let key = format!("c{i}");
      let score = score_map.get(&key).copied().unwrap_or(original);
      (chunk, score)
    })
    .collect();

  let memory_results: Vec<(Memory, f32)> = memory_results
    .into_iter()
    .enumerate()
    .map(|(i, (mem, original))| {
      let key = format!("m{i}");
      let score = score_map.get(&key).copied().unwrap_or(original);
      (mem, score)
    })
    .collect();

  let doc_results: Vec<(DocumentChunk, f32)> = doc_results
    .into_iter()
    .enumerate()
    .map(|(i, (doc, original))| {
      let key = format!("d{i}");
      let score = score_map.get(&key).copied().unwrap_or(original);
      (doc, score)
    })
    .collect();

  (code_results, memory_results, doc_results)
}

// ============================================================================
// Hints Computation
// ============================================================================

/// Compute navigation hints for a code chunk.
async fn compute_code_hints(db: &ProjectDb, chunk: &CodeChunk) -> ExploreHints {
  // Use pre-computed caller count from chunk
  let callers = chunk.caller_count as usize;

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
  let related_memories = count_related_memories(db, chunk).await;

  ExploreHints {
    callers: Some(callers),
    callees: Some(callees),
    siblings: Some(siblings),
    related_memories: Some(related_memories),
    related_code: None, // Not applicable to code chunks
    timeline_depth: None,
    total_chunks: None,
  }
}

/// Count related memories for a code chunk.
async fn count_related_memories(db: &ProjectDb, chunk: &CodeChunk) -> usize {
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
async fn compute_memory_hints(db: &ProjectDb, memory: &Memory) -> ExploreHints {
  // Count related memories via relationships
  let related = db.get_all_relationships(&memory.id).await.map(|r| r.len()).unwrap_or(0);

  // For related_code, we could do a vector search but it's expensive for just a hint.
  // Instead, we set it to Some(0) to indicate the feature exists, and the actual
  // count will be computed when the full context is retrieved.
  // If the memory has an embedding, we know cross-domain search is possible.
  let has_embedding = db.get_memory_embedding(&memory.id).await.ok().flatten().is_some();

  ExploreHints {
    related_memories: Some(related),
    // Indicate related code search is available if memory has embedding
    related_code: if has_embedding { Some(0) } else { None },
    timeline_depth: Some(5), // Default
    ..Default::default()
  }
}

// ============================================================================
// Context Expansion (for search results)
// ============================================================================

/// Threshold for truncating large chunks in expanded context (in lines)
const EXPANSION_LINE_THRESHOLD: usize = 80;

/// Number of lines to show when truncating large chunks
const EXPANSION_PREVIEW_LINES: usize = 20;

/// Create adaptive content for a chunk - truncates large chunks to signature + preview.
fn adaptive_content(content: &str, signature: Option<&str>) -> String {
  let line_count = content.lines().count();

  if line_count <= EXPANSION_LINE_THRESHOLD {
    return content.to_string();
  }

  // Large chunk - show signature + first N lines + truncation indicator
  let mut result = String::new();

  // Include signature if available and not already at start of content
  if let Some(sig) = signature {
    let first_line = content.lines().next().unwrap_or("");
    if !first_line.trim().starts_with(sig.lines().next().unwrap_or("").trim()) {
      result.push_str(sig);
      result.push_str("\n\n");
    }
  }

  // Add first N lines of content
  let preview: String = content
    .lines()
    .take(EXPANSION_PREVIEW_LINES)
    .collect::<Vec<_>>()
    .join("\n");
  result.push_str(&preview);

  // Add truncation indicator
  let remaining = line_count - EXPANSION_PREVIEW_LINES;
  result.push_str(&format!(
    "\n\n... ({} more lines, use `context` tool for full content)",
    remaining
  ));

  result
}

/// Expand a code result with full context.
async fn expand_code_result(db: &ProjectDb, chunk_id: &str, depth: usize) -> Option<ExpandedContext> {
  // Look up the chunk
  let chunk = match db.get_code_chunk_by_id_or_prefix(chunk_id).await {
    Ok(Some(c)) => c,
    _ => return None,
  };

  // Use adaptive content to handle large chunks
  let content = adaptive_content(&chunk.content, chunk.signature.as_deref());

  // Fetch all context in parallel for better performance
  let (callers, callees, siblings, memories) = tokio::join!(
    super::context::get_callers(db, &chunk, depth),
    super::context::get_callees(db, &chunk, depth),
    super::context::get_siblings(db, &chunk, depth),
    super::context::get_related_memories_for_code(db, &chunk, depth)
  );

  Some(ExpandedContext {
    content,
    callers,
    callees,
    siblings,
    memories,
  })
}
