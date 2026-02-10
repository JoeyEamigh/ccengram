//! Document search service.
//!
//! Provides search functionality for documents with vector/text fallback,
//! optional FTS hybrid retrieval with RRF fusion, and optional reranking.

use std::collections::HashMap;

use tracing::{debug, warn};

use crate::{
  db::ProjectDb,
  domain::config::SearchConfig,
  embedding::EmbeddingProvider,
  ipc::types::docs::{DocSearchItem, DocsSearchParams},
  rerank::{RerankCandidate, RerankRequest, RerankerProvider},
  service::util::{ServiceError, fusion},
};

// ============================================================================
// Service Context
// ============================================================================

/// Context for docs service operations.
pub struct DocsContext<'a> {
  /// Project database connection
  pub db: &'a ProjectDb,
  /// Optional embedding provider for vector search
  pub embedding: &'a dyn EmbeddingProvider,
}

impl<'a> DocsContext<'a> {
  /// Create a new docs context
  pub fn new(db: &'a ProjectDb, embedding: &'a dyn EmbeddingProvider) -> Self {
    Self { db, embedding }
  }

  /// Get an embedding for the given text, if a provider is available
  pub async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
    // Query mode - this is used for docs search queries
    Ok(
      self
        .embedding
        .embed(text, crate::embedding::EmbeddingMode::Query)
        .await?,
    )
  }
}

// ============================================================================
// Search
// ============================================================================

/// Search parameters for documents.
#[derive(Debug, Clone)]
pub struct SearchParams {
  /// The search query
  pub query: String,
  /// Maximum number of results
  pub limit: Option<usize>,
}

impl From<DocsSearchParams> for SearchParams {
  fn from(p: DocsSearchParams) -> Self {
    Self {
      query: p.query,
      limit: p.limit,
    }
  }
}

/// Search documents with hybrid retrieval, optional reranking.
///
/// When `search_config.fts_enabled` is true, runs vector and FTS in parallel
/// then fuses with RRF. Otherwise falls back to vector-only.
pub async fn search(
  ctx: &DocsContext<'_>,
  params: SearchParams,
  search_config: Option<&SearchConfig>,
  reranker: Option<&dyn RerankerProvider>,
) -> Result<Vec<DocSearchItem>, ServiceError> {
  let limit = params.limit.unwrap_or(10);
  let fts_enabled = search_config.is_some_and(|c| c.fts_enabled);
  let rrf_k = search_config.map_or(60, |c| c.rrf_k);
  let rerank_candidates = search_config.map_or(30, |c| c.rerank_candidates);

  let query_vec = ctx.get_embedding(&params.query).await?;

  if fts_enabled {
    let oversample = 50;

    let (vector_results, fts_results) = tokio::join!(
      ctx.db.search_documents(&query_vec, oversample, None),
      ctx.db.fts_search_documents(&params.query, oversample, None),
    );

    let vector_results = vector_results?;
    let fts_results = fts_results.unwrap_or_else(|e| {
      warn!(error = %e, "FTS document search failed, falling back to vector-only");
      Vec::new()
    });

    debug!(
      vector_count = vector_results.len(),
      fts_count = fts_results.len(),
      "Hybrid document retrieval complete"
    );

    // Build lookup map
    let mut doc_map: HashMap<String, crate::domain::document::DocumentChunk> = HashMap::new();
    let mut distance_map: HashMap<String, f32> = HashMap::new();
    for (doc, dist) in &vector_results {
      let id = doc.id.to_string();
      doc_map.insert(id.clone(), doc.clone());
      distance_map.insert(id, *dist);
    }
    for (doc, dist) in &fts_results {
      let id = doc.id.to_string();
      doc_map.entry(id.clone()).or_insert_with(|| doc.clone());
      distance_map.entry(id).or_insert(*dist);
    }

    // RRF fusion
    let vector_ids: Vec<String> = vector_results.iter().map(|(d, _)| d.id.to_string()).collect();
    let fts_ids: Vec<String> = fts_results.iter().map(|(d, _)| d.id.to_string()).collect();
    let fused = fusion::reciprocal_rank_fusion(&[vector_ids, fts_ids], rrf_k);
    let candidates: Vec<(String, f32)> = fused.into_iter().take(rerank_candidates).collect();

    // Optional reranking
    let ranked_ids = if let Some(reranker) = reranker {
      rerank_doc_candidates(&candidates, &doc_map, reranker, &params.query).await
    } else {
      candidates
    };

    let items: Vec<DocSearchItem> = ranked_ids
      .into_iter()
      .take(limit)
      .filter_map(|(id, score)| doc_map.remove(&id).map(|doc| DocSearchItem::from_search(&doc, score)))
      .collect();

    Ok(items)
  } else {
    // Vector-only path
    let results = ctx.db.search_documents(&query_vec, limit, None).await?;

    // Optional reranking even without FTS
    if let Some(reranker) = reranker {
      let mut doc_map: HashMap<String, crate::domain::document::DocumentChunk> = HashMap::new();
      for (doc, _) in &results {
        doc_map.insert(doc.id.to_string(), doc.clone());
      }

      let vector_ids: Vec<String> = results.iter().map(|(d, _)| d.id.to_string()).collect();
      let fused = fusion::reciprocal_rank_fusion(&[vector_ids], rrf_k);
      let candidates: Vec<(String, f32)> = fused.into_iter().take(rerank_candidates).collect();

      let ranked_ids = rerank_doc_candidates(&candidates, &doc_map, reranker, &params.query).await;

      let items: Vec<DocSearchItem> = ranked_ids
        .into_iter()
        .take(limit)
        .filter_map(|(id, score)| doc_map.remove(&id).map(|doc| DocSearchItem::from_search(&doc, score)))
        .collect();

      Ok(items)
    } else {
      let items: Vec<DocSearchItem> = results
        .into_iter()
        .map(|(doc, distance)| {
          let similarity = 1.0 - distance.min(1.0);
          DocSearchItem::from_search(&doc, similarity)
        })
        .collect();
      Ok(items)
    }
  }
}

/// Rerank document candidates using the provided reranker.
async fn rerank_doc_candidates(
  candidates: &[(String, f32)],
  doc_map: &HashMap<String, crate::domain::document::DocumentChunk>,
  reranker: &dyn RerankerProvider,
  query: &str,
) -> Vec<(String, f32)> {
  if !reranker.is_available() {
    warn!(
      provider = reranker.name(),
      "Reranker not available, using RRF scores only"
    );
    return candidates.to_vec();
  }

  let max = reranker.max_candidates();
  let rerank_candidates: Vec<RerankCandidate> = candidates
    .iter()
    .take(max)
    .filter_map(|(id, _)| {
      doc_map.get(id).map(|doc| RerankCandidate {
        id: id.clone(),
        text: doc.content.chars().take(4000).collect(),
      })
    })
    .collect();

  if rerank_candidates.is_empty() {
    return candidates.to_vec();
  }

  let request = RerankRequest {
    query: query.to_string(),
    instruction: None,
    candidates: rerank_candidates,
    top_n: None,
  };

  match reranker.rerank(request).await {
    Ok(response) => {
      debug!(
        duration_ms = response.duration_ms,
        results = response.results.len(),
        "Document reranking complete"
      );
      fusion::blend_scores(candidates, &response.results)
    }
    Err(e) => {
      warn!(error = %e, "Document reranking failed, using RRF scores only");
      candidates.to_vec()
    }
  }
}
