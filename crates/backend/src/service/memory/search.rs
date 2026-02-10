//! Memory search service.
//!
//! Provides memory search with vector/text fallback, optional FTS hybrid retrieval
//! with RRF fusion, optional reranking, and post-search ranking.
//!
//! ## Design Note
//!
//! This search implementation **does NOT auto-reinforce** top results.
//! The previous behavior of automatically reinforcing memories during search
//! was a side effect in a read operation. If you want to track memory access,
//! call `lifecycle::reinforce` explicitly after search.

use std::collections::HashMap;

use tracing::{debug, warn};

use super::{MemoryContext, RankingConfig, ranking};
use crate::{
  domain::config::Config,
  ipc::types::{
    code::SearchQuality,
    memory::{MemoryItem, MemorySearchParams},
  },
  rerank::{RerankCandidate, RerankRequest, RerankerProvider},
  service::util::{FilterBuilder, ServiceError, fusion},
};

/// Result of a memory search operation.
pub struct SearchResult {
  /// The search results
  pub items: Vec<MemoryItem>,
  /// Search quality metadata
  pub search_quality: SearchQuality,
}

/// Extended search parameters with internal config.
pub struct SearchParams {
  /// Base parameters from the request
  pub base: MemorySearchParams,
  /// Optional ranking configuration override
  pub ranking_config: Option<RankingConfig>,
}

impl From<MemorySearchParams> for SearchParams {
  fn from(params: MemorySearchParams) -> Self {
    Self {
      base: params,
      ranking_config: None,
    }
  }
}

/// Search memories with hybrid retrieval, optional reranking, and ranking.
///
/// When `config.search.fts_enabled` is true, runs vector and FTS search in parallel,
/// then fuses results with RRF. Otherwise falls back to vector-only search.
///
/// When a reranker is provided, top candidates after fusion are reranked
/// with position-aware score blending.
pub async fn search(
  ctx: &MemoryContext<'_>,
  params: impl Into<SearchParams>,
  config: &Config,
  reranker: Option<&dyn RerankerProvider>,
) -> Result<SearchResult, ServiceError> {
  let params = params.into();
  let base = params.base;

  // Build filter from parameters
  let filter = FilterBuilder::new()
    .exclude_inactive(base.include_superseded)
    .add_eq_opt("sector", base.sector.as_deref())
    .add_eq_opt("tier", base.tier.as_deref())
    .add_eq_opt("memory_type", base.memory_type.as_deref())
    .add_min_opt("salience", base.min_salience)
    .add_prefix_opt("scope_path", base.scope_path.as_deref())
    .add_eq_opt("scope_module", base.scope_module.as_deref())
    .add_eq_opt("session_id", base.session_id.as_deref())
    .build();

  let limit = base.limit.unwrap_or(config.search.default_limit);
  let fetch_limit = limit * 2;

  let ranking_config = params
    .ranking_config
    .unwrap_or_else(|| RankingConfig::from(&config.search));

  let query_vec = ctx.get_embedding(&base.query).await?;
  debug!("Using vector search for query: {}", base.query);

  let fts_enabled = config.search.fts_enabled;
  let rrf_k = config.search.rrf_k;
  let rerank_candidates = config.search.rerank_candidates;

  if fts_enabled {
    // Hybrid path: parallel vector + FTS, RRF fusion
    let oversample = 50;

    let (vector_results, fts_results) = tokio::join!(
      ctx.db.search_memories(&query_vec, oversample, filter.as_deref()),
      ctx.db.fts_search_memories(&base.query, oversample, filter.as_deref()),
    );

    let vector_results = vector_results?;
    let fts_results = fts_results.unwrap_or_else(|e| {
      warn!(error = %e, "FTS memory search failed, falling back to vector-only");
      Vec::new()
    });

    debug!(
      vector_count = vector_results.len(),
      fts_count = fts_results.len(),
      "Hybrid memory retrieval complete"
    );

    // Build lookup map
    let mut memory_map: HashMap<String, crate::domain::memory::Memory> = HashMap::new();
    let mut distance_map: HashMap<String, f32> = HashMap::new();
    for (mem, dist) in &vector_results {
      let id = mem.id.to_string();
      memory_map.insert(id.clone(), mem.clone());
      distance_map.insert(id, *dist);
    }
    for (mem, dist) in &fts_results {
      let id = mem.id.to_string();
      memory_map.entry(id.clone()).or_insert_with(|| mem.clone());
      distance_map.entry(id).or_insert(*dist);
    }

    // RRF fusion
    let vector_ids: Vec<String> = vector_results.iter().map(|(m, _)| m.id.to_string()).collect();
    let fts_ids: Vec<String> = fts_results.iter().map(|(m, _)| m.id.to_string()).collect();
    let fused = fusion::reciprocal_rank_fusion(&[vector_ids, fts_ids], rrf_k);
    let candidates: Vec<(String, f32)> = fused.into_iter().take(rerank_candidates).collect();

    // Optional reranking
    let ranked_ids = if let Some(reranker) = reranker {
      rerank_memory_candidates(&candidates, &memory_map, reranker, &base.query).await
    } else {
      candidates
    };

    // Convert back to (Memory, distance) for the existing ranking pipeline
    let fused_results: Vec<(crate::domain::memory::Memory, f32)> = ranked_ids
      .into_iter()
      .filter_map(|(id, _rrf_score)| {
        memory_map.remove(&id).map(|mem| {
          let dist = distance_map.get(&id).copied().unwrap_or(0.5);
          (mem, dist)
        })
      })
      .collect();

    let ranked = ranking::rank_memories(fused_results, limit, Some(&ranking_config));

    let distances: Vec<f32> = ranked.iter().map(|(_, distance, _)| *distance).collect();
    let search_quality = SearchQuality::from_distances(&distances);

    let items = ranked
      .into_iter()
      .map(|(m, distance, rank_score)| {
        let similarity = 1.0 - distance.min(1.0);
        MemoryItem::from_search(&m, similarity, rank_score)
      })
      .collect();

    Ok(SearchResult { items, search_quality })
  } else {
    // Vector-only path
    let results = ctx
      .db
      .search_memories(&query_vec, fetch_limit, filter.as_deref())
      .await?;

    // Optional reranking even without FTS
    let results = if let Some(reranker) = reranker {
      let mut memory_map: HashMap<String, crate::domain::memory::Memory> = HashMap::new();
      let mut distance_map: HashMap<String, f32> = HashMap::new();
      for (mem, dist) in &results {
        let id = mem.id.to_string();
        memory_map.insert(id.clone(), mem.clone());
        distance_map.insert(id, *dist);
      }

      let vector_ids: Vec<String> = results.iter().map(|(m, _)| m.id.to_string()).collect();
      let fused = fusion::reciprocal_rank_fusion(&[vector_ids], rrf_k);
      let candidates: Vec<(String, f32)> = fused.into_iter().take(rerank_candidates).collect();

      let ranked_ids = rerank_memory_candidates(&candidates, &memory_map, reranker, &base.query).await;

      ranked_ids
        .into_iter()
        .filter_map(|(id, _)| {
          memory_map.remove(&id).map(|mem| {
            let dist = distance_map.get(&id).copied().unwrap_or(0.5);
            (mem, dist)
          })
        })
        .collect()
    } else {
      results
    };

    let ranked = ranking::rank_memories(results, limit, Some(&ranking_config));

    let distances: Vec<f32> = ranked.iter().map(|(_, distance, _)| *distance).collect();
    let search_quality = SearchQuality::from_distances(&distances);

    let items = ranked
      .into_iter()
      .map(|(m, distance, rank_score)| {
        let similarity = 1.0 - distance.min(1.0);
        MemoryItem::from_search(&m, similarity, rank_score)
      })
      .collect();

    Ok(SearchResult { items, search_quality })
  }
}

/// Rerank memory candidates using the provided reranker.
async fn rerank_memory_candidates(
  candidates: &[(String, f32)],
  memory_map: &HashMap<String, crate::domain::memory::Memory>,
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
      memory_map.get(id).map(|mem| RerankCandidate {
        id: id.clone(),
        text: mem.content.chars().take(4000).collect(),
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
        "Memory reranking complete"
      );
      fusion::blend_scores(candidates, &response.results)
    }
    Err(e) => {
      warn!(error = %e, "Memory reranking failed, using RRF scores only");
      candidates.to_vec()
    }
  }
}

/// Search memories using a pre-computed embedding vector.
///
/// This is useful for cross-domain searches where you already have an embedding
/// (e.g., from a code chunk) and want to find semantically related memories
/// without recomputing the embedding.
///
/// Automatically filters out deleted memories.
pub async fn search_by_embedding(
  db: &crate::db::ProjectDb,
  embedding: &[f32],
  limit: usize,
  filter: Option<&str>,
) -> Result<Vec<(crate::domain::memory::Memory, f32)>, ServiceError> {
  // Combine user filter with is_deleted check
  let full_filter = match filter {
    Some(f) => Some(format!("is_deleted = false AND {}", f)),
    None => Some("is_deleted = false".to_string()),
  };

  let results = db.search_memories(embedding, limit, full_filter.as_deref()).await?;

  Ok(results)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_filter_building() {
    let filter = FilterBuilder::new()
      .exclude_inactive(false)
      .add_eq_opt("sector", Some("semantic"))
      .add_prefix_opt("scope_path", Some("src/"))
      .build();

    let filter_str = filter.unwrap();
    assert!(filter_str.contains("sector = 'semantic'"));
    assert!(filter_str.contains("scope_path LIKE 'src/%'"));
  }
}
