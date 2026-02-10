//! Reciprocal Rank Fusion and score blending for hybrid search.
//!
//! Combines ranked lists from multiple retrieval methods (vector search, FTS)
//! into a single fused ranking using RRF. When a reranker is present,
//! position-aware blending merges RRF scores with reranker relevance scores.

use std::{collections::HashMap, hash::Hash};

use crate::rerank::RerankResult;

/// Reciprocal Rank Fusion: merges multiple ranked lists into one.
///
/// For each item across all lists:
///   score(d) = sum(1 / (k + rank_i(d))) for each ranker i
///
/// Items are returned sorted by fused score descending.
///
/// `k=60` is the standard constant from the original RRF paper.
pub fn reciprocal_rank_fusion<I: Eq + Hash + Clone>(ranked_lists: &[Vec<I>], k: u32) -> Vec<(I, f32)> {
  let mut scores: HashMap<I, f32> = HashMap::new();

  for list in ranked_lists {
    for (rank, item) in list.iter().enumerate() {
      let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);
      *scores.entry(item.clone()).or_insert(0.0) += rrf_score;
    }
  }

  let mut fused: Vec<(I, f32)> = scores.into_iter().collect();
  fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
  fused
}

/// Position-aware blending of RRF scores with reranker scores.
///
/// Top-ranked RRF results get higher RRF weight (trust retrieval for top hits),
/// while lower-ranked results trust the reranker more.
///
/// Weight distribution by RRF position:
/// - Positions 0-2:  rrf_weight = 0.75
/// - Positions 3-9:  rrf_weight = 0.60
/// - Positions 10+:  rrf_weight = 0.40
pub fn blend_scores(rrf_ranked: &[(String, f32)], rerank_results: &[RerankResult]) -> Vec<(String, f32)> {
  let rerank_map: HashMap<&str, f32> = rerank_results.iter().map(|r| (r.id.as_str(), r.score)).collect();

  let mut blended: Vec<(String, f32)> = rrf_ranked
    .iter()
    .enumerate()
    .map(|(rrf_rank, (id, rrf_score))| {
      let rerank_score = rerank_map.get(id.as_str()).copied().unwrap_or(0.0);

      let rrf_weight = if rrf_rank < 3 {
        0.75
      } else if rrf_rank < 10 {
        0.60
      } else {
        0.40
      };

      let blended_score = rrf_weight * rrf_score + (1.0 - rrf_weight) * rerank_score;
      (id.clone(), blended_score)
    })
    .collect();

  blended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
  blended
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_rrf_single_list() {
    let lists = vec![vec!["a", "b", "c"]];
    let result = reciprocal_rank_fusion(&lists, 60);

    assert_eq!(result.len(), 3, "should have 3 items");
    assert_eq!(result[0].0, "a", "first item should be 'a'");
    assert!(result[0].1 > result[1].1, "'a' should score higher than 'b'");
    assert!(result[1].1 > result[2].1, "'b' should score higher than 'c'");
  }

  #[test]
  fn test_rrf_two_lists_agreement() {
    let lists = vec![vec!["a", "b", "c"], vec!["a", "c", "b"]];
    let result = reciprocal_rank_fusion(&lists, 60);

    assert_eq!(result[0].0, "a", "'a' ranked first in both lists should be first");
  }

  #[test]
  fn test_rrf_two_lists_disagreement() {
    // List 1: x ranked 1st, y ranked 2nd
    // List 2: y ranked 1st, x ranked 2nd
    // Both should get equal RRF scores, plus unique items
    let lists = vec![vec!["x", "y", "z"], vec!["y", "x", "w"]];
    let result = reciprocal_rank_fusion(&lists, 60);

    // x and y both appear in both lists at reciprocal positions,
    // so they should have the same fused score
    let x_score = result.iter().find(|(id, _)| *id == "x").map(|(_, s)| *s).unwrap_or(0.0);
    let y_score = result.iter().find(|(id, _)| *id == "y").map(|(_, s)| *s).unwrap_or(0.0);
    assert!(
      (x_score - y_score).abs() < 1e-6,
      "x and y at reciprocal positions should have equal RRF scores: x={x_score}, y={y_score}"
    );
  }

  #[test]
  fn test_rrf_disjoint_lists() {
    let lists = vec![vec!["a", "b"], vec!["c", "d"]];
    let result = reciprocal_rank_fusion(&lists, 60);

    assert_eq!(result.len(), 4, "should have all 4 unique items");
    // First items from each list should tie
    let a_score = result.iter().find(|(id, _)| *id == "a").unwrap().1;
    let c_score = result.iter().find(|(id, _)| *id == "c").unwrap().1;
    assert!(
      (a_score - c_score).abs() < 1e-6,
      "first-ranked items from disjoint lists should tie"
    );
  }

  #[test]
  fn test_rrf_empty_lists() {
    let lists: Vec<Vec<&str>> = vec![];
    let result = reciprocal_rank_fusion(&lists, 60);
    assert!(result.is_empty(), "empty input should produce empty output");
  }

  #[test]
  fn test_blend_scores_top_positions_favor_rrf() {
    let rrf_ranked = vec![("a".to_string(), 0.8), ("b".to_string(), 0.6), ("c".to_string(), 0.4)];
    let rerank_results = vec![
      RerankResult {
        id: "a".to_string(),
        score: 0.1, // reranker disagrees strongly
      },
      RerankResult {
        id: "b".to_string(),
        score: 0.9,
      },
      RerankResult {
        id: "c".to_string(),
        score: 0.95,
      },
    ];

    let blended = blend_scores(&rrf_ranked, &rerank_results);

    // 'a' has rrf_rank=0 so rrf_weight=0.75
    // blended_a = 0.75 * 0.8 + 0.25 * 0.1 = 0.625
    let a_score = blended.iter().find(|(id, _)| id == "a").unwrap().1;
    assert!(
      (a_score - 0.625).abs() < 1e-5,
      "blended score for 'a' should be ~0.625, got {a_score}"
    );
  }

  #[test]
  fn test_blend_scores_missing_rerank() {
    let rrf_ranked = vec![("a".to_string(), 0.5)];
    let rerank_results: Vec<RerankResult> = vec![];

    let blended = blend_scores(&rrf_ranked, &rerank_results);
    // With no rerank score, rerank_score defaults to 0.0
    // blended = 0.75 * 0.5 + 0.25 * 0.0 = 0.375
    let a_score = blended[0].1;
    assert!(
      (a_score - 0.375).abs() < 1e-5,
      "missing rerank score should default to 0.0, blended should be 0.375, got {a_score}"
    );
  }

  #[test]
  fn test_rrf_score_values_match_formula() {
    // Verify actual RRF score values: score = 1/(k + rank + 1)
    let lists = vec![vec!["a", "b"]];
    let result = reciprocal_rank_fusion(&lists, 60);

    let a_score = result.iter().find(|(id, _)| *id == "a").unwrap().1;
    let b_score = result.iter().find(|(id, _)| *id == "b").unwrap().1;

    // a at rank 0: 1/(60+0+1) = 1/61
    let expected_a = 1.0 / 61.0;
    assert!(
      (a_score - expected_a).abs() < 1e-6,
      "RRF score for rank-0 item should be 1/61 = {expected_a}, got {a_score}"
    );

    // b at rank 1: 1/(60+1+1) = 1/62
    let expected_b = 1.0 / 62.0;
    assert!(
      (b_score - expected_b).abs() < 1e-6,
      "RRF score for rank-1 item should be 1/62 = {expected_b}, got {b_score}"
    );
  }

  #[test]
  fn test_rrf_item_in_both_lists_beats_single_list() {
    // An item appearing in both lists should always beat items in only one list
    // even if the item in one list is ranked higher
    let lists = vec![
      vec!["shared", "only_vec_1", "only_vec_2"],
      vec!["only_fts_1", "shared", "only_fts_2"],
    ];
    let result = reciprocal_rank_fusion(&lists, 60);

    let shared_score = result.iter().find(|(id, _)| *id == "shared").unwrap().1;
    let only_vec_1_score = result.iter().find(|(id, _)| *id == "only_vec_1").unwrap().1;
    let only_fts_1_score = result.iter().find(|(id, _)| *id == "only_fts_1").unwrap().1;

    // shared: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
    // only_vec_1: 1/(60+1+1) = 1/62
    // only_fts_1: 1/(60+0+1) = 1/61
    assert!(
      shared_score > only_vec_1_score,
      "item in both lists should beat item only in vector list: shared={shared_score} > only_vec_1={only_vec_1_score}"
    );
    assert!(
      shared_score > only_fts_1_score,
      "item in both lists should beat item only in FTS list: shared={shared_score} > only_fts_1={only_fts_1_score}"
    );
  }

  #[test]
  fn test_rrf_three_lists() {
    // Three retrieval methods - item in all three dominates
    let lists = vec![
      vec!["all_three", "vec_only"],
      vec!["all_three", "fts_only"],
      vec!["all_three", "rerank_only"],
    ];
    let result = reciprocal_rank_fusion(&lists, 60);

    assert_eq!(result[0].0, "all_three", "item in all 3 lists should be ranked first");

    // The three items only in one list should tie
    let vec_score = result.iter().find(|(id, _)| *id == "vec_only").unwrap().1;
    let fts_score = result.iter().find(|(id, _)| *id == "fts_only").unwrap().1;
    assert!(
      (vec_score - fts_score).abs() < 1e-6,
      "items at same rank in different lists should tie: {vec_score} vs {fts_score}"
    );
  }

  #[test]
  fn test_rrf_k_sensitivity() {
    // Higher k reduces the score difference between ranks (flatter distribution)
    let lists = vec![vec!["a", "b"]];

    let result_k1 = reciprocal_rank_fusion(&lists, 1);
    let a_k1 = result_k1.iter().find(|(id, _)| *id == "a").unwrap().1;
    let b_k1 = result_k1.iter().find(|(id, _)| *id == "b").unwrap().1;
    let diff_k1 = a_k1 - b_k1;

    let result_k60 = reciprocal_rank_fusion(&lists, 60);
    let a_k60 = result_k60.iter().find(|(id, _)| *id == "a").unwrap().1;
    let b_k60 = result_k60.iter().find(|(id, _)| *id == "b").unwrap().1;
    let diff_k60 = a_k60 - b_k60;

    assert!(
      diff_k1 > diff_k60,
      "lower k should produce larger score gaps: diff_k1={diff_k1} > diff_k60={diff_k60}"
    );
  }

  #[test]
  fn test_blend_position_aware_weight_tiers() {
    // Verify all three weight tiers work correctly
    // Need at least 11 items to test all tiers: 0-2 (0.75), 3-9 (0.60), 10+ (0.40)
    let rrf_ranked: Vec<(String, f32)> = (0..12).map(|i| (format!("item_{i}"), 0.5)).collect();

    let rerank_results: Vec<RerankResult> = (0..12)
      .map(|i| RerankResult {
        id: format!("item_{i}"),
        score: 0.5,
      })
      .collect();

    let _blended = blend_scores(&rrf_ranked, &rerank_results);

    // All items have same rrf and rerank scores (0.5), so blended should be:
    // tier 0-2: 0.75 * 0.5 + 0.25 * 0.5 = 0.5
    // tier 3-9: 0.60 * 0.5 + 0.40 * 0.5 = 0.5
    // tier 10+: 0.40 * 0.5 + 0.60 * 0.5 = 0.5
    // When scores are equal, position-aware weighting produces the same result.
    // Let's test with divergent scores to see the tiers matter.

    let rrf_ranked2: Vec<(String, f32)> = (0..12)
      .map(|i| (format!("item_{i}"), 1.0)) // RRF says all are top
      .collect();

    let rerank_results2: Vec<RerankResult> = (0..12)
      .map(|i| RerankResult {
        id: format!("item_{i}"),
        score: 0.0, // reranker says all are bad
      })
      .collect();

    let blended2 = blend_scores(&rrf_ranked2, &rerank_results2);

    // Scores should differ by tier:
    // tier 0-2: 0.75 * 1.0 + 0.25 * 0.0 = 0.75
    // tier 3-9: 0.60 * 1.0 + 0.40 * 0.0 = 0.60
    // tier 10+: 0.40 * 1.0 + 0.60 * 0.0 = 0.40
    let item_0 = blended2.iter().find(|(id, _)| id == "item_0").unwrap().1;
    let item_5 = blended2.iter().find(|(id, _)| id == "item_5").unwrap().1;
    let item_11 = blended2.iter().find(|(id, _)| id == "item_11").unwrap().1;

    assert!(
      (item_0 - 0.75).abs() < 1e-5,
      "tier 0-2 with rrf=1.0, rerank=0.0 should be 0.75, got {item_0}"
    );
    assert!(
      (item_5 - 0.60).abs() < 1e-5,
      "tier 3-9 with rrf=1.0, rerank=0.0 should be 0.60, got {item_5}"
    );
    assert!(
      (item_11 - 0.40).abs() < 1e-5,
      "tier 10+ with rrf=1.0, rerank=0.0 should be 0.40, got {item_11}"
    );
  }

  #[test]
  fn test_blend_top3_stability() {
    // When reranker slightly disagrees with RRF on top-3, the top-3 should stay stable
    // because they get 0.75 RRF weight (trusted more)
    let rrf_ranked = vec![
      ("first".to_string(), 0.9),
      ("second".to_string(), 0.8),
      ("third".to_string(), 0.7),
      ("fourth".to_string(), 0.6), // position 3, tier 3-9 (0.60 weight)
    ];

    // Reranker says fourth is best, but only slightly
    let rerank_results = vec![
      RerankResult {
        id: "first".to_string(),
        score: 0.6,
      },
      RerankResult {
        id: "second".to_string(),
        score: 0.65,
      },
      RerankResult {
        id: "third".to_string(),
        score: 0.7,
      },
      RerankResult {
        id: "fourth".to_string(),
        score: 0.95,
      },
    ];

    let blended = blend_scores(&rrf_ranked, &rerank_results);

    // first: 0.75*0.9 + 0.25*0.6 = 0.675 + 0.15 = 0.825
    // second: 0.75*0.8 + 0.25*0.65 = 0.6 + 0.1625 = 0.7625
    // third: 0.75*0.7 + 0.25*0.7 = 0.525 + 0.175 = 0.7
    // fourth: 0.60*0.6 + 0.40*0.95 = 0.36 + 0.38 = 0.74
    // So order should be: first, second, fourth, third
    // Top-2 stayed in the original top-3
    assert_eq!(
      blended[0].0, "first",
      "top RRF item should remain first even when reranker disagrees"
    );
    assert_eq!(
      blended[1].0, "second",
      "second RRF item should remain near top even when reranker disagrees"
    );
  }

  #[test]
  fn test_blend_reranker_can_promote_lower_items() {
    // When reranker strongly disagrees with RRF on items outside top-3,
    // those items can be promoted because they have lower RRF weight
    let rrf_ranked = vec![
      ("rrf_top".to_string(), 0.5),
      ("rrf_second".to_string(), 0.4),
      ("rrf_third".to_string(), 0.3),
      ("rrf_fourth".to_string(), 0.2),    // tier 3-9: 0.60 weight
      ("rrf_tenth".to_string(), 0.1),     // tier 3-9: 0.60 weight
      ("rrf_eleventh".to_string(), 0.05), // would be at index 5
      ("rrf_sixth".to_string(), 0.04),
      ("rrf_seventh".to_string(), 0.03),
      ("rrf_eighth".to_string(), 0.02),
      ("rrf_ninth".to_string(), 0.015),
      ("rrf_deep".to_string(), 0.01), // tier 10+: 0.40 weight
    ];

    let rerank_results = vec![
      RerankResult {
        id: "rrf_deep".to_string(),
        score: 0.99,
      }, // reranker LOVES this
      RerankResult {
        id: "rrf_top".to_string(),
        score: 0.01,
      }, // reranker hates top
    ];

    let blended = blend_scores(&rrf_ranked, &rerank_results);

    // rrf_top: 0.75*0.5 + 0.25*0.01 = 0.375 + 0.0025 = 0.3775
    // rrf_deep (index 10): 0.40*0.01 + 0.60*0.99 = 0.004 + 0.594 = 0.598
    let top_score = blended.iter().find(|(id, _)| id == "rrf_top").unwrap().1;
    let deep_score = blended.iter().find(|(id, _)| id == "rrf_deep").unwrap().1;

    assert!(
      deep_score > top_score,
      "reranker with strong signal should be able to promote deeply ranked items: deep={deep_score} > top={top_score}"
    );
  }
}
