use std::sync::Arc;

use async_trait::async_trait;
use llama_cpp_2::{
  context::params::{LlamaContextParams, LlamaPoolingType},
  llama_backend::LlamaBackend,
  llama_batch::LlamaBatch,
  model::{AddBos, LlamaModel, params::LlamaModelParams},
};
use tracing::{debug, info, trace};

use super::{RerankCandidate, RerankRequest, RerankResponse, RerankResult, RerankerError, RerankerProvider};

const DEFAULT_RERANKER_REPO: &str = "gpustack/jina-reranker-v2-base-multilingual-GGUF";
const DEFAULT_RERANKER_FILE: &str = "jina-reranker-v2-base-multilingual-Q8_0.gguf";

const MAX_DOC_CHARS: usize = 4000;

pub struct LlamaCppReranker {
  backend: Arc<LlamaBackend>,
  model: Arc<LlamaModel>,
  max_candidates: usize,
}

impl LlamaCppReranker {
  pub async fn new(config: &crate::config::RerankerConfig) -> Result<Self, RerankerError> {
    let repo = config.llamacpp_model_repo.as_deref().unwrap_or(DEFAULT_RERANKER_REPO);
    let file = config.llamacpp_model_file.as_deref().unwrap_or(DEFAULT_RERANKER_FILE);
    let gpu_layers = config.llamacpp_gpu_layers.unwrap_or(-1);
    let max_candidates = config.max_candidates;

    info!(
      repo,
      file, gpu_layers, max_candidates, "Loading llama.cpp reranker model"
    );

    let model_path = crate::embedding::llamacpp::download_model(repo, file)
      .await
      .map_err(|e| RerankerError::ProviderError(format!("Model download failed: {e}")))?;

    let backend = Arc::new(
      LlamaBackend::init().map_err(|e| RerankerError::ProviderError(format!("Failed to init llama backend: {e}")))?,
    );

    // LlamaModelParams is not Send (contains raw pointers), so create inside spawn_blocking
    let backend_clone = backend.clone();
    let model = tokio::task::spawn_blocking(move || {
      let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32);
      LlamaModel::load_from_file(&backend_clone, model_path, &model_params)
        .map_err(|e| RerankerError::ProviderError(format!("Failed to load reranker model: {e}")))
    })
    .await
    .map_err(|e| RerankerError::ProviderError(format!("Join error: {e}")))??;

    info!("llama.cpp reranker model loaded");

    Ok(Self {
      backend,
      model: Arc::new(model),
      max_candidates,
    })
  }
}

fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

fn truncate_text(text: &str, max_chars: usize) -> &str {
  if text.len() <= max_chars {
    return text;
  }
  let mut end = max_chars;
  while end > 0 && !text.is_char_boundary(end) {
    end -= 1;
  }
  &text[..end]
}

#[async_trait]
impl RerankerProvider for LlamaCppReranker {
  fn name(&self) -> &str {
    "llamacpp"
  }

  fn is_available(&self) -> bool {
    true
  }

  fn max_candidates(&self) -> usize {
    self.max_candidates
  }

  #[tracing::instrument(level = "trace", skip(self, request), fields(candidates = request.candidates.len()))]
  async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, RerankerError> {
    let start = std::time::Instant::now();
    let query = request.query.clone();
    let candidates: Vec<RerankCandidate> = request.candidates.into_iter().take(self.max_candidates).collect();
    let top_n = request.top_n;

    let model = self.model.clone();
    let backend = self.backend.clone();

    let mut results = tokio::task::spawn_blocking(move || rerank_blocking(&backend, &model, &query, &candidates))
      .await
      .map_err(|e| RerankerError::ProviderError(format!("Join error: {e}")))??;

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(n) = top_n {
      results.truncate(n);
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    debug!(candidates = results.len(), duration_ms, "Reranking complete");

    Ok(RerankResponse { results, duration_ms })
  }
}

fn rerank_blocking(
  backend: &LlamaBackend,
  model: &LlamaModel,
  query: &str,
  candidates: &[RerankCandidate],
) -> Result<Vec<RerankResult>, RerankerError> {
  let mut results = Vec::with_capacity(candidates.len());

  for candidate in candidates {
    let doc_text = truncate_text(&candidate.text, MAX_DOC_CHARS);
    // Cross-encoder format: query [SEP] document
    let prompt = format!("{}</s><s>{}", query, doc_text);

    let tokens = model
      .str_to_token(&prompt, AddBos::Always)
      .map_err(|e| RerankerError::ProviderError(format!("Tokenization failed: {e}")))?;

    let ctx_params = LlamaContextParams::default()
      .with_embeddings(true)
      .with_pooling_type(LlamaPoolingType::Rank)
      .with_n_ctx(std::num::NonZeroU32::new(tokens.len() as u32 + 64));

    let mut ctx = model
      .new_context(backend, ctx_params)
      .map_err(|e| RerankerError::ProviderError(format!("Failed to create context: {e}")))?;

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch
      .add_sequence(&tokens, 0, false)
      .map_err(|e| RerankerError::ProviderError(format!("Failed to add sequence: {e}")))?;

    ctx
      .decode(&mut batch)
      .map_err(|e| RerankerError::ProviderError(format!("Decode failed: {e}")))?;

    let embeddings = ctx
      .embeddings_seq_ith(0)
      .map_err(|e| RerankerError::ProviderError(format!("Failed to get rank output: {e}")))?;

    let raw_score = embeddings.first().copied().unwrap_or(0.0);
    let score = sigmoid(raw_score);

    trace!(id = %candidate.id, raw_score, score, tokens = tokens.len(), "Scored candidate");

    results.push(RerankResult {
      id: candidate.id.clone(),
      score,
    });
  }

  debug!(count = results.len(), "Reranking batch complete");
  Ok(results)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::RerankerConfig;

  fn test_reranker_config() -> RerankerConfig {
    RerankerConfig {
      enabled: true,
      provider: crate::config::RerankerProviderKind::LlamaCpp,
      llamacpp_gpu_layers: Some(0),
      ..Default::default()
    }
  }

  #[tokio::test]
  async fn llamacpp_reranker_load_and_rank_relevance() {
    let config = test_reranker_config();
    let reranker = LlamaCppReranker::new(&config)
      .await
      .expect("reranker model download and load should succeed");

    assert!(reranker.is_available(), "reranker should report available");

    let request = RerankRequest {
      query: "How do I read a file asynchronously in Rust?".to_string(),
      instruction: None,
      candidates: vec![
        RerankCandidate {
          id: "relevant_1".to_string(),
          text: "Use tokio::fs::read_to_string for async file reading in Rust. \
                 It returns a Future that resolves to the file contents."
            .to_string(),
        },
        RerankCandidate {
          id: "relevant_2".to_string(),
          text: "The tokio runtime provides asynchronous filesystem operations \
                 through tokio::fs, including read, write, and metadata queries."
            .to_string(),
        },
        RerankCandidate {
          id: "irrelevant_1".to_string(),
          text: "The best chocolate cake recipe requires Dutch-process cocoa powder, \
                 buttermilk, and a cream cheese frosting."
            .to_string(),
        },
        RerankCandidate {
          id: "irrelevant_2".to_string(),
          text: "The Eiffel Tower was completed in 1889 and stands 330 meters tall \
                 in the center of Paris, France."
            .to_string(),
        },
      ],
      top_n: None,
    };

    let response = reranker.rerank(request).await.expect("reranking should succeed");

    assert_eq!(response.results.len(), 4, "should return all 4 candidates");

    // Results should be sorted descending by score
    for pair in response.results.windows(2) {
      assert!(
        pair[0].score >= pair[1].score,
        "results should be sorted descending: {} ({:.4}) should be >= {} ({:.4})",
        pair[0].id,
        pair[0].score,
        pair[1].id,
        pair[1].score,
      );
    }

    // All scores should be finite and in [0, 1] (sigmoid output)
    for r in &response.results {
      assert!(
        r.score.is_finite() && r.score >= 0.0 && r.score <= 1.0,
        "score for {} should be finite in [0,1], got {}",
        r.id,
        r.score
      );
    }

    // The top 2 results should be the relevant ones
    let top_ids: Vec<&str> = response.results.iter().take(2).map(|r| r.id.as_str()).collect();
    assert!(
      top_ids.contains(&"relevant_1") && top_ids.contains(&"relevant_2"),
      "top 2 results should be the relevant candidates about async file reading, \
       but got: {top_ids:?} (scores: {})",
      response
        .results
        .iter()
        .map(|r| format!("{}={:.4}", r.id, r.score))
        .collect::<Vec<_>>()
        .join(", ")
    );

    assert!(
      response.duration_ms > 0,
      "duration should be positive, got {}ms",
      response.duration_ms
    );
  }

  #[tokio::test]
  async fn llamacpp_reranker_top_n_truncation() {
    let config = test_reranker_config();
    let reranker = LlamaCppReranker::new(&config).await.expect("reranker should load");

    let request = RerankRequest {
      query: "memory management in Rust".to_string(),
      instruction: None,
      candidates: vec![
        RerankCandidate {
          id: "a".to_string(),
          text: "Rust uses ownership and borrowing for memory safety without a garbage collector".to_string(),
        },
        RerankCandidate {
          id: "b".to_string(),
          text: "The borrow checker ensures references are always valid at compile time".to_string(),
        },
        RerankCandidate {
          id: "c".to_string(),
          text: "Python uses reference counting and a cyclic garbage collector".to_string(),
        },
        RerankCandidate {
          id: "d".to_string(),
          text: "How to make sourdough bread from scratch with a starter".to_string(),
        },
      ],
      top_n: Some(2),
    };

    let response = reranker
      .rerank(request)
      .await
      .expect("reranking with top_n should succeed");

    assert_eq!(
      response.results.len(),
      2,
      "top_n=2 should return exactly 2 results, got {}",
      response.results.len()
    );

    // The top 2 should be Rust memory-related, not bread
    assert!(
      response.results.iter().all(|r| r.id != "d"),
      "sourdough bread should not appear in top 2 results about Rust memory management"
    );
  }
}
