use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::{debug, trace, warn};

use super::{RerankRequest, RerankResponse, RerankResult, RerankerError, RerankerProvider};
use crate::domain::config::RerankerConfig;

const DEEPINFRA_INFERENCE_URL: &str = "https://api.deepinfra.com/v1/inference";
const DEFAULT_MODEL: &str = "Qwen/Qwen3-Reranker-8B";
const DEFAULT_INSTRUCTION: &str = "Given a code search query, retrieve relevant code snippets and documentation";

pub struct DeepInfraReranker {
  client: reqwest::Client,
  api_key: String,
  model: String,
  default_instruction: String,
  max_candidates: usize,
}

impl DeepInfraReranker {
  pub fn new(config: &RerankerConfig) -> Result<Self, RerankerError> {
    let api_key = config
      .deepinfra_api_key
      .clone()
      .or_else(Self::key_from_env)
      .ok_or_else(|| RerankerError::ProviderError("No DeepInfra API key configured".to_string()))?;

    let model = if config.model.is_empty() {
      DEFAULT_MODEL.to_string()
    } else {
      config.model.clone()
    };

    let default_instruction = if config.instruction.is_empty() {
      DEFAULT_INSTRUCTION.to_string()
    } else {
      config.instruction.clone()
    };

    debug!(
      model = %model,
      max_candidates = config.max_candidates,
      "DeepInfra reranker initialized"
    );

    Ok(Self {
      client: reqwest::Client::new(),
      api_key,
      model,
      default_instruction,
      max_candidates: config.max_candidates,
    })
  }

  fn key_from_env() -> Option<String> {
    match std::env::var("DEEPINFRA_API_KEY") {
      Ok(key) => {
        debug!("DEEPINFRA_API_KEY found in environment");
        Some(key)
      }
      Err(_) => {
        debug!("DEEPINFRA_API_KEY not set");
        None
      }
    }
  }
}

#[derive(Debug, Serialize)]
struct DeepInfraRerankRequest<'a> {
  queries: Vec<&'a str>,
  documents: Vec<&'a str>,
  instruction: &'a str,
  return_documents: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_n: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct DeepInfraRerankResponse {
  scores: Vec<f32>,
}

#[async_trait::async_trait]
impl RerankerProvider for DeepInfraReranker {
  fn name(&self) -> &str {
    "deepinfra"
  }

  fn is_available(&self) -> bool {
    true
  }

  fn max_candidates(&self) -> usize {
    self.max_candidates
  }

  #[tracing::instrument(level = "trace", skip(self, request), fields(query_len = request.query.len(), candidates = request.candidates.len()))]
  async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, RerankerError> {
    if !self.is_available() {
      return Err(RerankerError::NotAvailable);
    }

    if request.candidates.is_empty() {
      return Ok(RerankResponse {
        results: Vec::new(),
        duration_ms: 0,
      });
    }

    let instruction = request.instruction.as_deref().unwrap_or(&self.default_instruction);

    let documents: Vec<&str> = request.candidates.iter().map(|c| c.text.as_str()).collect();

    let api_request = DeepInfraRerankRequest {
      queries: vec![request.query.as_str()],
      documents: documents.clone(),
      instruction,
      return_documents: false,
      top_n: request.top_n,
    };

    let url = format!("{}/{}", DEEPINFRA_INFERENCE_URL, self.model);

    trace!(
      url = %url,
      candidates = documents.len(),
      "Sending rerank request to DeepInfra"
    );
    let start = Instant::now();

    let response = self
      .client
      .post(&url)
      .header("Authorization", format!("Bearer {}", self.api_key))
      .header("Content-Type", "application/json")
      .timeout(std::time::Duration::from_secs(30))
      .json(&api_request)
      .send()
      .await
      .map_err(|e| {
        if e.is_timeout() {
          RerankerError::Timeout
        } else {
          RerankerError::Request(e)
        }
      })?;

    let status = response.status();
    let elapsed = start.elapsed();

    trace!(
      status = %status,
      elapsed_ms = elapsed.as_millis(),
      "Received response from DeepInfra"
    );

    if !status.is_success() {
      let body = response.text().await.unwrap_or_default();
      let body_preview: String = body.chars().take(500).collect();
      warn!(
        status = %status,
        model = %self.model,
        body_preview = %body_preview,
        "DeepInfra rerank request failed"
      );
      return Err(RerankerError::ProviderError(format!(
        "DeepInfra returned {}: {}",
        status,
        body_preview.chars().take(300).collect::<String>()
      )));
    }

    let body_text = response.text().await.map_err(|e| {
      warn!(error = %e, "Failed to read DeepInfra response body");
      RerankerError::ProviderError(format!("Failed to read response body: {}", e))
    })?;

    let result: DeepInfraRerankResponse = serde_json::from_str(&body_text).map_err(|e| {
      let body_preview: String = body_text.chars().take(500).collect();
      warn!(
        error = %e,
        body_preview = %body_preview,
        "Failed to parse DeepInfra rerank response"
      );
      RerankerError::ProviderError(format!("JSON parse error: {}", e))
    })?;

    if result.scores.len() != request.candidates.len() {
      warn!(
        expected = request.candidates.len(),
        got = result.scores.len(),
        "DeepInfra returned unexpected number of scores"
      );
    }

    // Pair each candidate with its score, then sort by score descending
    let mut scored: Vec<RerankResult> = request
      .candidates
      .iter()
      .zip(result.scores.iter())
      .map(|(candidate, &score)| RerankResult {
        id: candidate.id.clone(),
        score,
      })
      .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top_n limit if specified
    if let Some(top_n) = request.top_n {
      scored.truncate(top_n);
    }

    let duration_ms = elapsed.as_millis() as u64;

    debug!(
      candidates = request.candidates.len(),
      results = scored.len(),
      duration_ms,
      top_score = scored.first().map(|r| r.score).unwrap_or(0.0),
      "DeepInfra rerank complete"
    );

    Ok(RerankResponse {
      results: scored,
      duration_ms,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    domain::config::RerankerConfig,
    rerank::{RerankCandidate, RerankRequest, RerankerProvider},
  };

  fn make_reranker() -> DeepInfraReranker {
    let config = RerankerConfig::default();
    DeepInfraReranker::new(&config).expect("DEEPINFRA_API_KEY must be set in the environment")
  }

  fn candidate(id: &str, text: &str) -> RerankCandidate {
    RerankCandidate {
      id: id.to_string(),
      text: text.to_string(),
    }
  }

  #[tokio::test]
  async fn test_rerank_relevance_and_ordering() {
    let reranker = make_reranker();

    let request = RerankRequest {
      query: "How to read a file in Rust using tokio".to_string(),
      instruction: None,
      candidates: vec![
        candidate(
          "rust_file_io",
          "Use tokio::fs::read_to_string to asynchronously read a file in Rust. This returns a Future that resolves to the file contents as a String.",
        ),
        candidate(
          "pizza_recipe",
          "To make pizza dough, combine flour, water, yeast, salt, and olive oil. Knead for 10 minutes and let rise for 1 hour.",
        ),
        candidate(
          "rust_http",
          "The reqwest crate provides an ergonomic HTTP client for Rust. Use reqwest::get to make GET requests.",
        ),
        candidate(
          "tokio_spawn",
          "tokio::spawn creates a new asynchronous task. The spawned task runs concurrently with the current task on the tokio runtime.",
        ),
        candidate(
          "gardening",
          "Plant tomatoes in spring after the last frost. They need full sun and regular watering.",
        ),
      ],
      top_n: None,
    };

    let response = reranker.rerank(request).await.expect("rerank request should succeed");

    // Should return all 5 candidates
    assert_eq!(response.results.len(), 5, "should return all 5 candidates");

    // Results must be sorted by score descending
    for window in response.results.windows(2) {
      assert!(
        window[0].score >= window[1].score,
        "results should be sorted descending: {} (score {}) should be >= {} (score {})",
        window[0].id,
        window[0].score,
        window[1].id,
        window[1].score
      );
    }

    // The most relevant candidate (rust_file_io) should be ranked first
    assert_eq!(
      response.results[0].id, "rust_file_io",
      "rust_file_io should be the top result, got '{}' with score {}",
      response.results[0].id, response.results[0].score
    );

    // The irrelevant candidates (pizza, gardening) should be in the bottom half
    let bottom_half_ids: Vec<&str> = response.results[3..].iter().map(|r| r.id.as_str()).collect();
    assert!(
      bottom_half_ids.contains(&"pizza_recipe") || bottom_half_ids.contains(&"gardening"),
      "irrelevant candidates should be ranked low, bottom half: {:?}",
      bottom_half_ids
    );

    // All scores should be finite
    for r in &response.results {
      assert!(
        r.score.is_finite(),
        "score for '{}' should be finite, got {}",
        r.id,
        r.score
      );
    }

    assert!(response.duration_ms > 0, "duration should be positive");
  }

  #[tokio::test]
  async fn test_rerank_single_candidate() {
    let reranker = make_reranker();

    let request = RerankRequest {
      query: "Rust error handling".to_string(),
      instruction: None,
      candidates: vec![candidate(
        "only_one",
        "Use the ? operator in Rust to propagate errors up the call stack.",
      )],
      top_n: None,
    };

    let response = reranker
      .rerank(request)
      .await
      .expect("single candidate rerank should succeed");
    assert_eq!(response.results.len(), 1, "should return exactly 1 result");
    assert_eq!(response.results[0].id, "only_one");
    assert!(response.results[0].score.is_finite(), "score should be finite");
  }

  #[tokio::test]
  async fn test_rerank_empty_candidates() {
    let reranker = make_reranker();

    let request = RerankRequest {
      query: "anything".to_string(),
      instruction: None,
      candidates: vec![],
      top_n: None,
    };

    let response = reranker.rerank(request).await.expect("empty candidates should succeed");
    assert!(
      response.results.is_empty(),
      "empty candidates should return empty results"
    );
    assert_eq!(response.duration_ms, 0, "duration should be 0 for empty request");
  }

  #[tokio::test]
  async fn test_rerank_empty_query() {
    let reranker = make_reranker();

    let request = RerankRequest {
      query: String::new(),
      instruction: None,
      candidates: vec![
        candidate("a", "Some text about Rust programming"),
        candidate("b", "Another text about Python scripting"),
      ],
      top_n: None,
    };

    let response = reranker.rerank(request).await.expect("empty query should not error");
    assert_eq!(response.results.len(), 2, "should still return both candidates");
    for r in &response.results {
      assert!(
        r.score.is_finite(),
        "score for '{}' should be finite even with empty query",
        r.id
      );
    }
  }

  #[tokio::test]
  async fn test_rerank_long_document() {
    let reranker = make_reranker();

    let long_text = "fn main() { println!(\"hello\"); }\n".repeat(500);

    let request = RerankRequest {
      query: "Rust main function".to_string(),
      instruction: None,
      candidates: vec![
        candidate("long_doc", &long_text),
        candidate(
          "short_doc",
          "A short hello world in Rust: fn main() { println!(\"hello\"); }",
        ),
      ],
      top_n: None,
    };

    let response = reranker.rerank(request).await.expect("long document should not error");
    assert_eq!(response.results.len(), 2, "should return both candidates");
    for r in &response.results {
      assert!(r.score.is_finite(), "score for '{}' should be finite", r.id);
    }
  }

  #[tokio::test]
  async fn test_rerank_top_n_limits_results() {
    let reranker = make_reranker();

    let request = RerankRequest {
      query: "database query optimization".to_string(),
      instruction: None,
      candidates: vec![
        candidate("a", "SQL query optimization involves analyzing execution plans"),
        candidate("b", "Index creation speeds up database lookups significantly"),
        candidate("c", "Flower arrangement tips for spring weddings"),
        candidate("d", "Use EXPLAIN ANALYZE in PostgreSQL to profile queries"),
      ],
      top_n: Some(2),
    };

    let response = reranker.rerank(request).await.expect("top_n rerank should succeed");
    assert_eq!(response.results.len(), 2, "top_n=2 should return exactly 2 results");

    // The 2 returned should be sorted descending
    assert!(
      response.results[0].score >= response.results[1].score,
      "top_n results should be sorted descending"
    );
  }

  #[tokio::test]
  async fn test_rerank_invalid_api_key() {
    let config = RerankerConfig {
      deepinfra_api_key: Some("invalid-key-12345".to_string()),
      ..Default::default()
    };
    let reranker = DeepInfraReranker::new(&config).expect("construction should succeed with any key");

    let request = RerankRequest {
      query: "test".to_string(),
      instruction: None,
      candidates: vec![candidate("a", "some text")],
      top_n: None,
    };

    let result = reranker.rerank(request).await;
    match result {
      Ok(_) => panic!("invalid API key should produce an error, but got Ok"),
      Err(RerankerError::ProviderError(msg)) => {
        assert!(
          msg.contains("401") || msg.contains("403") || msg.contains("Unauthorized") || msg.contains("Forbidden"),
          "error should indicate auth failure, got: {}",
          msg
        );
      }
      Err(other) => panic!("expected ProviderError for invalid key, got: {:?}", other),
    }
  }

  #[tokio::test]
  async fn test_rerank_provider_metadata() {
    let reranker = make_reranker();
    assert_eq!(reranker.name(), "deepinfra");
    assert!(reranker.is_available());
    assert_eq!(reranker.max_candidates(), 30);
  }
}
