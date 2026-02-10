use std::sync::Arc;

use async_trait::async_trait;
use llama_cpp_2::{
  context::params::LlamaContextParams,
  llama_backend::LlamaBackend,
  llama_batch::LlamaBatch,
  model::{AddBos, LlamaModel, params::LlamaModelParams},
};
use tracing::{debug, info, trace};

use super::{EmbeddingError, EmbeddingMode, EmbeddingProvider};

const DEFAULT_EMBEDDING_REPO: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF";
const DEFAULT_EMBEDDING_FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";

const DEFAULT_QUERY_INSTRUCTION: &str = "Given a search query, retrieve relevant passages\nQuery: ";

pub struct LlamaCppEmbeddingProvider {
  backend: Arc<LlamaBackend>,
  model: Arc<LlamaModel>,
  dimensions: usize,
  query_instruction: Option<String>,
}

impl LlamaCppEmbeddingProvider {
  pub async fn new(config: &crate::config::EmbeddingConfig) -> Result<Self, EmbeddingError> {
    let repo = config.llamacpp_model_repo.as_deref().unwrap_or(DEFAULT_EMBEDDING_REPO);
    let file = config.llamacpp_model_file.as_deref().unwrap_or(DEFAULT_EMBEDDING_FILE);
    let gpu_layers = config.llamacpp_gpu_layers.unwrap_or(-1);
    let dimensions = config.dimensions;
    let query_instruction = config
      .query_instruction
      .clone()
      .or_else(|| Some(DEFAULT_QUERY_INSTRUCTION.to_string()));

    info!(repo, file, gpu_layers, dimensions, "Loading llama.cpp embedding model");

    let model_path = download_model(repo, file).await?;

    // LlamaModelParams contains raw pointers and is not Send, so we must
    // create it inside spawn_blocking alongside the model load.
    let backend = Arc::new(
      LlamaBackend::init().map_err(|e| EmbeddingError::ProviderError(format!("Failed to init llama backend: {e}")))?,
    );

    let backend_clone = backend.clone();
    let model = tokio::task::spawn_blocking(move || {
      let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32);
      LlamaModel::load_from_file(&backend_clone, model_path, &model_params)
        .map_err(|e| EmbeddingError::ProviderError(format!("Failed to load embedding model: {e}")))
    })
    .await
    .map_err(|e| EmbeddingError::ProviderError(format!("Join error: {e}")))??;

    info!(dimensions, "llama.cpp embedding model loaded");

    Ok(Self {
      backend,
      model: Arc::new(model),
      dimensions,
      query_instruction,
    })
  }

  fn format_for_embedding(&self, text: &str, mode: EmbeddingMode) -> String {
    match mode {
      EmbeddingMode::Query => {
        if let Some(ref instruction) = self.query_instruction
          && !instruction.is_empty()
        {
          return format!("Instruct: {}{}", instruction, text);
        }
        text.to_string()
      }
      EmbeddingMode::Document => text.to_string(),
    }
  }
}

fn l2_normalize(v: &mut [f32]) {
  let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
  if magnitude > 0.0 {
    for x in v.iter_mut() {
      *x /= magnitude;
    }
  }
}

#[async_trait]
impl EmbeddingProvider for LlamaCppEmbeddingProvider {
  fn name(&self) -> &str {
    "llamacpp"
  }

  fn model_id(&self) -> &str {
    "llamacpp-embedding"
  }

  fn dimensions(&self) -> usize {
    self.dimensions
  }

  async fn embed(&self, text: &str, mode: EmbeddingMode) -> Result<Vec<f32>, EmbeddingError> {
    let results = self.embed_batch(&[text], mode).await?;
    results
      .into_iter()
      .next()
      .ok_or_else(|| EmbeddingError::ProviderError("No embedding returned".to_string()))
  }

  #[tracing::instrument(level = "trace", skip(self, texts), fields(batch_size = texts.len()))]
  async fn embed_batch(&self, texts: &[&str], mode: EmbeddingMode) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }

    let formatted: Vec<String> = texts.iter().map(|t| self.format_for_embedding(t, mode)).collect();

    let model = self.model.clone();
    let backend = self.backend.clone();
    let dimensions = self.dimensions;

    tokio::task::spawn_blocking(move || embed_texts_blocking(&backend, &model, &formatted, dimensions))
      .await
      .map_err(|e| EmbeddingError::ProviderError(format!("Join error: {e}")))?
  }
}

fn embed_texts_blocking(
  backend: &LlamaBackend,
  model: &LlamaModel,
  texts: &[String],
  dimensions: usize,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
  let n_ctx = (texts.iter().map(|t| t.len() / 3).sum::<usize>()).clamp(512, 32768) as u32;

  let ctx_params = LlamaContextParams::default()
    .with_embeddings(true)
    .with_n_ctx(std::num::NonZeroU32::new(n_ctx));

  let mut ctx = model
    .new_context(backend, ctx_params)
    .map_err(|e| EmbeddingError::ProviderError(format!("Failed to create context: {e}")))?;

  let mut all_embeddings = Vec::with_capacity(texts.len());

  for (seq_idx, text) in texts.iter().enumerate() {
    let tokens = model
      .str_to_token(text, AddBos::Always)
      .map_err(|e| EmbeddingError::ProviderError(format!("Tokenization failed: {e}")))?;

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    // Always use seq_id 0 since we process texts one at a time with KV cache clears
    batch
      .add_sequence(&tokens, 0, false)
      .map_err(|e| EmbeddingError::ProviderError(format!("Failed to add sequence to batch: {e}")))?;

    ctx.clear_kv_cache();
    ctx
      .decode(&mut batch)
      .map_err(|e| EmbeddingError::ProviderError(format!("Decode failed: {e}")))?;

    let raw_embeddings = ctx
      .embeddings_seq_ith(0)
      .map_err(|e| EmbeddingError::ProviderError(format!("Failed to get embeddings: {e}")))?;

    let mut embedding: Vec<f32> = if raw_embeddings.len() >= dimensions {
      raw_embeddings[..dimensions].to_vec()
    } else {
      let mut v = raw_embeddings.to_vec();
      v.resize(dimensions, 0.0);
      v
    };

    l2_normalize(&mut embedding);
    all_embeddings.push(embedding);

    trace!(seq_idx, tokens = tokens.len(), "Embedded sequence");
  }

  debug!(count = all_embeddings.len(), dimensions, "Batch embedding complete");
  Ok(all_embeddings)
}

/// Download a model file from HuggingFace Hub, returning the local path.
pub async fn download_model(repo: &str, filename: &str) -> Result<std::path::PathBuf, EmbeddingError> {
  use hf_hub::api::tokio::Api;

  info!(repo, filename, "Ensuring model is downloaded");

  let api = Api::new().map_err(|e| EmbeddingError::ProviderError(format!("Failed to create HF Hub API: {e}")))?;

  let model_repo = api.model(repo.to_string());
  let path = model_repo
    .get(filename)
    .await
    .map_err(|e| EmbeddingError::ProviderError(format!("Failed to download model {repo}/{filename}: {e}")))?;

  info!(?path, "Model available at local path");
  Ok(path)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::{EmbeddingConfig, EmbeddingProvider as ConfigEmbeddingProvider};

  fn test_embedding_config() -> EmbeddingConfig {
    EmbeddingConfig {
      provider: ConfigEmbeddingProvider::LlamaCpp,
      dimensions: 1024,
      llamacpp_gpu_layers: Some(0),
      ..Default::default()
    }
  }

  fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
  }

  #[tokio::test]
  async fn llamacpp_embedding_load_and_single_embed() {
    let config = test_embedding_config();
    let provider = LlamaCppEmbeddingProvider::new(&config)
      .await
      .expect("model download and load should succeed");

    let embedding = provider
      .embed("fn main() { println!(\"hello\"); }", EmbeddingMode::Document)
      .await
      .expect("single embed should succeed");

    assert_eq!(
      embedding.len(),
      1024,
      "embedding dimensions should match configured value of 1024"
    );
    assert!(
      embedding.iter().all(|v| v.is_finite()),
      "all embedding values should be finite"
    );
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
      (norm - 1.0).abs() < 0.01,
      "L2 norm should be ~1.0 after normalization, got {norm}"
    );
  }

  #[tokio::test]
  async fn llamacpp_embedding_batch_and_semantic_similarity() {
    let config = test_embedding_config();
    let provider = LlamaCppEmbeddingProvider::new(&config)
      .await
      .expect("model should load");

    let texts: &[&str] = &[
      "Rust async runtime using tokio",
      "Tokio is an asynchronous runtime for Rust",
      "The weather in Paris is sunny today",
      "A recipe for chocolate cake with frosting",
    ];

    let embeddings = provider
      .embed_batch(texts, EmbeddingMode::Document)
      .await
      .expect("batch embed should succeed");

    assert_eq!(
      embeddings.len(),
      texts.len(),
      "batch should return one embedding per input text"
    );
    for (i, emb) in embeddings.iter().enumerate() {
      assert_eq!(emb.len(), 1024, "embedding {i} should have 1024 dimensions");
      let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
      assert!(
        (norm - 1.0).abs() < 0.01,
        "embedding {i} L2 norm should be ~1.0, got {norm}"
      );
    }

    // Semantically related texts (Rust+tokio) should be more similar than unrelated ones
    let sim_related = cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_unrelated_a = cosine_similarity(&embeddings[0], &embeddings[2]);
    let sim_unrelated_b = cosine_similarity(&embeddings[0], &embeddings[3]);
    assert!(
      sim_related > sim_unrelated_a,
      "Rust/tokio texts should be more similar ({sim_related:.4}) \
       than Rust vs weather ({sim_unrelated_a:.4})"
    );
    assert!(
      sim_related > sim_unrelated_b,
      "Rust/tokio texts should be more similar ({sim_related:.4}) \
       than Rust vs cake ({sim_unrelated_b:.4})"
    );
  }

  #[tokio::test]
  async fn llamacpp_embedding_query_vs_document_mode() {
    let config = test_embedding_config();
    let provider = LlamaCppEmbeddingProvider::new(&config)
      .await
      .expect("model should load");

    let text = "how to spawn an async task in tokio";

    let doc_embedding = provider
      .embed(text, EmbeddingMode::Document)
      .await
      .expect("document mode embed should succeed");
    let query_embedding = provider
      .embed(text, EmbeddingMode::Query)
      .await
      .expect("query mode embed should succeed");

    assert_eq!(doc_embedding.len(), 1024, "document embedding dimensions");
    assert_eq!(query_embedding.len(), 1024, "query embedding dimensions");

    // Query mode prepends an instruction prefix so the vectors should differ
    let sim = cosine_similarity(&doc_embedding, &query_embedding);
    assert!(
      sim < 0.99,
      "document and query embeddings of the same text should differ due to instruction prefix, \
       but cosine similarity was {sim:.4}"
    );
  }
}
