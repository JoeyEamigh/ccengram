//! Tool handlers for MCP requests
//!
//! This module provides the ToolHandler struct and all tool methods organized
//! by domain area.

mod code;
mod documents;
mod entities;
mod explore;
mod format;
mod memory;
mod ranking;
mod suggestions;
mod system;
mod watch;

pub use format::{format_context_response, format_explore_response};

use crate::projects::ProjectRegistry;
use embedding::EmbeddingProvider;
use engram_core::EmbeddingConfig;
use std::sync::Arc;
use tracing::warn;

pub use ranking::{RankingWeights, rank_memories};

/// Handler for MCP tool calls
pub struct ToolHandler {
  pub(crate) registry: Arc<ProjectRegistry>,
  pub(crate) embedding: Option<Arc<dyn EmbeddingProvider>>,
  /// Embedding configuration for health check context_length comparison
  pub(crate) embedding_config: Option<EmbeddingConfig>,
}

impl ToolHandler {
  pub fn new(registry: Arc<ProjectRegistry>) -> Self {
    Self {
      registry,
      embedding: None,
      embedding_config: None,
    }
  }

  pub fn with_embedding(registry: Arc<ProjectRegistry>, embedding: Arc<dyn EmbeddingProvider>) -> Self {
    Self {
      registry,
      embedding: Some(embedding),
      embedding_config: None,
    }
  }

  pub fn with_embedding_and_config(
    registry: Arc<ProjectRegistry>,
    embedding: Arc<dyn EmbeddingProvider>,
    config: EmbeddingConfig,
  ) -> Self {
    Self {
      registry,
      embedding: Some(embedding),
      embedding_config: Some(config),
    }
  }

  /// Get embedding for a query, with fallback to None if provider unavailable
  pub(crate) async fn get_embedding(&self, text: &str) -> Option<Vec<f32>> {
    if let Some(ref provider) = self.embedding {
      match provider.embed(text).await {
        Ok(vec) => Some(vec),
        Err(e) => {
          warn!("Embedding failed: {}", e);
          None
        }
      }
    } else {
      None
    }
  }

  /// Get embeddings for multiple texts in a batch (more efficient for bulk operations)
  pub(crate) async fn get_embeddings_batch(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>> {
    if texts.is_empty() {
      return vec![];
    }
    if let Some(ref provider) = self.embedding {
      match provider.embed_batch(texts).await {
        Ok(vecs) => vecs.into_iter().map(Some).collect(),
        Err(e) => {
          warn!("Batch embedding failed: {}", e);
          vec![None; texts.len()]
        }
      }
    } else {
      vec![None; texts.len()]
    }
  }
}

#[cfg(test)]
pub(crate) fn create_test_handler() -> (tempfile::TempDir, ToolHandler) {
  let data_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
  let registry = Arc::new(ProjectRegistry::with_data_dir(data_dir.path().to_path_buf()));
  let handler = ToolHandler::new(registry);
  (data_dir, handler)
}
