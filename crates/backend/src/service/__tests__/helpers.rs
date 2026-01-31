//! Shared test helpers for service-level integration tests.

use std::{path::Path, sync::Arc};

use tempfile::TempDir;
use uuid::Uuid;

use crate::{
  config::Config,
  context::files::code::chunker::{Chunker, ChunkerConfig},
  db::ProjectDb,
  domain::{code::Language, project::ProjectId},
  embedding::EmbeddingProvider,
  service::memory::MemoryContext,
};

/// Test context providing temp directory, database, and embedding for integration tests.
///
/// The temp directory is automatically cleaned up when the context is dropped.
pub struct TestContext {
  /// Temp directory - must be kept alive for the duration of the test
  _temp_dir: TempDir,
  /// Project database
  pub db: ProjectDb,
  /// Project configuration
  pub config: Arc<Config>,
  /// Project UUID for memory operations
  pub project_uuid: Uuid,
  /// Embedding provider (OpenRouter)
  pub embedding: Arc<dyn EmbeddingProvider>,
}

impl TestContext {
  /// Create a new test context with a fresh temp directory, database, and OpenRouter embedding.
  pub async fn new() -> Self {
    let temp_dir = TempDir::new().expect("create temp dir");
    let project_id = ProjectId::from_path(Path::new("/test/project")).await;
    let config = Arc::new(Config::default());
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), config.clone())
      .await
      .expect("open test database");

    let project_uuid = Uuid::new_v4();

    // Create real embedding provider from config
    let embedding = <dyn EmbeddingProvider>::from_config(&config.embedding).expect("create embedding provider");

    Self {
      _temp_dir: temp_dir,
      db,
      config,
      project_uuid,
      embedding,
    }
  }

  /// Create a memory context for memory service operations with embedding support.
  pub fn memory_context(&self) -> MemoryContext<'_> {
    MemoryContext::new(&self.db, self.embedding.as_ref(), self.project_uuid)
  }

  /// Index code content using the chunker and store in the database.
  ///
  /// This uses the full AST-based chunker to properly extract symbols,
  /// signatures, docstrings, and generate enriched embedding text.
  pub async fn index_code(&self, file_path: &str, content: &str, language: Language) {
    let mut chunker = Chunker::with_owned_parser(ChunkerConfig::default());
    // Simple hash for test purposes
    let file_hash = format!(
      "{:016x}",
      content
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
    );
    let chunks = chunker.chunk(content, file_path, language, &file_hash, None);

    let mut chunks_with_embeddings = Vec::new();
    for chunk in chunks {
      // Generate embedding from the enriched embedding_text or fall back to content
      let text_to_embed = chunk.embedding_text.as_deref().unwrap_or(&chunk.content);
      // Document mode - we're indexing code chunks
      let embedding = self
        .embedding
        .embed(text_to_embed, crate::embedding::EmbeddingMode::Document)
        .await
        .expect("embedding required");

      chunks_with_embeddings.push((chunk, embedding));
    }

    self
      .db
      .upsert_code_chunks(file_path, &chunks_with_embeddings)
      .await
      .expect("upsert code chunks");
  }
}
