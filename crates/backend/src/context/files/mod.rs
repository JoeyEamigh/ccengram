//! Unified file indexing
//!
//! This module provides a single `Indexer` that handles both code and document files.
//! File type is detected automatically by extension.
//!
//! ## Architecture
//!
//! ```text
//! Indexer
//!   ├── Code files (.rs, .ts, .py, etc.) → AST-aware chunking via tree-sitter
//!   └── Document files (.md, .txt, etc.) → Sentence-aware text chunking
//! ```

pub mod code;

use std::{collections::HashMap, path::Path};

use sha2::{Digest, Sha256};
use uuid::Uuid;

pub use self::code::chunker::Chunker;
use crate::{
  db::ProjectDb,
  domain::{
    code::{CodeChunk, Language},
    document::{ChunkParams, DocumentChunk, DocumentId, DocumentSource, chunk_text},
  },
};

#[cfg(feature = "statm")]
/// Get current process RSS in MB from /proc/self/statm
async fn get_rss_mb() -> f64 {
  tokio::fs::read_to_string("/proc/self/statm")
    .await
    .ok()
    .and_then(|c| c.split_whitespace().nth(1)?.parse::<u64>().ok())
    .map(|pages| (pages * 4096) as f64 / (1024.0 * 1024.0))
    .unwrap_or(0.0)
}

// ============================================================================
// Error Type
// ============================================================================

/// Errors that can occur during file indexing
#[derive(Debug, Clone, thiserror::Error)]
pub enum FileIndexError {
  #[error("io error: {0}")]
  IoError(String),
}

impl From<std::io::Error> for FileIndexError {
  fn from(e: std::io::Error) -> Self {
    FileIndexError::IoError(e.to_string())
  }
}

// ============================================================================
// Document Extensions
// ============================================================================

/// File extensions that are treated as documents (not code)
const DOCUMENT_EXTENSIONS: &[&str] = &[
  "md", "markdown", "txt", "text", "rst", "adoc", "asciidoc", "org", "wiki", "textile",
];

/// Check if a file extension indicates a document file
pub fn is_document_extension(ext: &str) -> bool {
  DOCUMENT_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

// ============================================================================
// Unified Chunk Type
// ============================================================================

#[allow(clippy::large_enum_variant)]
/// A chunk from any file type (code or document)
#[derive(Debug, Clone)]
pub enum Chunk {
  Code(CodeChunk),
  Document(DocumentChunk),
}

impl Chunk {
  pub fn file_hash(&self) -> &str {
    match self {
      Chunk::Code(c) => &c.file_hash,
      Chunk::Document(_) => "", // Documents use content_hash at chunk level
    }
  }
}

// ============================================================================
// Unified Metadata Type
// ============================================================================

/// Metadata for any file type
#[derive(Debug, Clone)]
pub enum FileMetadata {
  Code {
    language: Language,
    relative_path: String,
  },
  Document {
    relative_path: String,
    title: String,
    project_id: Uuid,
  },
}

// ============================================================================
// Unified Indexer
// ============================================================================

/// Unified file indexer that handles both code and documents.
///
/// Detects file type by extension and routes to appropriate chunking logic.
/// This is the ONLY indexer - there are no separate code/document indexers.
#[derive(Clone)]
pub struct Indexer {
  /// Code chunker (tree-sitter based)
  chunker: Chunker,
  /// Document chunking parameters
  chunk_params: ChunkParams,
  /// Project ID for document chunks
  project_id: Uuid,
}

impl Indexer {
  /// Create a new unified indexer
  pub fn new(project_id: Uuid) -> Self {
    Self {
      chunker: Chunker::default(),
      chunk_params: ChunkParams::default(),
      project_id,
    }
  }

  /// Compute SHA-256 hash of content (truncated to 16 hex chars)
  fn compute_file_hash(content: &str) -> String {
    let result = Sha256::digest(content.as_bytes());
    format!("{:016x}", u64::from_be_bytes(result[0..8].try_into().unwrap()))
  }

  /// Extract title from filename
  fn extract_title(path: &Path) -> String {
    path
      .file_stem()
      .and_then(|s| s.to_str())
      .map(|s| s.to_string())
      .unwrap_or_else(|| "Untitled".to_string())
  }

  // ============================================================================
  // Indexer Implementation
  // ============================================================================

  /// Scan a file and extract metadata. Returns None if file type is not supported.
  pub fn scan_file(&self, path: &Path, root: &Path) -> Option<FileMetadata> {
    let extension = path.extension()?.to_str()?;
    let relative_path = path.strip_prefix(root).ok()?.to_string_lossy().to_string();

    // Check if it's a document file
    if is_document_extension(extension) {
      let title = Self::extract_title(path);
      return Some(FileMetadata::Document {
        relative_path,
        title,
        project_id: self.project_id,
      });
    }

    // Check if it's a code file
    if let Some(language) = Language::from_extension(extension) {
      return Some(FileMetadata::Code {
        language,
        relative_path,
      });
    }

    // Unsupported file type
    None
  }

  /// Chunk file content based on its type
  pub fn chunk_file(
    &mut self,
    content: &str,
    metadata: &FileMetadata,
    old_content: Option<&str>,
  ) -> Result<Vec<Chunk>, FileIndexError> {
    match metadata {
      FileMetadata::Code {
        language,
        relative_path,
      } => {
        let file_hash = Self::compute_file_hash(content);
        let chunks = self
          .chunker
          .chunk(content, relative_path, *language, &file_hash, old_content);
        Ok(chunks.into_iter().map(Chunk::Code).collect())
      }
      FileMetadata::Document {
        relative_path,
        title,
        project_id,
      } => {
        let raw_chunks = chunk_text(content, &self.chunk_params);
        let total_chunks = raw_chunks.len();
        let document_id = DocumentId::new();

        let chunks: Vec<Chunk> = raw_chunks
          .into_iter()
          .enumerate()
          .map(|(idx, (chunk_content, char_offset))| {
            Chunk::Document(DocumentChunk::new(
              document_id,
              *project_id,
              chunk_content,
              title.clone(),
              relative_path.clone(),
              DocumentSource::File,
              idx,
              total_chunks,
              char_offset,
            ))
          })
          .collect();

        Ok(chunks)
      }
    }
  }

  /// Prepare text for embedding
  pub fn prepare_embedding_text(&self, chunk: &Chunk) -> String {
    match chunk {
      Chunk::Code(c) => c.embedding_text.clone().unwrap_or_else(|| c.content.clone()),
      Chunk::Document(c) => c.content.clone(),
    }
  }

  /// Get cache key for embedding reuse
  pub fn cache_key(&self, chunk: &Chunk) -> Option<String> {
    match chunk {
      Chunk::Code(c) => c.content_hash.clone(),
      Chunk::Document(c) => Some(c.content_hash.clone()),
    }
  }

  /// Store chunks with embeddings to the database using atomic upsert
  ///
  /// Uses merge_insert to atomically:
  /// - Update existing chunks with matching content_hash
  /// - Insert new chunks
  /// - Delete stale chunks for this file
  pub async fn store_chunks(
    &self,
    db: &ProjectDb,
    file_path: &str,
    chunks: Vec<(Chunk, Vec<f32>)>,
  ) -> Result<(), FileIndexError> {
    // Separate code and document chunks
    let mut code_chunks: Vec<(CodeChunk, Vec<f32>)> = Vec::new();
    let mut doc_chunks: Vec<DocumentChunk> = Vec::new();
    let mut doc_vectors: Vec<Vec<f32>> = Vec::new();

    for (chunk, vector) in chunks {
      match chunk {
        Chunk::Code(c) => code_chunks.push((c, vector)),
        Chunk::Document(c) => {
          doc_chunks.push(c);
          doc_vectors.push(vector);
        }
      }
    }

    // Upsert code chunks (handles insert, update, and delete atomically)
    if !code_chunks.is_empty() {
      db.upsert_code_chunks(file_path, &code_chunks)
        .await
        .map_err(|e| FileIndexError::IoError(e.to_string()))?;
    }

    // Upsert document chunks (handles insert, update, and delete atomically)
    if !doc_chunks.is_empty() {
      db.upsert_document_chunks(file_path, &doc_chunks, &doc_vectors)
        .await
        .map_err(|e| FileIndexError::IoError(e.to_string()))?;
    }

    Ok(())
  }

  /// Store chunks for multiple files in a single batch DB operation
  ///
  /// Consumes the input files to avoid cloning chunks and vectors.
  /// More efficient than calling `store_chunks` per file when flushing
  /// multiple files at once. Uses batch upsert with IN clause for deletes.
  pub async fn store_chunks_batch(
    &self,
    db: &ProjectDb,
    files: Vec<crate::actor::pipeline::ProcessedFile>,
  ) -> Result<(), FileIndexError> {
    #[cfg(feature = "statm")]
    let mem_entry = get_rss_mb().await;

    // Collect all code and document chunks across all files
    let mut all_code_chunks: Vec<(CodeChunk, Vec<f32>)> = Vec::new();
    let mut all_doc_chunks: Vec<DocumentChunk> = Vec::new();
    let mut all_doc_vectors: Vec<Vec<f32>> = Vec::new();
    let mut code_file_paths: Vec<String> = Vec::new();
    let mut doc_sources: Vec<String> = Vec::new();

    for file in files {
      let file_path = file.relative;
      let mut has_code = false;
      let mut has_doc = false;

      // Move chunks out instead of cloning
      for (chunk, vector) in file.chunks_with_vectors {
        match chunk {
          Chunk::Code(c) => {
            all_code_chunks.push((c, vector));
            has_code = true;
          }
          Chunk::Document(c) => {
            all_doc_chunks.push(c);
            all_doc_vectors.push(vector);
            has_doc = true;
          }
        }
      }

      if has_code {
        code_file_paths.push(file_path.clone());
      }
      if has_doc {
        doc_sources.push(file_path);
      }
    }

    #[cfg(feature = "statm")]
    let mem_after_prep = get_rss_mb().await;

    // Batch upsert code and document chunks in parallel (different tables)
    let code_future = async {
      if !all_code_chunks.is_empty() {
        let paths_refs: Vec<&str> = code_file_paths.iter().map(|s| s.as_str()).collect();
        db.upsert_code_chunks_batch(&paths_refs, &all_code_chunks)
          .await
          .map_err(|e| FileIndexError::IoError(e.to_string()))
      } else {
        Ok(())
      }
    };

    let doc_future = async {
      if !all_doc_chunks.is_empty() {
        let sources_refs: Vec<&str> = doc_sources.iter().map(|s| s.as_str()).collect();
        db.upsert_document_chunks_batch(&sources_refs, &all_doc_chunks, &all_doc_vectors)
          .await
          .map_err(|e| FileIndexError::IoError(e.to_string()))
      } else {
        Ok(())
      }
    };

    // Run both upserts in parallel
    let (code_result, doc_result) = tokio::join!(code_future, doc_future);
    code_result?;
    doc_result?;

    #[cfg(feature = "statm")]
    {
      let mem_final = get_rss_mb().await;

      tracing::trace!(
        prep_delta = format!("{:+.1}", mem_after_prep - mem_entry),
        upsert_delta = format!("{:+.1}", mem_final - mem_after_prep),
        "[MEM] store_chunks_batch breakdown"
      );
    }

    Ok(())
  }

  /// Get existing embeddings for a file (for embedding reuse)
  pub async fn get_existing_embeddings(
    &self,
    db: &ProjectDb,
    file_path: &str,
  ) -> Result<HashMap<String, Vec<f32>>, FileIndexError> {
    // Only code chunks support embedding reuse currently
    let existing = db
      .get_chunks_with_embeddings_for_file(file_path)
      .await
      .map_err(|e| FileIndexError::IoError(e.to_string()))?;

    let mut map = HashMap::new();
    for (chunk, embedding) in existing {
      if let Some(hash) = chunk.content_hash {
        map.insert(hash, embedding);
      }
    }
    Ok(map)
  }

  /// Rename a file in the index (preserves embeddings)
  pub async fn rename_file(&self, db: &ProjectDb, from: &str, to: &str) -> Result<(), FileIndexError> {
    // Run both renames in parallel - they operate on different tables
    // One will be a no-op depending on file type
    let (_, _, doc_meta) = tokio::join!(
      db.rename_file(from, to),
      db.rename_document(from, to),
      db.get_document_by_source(from)
    );

    // Also update document metadata source if this was a document file
    if let Ok(Some(mut doc)) = doc_meta {
      doc.source = to.to_string();
      doc.updated_at = chrono::Utc::now();
      let _ = db.upsert_document_metadata(&doc).await;
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use super::*;

  fn test_project_id() -> Uuid {
    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
  }

  #[test]
  fn test_indexer_scan_code_file() {
    let indexer = Indexer::new(test_project_id());
    let root = PathBuf::from("/project");
    let path = PathBuf::from("/project/src/main.rs");

    let metadata = indexer.scan_file(&path, &root);
    assert!(metadata.is_some());

    match metadata.unwrap() {
      FileMetadata::Code {
        language,
        relative_path,
      } => {
        assert_eq!(language, Language::Rust);
        assert_eq!(relative_path, "src/main.rs");
      }
      _ => panic!("Expected code metadata"),
    }
  }

  #[test]
  fn test_indexer_scan_document_file() {
    let indexer = Indexer::new(test_project_id());
    let root = PathBuf::from("/project");
    let path = PathBuf::from("/project/docs/README.md");

    let metadata = indexer.scan_file(&path, &root);
    assert!(metadata.is_some());

    match metadata.unwrap() {
      FileMetadata::Document {
        relative_path, title, ..
      } => {
        assert_eq!(relative_path, "docs/README.md");
        assert_eq!(title, "README");
      }
      _ => panic!("Expected document metadata"),
    }
  }

  #[test]
  fn test_indexer_scan_unsupported_file() {
    let indexer = Indexer::new(test_project_id());
    let root = PathBuf::from("/project");
    let path = PathBuf::from("/project/file.xyz");

    let metadata = indexer.scan_file(&path, &root);
    assert!(metadata.is_none());
  }

  #[test]
  fn test_indexer_chunk_code_file() {
    let mut indexer = Indexer::new(test_project_id());
    let metadata = FileMetadata::Code {
      language: Language::Rust,
      relative_path: "test.rs".to_string(),
    };

    let content = r#"
fn hello() {
    println!("Hello, world!");
}

fn goodbye() {
    println!("Goodbye!");
}
"#;

    let chunks = indexer.chunk_file(content, &metadata, None).unwrap();
    assert!(!chunks.is_empty());

    // All should be code chunks
    for chunk in &chunks {
      assert!(matches!(chunk, Chunk::Code(_)));
    }
  }

  #[test]
  fn test_indexer_chunk_document_file() {
    let mut indexer = Indexer::new(test_project_id());
    let metadata = FileMetadata::Document {
      relative_path: "test.md".to_string(),
      title: "Test".to_string(),
      project_id: test_project_id(),
    };

    let content = "This is a test document with some content.";

    let chunks = indexer.chunk_file(content, &metadata, None).unwrap();
    assert!(!chunks.is_empty());

    // All should be document chunks
    for chunk in &chunks {
      assert!(matches!(chunk, Chunk::Document(_)));
    }
  }

  #[test]
  fn test_indexer_prepare_embedding_text() {
    let indexer = Indexer::new(test_project_id());

    // Code chunk with embedding_text
    let code_chunk = Chunk::Code(CodeChunk {
      id: uuid::Uuid::new_v4(),
      file_path: "test.rs".to_string(),
      content: "fn test() {}".to_string(),
      language: Language::Rust,
      chunk_type: crate::domain::code::ChunkType::Function,
      symbols: vec!["test".to_string()],
      imports: vec![],
      calls: vec![],
      start_line: 1,
      end_line: 1,
      file_hash: "abc".to_string(),
      indexed_at: chrono::Utc::now(),
      tokens_estimate: 3,
      definition_kind: None,
      definition_name: None,
      visibility: None,
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: Some("[ENRICHED] fn test() {}".to_string()),
      content_hash: Some("hash123".to_string()),
      caller_count: 0,
      callee_count: 0,
    });

    assert_eq!(indexer.prepare_embedding_text(&code_chunk), "[ENRICHED] fn test() {}");

    // Document chunk
    let doc_chunk = Chunk::Document(DocumentChunk::new(
      DocumentId::new(),
      test_project_id(),
      "Test content".to_string(),
      "Title".to_string(),
      "test.md".to_string(),
      DocumentSource::File,
      0,
      1,
      0,
    ));

    assert_eq!(indexer.prepare_embedding_text(&doc_chunk), "Test content");
  }

  #[test]
  fn test_indexer_cache_key() {
    let indexer = Indexer::new(test_project_id());

    // Code chunk with content_hash
    let code_chunk = Chunk::Code(CodeChunk {
      id: uuid::Uuid::new_v4(),
      file_path: "test.rs".to_string(),
      content: "fn test() {}".to_string(),
      language: Language::Rust,
      chunk_type: crate::domain::code::ChunkType::Function,
      symbols: vec![],
      imports: vec![],
      calls: vec![],
      start_line: 1,
      end_line: 1,
      file_hash: "abc".to_string(),
      indexed_at: chrono::Utc::now(),
      tokens_estimate: 3,
      definition_kind: None,
      definition_name: None,
      visibility: None,
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: None,
      content_hash: Some("hash123".to_string()),
      caller_count: 0,
      callee_count: 0,
    });

    assert_eq!(indexer.cache_key(&code_chunk), Some("hash123".to_string()));

    // Document chunk - has cache key (content_hash)
    let doc_chunk = Chunk::Document(DocumentChunk::new(
      DocumentId::new(),
      test_project_id(),
      "Test".to_string(),
      "Title".to_string(),
      "test.md".to_string(),
      DocumentSource::File,
      0,
      1,
      0,
    ));

    // Document chunks now have content_hash computed from content
    assert!(indexer.cache_key(&doc_chunk).is_some());
  }
}
