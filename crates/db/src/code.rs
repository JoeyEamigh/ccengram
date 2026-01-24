// Code chunks table operations

use arrow_array::{
  Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use chrono::{TimeZone, Utc};
use engram_core::{ChunkType, CodeChunk, Language};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::code_chunks_schema;

impl ProjectDb {
  /// Add a new code chunk to the database
  pub async fn add_code_chunk(&self, chunk: &CodeChunk, vector: Option<&[f32]>) -> Result<()> {
    let table = self.code_chunks_table().await?;

    let batch = code_chunk_to_batch(chunk, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], code_chunks_schema(self.vector_dim));

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Add multiple code chunks (batch insert)
  pub async fn add_code_chunks(&self, chunks: &[(CodeChunk, Vec<f32>)]) -> Result<()> {
    if chunks.is_empty() {
      return Ok(());
    }

    let table = self.code_chunks_table().await?;

    let batches: Vec<_> = chunks
      .iter()
      .map(|(chunk, vec)| code_chunk_to_batch(chunk, Some(vec), self.vector_dim))
      .collect::<Result<Vec<_>>>()?;

    let iter = RecordBatchIterator::new(batches.into_iter().map(Ok), code_chunks_schema(self.vector_dim));

    table.add(Box::new(iter)).execute().await?;
    Ok(())
  }

  /// Get a code chunk by ID
  pub async fn get_code_chunk(&self, id: &Uuid) -> Result<Option<CodeChunk>> {
    let table = self.code_chunks_table().await?;
    let id_str = id.to_string();

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id_str))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() {
      return Ok(None);
    }

    let batch = &results[0];
    if batch.num_rows() == 0 {
      return Ok(None);
    }

    Ok(Some(batch_to_code_chunk(batch, 0)?))
  }

  /// Delete all chunks for a file
  pub async fn delete_chunks_for_file(&self, file_path: &str) -> Result<()> {
    let table = self.code_chunks_table().await?;
    table.delete(&format!("file_path = '{}'", file_path)).await?;
    Ok(())
  }

  /// Delete all chunks for multiple files in a single operation
  /// Much more efficient than calling delete_chunks_for_file in a loop
  pub async fn delete_chunks_for_files(&self, file_paths: &[&str]) -> Result<()> {
    if file_paths.is_empty() {
      return Ok(());
    }

    let table = self.code_chunks_table().await?;

    // Build IN clause: file_path IN ('path1', 'path2', ...)
    let paths_list = file_paths
      .iter()
      .map(|p| format!("'{}'", p.replace('\'', "''"))) // Escape single quotes
      .collect::<Vec<_>>()
      .join(", ");

    table.delete(&format!("file_path IN ({})", paths_list)).await?;
    Ok(())
  }

  /// Delete a code chunk by ID
  pub async fn delete_code_chunk(&self, id: &Uuid) -> Result<()> {
    let table = self.code_chunks_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Rename a file by updating file_path for all its chunks (preserves embeddings)
  ///
  /// This is more efficient than delete + re-index because it preserves existing
  /// embeddings and other computed data.
  pub async fn rename_file(&self, old_path: &str, new_path: &str) -> Result<usize> {
    let table = self.code_chunks_table().await?;

    // Get chunks for the old path to count and update
    let chunks = self.get_chunks_for_file(old_path).await?;
    let count = chunks.len();

    if count == 0 {
      return Ok(0);
    }

    // LanceDB update: set file_path = new_path where file_path = old_path
    // Escape single quotes in paths
    let old_escaped = old_path.replace('\'', "''");
    let new_escaped = new_path.replace('\'', "''");

    table
      .update()
      .only_if(format!("file_path = '{}'", old_escaped))
      .column("file_path", format!("'{}'", new_escaped))
      .execute()
      .await?;

    Ok(count)
  }

  /// Batch rename files with common prefix change (for folder renames)
  ///
  /// Example: rename_files_with_prefix("src/old/", "src/new/") will update
  /// all files starting with "src/old/" to start with "src/new/".
  pub async fn rename_files_with_prefix(&self, old_prefix: &str, new_prefix: &str) -> Result<usize> {
    // Get all files matching the old prefix
    let old_escaped = old_prefix.replace('\'', "''");
    let filter = format!("file_path LIKE '{}%'", old_escaped);

    let chunks = self.list_code_chunks(Some(&filter), None).await?;

    if chunks.is_empty() {
      return Ok(0);
    }

    // Group chunks by file path for counting unique files
    let mut files_renamed = std::collections::HashSet::new();

    // Update each file's path
    let table = self.code_chunks_table().await?;

    for chunk in &chunks {
      if files_renamed.contains(&chunk.file_path) {
        continue;
      }

      let new_path = chunk.file_path.replacen(old_prefix, new_prefix, 1);
      let old_path_escaped = chunk.file_path.replace('\'', "''");
      let new_path_escaped = new_path.replace('\'', "''");

      table
        .update()
        .only_if(format!("file_path = '{}'", old_path_escaped))
        .column("file_path", format!("'{}'", new_path_escaped))
        .execute()
        .await?;

      files_renamed.insert(chunk.file_path.clone());
    }

    Ok(files_renamed.len())
  }

  /// Update a code chunk (delete + add)
  pub async fn update_code_chunk(&self, chunk: &CodeChunk, vector: Option<&[f32]>) -> Result<()> {
    let table = self.code_chunks_table().await?;

    // Delete existing
    let _ = table.delete(&format!("id = '{}'", chunk.id)).await;

    // Add new
    let batch = code_chunk_to_batch(chunk, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], code_chunks_schema(self.vector_dim));
    table.add(Box::new(batches)).execute().await?;

    Ok(())
  }

  /// Search code chunks by vector similarity
  pub async fn search_code_chunks(
    &self,
    query_vector: &[f32],
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(CodeChunk, f32)>> {
    let table = self.code_chunks_table().await?;

    let query = if let Some(f) = filter {
      table.vector_search(query_vector.to_vec())?.limit(limit).only_if(f)
    } else {
      table.vector_search(query_vector.to_vec())?.limit(limit)
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut chunks = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        let chunk = batch_to_code_chunk(&batch, i)?;
        // Get distance score from _distance column if present
        let distance = batch
          .column_by_name("_distance")
          .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
          .map(|arr| arr.value(i))
          .unwrap_or(0.0);
        chunks.push((chunk, distance));
      }
    }

    Ok(chunks)
  }

  /// List code chunks with optional filters
  pub async fn list_code_chunks(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<CodeChunk>> {
    let table = self.code_chunks_table().await?;

    let query = match (filter, limit) {
      (Some(f), Some(l)) => table.query().only_if(f).limit(l),
      (Some(f), None) => table.query().only_if(f),
      (None, Some(l)) => table.query().limit(l),
      (None, None) => table.query(),
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut chunks = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        chunks.push(batch_to_code_chunk(&batch, i)?);
      }
    }

    Ok(chunks)
  }

  /// Count code chunks matching a filter (uses native count_rows for efficiency)
  pub async fn count_code_chunks(&self, filter: Option<&str>) -> Result<usize> {
    let table = self.code_chunks_table().await?;
    let count = table.count_rows(filter.map(|s| s.to_string())).await?;
    Ok(count)
  }

  /// Get chunks for a specific file
  pub async fn get_chunks_for_file(&self, file_path: &str) -> Result<Vec<CodeChunk>> {
    self
      .list_code_chunks(Some(&format!("file_path = '{}'", file_path)), None)
      .await
  }

  /// Get chunks with their embeddings for a file
  ///
  /// Used for differential re-indexing: when a file changes, we can reuse
  /// embeddings for chunks whose content hasn't changed.
  pub async fn get_chunks_with_embeddings_for_file(
    &self,
    file_path: &str,
  ) -> Result<Vec<(CodeChunk, Option<Vec<f32>>)>> {
    let table = self.code_chunks_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("file_path = '{}'", file_path))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut chunks_with_embeddings = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        let chunk = batch_to_code_chunk(&batch, i)?;
        let embedding = extract_vector_from_batch(&batch, i, self.vector_dim);
        chunks_with_embeddings.push((chunk, embedding));
      }
    }

    Ok(chunks_with_embeddings)
  }

  /// Find code chunks by ID prefix
  ///
  /// Searches for code chunks whose ID starts with the given prefix.
  pub async fn find_code_chunks_by_prefix(&self, prefix: &str) -> Result<Vec<CodeChunk>> {
    if prefix.len() < 6 {
      return Err(DbError::InvalidInput("ID prefix must be at least 6 characters".into()));
    }

    // Use LIKE query for prefix matching
    let filter = format!("id LIKE '{}%'", prefix);
    self.list_code_chunks(Some(&filter), Some(10)).await
  }

  /// Get a code chunk by ID or prefix
  ///
  /// First tries exact match, then falls back to prefix matching.
  /// Returns error if prefix matches multiple chunks (ambiguous).
  pub async fn get_code_chunk_by_id_or_prefix(&self, id_or_prefix: &str) -> Result<Option<CodeChunk>> {
    // Try exact UUID match first
    if let Ok(chunk_id) = Uuid::parse_str(id_or_prefix)
      && let Ok(Some(chunk)) = self.get_code_chunk(&chunk_id).await
    {
      return Ok(Some(chunk));
    }

    // Try prefix match if at least 6 characters
    if id_or_prefix.len() >= 6 {
      let matches = self.find_code_chunks_by_prefix(id_or_prefix).await?;
      match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches.into_iter().next().expect("just checked len"))),
        count => Err(DbError::AmbiguousPrefix {
          prefix: id_or_prefix.to_string(),
          count,
        }),
      }
    } else if id_or_prefix.len() < 6 {
      Err(DbError::InvalidInput("ID prefix must be at least 6 characters".into()))
    } else {
      Ok(None)
    }
  }
}

/// Convert a CodeChunk to an Arrow RecordBatch
fn code_chunk_to_batch(chunk: &CodeChunk, vector: Option<&[f32]>, vector_dim: usize) -> Result<RecordBatch> {
  let id = StringArray::from(vec![chunk.id.to_string()]);
  let project_id = StringArray::from(vec![""]); // We don't have project_id on CodeChunk, using empty
  let file_path = StringArray::from(vec![chunk.file_path.clone()]);
  let content = StringArray::from(vec![chunk.content.clone()]);
  let language = StringArray::from(vec![format!("{:?}", chunk.language).to_lowercase()]);
  let chunk_type = StringArray::from(vec![format!("{:?}", chunk.chunk_type).to_lowercase()]);
  let symbols = StringArray::from(vec![serde_json::to_string(&chunk.symbols)?]);
  let imports = StringArray::from(vec![serde_json::to_string(&chunk.imports)?]);
  let calls = StringArray::from(vec![serde_json::to_string(&chunk.calls)?]);
  let start_line = UInt32Array::from(vec![chunk.start_line]);
  let end_line = UInt32Array::from(vec![chunk.end_line]);
  let file_hash = StringArray::from(vec![chunk.file_hash.clone()]);
  let indexed_at = Int64Array::from(vec![chunk.indexed_at.timestamp_millis()]);

  // Definition metadata fields
  let definition_kind = StringArray::from(vec![chunk.definition_kind.clone()]);
  let definition_name = StringArray::from(vec![chunk.definition_name.clone()]);
  let visibility = StringArray::from(vec![chunk.visibility.clone()]);
  let signature = StringArray::from(vec![chunk.signature.clone()]);
  let docstring = StringArray::from(vec![chunk.docstring.clone()]);
  let parent_definition = StringArray::from(vec![chunk.parent_definition.clone()]);
  let embedding_text = StringArray::from(vec![chunk.embedding_text.clone()]);
  let content_hash = StringArray::from(vec![chunk.content_hash.clone()]);

  // Pre-computed relationship counts
  let caller_count = UInt32Array::from(vec![chunk.caller_count]);
  let callee_count = UInt32Array::from(vec![chunk.callee_count]);

  // Handle vector - pad or truncate to match expected dimensions
  let vector_arr = if let Some(v) = vector {
    let mut vec_padded = v.to_vec();
    vec_padded.resize(vector_dim, 0.0);
    Some(Float32Array::from(vec_padded))
  } else {
    None
  };

  let vector_list = if let Some(v) = vector_arr {
    let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
    FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(v), None)?
  } else {
    // Create null vector
    let null_vec = Float32Array::from(vec![0.0f32; vector_dim]);
    let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
    FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(null_vec), Some(vec![false].into()))?
  };

  let batch = RecordBatch::try_new(
    code_chunks_schema(vector_dim),
    vec![
      Arc::new(id),
      Arc::new(project_id),
      Arc::new(file_path),
      Arc::new(content),
      Arc::new(language),
      Arc::new(chunk_type),
      Arc::new(symbols),
      Arc::new(imports),
      Arc::new(calls),
      Arc::new(start_line),
      Arc::new(end_line),
      Arc::new(file_hash),
      Arc::new(indexed_at),
      Arc::new(definition_kind),
      Arc::new(definition_name),
      Arc::new(visibility),
      Arc::new(signature),
      Arc::new(docstring),
      Arc::new(parent_definition),
      Arc::new(embedding_text),
      Arc::new(content_hash),
      Arc::new(caller_count),
      Arc::new(callee_count),
      Arc::new(vector_list),
    ],
  )?;

  Ok(batch)
}

/// Extract vector embedding from a RecordBatch row
fn extract_vector_from_batch(batch: &RecordBatch, row: usize, vector_dim: usize) -> Option<Vec<f32>> {
  batch
    .column_by_name("vector")
    .and_then(|col| col.as_any().downcast_ref::<FixedSizeListArray>())
    .and_then(|arr| {
      if arr.is_null(row) {
        return None;
      }
      let values = arr.value(row);
      let float_arr = values.as_any().downcast_ref::<Float32Array>()?;
      let vec: Vec<f32> = (0..vector_dim).map(|i| float_arr.value(i)).collect();
      // Check if it's a null/zero vector
      if vec.iter().all(|&v| v == 0.0) { None } else { Some(vec) }
    })
}

/// Convert a RecordBatch row to a CodeChunk
fn batch_to_code_chunk(batch: &RecordBatch, row: usize) -> Result<CodeChunk> {
  let get_string = |name: &str| -> Result<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_string_opt = |name: &str| -> Option<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
  };

  let get_u32 = |name: &str| -> Result<u32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_i64 = |name: &str| -> Result<i64> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let id_str = get_string("id")?;
  let language_str = get_string("language")?;
  let chunk_type_str = get_string("chunk_type")?;
  let symbols_json = get_string("symbols")?;

  // imports and calls may not exist in older databases
  let imports_json = get_string_opt("imports");
  let calls_json = get_string_opt("calls");

  let language = match language_str.as_str() {
    "typescript" => Language::TypeScript,
    "javascript" => Language::JavaScript,
    "tsx" => Language::Tsx,
    "jsx" => Language::Jsx,
    "html" => Language::Html,
    "css" => Language::Css,
    "rust" => Language::Rust,
    "python" => Language::Python,
    "go" => Language::Go,
    "json" => Language::Json,
    "yaml" => Language::Yaml,
    "toml" => Language::Toml,
    "markdown" => Language::Markdown,
    "shell" => Language::Shell,
    _ => Language::Markdown, // Fallback
  };

  let chunk_type = match chunk_type_str.as_str() {
    "function" => ChunkType::Function,
    "class" => ChunkType::Class,
    "module" => ChunkType::Module,
    "block" => ChunkType::Block,
    "import" => ChunkType::Import,
    _ => ChunkType::Block, // Fallback
  };

  let indexed_at = Utc
    .timestamp_millis_opt(get_i64("indexed_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid indexed_at timestamp".into()))?;

  let content = get_string("content")?;
  let tokens_estimate = (content.len() / 4) as u32; // Estimate tokens from content

  // Parse imports/calls, defaulting to empty vec if not present
  let imports = imports_json
    .and_then(|j| serde_json::from_str(&j).ok())
    .unwrap_or_default();
  let calls = calls_json
    .and_then(|j| serde_json::from_str(&j).ok())
    .unwrap_or_default();

  // Definition metadata (all optional, for backwards compatibility)
  let definition_kind = get_string_opt("definition_kind").filter(|s| !s.is_empty());
  let definition_name = get_string_opt("definition_name").filter(|s| !s.is_empty());
  let visibility = get_string_opt("visibility").filter(|s| !s.is_empty());
  let signature = get_string_opt("signature").filter(|s| !s.is_empty());
  let docstring = get_string_opt("docstring").filter(|s| !s.is_empty());
  let parent_definition = get_string_opt("parent_definition").filter(|s| !s.is_empty());
  let embedding_text = get_string_opt("embedding_text").filter(|s| !s.is_empty());
  let content_hash = get_string_opt("content_hash").filter(|s| !s.is_empty());

  // Pre-computed counts (optional for backwards compatibility with existing databases)
  let get_u32_opt = |name: &str| -> Option<u32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
      .map(|a| a.value(row))
  };
  let caller_count = get_u32_opt("caller_count").unwrap_or(0);
  let callee_count = get_u32_opt("callee_count").unwrap_or(0);

  Ok(CodeChunk {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    file_path: get_string("file_path")?,
    content,
    language,
    chunk_type,
    symbols: serde_json::from_str(&symbols_json)?,
    imports,
    calls,
    start_line: get_u32("start_line")?,
    end_line: get_u32("end_line")?,
    file_hash: get_string("file_hash")?,
    indexed_at,
    tokens_estimate,
    definition_kind,
    definition_name,
    visibility,
    signature,
    docstring,
    parent_definition,
    embedding_text,
    content_hash,
    caller_count,
    callee_count,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use engram_core::ProjectId;
  use std::path::Path;
  use tempfile::TempDir;

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();
    (temp_dir, db)
  }

  fn create_test_chunk() -> CodeChunk {
    let content = "fn test() {}".to_string();
    CodeChunk {
      id: Uuid::new_v4(),
      file_path: "/test/file.rs".to_string(),
      tokens_estimate: (content.len() / 4) as u32,
      content,
      language: Language::Rust,
      chunk_type: ChunkType::Function,
      symbols: vec!["test".to_string()],
      imports: Vec::new(),
      calls: Vec::new(),
      start_line: 1,
      end_line: 1,
      file_hash: "abc123".to_string(),
      indexed_at: Utc::now(),
      definition_kind: None,
      definition_name: None,
      visibility: None,
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: None,
      content_hash: None,
      caller_count: 0,
      callee_count: 0,
    }
  }

  #[tokio::test]
  async fn test_add_and_get_code_chunk() {
    let (_temp, db) = create_test_db().await;
    let chunk = create_test_chunk();

    db.add_code_chunk(&chunk, None).await.unwrap();

    let retrieved = db.get_code_chunk(&chunk.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.file_path, chunk.file_path);
    assert_eq!(retrieved.content, chunk.content);
  }

  #[tokio::test]
  async fn test_list_code_chunks() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk();
    c1.file_path = "/test/a.rs".to_string();
    let mut c2 = create_test_chunk();
    c2.file_path = "/test/b.rs".to_string();

    db.add_code_chunk(&c1, None).await.unwrap();
    db.add_code_chunk(&c2, None).await.unwrap();

    let chunks = db.list_code_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 2);
  }

  #[tokio::test]
  async fn test_delete_chunks_for_file() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk();
    c1.file_path = "/test/target.rs".to_string();
    let mut c2 = create_test_chunk();
    c2.file_path = "/test/target.rs".to_string();
    let mut c3 = create_test_chunk();
    c3.file_path = "/test/other.rs".to_string();

    db.add_code_chunk(&c1, None).await.unwrap();
    db.add_code_chunk(&c2, None).await.unwrap();
    db.add_code_chunk(&c3, None).await.unwrap();

    db.delete_chunks_for_file("/test/target.rs").await.unwrap();

    let chunks = db.list_code_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].file_path, "/test/other.rs");
  }

  #[tokio::test]
  async fn test_delete_chunks_for_files_batch() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk();
    c1.file_path = "/test/a.rs".to_string();
    let mut c2 = create_test_chunk();
    c2.file_path = "/test/b.rs".to_string();
    let mut c3 = create_test_chunk();
    c3.file_path = "/test/c.rs".to_string();
    let mut c4 = create_test_chunk();
    c4.file_path = "/test/keep.rs".to_string();

    db.add_code_chunk(&c1, None).await.unwrap();
    db.add_code_chunk(&c2, None).await.unwrap();
    db.add_code_chunk(&c3, None).await.unwrap();
    db.add_code_chunk(&c4, None).await.unwrap();

    // Delete multiple files in one operation
    db.delete_chunks_for_files(&["/test/a.rs", "/test/b.rs", "/test/c.rs"])
      .await
      .unwrap();

    let chunks = db.list_code_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].file_path, "/test/keep.rs");
  }

  #[tokio::test]
  async fn test_delete_chunks_for_files_empty() {
    let (_temp, db) = create_test_db().await;
    // Should not error on empty input
    db.delete_chunks_for_files(&[]).await.unwrap();
  }

  #[tokio::test]
  async fn test_rename_file() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk();
    c1.file_path = "/old/path/file.rs".to_string();
    let mut c2 = create_test_chunk();
    c2.file_path = "/old/path/file.rs".to_string();
    let mut c3 = create_test_chunk();
    c3.file_path = "/other/file.rs".to_string();

    db.add_code_chunk(&c1, None).await.unwrap();
    db.add_code_chunk(&c2, None).await.unwrap();
    db.add_code_chunk(&c3, None).await.unwrap();

    // Rename the file
    let renamed = db.rename_file("/old/path/file.rs", "/new/path/file.rs").await.unwrap();
    assert_eq!(renamed, 2); // Two chunks were renamed

    // Verify chunks are at new path
    let chunks = db.get_chunks_for_file("/new/path/file.rs").await.unwrap();
    assert_eq!(chunks.len(), 2);

    // Verify old path has no chunks
    let old_chunks = db.get_chunks_for_file("/old/path/file.rs").await.unwrap();
    assert!(old_chunks.is_empty());

    // Verify other file is unchanged
    let other_chunks = db.get_chunks_for_file("/other/file.rs").await.unwrap();
    assert_eq!(other_chunks.len(), 1);
  }

  #[tokio::test]
  async fn test_rename_file_not_found() {
    let (_temp, db) = create_test_db().await;

    // Renaming a non-existent file should return 0
    let renamed = db.rename_file("/nonexistent.rs", "/new.rs").await.unwrap();
    assert_eq!(renamed, 0);
  }

  #[tokio::test]
  async fn test_rename_files_with_prefix() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk();
    c1.file_path = "src/old/a.rs".to_string();
    let mut c2 = create_test_chunk();
    c2.file_path = "src/old/b.rs".to_string();
    let mut c3 = create_test_chunk();
    c3.file_path = "src/old/nested/c.rs".to_string();
    let mut c4 = create_test_chunk();
    c4.file_path = "src/other/d.rs".to_string();

    db.add_code_chunk(&c1, None).await.unwrap();
    db.add_code_chunk(&c2, None).await.unwrap();
    db.add_code_chunk(&c3, None).await.unwrap();
    db.add_code_chunk(&c4, None).await.unwrap();

    // Rename folder
    let renamed = db.rename_files_with_prefix("src/old/", "src/new/").await.unwrap();
    assert_eq!(renamed, 3); // Three files were renamed

    // Verify files are at new paths
    let a_chunks = db.get_chunks_for_file("src/new/a.rs").await.unwrap();
    assert_eq!(a_chunks.len(), 1);

    let b_chunks = db.get_chunks_for_file("src/new/b.rs").await.unwrap();
    assert_eq!(b_chunks.len(), 1);

    let c_chunks = db.get_chunks_for_file("src/new/nested/c.rs").await.unwrap();
    assert_eq!(c_chunks.len(), 1);

    // Verify other file is unchanged
    let d_chunks = db.get_chunks_for_file("src/other/d.rs").await.unwrap();
    assert_eq!(d_chunks.len(), 1);
  }
}
