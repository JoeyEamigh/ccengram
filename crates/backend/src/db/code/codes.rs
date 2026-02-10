// Code chunks table operations

use std::sync::Arc;

use arrow_array::{
  Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use tracing::{debug, trace};
use uuid::Uuid;

use crate::{
  db::{
    connection::{DbError, ProjectDb, Result},
    schema::code_chunks_schema,
  },
  domain::code::{ChunkType, CodeChunk, Language},
};

impl ProjectDb {
  /// Upsert code chunks for a file using merge_insert
  ///
  /// This atomically:
  /// - Updates existing chunks with matching content_hash
  /// - Inserts new chunks that don't exist
  /// - Deletes old chunks for this file that aren't in the new set
  #[tracing::instrument(level = "trace", skip(self, chunks), fields(batch_size = chunks.len()))]
  pub async fn upsert_code_chunks(&self, file_path: &str, chunks: &[(CodeChunk, Vec<f32>)]) -> Result<()> {
    if chunks.is_empty() {
      // If no chunks, just delete any existing chunks for this file
      return self.delete_chunks_for_file(file_path).await;
    }

    trace!(
      table = "code_chunks",
      operation = "merge_insert",
      file_path = %file_path,
      batch_size = chunks.len(),
      "Upserting code chunks"
    );

    let table = self.code_chunks_table();

    // Create a batched RecordBatch with all chunks
    let batch = code_chunks_to_batch(chunks, self.vector_dim)?;
    let iter = RecordBatchIterator::new(vec![Ok(batch)], code_chunks_schema(self.vector_dim));

    // Use merge_insert with file_path + start_line + end_line as the key
    // This uniquely identifies each chunk's span and handles nested structures
    // (e.g., a class and its method can share start_line but have different end_line).
    // - when_matched_update_all: update existing chunks at same span
    // - when_not_matched_insert_all: insert new chunks
    // - when_not_matched_by_source_delete: delete old chunks for this file not in new data
    let escaped_path = file_path.replace('\'', "''");
    let mut builder = table.merge_insert(&["file_path", "start_line", "end_line"]);
    builder
      .when_matched_update_all(None)
      .when_not_matched_insert_all()
      .when_not_matched_by_source_delete(Some(format!("file_path = '{}'", escaped_path)));
    builder.execute(Box::new(iter)).await?;

    Ok(())
  }

  /// Batch upsert code chunks for multiple files using a single merge_insert
  ///
  /// More efficient than calling `upsert_code_chunks` per file when flushing
  /// multiple files at once. Uses `file_path IN (...)` for the delete filter.
  #[tracing::instrument(level = "trace", skip(self, chunks), fields(file_count = file_paths.len(), chunk_count = chunks.len()))]
  pub async fn upsert_code_chunks_batch(&self, file_paths: &[&str], chunks: &[(CodeChunk, Vec<f32>)]) -> Result<()> {
    if chunks.is_empty() {
      // Delete all chunks for these files with a single bulk delete
      if !file_paths.is_empty() {
        let table = self.code_chunks_table();
        let paths_filter = file_paths
          .iter()
          .map(|p| format!("'{}'", p.replace('\'', "''")))
          .collect::<Vec<_>>()
          .join(", ");
        table.delete(&format!("file_path IN ({})", paths_filter)).await?;
      }
      return Ok(());
    }

    trace!(
      table = "code_chunks",
      operation = "merge_insert_batch",
      file_count = file_paths.len(),
      chunk_count = chunks.len(),
      "Batch upserting code chunks"
    );

    let table = self.code_chunks_table();

    // Create a single RecordBatch with all chunks from all files
    let batch = code_chunks_to_batch(chunks, self.vector_dim)?;
    let iter = RecordBatchIterator::new(vec![Ok(batch)], code_chunks_schema(self.vector_dim));

    // Build IN clause for delete filter: file_path IN ('path1', 'path2', ...)
    let paths_filter = file_paths
      .iter()
      .map(|p| format!("'{}'", p.replace('\'', "''")))
      .collect::<Vec<_>>()
      .join(", ");

    // Use file_path + start_line + end_line as key to handle nested structures
    let mut builder = table.merge_insert(&["file_path", "start_line", "end_line"]);
    builder
      .when_matched_update_all(None)
      .when_not_matched_insert_all()
      .when_not_matched_by_source_delete(Some(format!("file_path IN ({})", paths_filter)));
    builder.execute(Box::new(iter)).await?;

    Ok(())
  }

  /// Get a code chunk by ID
  pub async fn get_code_chunk(&self, id: &Uuid) -> Result<Option<CodeChunk>> {
    let table = self.code_chunks_table();
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
    debug!(table = "code_chunks", operation = "delete_for_file", file = %file_path, "Deleting chunks for file");
    let table = self.code_chunks_table();
    let escaped_path = file_path.replace('\'', "''");
    table.delete(&format!("file_path = '{}'", escaped_path)).await?;
    Ok(())
  }

  /// Delete a code chunk by ID
  pub async fn delete_code_chunk(&self, id: &Uuid) -> Result<()> {
    debug!(table = "code_chunks", operation = "delete", id = %id, "Deleting code chunk");
    let table = self.code_chunks_table();
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Rename a file by updating file_path for all its chunks (preserves embeddings)
  ///
  /// This is more efficient than delete + re-index because it preserves existing
  /// embeddings and other computed data.
  pub async fn rename_file(&self, old_path: &str, new_path: &str) -> Result<usize> {
    debug!(
      table = "code_chunks",
      operation = "rename_file",
      old_path = %old_path,
      new_path = %new_path,
      "Renaming file in index"
    );

    let table = self.code_chunks_table();

    // Get chunks for the old path to count and update
    let chunks = self.get_chunks_for_file(old_path).await?;
    let count = chunks.len();

    if count == 0 {
      debug!(old_path = %old_path, "No chunks found for file rename");
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

    debug!(
      old_path = %old_path,
      new_path = %new_path,
      chunks_renamed = count,
      "File rename complete"
    );

    Ok(count)
  }

  /// Search code chunks by vector similarity
  pub async fn search_code_chunks(
    &self,
    query_vector: &[f32],
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(CodeChunk, f32)>> {
    debug!(
      table = "code_chunks",
      operation = "search",
      query_len = query_vector.len(),
      limit = limit,
      has_filter = filter.is_some(),
      "Searching code chunks"
    );

    let table = self.code_chunks_table();

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

    debug!(
      table = "code_chunks",
      operation = "search",
      results = chunks.len(),
      "Search complete"
    );

    Ok(chunks)
  }

  /// Search code chunks by full-text search (BM25)
  ///
  /// Searches the `embedding_text` column which contains enriched, tokenized text.
  /// Results are ordered by BM25 relevance score.
  pub async fn fts_search_code_chunks(
    &self,
    query: &str,
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(CodeChunk, f32)>> {
    use lance_index::scalar::FullTextSearchQuery;

    debug!(
      table = "code_chunks",
      operation = "fts_search",
      query = %query,
      limit = limit,
      has_filter = filter.is_some(),
      "FTS searching code chunks"
    );

    let table = self.code_chunks_table();

    let fts_query = FullTextSearchQuery::new(query.to_owned())
      .with_column("embedding_text".to_string())
      .map_err(|e| DbError::Query(format!("FTS query construction failed: {e}")))?;

    let builder = if let Some(f) = filter {
      table.query().full_text_search(fts_query).limit(limit).only_if(f)
    } else {
      table.query().full_text_search(fts_query).limit(limit)
    };

    let results: Vec<RecordBatch> = builder.execute().await?.try_collect().await?;

    let mut chunks = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        let chunk = batch_to_code_chunk(&batch, i)?;
        let score = batch
          .column_by_name("_score")
          .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
          .map(|arr| arr.value(i))
          .unwrap_or(0.0);
        chunks.push((chunk, score));
      }
    }

    debug!(
      table = "code_chunks",
      operation = "fts_search",
      results = chunks.len(),
      "FTS search complete"
    );

    Ok(chunks)
  }

  /// List code chunks with optional filters
  #[tracing::instrument(level = "trace", skip(self), fields(has_filter = filter.is_some(), limit = ?limit))]
  pub async fn list_code_chunks(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<CodeChunk>> {
    let table = self.code_chunks_table();

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
  #[tracing::instrument(level = "trace", skip(self), fields(file = %file_path))]
  pub async fn get_chunks_with_embeddings_for_file(&self, file_path: &str) -> Result<Vec<(CodeChunk, Vec<f32>)>> {
    let table = self.code_chunks_table();

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
        let embedding = extract_vector_from_batch(&batch, i, self.vector_dim)?;
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

  /// Get the embedding vector for a code chunk by ID.
  ///
  /// Returns None if the chunk doesn't exist or has no embedding.
  /// This is useful for reusing embeddings in cross-domain searches.
  pub async fn get_code_chunk_embedding(&self, id: &Uuid) -> Result<Option<Vec<f32>>> {
    let table = self.code_chunks_table();
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

    Ok(Some(extract_vector_from_batch(batch, 0, self.vector_dim)?))
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

/// Convert multiple CodeChunks to a single Arrow RecordBatch (true batch insert)
fn code_chunks_to_batch(chunks: &[(CodeChunk, Vec<f32>)], vector_dim: usize) -> Result<RecordBatch> {
  let n = chunks.len();

  let ids: Vec<String> = chunks.iter().map(|(c, _)| c.id.to_string()).collect();
  let project_ids: Vec<&str> = vec![""; n];
  let file_paths: Vec<&str> = chunks.iter().map(|(c, _)| c.file_path.as_str()).collect();
  let contents: Vec<&str> = chunks.iter().map(|(c, _)| c.content.as_str()).collect();
  let languages: Vec<String> = chunks
    .iter()
    .map(|(c, _)| format!("{:?}", c.language).to_lowercase())
    .collect();
  let chunk_types: Vec<String> = chunks
    .iter()
    .map(|(c, _)| format!("{:?}", c.chunk_type).to_lowercase())
    .collect();
  let symbols_json: Vec<String> = chunks
    .iter()
    .map(|(c, _)| serde_json::to_string(&c.symbols).unwrap_or_default())
    .collect();
  let imports_json: Vec<String> = chunks
    .iter()
    .map(|(c, _)| serde_json::to_string(&c.imports).unwrap_or_default())
    .collect();
  let calls_json: Vec<String> = chunks
    .iter()
    .map(|(c, _)| serde_json::to_string(&c.calls).unwrap_or_default())
    .collect();
  let start_lines: Vec<u32> = chunks.iter().map(|(c, _)| c.start_line).collect();
  let end_lines: Vec<u32> = chunks.iter().map(|(c, _)| c.end_line).collect();
  let file_hashes: Vec<&str> = chunks.iter().map(|(c, _)| c.file_hash.as_str()).collect();
  let indexed_ats: Vec<i64> = chunks.iter().map(|(c, _)| c.indexed_at.timestamp_millis()).collect();

  // Definition metadata
  let def_kinds: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.definition_kind.as_deref()).collect();
  let def_names: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.definition_name.as_deref()).collect();
  let visibilities: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.visibility.as_deref()).collect();
  let signatures: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.signature.as_deref()).collect();
  let docstrings: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.docstring.as_deref()).collect();
  let parent_defs: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.parent_definition.as_deref()).collect();
  let embed_texts: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.embedding_text.as_deref()).collect();
  let content_hashes: Vec<Option<&str>> = chunks.iter().map(|(c, _)| c.content_hash.as_deref()).collect();

  // Counts
  let caller_counts: Vec<u32> = chunks.iter().map(|(c, _)| c.caller_count).collect();
  let callee_counts: Vec<u32> = chunks.iter().map(|(c, _)| c.callee_count).collect();

  // Vectors - flatten all into one array
  let mut all_vectors: Vec<f32> = Vec::with_capacity(n * vector_dim);
  for (_, vec) in chunks {
    let mut v = vec.clone();
    v.resize(vector_dim, 0.0);
    all_vectors.extend(v);
  }

  let vector_values = Float32Array::from(all_vectors);
  let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
  let vector_list = FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(vector_values), None)?;

  let batch = RecordBatch::try_new(
    code_chunks_schema(vector_dim),
    vec![
      Arc::new(StringArray::from(ids)),
      Arc::new(StringArray::from(project_ids)),
      Arc::new(StringArray::from(file_paths)),
      Arc::new(StringArray::from(contents)),
      Arc::new(StringArray::from(languages)),
      Arc::new(StringArray::from(chunk_types)),
      Arc::new(StringArray::from(symbols_json)),
      Arc::new(StringArray::from(imports_json)),
      Arc::new(StringArray::from(calls_json)),
      Arc::new(UInt32Array::from(start_lines)),
      Arc::new(UInt32Array::from(end_lines)),
      Arc::new(StringArray::from(file_hashes)),
      Arc::new(Int64Array::from(indexed_ats)),
      Arc::new(StringArray::from(def_kinds)),
      Arc::new(StringArray::from(def_names)),
      Arc::new(StringArray::from(visibilities)),
      Arc::new(StringArray::from(signatures)),
      Arc::new(StringArray::from(docstrings)),
      Arc::new(StringArray::from(parent_defs)),
      Arc::new(StringArray::from(embed_texts)),
      Arc::new(StringArray::from(content_hashes)),
      Arc::new(UInt32Array::from(caller_counts)),
      Arc::new(UInt32Array::from(callee_counts)),
      Arc::new(vector_list),
    ],
  )?;

  Ok(batch)
}

/// Extract vector embedding from a RecordBatch row
fn extract_vector_from_batch(batch: &RecordBatch, row: usize, vector_dim: usize) -> Result<Vec<f32>> {
  batch
    .column_by_name("vector")
    .and_then(|col| col.as_any().downcast_ref::<FixedSizeListArray>())
    .and_then(|arr| {
      if arr.is_null(row) {
        return None;
      }
      let values = arr.value(row);
      let float_arr = values.as_any().downcast_ref::<Float32Array>()?;
      Some((0..vector_dim).map(|i| float_arr.value(i)).collect())
    })
    .ok_or_else(|| DbError::NotFound("vector column missing or null".into()))
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
  use std::path::Path;

  use tempfile::TempDir;

  use super::*;
  use crate::{config::Config, domain::project::ProjectId};

  fn dummy_vector(dim: usize) -> Vec<f32> {
    vec![0.0f32; dim]
  }

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test")).await;
    let db = ProjectDb::open_at_path(
      project_id,
      temp_dir.path().join("test.lancedb"),
      Arc::new(Config::default()),
    )
    .await
    .unwrap();
    (temp_dir, db)
  }

  fn create_test_chunk() -> CodeChunk {
    create_test_chunk_with_hash("test_hash_default")
  }

  fn create_test_chunk_with_hash(hash: &str) -> CodeChunk {
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
      content_hash: Some(hash.to_string()),
      caller_count: 0,
      callee_count: 0,
    }
  }

  #[tokio::test]
  async fn test_upsert_and_get_code_chunk() {
    let (_temp, db) = create_test_db().await;
    let chunk = create_test_chunk();
    let vec = dummy_vector(db.vector_dim);

    db.upsert_code_chunks(&chunk.file_path.clone(), &[(chunk.clone(), vec)])
      .await
      .unwrap();

    let retrieved = db.get_code_chunk(&chunk.id).await.unwrap();
    assert!(retrieved.is_some(), "should retrieve the chunk");
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.file_path, chunk.file_path);
    assert_eq!(retrieved.content, chunk.content);
  }

  #[tokio::test]
  async fn test_list_code_chunks() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk_with_hash("hash_a");
    c1.file_path = "/test/a.rs".to_string();
    let mut c2 = create_test_chunk_with_hash("hash_b");
    c2.file_path = "/test/b.rs".to_string();
    let vec = dummy_vector(db.vector_dim);

    db.upsert_code_chunks("/test/a.rs", &[(c1, vec.clone())]).await.unwrap();
    db.upsert_code_chunks("/test/b.rs", &[(c2, vec)]).await.unwrap();

    let chunks = db.list_code_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 2, "should list both chunks");
  }

  #[tokio::test]
  async fn test_delete_chunks_for_file() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk_with_hash("hash_target_1");
    c1.file_path = "/test/target.rs".to_string();
    let mut c2 = create_test_chunk_with_hash("hash_target_2");
    c2.file_path = "/test/target.rs".to_string();
    let mut c3 = create_test_chunk_with_hash("hash_other");
    c3.file_path = "/test/other.rs".to_string();
    let vec = dummy_vector(db.vector_dim);

    db.upsert_code_chunks("/test/target.rs", &[(c1, vec.clone()), (c2, vec.clone())])
      .await
      .unwrap();
    db.upsert_code_chunks("/test/other.rs", &[(c3, vec)]).await.unwrap();

    db.delete_chunks_for_file("/test/target.rs").await.unwrap();

    let chunks = db.list_code_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 1, "should have only the other file chunk");
    assert_eq!(chunks[0].file_path, "/test/other.rs");
  }

  #[tokio::test]
  async fn test_rename_file() {
    let (_temp, db) = create_test_db().await;

    let mut c1 = create_test_chunk_with_hash("hash_old_1");
    c1.file_path = "/old/path/file.rs".to_string();
    let mut c2 = create_test_chunk_with_hash("hash_old_2");
    c2.file_path = "/old/path/file.rs".to_string();
    let mut c3 = create_test_chunk_with_hash("hash_other");
    c3.file_path = "/other/file.rs".to_string();
    let vec = dummy_vector(db.vector_dim);

    db.upsert_code_chunks("/old/path/file.rs", &[(c1, vec.clone()), (c2, vec.clone())])
      .await
      .unwrap();
    db.upsert_code_chunks("/other/file.rs", &[(c3, vec)]).await.unwrap();

    // Rename the file
    let renamed = db.rename_file("/old/path/file.rs", "/new/path/file.rs").await.unwrap();
    assert_eq!(renamed, 2, "two chunks should be renamed");

    // Verify chunks are at new path
    let chunks = db.get_chunks_for_file("/new/path/file.rs").await.unwrap();
    assert_eq!(chunks.len(), 2);

    // Verify old path has no chunks
    let old_chunks = db.get_chunks_for_file("/old/path/file.rs").await.unwrap();
    assert!(old_chunks.is_empty(), "old path should have no chunks");

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
}
