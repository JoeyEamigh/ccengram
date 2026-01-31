use std::sync::Arc;

use arrow_array::{
  Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use tracing::debug;
use uuid::Uuid;

use crate::{
  db::{
    connection::{DbError, ProjectDb, Result},
    schema::documents_schema,
  },
  domain::document::{DocumentChunk, DocumentId, DocumentSource},
};

impl ProjectDb {
  /// Upsert document chunks for a source using merge_insert
  ///
  /// This atomically:
  /// - Updates existing chunks with matching content_hash
  /// - Inserts new chunks that don't exist
  /// - Deletes old chunks for this source that aren't in the new set
  #[tracing::instrument(level = "trace", skip(self, chunks, vectors), fields(batch_size = chunks.len()))]
  pub async fn upsert_document_chunks(
    &self,
    source: &str,
    chunks: &[DocumentChunk],
    vectors: &[Vec<f32>],
  ) -> Result<()> {
    if chunks.is_empty() {
      // If no chunks, just delete any existing chunks for this source
      return self.delete_document_chunks_by_source(source).await;
    }

    debug!(
      table = "documents",
      operation = "merge_insert",
      source = %source,
      batch_size = chunks.len(),
      "Upserting document chunks"
    );

    let table = self.documents_table();

    let batches: Vec<RecordBatch> = chunks
      .iter()
      .zip(vectors.iter())
      .map(|(chunk, vec)| chunk_to_batch(chunk, vec, self.vector_dim))
      .collect::<Result<Vec<_>>>()?;

    let schema = documents_schema(self.vector_dim);
    let iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    // Use merge_insert with source + chunk_index as the key
    // This uniquely identifies each chunk's position and avoids collisions when
    // multiple chunks have identical content (same content_hash).
    let escaped_source = source.replace('\'', "''");
    let mut builder = table.merge_insert(&["source", "chunk_index"]);
    builder
      .when_matched_update_all(None)
      .when_not_matched_insert_all()
      .when_not_matched_by_source_delete(Some(format!("source = '{}'", escaped_source)));
    builder.execute(Box::new(iter)).await?;

    Ok(())
  }

  /// Batch upsert document chunks for multiple sources
  ///
  /// More efficient than calling `upsert_document_chunks` per source when flushing
  /// multiple files at once. Uses bulk delete + add for consistency with code chunks.
  #[tracing::instrument(level = "trace", skip(self, chunks, vectors), fields(source_count = sources.len(), chunk_count = chunks.len()))]
  pub async fn upsert_document_chunks_batch(
    &self,
    sources: &[&str],
    chunks: &[DocumentChunk],
    vectors: &[Vec<f32>],
  ) -> Result<()> {
    let table = self.documents_table();

    // Single bulk delete for all sources
    if !sources.is_empty() {
      let sources_filter = sources
        .iter()
        .map(|s| format!("'{}'", s.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");
      table.delete(&format!("source IN ({})", sources_filter)).await?;
    }

    if chunks.is_empty() {
      return Ok(());
    }

    debug!(
      table = "documents",
      operation = "batch_add",
      source_count = sources.len(),
      chunk_count = chunks.len(),
      "Batch adding document chunks"
    );

    // Create batched RecordBatch with all chunks and add them
    let batch = chunks_to_batch(chunks, vectors, self.vector_dim)?;
    let schema = documents_schema(self.vector_dim);
    let iter = RecordBatchIterator::new(vec![Ok(batch)], schema);
    table.add(Box::new(iter)).execute().await?;

    Ok(())
  }

  /// Get a document chunk by ID
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_document_chunk(&self, id: &DocumentId) -> Result<Option<DocumentChunk>> {
    let table = self.documents_table();
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

    Ok(Some(batch_to_chunk(batch, 0)?))
  }

  /// Search document chunks by vector similarity
  #[tracing::instrument(level = "trace", skip(self, query_vector))]
  pub async fn search_documents(
    &self,
    query_vector: &[f32],
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(DocumentChunk, f32)>> {
    debug!(
      table = "documents",
      operation = "search",
      query_len = query_vector.len(),
      limit = limit,
      has_filter = filter.is_some(),
      "Searching documents"
    );

    let table = self.documents_table();

    let query = if let Some(f) = filter {
      table.vector_search(query_vector.to_vec())?.limit(limit).only_if(f)
    } else {
      table.vector_search(query_vector.to_vec())?.limit(limit)
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut chunks = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        let chunk = batch_to_chunk(&batch, i)?;
        let distance = batch
          .column_by_name("_distance")
          .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
          .map(|arr| arr.value(i))
          .unwrap_or(0.0);
        chunks.push((chunk, distance));
      }
    }

    debug!(
      table = "documents",
      operation = "search",
      results = chunks.len(),
      "Search complete"
    );

    Ok(chunks)
  }

  /// List document chunks with optional filter
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn list_document_chunks(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<DocumentChunk>> {
    let table = self.documents_table();

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
        chunks.push(batch_to_chunk(&batch, i)?);
      }
    }

    Ok(chunks)
  }

  /// Delete a single document chunk
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn delete_document_chunk(&self, id: &DocumentId) -> Result<()> {
    debug!(table = "documents", operation = "delete_chunk", id = %id, "Deleting document chunk");
    let table = self.documents_table();
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Find document chunks by ID prefix
  ///
  /// Searches for document chunks whose ID starts with the given prefix.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn find_document_chunks_by_prefix(&self, prefix: &str) -> Result<Vec<DocumentChunk>> {
    if prefix.len() < 6 {
      return Err(DbError::InvalidInput("ID prefix must be at least 6 characters".into()));
    }

    // Use LIKE query for prefix matching
    let filter = format!("id LIKE '{}%'", prefix);
    self.list_document_chunks(Some(&filter), Some(10)).await
  }

  /// Get a document chunk by ID or prefix
  ///
  /// First tries exact match, then falls back to prefix matching.
  /// Returns error if prefix matches multiple chunks (ambiguous).
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_document_chunk_by_id_or_prefix(&self, id_or_prefix: &str) -> Result<Option<DocumentChunk>> {
    // Try exact match first
    if let Ok(chunk_id) = id_or_prefix.parse::<DocumentId>()
      && let Ok(Some(chunk)) = self.get_document_chunk(&chunk_id).await
    {
      return Ok(Some(chunk));
    }

    // Try prefix match if at least 6 characters
    if id_or_prefix.len() >= 6 {
      let matches = self.find_document_chunks_by_prefix(id_or_prefix).await?;
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

  /// Get adjacent document chunks from the same document
  ///
  /// Returns chunks with chunk_index in range [center_index - before, center_index + after],
  /// ordered by chunk_index.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_adjacent_document_chunks(
    &self,
    document_id: &DocumentId,
    center_index: usize,
    chunks_before: usize,
    chunks_after: usize,
  ) -> Result<Vec<DocumentChunk>> {
    let start_index = center_index.saturating_sub(chunks_before);
    let end_index = center_index + chunks_after;

    let filter = format!(
      "document_id = '{}' AND chunk_index >= {} AND chunk_index <= {}",
      document_id, start_index, end_index
    );

    let mut chunks = self.list_document_chunks(Some(&filter), None).await?;

    // Sort by chunk_index to ensure correct order
    chunks.sort_by_key(|c| c.chunk_index);

    Ok(chunks)
  }

  /// Delete all document chunks by source path
  ///
  /// Used by the indexing pipeline to delete all chunks for a file
  /// before re-indexing.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn delete_document_chunks_by_source(&self, source: &str) -> Result<()> {
    debug!(
      table = "documents",
      operation = "delete_chunks_by_source",
      source = %source,
      "Deleting document chunks by source"
    );
    let table = self.documents_table();
    let escaped = source.replace('\'', "''");
    table.delete(&format!("source = '{}'", escaped)).await?;
    Ok(())
  }

  /// Rename document source path (preserves embeddings)
  ///
  /// Updates the source field for all chunks matching the old path.
  /// More efficient than delete + re-index since embeddings are preserved.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn rename_document(&self, from: &str, to: &str) -> Result<usize> {
    debug!(
      table = "documents",
      operation = "rename",
      from = %from,
      to = %to,
      "Renaming document source"
    );

    let table = self.documents_table();

    // Count chunks before rename
    let filter = format!("source = '{}'", from.replace('\'', "''"));
    let count = table.count_rows(Some(filter.clone())).await?;

    if count == 0 {
      return Ok(0);
    }

    // Update source field
    let old_escaped = from.replace('\'', "''");
    let new_escaped = to.replace('\'', "''");

    table
      .update()
      .only_if(format!("source = '{}'", old_escaped))
      .column("source", format!("'{}'", new_escaped))
      .execute()
      .await?;

    Ok(count)
  }
}

/// Convert multiple DocumentChunks to a single Arrow RecordBatch (true batch insert)
fn chunks_to_batch(chunks: &[DocumentChunk], vectors: &[Vec<f32>], vector_dim: usize) -> Result<RecordBatch> {
  let n = chunks.len();

  let ids: Vec<String> = chunks.iter().map(|c| c.id.to_string()).collect();
  let document_ids: Vec<String> = chunks.iter().map(|c| c.document_id.to_string()).collect();
  let project_ids: Vec<String> = chunks.iter().map(|c| c.project_id.to_string()).collect();
  let contents: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
  let titles: Vec<&str> = chunks.iter().map(|c| c.title.as_str()).collect();
  let sources: Vec<&str> = chunks.iter().map(|c| c.source.as_str()).collect();
  let source_types: Vec<String> = chunks.iter().map(|c| c.source_type.as_str().to_string()).collect();
  let chunk_indices: Vec<u32> = chunks.iter().map(|c| c.chunk_index as u32).collect();
  let total_chunks: Vec<u32> = chunks.iter().map(|c| c.total_chunks as u32).collect();
  let char_offsets: Vec<u32> = chunks.iter().map(|c| c.char_offset as u32).collect();
  let content_hashes: Vec<&str> = chunks.iter().map(|c| c.content_hash.as_str()).collect();
  let created_ats: Vec<i64> = chunks.iter().map(|c| c.created_at.timestamp_millis()).collect();
  let updated_ats: Vec<i64> = chunks.iter().map(|c| c.updated_at.timestamp_millis()).collect();

  // Vectors - flatten all into one array
  let mut all_vectors: Vec<f32> = Vec::with_capacity(n * vector_dim);
  for vec in vectors {
    let mut v = vec.clone();
    v.resize(vector_dim, 0.0);
    all_vectors.extend(v);
  }

  let vector_values = Float32Array::from(all_vectors);
  let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
  let vector_list = FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(vector_values), None)?;

  let batch = RecordBatch::try_new(
    documents_schema(vector_dim),
    vec![
      Arc::new(StringArray::from(ids)),
      Arc::new(StringArray::from(document_ids)),
      Arc::new(StringArray::from(project_ids)),
      Arc::new(StringArray::from(contents)),
      Arc::new(StringArray::from(titles)),
      Arc::new(StringArray::from(sources)),
      Arc::new(StringArray::from(source_types)),
      Arc::new(UInt32Array::from(chunk_indices)),
      Arc::new(UInt32Array::from(total_chunks)),
      Arc::new(UInt32Array::from(char_offsets)),
      Arc::new(StringArray::from(content_hashes)),
      Arc::new(Int64Array::from(created_ats)),
      Arc::new(Int64Array::from(updated_ats)),
      Arc::new(vector_list),
    ],
  )?;

  Ok(batch)
}

/// Convert a DocumentChunk to an Arrow RecordBatch
fn chunk_to_batch(chunk: &DocumentChunk, vector: &[f32], vector_dim: usize) -> Result<RecordBatch> {
  let id = StringArray::from(vec![chunk.id.to_string()]);
  let document_id = StringArray::from(vec![chunk.document_id.to_string()]);
  let project_id = StringArray::from(vec![chunk.project_id.to_string()]);
  let content = StringArray::from(vec![chunk.content.clone()]);
  let title = StringArray::from(vec![chunk.title.clone()]);
  let source = StringArray::from(vec![chunk.source.clone()]);
  let source_type = StringArray::from(vec![chunk.source_type.as_str().to_string()]);
  let chunk_index = UInt32Array::from(vec![chunk.chunk_index as u32]);
  let total_chunks = UInt32Array::from(vec![chunk.total_chunks as u32]);
  let char_offset = UInt32Array::from(vec![chunk.char_offset as u32]);
  let content_hash = StringArray::from(vec![chunk.content_hash.clone()]);
  let created_at = Int64Array::from(vec![chunk.created_at.timestamp_millis()]);
  let updated_at = Int64Array::from(vec![chunk.updated_at.timestamp_millis()]);

  // Handle vector

  let mut vec_padded = vector.to_vec();
  vec_padded.resize(vector_dim, 0.0);
  let vector_arr = Float32Array::from(vec_padded);

  let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
  let vector_list = FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(vector_arr), None)?;

  let batch = RecordBatch::try_new(
    documents_schema(vector_dim),
    vec![
      Arc::new(id),
      Arc::new(document_id),
      Arc::new(project_id),
      Arc::new(content),
      Arc::new(title),
      Arc::new(source),
      Arc::new(source_type),
      Arc::new(chunk_index),
      Arc::new(total_chunks),
      Arc::new(char_offset),
      Arc::new(content_hash),
      Arc::new(created_at),
      Arc::new(updated_at),
      Arc::new(vector_list),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a DocumentChunk
fn batch_to_chunk(batch: &RecordBatch, row: usize) -> Result<DocumentChunk> {
  let get_string = |name: &str| -> Result<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
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
  let document_id_str = get_string("document_id")?;
  let project_id_str = get_string("project_id")?;
  let source_type_str = get_string("source_type")?;

  let source_type = source_type_str.parse::<DocumentSource>().map_err(DbError::NotFound)?;

  let created_at = Utc
    .timestamp_millis_opt(get_i64("created_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid created_at timestamp".into()))?;
  let updated_at = Utc
    .timestamp_millis_opt(get_i64("updated_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid updated_at timestamp".into()))?;

  // content_hash may not exist in old databases, provide default
  let content_hash = batch
    .column_by_name("content_hash")
    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
    .map(|a| a.value(row).to_string())
    .unwrap_or_default();

  Ok(DocumentChunk {
    id: id_str.parse().map_err(|_| DbError::NotFound("invalid id".into()))?,
    document_id: document_id_str
      .parse()
      .map_err(|_| DbError::NotFound("invalid document_id".into()))?,
    project_id: Uuid::parse_str(&project_id_str).map_err(|_| DbError::NotFound("invalid project_id".into()))?,
    content: get_string("content")?,
    title: get_string("title")?,
    source: get_string("source")?,
    source_type,
    chunk_index: get_u32("chunk_index")? as usize,
    total_chunks: get_u32("total_chunks")? as usize,
    char_offset: get_u32("char_offset")? as usize,
    content_hash,
    created_at,
    updated_at,
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

  fn create_test_chunk() -> DocumentChunk {
    DocumentChunk::new(
      DocumentId::new(),
      Uuid::new_v4(),
      "Test document content for searching".to_string(),
      "Test Document".to_string(),
      "/path/to/doc.md".to_string(),
      DocumentSource::File,
      0,
      1,
      0,
    )
  }

  #[tokio::test]
  async fn test_upsert_and_get_document_chunk() {
    let (_temp, db) = create_test_db().await;
    let chunk = create_test_chunk();
    let vec = dummy_vector(db.vector_dim);

    db.upsert_document_chunks(&chunk.source, std::slice::from_ref(&chunk), &[vec])
      .await
      .unwrap();

    let retrieved = db.get_document_chunk(&chunk.id).await.unwrap();
    assert!(retrieved.is_some(), "should retrieve the chunk");
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.content, chunk.content);
    assert_eq!(retrieved.title, chunk.title);
  }

  #[tokio::test]
  async fn test_list_document_chunks() {
    let (_temp, db) = create_test_db().await;
    let doc_id = DocumentId::new();
    let project_id = Uuid::new_v4();

    // Create chunks with different content (therefore different content_hash)
    let c1 = DocumentChunk::new(
      doc_id,
      project_id,
      "Content for chunk 1".to_string(),
      "Test".to_string(),
      "/path/to/doc.md".to_string(),
      DocumentSource::File,
      0,
      2,
      0,
    );
    let c2 = DocumentChunk::new(
      doc_id,
      project_id,
      "Content for chunk 2".to_string(),
      "Test".to_string(),
      "/path/to/doc.md".to_string(),
      DocumentSource::File,
      1,
      2,
      50,
    );
    let vec = dummy_vector(db.vector_dim);

    db.upsert_document_chunks("/path/to/doc.md", &[c1, c2], &[vec.clone(), vec])
      .await
      .unwrap();

    let chunks = db.list_document_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 2, "should list both chunks");
  }

  #[tokio::test]
  async fn test_delete_document_chunks_by_source() {
    let (_temp, db) = create_test_db().await;
    let doc_id = DocumentId::new();
    let project_id = Uuid::new_v4();

    // Create two chunks for same document/source
    let c1 = DocumentChunk::new(
      doc_id,
      project_id,
      "Chunk 1".to_string(),
      "Doc".to_string(),
      "doc.md".to_string(),
      DocumentSource::File,
      0,
      2,
      0,
    );
    let c2 = DocumentChunk::new(
      doc_id,
      project_id,
      "Chunk 2".to_string(),
      "Doc".to_string(),
      "doc.md".to_string(),
      DocumentSource::File,
      1,
      2,
      100,
    );

    let vec = dummy_vector(db.vector_dim);
    db.upsert_document_chunks("doc.md", &[c1, c2], &[vec.clone(), vec])
      .await
      .unwrap();

    let before = db.list_document_chunks(None, None).await.unwrap();
    assert_eq!(before.len(), 2, "should have 2 chunks before delete");

    db.delete_document_chunks_by_source("doc.md").await.unwrap();

    let after = db.list_document_chunks(None, None).await.unwrap();
    assert_eq!(after.len(), 0, "should have 0 chunks after delete");
  }
}
