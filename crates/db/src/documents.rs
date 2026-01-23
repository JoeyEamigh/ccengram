use arrow_array::{
  Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use chrono::{TimeZone, Utc};
use engram_core::{DocumentChunk, DocumentId, DocumentSource};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::documents_schema;

impl ProjectDb {
  /// Add a document chunk to the database
  pub async fn add_document_chunk(&self, chunk: &DocumentChunk, vector: Option<&[f32]>) -> Result<()> {
    let table = self.documents_table().await?;

    let batch = chunk_to_batch(chunk, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], documents_schema(self.vector_dim));

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Add multiple document chunks in a batch
  pub async fn add_document_chunks(&self, chunks: &[DocumentChunk], vectors: &[Option<Vec<f32>>]) -> Result<()> {
    if chunks.is_empty() {
      return Ok(());
    }

    let table = self.documents_table().await?;

    let batches: Vec<RecordBatch> = chunks
      .iter()
      .zip(vectors.iter())
      .map(|(chunk, vec)| chunk_to_batch(chunk, vec.as_deref(), self.vector_dim))
      .collect::<Result<Vec<_>>>()?;

    let schema = documents_schema(self.vector_dim);
    let iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    table.add(Box::new(iter)).execute().await?;
    Ok(())
  }

  /// Get a document chunk by ID
  pub async fn get_document_chunk(&self, id: &DocumentId) -> Result<Option<DocumentChunk>> {
    let table = self.documents_table().await?;
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
  pub async fn search_documents(
    &self,
    query_vector: &[f32],
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(DocumentChunk, f32)>> {
    let table = self.documents_table().await?;

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

    Ok(chunks)
  }

  /// List document chunks with optional filter
  pub async fn list_document_chunks(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<DocumentChunk>> {
    let table = self.documents_table().await?;

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

  /// Delete all chunks for a document
  pub async fn delete_document(&self, document_id: &DocumentId) -> Result<()> {
    let table = self.documents_table().await?;
    table.delete(&format!("document_id = '{}'", document_id)).await?;
    Ok(())
  }

  /// Delete a single document chunk
  pub async fn delete_document_chunk(&self, id: &DocumentId) -> Result<()> {
    let table = self.documents_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Update a document chunk (delete + add)
  pub async fn update_document_chunk(&self, chunk: &DocumentChunk, vector: Option<&[f32]>) -> Result<()> {
    let table = self.documents_table().await?;

    // Delete existing
    let _ = table.delete(&format!("id = '{}'", chunk.id)).await;

    // Add new
    let batch = chunk_to_batch(chunk, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], documents_schema(self.vector_dim));
    table.add(Box::new(batches)).execute().await?;

    Ok(())
  }

  /// Get chunks for a specific document
  pub async fn get_document_chunks(&self, document_id: &DocumentId) -> Result<Vec<DocumentChunk>> {
    self
      .list_document_chunks(Some(&format!("document_id = '{}'", document_id)), None)
      .await
  }

  /// Count document chunks (uses native count_rows for efficiency)
  pub async fn count_document_chunks(&self, filter: Option<&str>) -> Result<usize> {
    let table = self.documents_table().await?;
    let count = table.count_rows(filter.map(|s| s.to_string())).await?;
    Ok(count)
  }

  /// Find document chunks by ID prefix
  ///
  /// Searches for document chunks whose ID starts with the given prefix.
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
}

/// Convert a DocumentChunk to an Arrow RecordBatch
fn chunk_to_batch(chunk: &DocumentChunk, vector: Option<&[f32]>, vector_dim: usize) -> Result<RecordBatch> {
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
  let created_at = Int64Array::from(vec![chunk.created_at.timestamp_millis()]);
  let updated_at = Int64Array::from(vec![chunk.updated_at.timestamp_millis()]);

  // Handle vector
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
    let null_vec = Float32Array::from(vec![0.0f32; vector_dim]);
    let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
    FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(null_vec), Some(vec![false].into()))?
  };

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
    created_at,
    updated_at,
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
  async fn test_add_and_get_document_chunk() {
    let (_temp, db) = create_test_db().await;
    let chunk = create_test_chunk();

    db.add_document_chunk(&chunk, None).await.unwrap();

    let retrieved = db.get_document_chunk(&chunk.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.content, chunk.content);
    assert_eq!(retrieved.title, chunk.title);
  }

  #[tokio::test]
  async fn test_list_document_chunks() {
    let (_temp, db) = create_test_db().await;

    let c1 = create_test_chunk();
    let c2 = create_test_chunk();

    db.add_document_chunk(&c1, None).await.unwrap();
    db.add_document_chunk(&c2, None).await.unwrap();

    let chunks = db.list_document_chunks(None, None).await.unwrap();
    assert_eq!(chunks.len(), 2);
  }

  #[tokio::test]
  async fn test_delete_document() {
    let (_temp, db) = create_test_db().await;
    let doc_id = DocumentId::new();
    let project_id = Uuid::new_v4();

    // Create two chunks for same document
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

    db.add_document_chunk(&c1, None).await.unwrap();
    db.add_document_chunk(&c2, None).await.unwrap();

    let before = db.list_document_chunks(None, None).await.unwrap();
    assert_eq!(before.len(), 2);

    db.delete_document(&doc_id).await.unwrap();

    let after = db.list_document_chunks(None, None).await.unwrap();
    assert_eq!(after.len(), 0);
  }
}
