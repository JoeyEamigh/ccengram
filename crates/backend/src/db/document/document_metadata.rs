// Document Metadata table operations
//
// Tracks document-level information for update detection:
// - Content hash for detecting changes
// - Source path/URL for lookup
// - Timestamps for tracking updates

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::{
  db::{
    connection::{DbError, ProjectDb, Result},
    schema::document_metadata_schema,
  },
  domain::document::{Document, DocumentSource},
};

impl ProjectDb {
  /// Add or update document metadata
  #[tracing::instrument(level = "trace", skip(self, doc), fields(id = %doc.id))]
  pub async fn upsert_document_metadata(&self, doc: &Document) -> Result<()> {
    // Delete existing if present
    let table = self.document_metadata_table();
    table.delete(&format!("id = '{}'", doc.id)).await.ok();

    // Insert new
    let batch = document_to_batch(doc)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], document_metadata_schema());
    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Batch add or update document metadata
  ///
  /// More efficient than calling `upsert_document_metadata` per document.
  /// Uses a single delete with IN clause and a single batch insert.
  #[tracing::instrument(level = "trace", skip(self, docs), fields(count = docs.len()))]
  pub async fn upsert_document_metadata_batch(&self, docs: &[Document]) -> Result<()> {
    if docs.is_empty() {
      return Ok(());
    }

    let table = self.document_metadata_table();

    // Delete all existing docs with one query using IN clause
    let ids_filter = docs
      .iter()
      .map(|d| format!("'{}'", d.id))
      .collect::<Vec<_>>()
      .join(", ");
    table.delete(&format!("id IN ({})", ids_filter)).await.ok();

    // Insert all docs in one batch
    let batch = documents_to_batch(docs)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], document_metadata_schema());
    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get document metadata by source path/URL
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_document_by_source(&self, source: &str) -> Result<Option<Document>> {
    let table = self.document_metadata_table();

    // Escape single quotes in source path
    let escaped_source = source.replace('\'', "''");

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("source = '{}'", escaped_source))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() || results[0].num_rows() == 0 {
      return Ok(None);
    }

    Ok(Some(batch_to_document(&results[0], 0)?))
  }

  /// Delete document metadata by source path
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn delete_document_by_source(&self, source: &str) -> Result<()> {
    let escaped_source = source.replace('\'', "''");
    let table = self.document_metadata_table();
    table.delete(&format!("source = '{}'", escaped_source)).await?;
    Ok(())
  }
}

/// Convert multiple Documents to a single Arrow RecordBatch (true batch insert)
fn documents_to_batch(docs: &[Document]) -> Result<RecordBatch> {
  let ids: Vec<String> = docs.iter().map(|d| d.id.to_string()).collect();
  let project_ids: Vec<String> = docs.iter().map(|d| d.project_id.to_string()).collect();
  let titles: Vec<&str> = docs.iter().map(|d| d.title.as_str()).collect();
  let sources: Vec<&str> = docs.iter().map(|d| d.source.as_str()).collect();
  let source_types: Vec<String> = docs.iter().map(|d| d.source_type.as_str().to_string()).collect();
  let content_hashes: Vec<&str> = docs.iter().map(|d| d.content_hash.as_str()).collect();
  let char_counts: Vec<u32> = docs.iter().map(|d| d.char_count as u32).collect();
  let chunk_counts: Vec<u32> = docs.iter().map(|d| d.chunk_count as u32).collect();
  let full_contents: Vec<Option<&str>> = docs.iter().map(|d| d.full_content.as_deref()).collect();
  let created_ats: Vec<i64> = docs.iter().map(|d| d.created_at.timestamp_millis()).collect();
  let updated_ats: Vec<i64> = docs.iter().map(|d| d.updated_at.timestamp_millis()).collect();

  let batch = RecordBatch::try_new(
    document_metadata_schema(),
    vec![
      Arc::new(StringArray::from(ids)),
      Arc::new(StringArray::from(project_ids)),
      Arc::new(StringArray::from(titles)),
      Arc::new(StringArray::from(sources)),
      Arc::new(StringArray::from(source_types)),
      Arc::new(StringArray::from(content_hashes)),
      Arc::new(UInt32Array::from(char_counts)),
      Arc::new(UInt32Array::from(chunk_counts)),
      Arc::new(StringArray::from(full_contents)),
      Arc::new(Int64Array::from(created_ats)),
      Arc::new(Int64Array::from(updated_ats)),
    ],
  )?;

  Ok(batch)
}

/// Convert a Document to an Arrow RecordBatch
fn document_to_batch(doc: &Document) -> Result<RecordBatch> {
  let id = StringArray::from(vec![doc.id.to_string()]);
  let project_id = StringArray::from(vec![doc.project_id.to_string()]);
  let title = StringArray::from(vec![doc.title.clone()]);
  let source = StringArray::from(vec![doc.source.clone()]);
  let source_type = StringArray::from(vec![doc.source_type.as_str().to_string()]);
  let content_hash = StringArray::from(vec![doc.content_hash.clone()]);
  let char_count = UInt32Array::from(vec![doc.char_count as u32]);
  let chunk_count = UInt32Array::from(vec![doc.chunk_count as u32]);
  let full_content = StringArray::from(vec![doc.full_content.clone()]);
  let created_at = Int64Array::from(vec![doc.created_at.timestamp_millis()]);
  let updated_at = Int64Array::from(vec![doc.updated_at.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    document_metadata_schema(),
    vec![
      Arc::new(id),
      Arc::new(project_id),
      Arc::new(title),
      Arc::new(source),
      Arc::new(source_type),
      Arc::new(content_hash),
      Arc::new(char_count),
      Arc::new(chunk_count),
      Arc::new(full_content),
      Arc::new(created_at),
      Arc::new(updated_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a Document
fn batch_to_document(batch: &RecordBatch, row: usize) -> Result<Document> {
  let get_string = |name: &str| -> Result<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_optional_string = |name: &str| -> Option<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .and_then(|a| {
        if a.is_null(row) {
          None
        } else {
          Some(a.value(row).to_string())
        }
      })
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

  Ok(Document {
    id: id_str.parse().map_err(|_| DbError::NotFound("invalid id".into()))?,
    project_id: uuid::Uuid::parse_str(&project_id_str).map_err(|_| DbError::NotFound("invalid project_id".into()))?,
    title: get_string("title")?,
    source: get_string("source")?,
    source_type,
    content_hash: get_string("content_hash")?,
    char_count: get_u32("char_count")? as usize,
    chunk_count: get_u32("chunk_count")? as usize,
    full_content: get_optional_string("full_content"),
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

  fn create_test_document() -> Document {
    Document::new(
      uuid::Uuid::new_v4(),
      "Test Document".to_string(),
      "/path/to/doc.md".to_string(),
      DocumentSource::File,
      "abc123".to_string(),
      1000,
      3,
    )
  }

  #[tokio::test]
  async fn test_get_document_by_source() {
    let (_temp, db) = create_test_db().await;
    let doc = create_test_document();

    db.upsert_document_metadata(&doc).await.unwrap();

    let retrieved = db.get_document_by_source(&doc.source).await.unwrap();
    assert!(retrieved.is_some(), "Document should be found by source");
    assert_eq!(retrieved.unwrap().id, doc.id);
  }

  #[tokio::test]
  async fn test_get_document_by_source_with_quotes() {
    let (_temp, db) = create_test_db().await;
    let mut doc = create_test_document();
    doc.source = "/path/to/doc's file.md".to_string();

    db.upsert_document_metadata(&doc).await.unwrap();

    let retrieved = db.get_document_by_source(&doc.source).await.unwrap();
    assert!(retrieved.is_some(), "Document with quotes in path should be found");
    assert_eq!(retrieved.unwrap().id, doc.id);
  }

  #[tokio::test]
  async fn test_delete_document_by_source() {
    let (_temp, db) = create_test_db().await;
    let doc = create_test_document();

    db.upsert_document_metadata(&doc).await.unwrap();
    db.delete_document_by_source(&doc.source).await.unwrap();

    let retrieved = db.get_document_by_source(&doc.source).await.unwrap();
    assert!(retrieved.is_none(), "Document should be deleted");
  }
}
