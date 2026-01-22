// Document Metadata table operations
//
// Tracks document-level information for update detection:
// - Content hash for detecting changes
// - Source path/URL for lookup
// - Timestamps for tracking updates

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use chrono::{TimeZone, Utc};
use engram_core::{Document, DocumentId, DocumentSource};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::document_metadata_schema;

/// Result of checking for document updates
#[derive(Debug, Clone)]
pub struct DocumentUpdateCheck {
  /// Documents that have been modified (hash mismatch)
  pub modified: Vec<DocumentId>,
  /// Documents that are missing from source (file deleted, URL unavailable)
  pub missing: Vec<DocumentId>,
}

impl ProjectDb {
  /// Add or update document metadata
  pub async fn upsert_document_metadata(&self, doc: &Document) -> Result<()> {
    // Delete existing if present
    let table = self.document_metadata_table().await?;
    table.delete(&format!("id = '{}'", doc.id)).await.ok();

    // Insert new
    let batch = document_to_batch(doc)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], document_metadata_schema());
    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get document metadata by ID
  pub async fn get_document_metadata(&self, id: &DocumentId) -> Result<Option<Document>> {
    let table = self.document_metadata_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() || results[0].num_rows() == 0 {
      return Ok(None);
    }

    Ok(Some(batch_to_document(&results[0], 0)?))
  }

  /// Get document metadata by source path/URL
  pub async fn get_document_by_source(&self, source: &str) -> Result<Option<Document>> {
    let table = self.document_metadata_table().await?;

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

  /// List all document metadata for a project
  pub async fn list_document_metadata(&self, project_id: &str) -> Result<Vec<Document>> {
    let table = self.document_metadata_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("project_id = '{}'", project_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut docs = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        docs.push(batch_to_document(&batch, i)?);
      }
    }

    Ok(docs)
  }

  /// Check for document updates by comparing content hashes
  ///
  /// For file-based documents, reads the file and computes hash.
  /// Returns IDs of documents that have changed or are missing.
  pub async fn check_document_updates(&self, project_id: &str) -> Result<DocumentUpdateCheck> {
    let docs = self.list_document_metadata(project_id).await?;

    let mut modified = Vec::new();
    let mut missing = Vec::new();

    for doc in docs {
      match doc.source_type {
        DocumentSource::File => {
          let path = std::path::Path::new(&doc.source);
          if path.exists() {
            // Read file and compute hash
            match std::fs::read_to_string(path) {
              Ok(content) => {
                let new_hash = compute_content_hash(&content);
                if new_hash != doc.content_hash {
                  modified.push(doc.id);
                }
              }
              Err(_) => {
                // File exists but unreadable - treat as modified
                modified.push(doc.id);
              }
            }
          } else {
            missing.push(doc.id);
          }
        }
        DocumentSource::Url => {
          // URL documents need to be re-fetched to check - mark as potentially stale
          // In practice, this should be handled by explicit re-fetch
        }
        DocumentSource::Content => {
          // Direct content documents can't change unless re-provided
        }
      }
    }

    Ok(DocumentUpdateCheck { modified, missing })
  }

  /// Delete document metadata by ID
  pub async fn delete_document_metadata(&self, id: &DocumentId) -> Result<()> {
    let table = self.document_metadata_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Delete document metadata by source path
  pub async fn delete_document_by_source(&self, source: &str) -> Result<()> {
    let escaped_source = source.replace('\'', "''");
    let table = self.document_metadata_table().await?;
    table.delete(&format!("source = '{}'", escaped_source)).await?;
    Ok(())
  }

  /// Count document metadata entries
  pub async fn count_document_metadata(&self, project_id: &str) -> Result<usize> {
    let docs = self.list_document_metadata(project_id).await?;
    Ok(docs.len())
  }
}

/// Compute SHA-256 hash of content
pub fn compute_content_hash(content: &str) -> String {
  use sha2::{Digest, Sha256};
  let mut hasher = Sha256::new();
  hasher.update(content.as_bytes());
  format!("{:x}", hasher.finalize())
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
  use super::*;
  use engram_core::ProjectId;
  use std::io::Write;
  use std::path::Path;
  use tempfile::TempDir;

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
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
  async fn test_upsert_and_get_document_metadata() {
    let (_temp, db) = create_test_db().await;
    let doc = create_test_document();

    db.upsert_document_metadata(&doc).await.unwrap();

    let retrieved = db.get_document_metadata(&doc.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.title, doc.title);
    assert_eq!(retrieved.content_hash, doc.content_hash);
  }

  #[tokio::test]
  async fn test_get_document_by_source() {
    let (_temp, db) = create_test_db().await;
    let doc = create_test_document();

    db.upsert_document_metadata(&doc).await.unwrap();

    let retrieved = db.get_document_by_source(&doc.source).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, doc.id);
  }

  #[tokio::test]
  async fn test_get_document_by_source_with_quotes() {
    let (_temp, db) = create_test_db().await;
    let mut doc = create_test_document();
    doc.source = "/path/to/doc's file.md".to_string();

    db.upsert_document_metadata(&doc).await.unwrap();

    let retrieved = db.get_document_by_source(&doc.source).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, doc.id);
  }

  #[tokio::test]
  async fn test_list_document_metadata() {
    let (_temp, db) = create_test_db().await;
    let project_id = uuid::Uuid::new_v4();

    let doc1 = Document::new(
      project_id,
      "Doc 1".to_string(),
      "/path/1.md".to_string(),
      DocumentSource::File,
      "hash1".to_string(),
      100,
      1,
    );
    let doc2 = Document::new(
      project_id,
      "Doc 2".to_string(),
      "/path/2.md".to_string(),
      DocumentSource::File,
      "hash2".to_string(),
      200,
      2,
    );

    db.upsert_document_metadata(&doc1).await.unwrap();
    db.upsert_document_metadata(&doc2).await.unwrap();

    let docs = db.list_document_metadata(&project_id.to_string()).await.unwrap();
    assert_eq!(docs.len(), 2);
  }

  #[tokio::test]
  async fn test_check_document_updates_modified() {
    let (temp, db) = create_test_db().await;
    let project_id = uuid::Uuid::new_v4();

    // Create a temp file
    let file_path = temp.path().join("test.md");
    let mut file = std::fs::File::create(&file_path).unwrap();
    writeln!(file, "Original content").unwrap();
    drop(file);

    // Compute hash and store metadata
    let content = std::fs::read_to_string(&file_path).unwrap();
    let hash = compute_content_hash(&content);
    let doc = Document::new(
      project_id,
      "Test".to_string(),
      file_path.to_string_lossy().to_string(),
      DocumentSource::File,
      hash,
      content.len(),
      1,
    );
    db.upsert_document_metadata(&doc).await.unwrap();

    // Modify the file
    let mut file = std::fs::File::create(&file_path).unwrap();
    writeln!(file, "Modified content").unwrap();
    drop(file);

    // Check for updates
    let updates = db.check_document_updates(&project_id.to_string()).await.unwrap();
    assert_eq!(updates.modified.len(), 1);
    assert_eq!(updates.modified[0], doc.id);
    assert!(updates.missing.is_empty());
  }

  #[tokio::test]
  async fn test_check_document_updates_missing() {
    let (_temp, db) = create_test_db().await;
    let project_id = uuid::Uuid::new_v4();

    // Create metadata for a non-existent file
    let doc = Document::new(
      project_id,
      "Missing".to_string(),
      "/nonexistent/path.md".to_string(),
      DocumentSource::File,
      "somehash".to_string(),
      100,
      1,
    );
    db.upsert_document_metadata(&doc).await.unwrap();

    // Check for updates
    let updates = db.check_document_updates(&project_id.to_string()).await.unwrap();
    assert!(updates.modified.is_empty());
    assert_eq!(updates.missing.len(), 1);
    assert_eq!(updates.missing[0], doc.id);
  }

  #[tokio::test]
  async fn test_delete_document_metadata() {
    let (_temp, db) = create_test_db().await;
    let doc = create_test_document();

    db.upsert_document_metadata(&doc).await.unwrap();
    db.delete_document_metadata(&doc.id).await.unwrap();

    let retrieved = db.get_document_metadata(&doc.id).await.unwrap();
    assert!(retrieved.is_none());
  }

  #[tokio::test]
  async fn test_delete_document_by_source() {
    let (_temp, db) = create_test_db().await;
    let doc = create_test_document();

    db.upsert_document_metadata(&doc).await.unwrap();
    db.delete_document_by_source(&doc.source).await.unwrap();

    let retrieved = db.get_document_by_source(&doc.source).await.unwrap();
    assert!(retrieved.is_none());
  }

  #[test]
  fn test_compute_content_hash() {
    let hash1 = compute_content_hash("hello world");
    let hash2 = compute_content_hash("hello world");
    let hash3 = compute_content_hash("different content");

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, hash3);
    // SHA-256 produces 64 hex characters
    assert_eq!(hash1.len(), 64);
  }
}
