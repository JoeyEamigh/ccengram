// Index checkpoint operations for resuming interrupted indexing
//
// Tracks progress during code/document indexing:
// - Processed files: Successfully indexed
// - Pending files: Still need to be processed
// - Error count: Files that failed during indexing
// - Gitignore hash: Detects if rules changed since checkpoint

use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use chrono::{DateTime, TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::index_checkpoints_schema;

/// Type of indexing checkpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointType {
  /// Code file indexing
  Code,
  /// Document ingestion
  Document,
}

impl CheckpointType {
  pub fn as_str(&self) -> &'static str {
    match self {
      CheckpointType::Code => "code",
      CheckpointType::Document => "document",
    }
  }
}

impl std::str::FromStr for CheckpointType {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "code" => Ok(CheckpointType::Code),
      "document" => Ok(CheckpointType::Document),
      _ => Err(format!("Unknown checkpoint type: {}", s)),
    }
  }
}

/// An indexing checkpoint record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexCheckpoint {
  pub id: String,
  pub project_id: String,
  pub checkpoint_type: CheckpointType,
  pub processed_files: HashSet<String>,
  pub pending_files: Vec<String>,
  pub total_files: u32,
  pub processed_count: u32,
  pub error_count: u32,
  pub gitignore_hash: Option<String>,
  pub started_at: DateTime<Utc>,
  pub updated_at: DateTime<Utc>,
  pub is_complete: bool,
}

impl IndexCheckpoint {
  /// Create a new checkpoint for indexing
  pub fn new(project_id: &str, checkpoint_type: CheckpointType, pending_files: Vec<String>) -> Self {
    let now = Utc::now();
    let total = pending_files.len() as u32;
    let id = format!("{}:{}", project_id, checkpoint_type.as_str());

    Self {
      id,
      project_id: project_id.to_string(),
      checkpoint_type,
      processed_files: HashSet::new(),
      pending_files,
      total_files: total,
      processed_count: 0,
      error_count: 0,
      gitignore_hash: None,
      started_at: now,
      updated_at: now,
      is_complete: false,
    }
  }

  /// Mark a file as successfully processed
  pub fn mark_processed(&mut self, file_path: &str) {
    self.processed_files.insert(file_path.to_string());
    self.pending_files.retain(|f| f != file_path);
    self.processed_count += 1;
    self.updated_at = Utc::now();
  }

  /// Mark a file as having an error
  pub fn mark_error(&mut self, file_path: &str) {
    self.pending_files.retain(|f| f != file_path);
    self.error_count += 1;
    self.updated_at = Utc::now();
  }

  /// Mark indexing as complete
  pub fn mark_complete(&mut self) {
    self.is_complete = true;
    self.updated_at = Utc::now();
  }

  /// Check if there are pending files
  pub fn has_pending(&self) -> bool {
    !self.pending_files.is_empty()
  }

  /// Get progress percentage (0-100)
  pub fn progress_percent(&self) -> f32 {
    if self.total_files == 0 {
      return 100.0;
    }
    ((self.processed_count + self.error_count) as f32 / self.total_files as f32) * 100.0
  }
}

impl ProjectDb {
  /// Save or update an indexing checkpoint
  pub async fn save_checkpoint(&self, checkpoint: &IndexCheckpoint) -> Result<()> {
    let table = self.index_checkpoints_table().await?;

    let batch = checkpoint_to_batch(checkpoint)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], index_checkpoints_schema());

    // Delete existing checkpoint with same ID first
    let _ = table.delete(&format!("id = '{}'", checkpoint.id)).await;

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get a checkpoint by project and type
  pub async fn get_checkpoint(
    &self,
    project_id: &str,
    checkpoint_type: CheckpointType,
  ) -> Result<Option<IndexCheckpoint>> {
    let table = self.index_checkpoints_table().await?;
    let id = format!("{}:{}", project_id, checkpoint_type.as_str());

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id))
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

    Ok(Some(batch_to_checkpoint(batch, 0)?))
  }

  /// Delete a checkpoint (called when indexing completes or is reset)
  pub async fn clear_checkpoint(&self, project_id: &str, checkpoint_type: CheckpointType) -> Result<()> {
    let table = self.index_checkpoints_table().await?;
    let id = format!("{}:{}", project_id, checkpoint_type.as_str());
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// List all checkpoints for a project
  pub async fn list_checkpoints(&self, project_id: &str) -> Result<Vec<IndexCheckpoint>> {
    let table = self.index_checkpoints_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("project_id = '{}'", project_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut checkpoints = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        checkpoints.push(batch_to_checkpoint(&batch, i)?);
      }
    }

    Ok(checkpoints)
  }
}

/// Convert an IndexCheckpoint to an Arrow RecordBatch
fn checkpoint_to_batch(checkpoint: &IndexCheckpoint) -> Result<RecordBatch> {
  let id = StringArray::from(vec![checkpoint.id.clone()]);
  let project_id = StringArray::from(vec![checkpoint.project_id.to_string()]);
  let checkpoint_type = StringArray::from(vec![checkpoint.checkpoint_type.as_str().to_string()]);
  let processed_files = StringArray::from(vec![serde_json::to_string(
    &checkpoint.processed_files.iter().collect::<Vec<_>>(),
  )?]);
  let pending_files = StringArray::from(vec![serde_json::to_string(&checkpoint.pending_files)?]);
  let total_files = UInt32Array::from(vec![checkpoint.total_files]);
  let processed_count = UInt32Array::from(vec![checkpoint.processed_count]);
  let error_count = UInt32Array::from(vec![checkpoint.error_count]);
  let gitignore_hash = StringArray::from(vec![checkpoint.gitignore_hash.clone()]);
  let started_at = Int64Array::from(vec![checkpoint.started_at.timestamp_millis()]);
  let updated_at = Int64Array::from(vec![checkpoint.updated_at.timestamp_millis()]);
  let is_complete = BooleanArray::from(vec![checkpoint.is_complete]);

  let batch = RecordBatch::try_new(
    index_checkpoints_schema(),
    vec![
      Arc::new(id),
      Arc::new(project_id),
      Arc::new(checkpoint_type),
      Arc::new(processed_files),
      Arc::new(pending_files),
      Arc::new(total_files),
      Arc::new(processed_count),
      Arc::new(error_count),
      Arc::new(gitignore_hash),
      Arc::new(started_at),
      Arc::new(updated_at),
      Arc::new(is_complete),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to an IndexCheckpoint
fn batch_to_checkpoint(batch: &RecordBatch, row: usize) -> Result<IndexCheckpoint> {
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

  let get_bool = |name: &str| -> Result<bool> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let id = get_string("id")?;
  let project_id_str = get_string("project_id")?;
  let checkpoint_type_str = get_string("checkpoint_type")?;
  let processed_files_json = get_string("processed_files")?;
  let pending_files_json = get_string("pending_files")?;

  let processed_files: Vec<String> = serde_json::from_str(&processed_files_json)?;
  let pending_files: Vec<String> = serde_json::from_str(&pending_files_json)?;

  let checkpoint_type = checkpoint_type_str
    .parse::<CheckpointType>()
    .map_err(DbError::NotFound)?;

  let started_at = Utc
    .timestamp_millis_opt(get_i64("started_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid started_at timestamp".into()))?;
  let updated_at = Utc
    .timestamp_millis_opt(get_i64("updated_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid updated_at timestamp".into()))?;

  Ok(IndexCheckpoint {
    id,
    project_id: project_id_str,
    checkpoint_type,
    processed_files: processed_files.into_iter().collect(),
    pending_files,
    total_files: get_u32("total_files")?,
    processed_count: get_u32("processed_count")?,
    error_count: get_u32("error_count")?,
    gitignore_hash: get_optional_string("gitignore_hash"),
    started_at,
    updated_at,
    is_complete: get_bool("is_complete")?,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ProjectDb;
  use engram_core::ProjectId;
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

  #[tokio::test]
  async fn test_save_and_get_checkpoint() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project_abc123";

    let files = vec![
      "src/main.rs".to_string(),
      "src/lib.rs".to_string(),
      "Cargo.toml".to_string(),
    ];
    let checkpoint = IndexCheckpoint::new(project_id, CheckpointType::Code, files);

    db.save_checkpoint(&checkpoint).await.unwrap();

    let retrieved = db.get_checkpoint(project_id, CheckpointType::Code).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.project_id, project_id);
    assert_eq!(retrieved.total_files, 3);
    assert_eq!(retrieved.pending_files.len(), 3);
  }

  #[tokio::test]
  async fn test_checkpoint_progress() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project_progress";

    let files = vec![
      "a.rs".to_string(),
      "b.rs".to_string(),
      "c.rs".to_string(),
      "d.rs".to_string(),
    ];
    let mut checkpoint = IndexCheckpoint::new(project_id, CheckpointType::Code, files);

    // Mark 2 processed, 1 error
    checkpoint.mark_processed("a.rs");
    checkpoint.mark_processed("b.rs");
    checkpoint.mark_error("c.rs");

    db.save_checkpoint(&checkpoint).await.unwrap();

    let retrieved = db
      .get_checkpoint(project_id, CheckpointType::Code)
      .await
      .unwrap()
      .unwrap();
    assert_eq!(retrieved.processed_count, 2);
    assert_eq!(retrieved.error_count, 1);
    assert_eq!(retrieved.pending_files.len(), 1);
    assert_eq!(retrieved.pending_files[0], "d.rs");
    assert!((retrieved.progress_percent() - 75.0).abs() < 0.01);
  }

  #[tokio::test]
  async fn test_clear_checkpoint() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project_clear";

    let checkpoint = IndexCheckpoint::new(project_id, CheckpointType::Code, vec!["test.rs".to_string()]);
    db.save_checkpoint(&checkpoint).await.unwrap();

    // Verify exists
    assert!(
      db.get_checkpoint(project_id, CheckpointType::Code)
        .await
        .unwrap()
        .is_some()
    );

    // Clear
    db.clear_checkpoint(project_id, CheckpointType::Code).await.unwrap();

    // Verify gone
    assert!(
      db.get_checkpoint(project_id, CheckpointType::Code)
        .await
        .unwrap()
        .is_none()
    );
  }

  #[tokio::test]
  async fn test_checkpoint_types() {
    assert_eq!("code".parse::<CheckpointType>().unwrap(), CheckpointType::Code);
    assert_eq!("document".parse::<CheckpointType>().unwrap(), CheckpointType::Document);
    assert!("invalid".parse::<CheckpointType>().is_err());
  }
}
