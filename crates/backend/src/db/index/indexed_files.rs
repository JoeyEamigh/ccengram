// Indexed files operations for tracking file metadata
//
// This module provides database operations for the indexed_files table,
// which stores metadata about indexed files to enable startup scan detection of:
// - Added files (file exists on disk but not in DB)
// - Deleted files (file in DB but not on disk)
// - Modified files (mtime changed -> verify with content_hash)
// - Moved files (same content_hash, different file_path)

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array};
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::db::{
  connection::{DbError, ProjectDb, Result},
  schema::indexed_files_schema,
};

/// Metadata about an indexed file
#[derive(Debug, Clone)]
pub struct IndexedFile {
  /// Relative path from project root
  pub file_path: String,
  /// Project identifier
  pub project_id: String,
  /// File modification time (Unix timestamp in seconds)
  pub mtime: i64,
  /// SHA-256 hash of file content
  pub content_hash: String,
  /// File size in bytes
  pub file_size: u64,
  /// When the file was last indexed (Unix timestamp in milliseconds)
  pub last_indexed_at: i64,
}

impl ProjectDb {
  /// Save or update file metadata after indexing
  #[tracing::instrument(level = "trace", skip(self, file), fields(file_path = %file.file_path))]
  pub async fn save_indexed_file(&self, file: &IndexedFile) -> Result<()> {
    let table = self.indexed_files_table();

    // Delete existing entry for this file path first
    let _ = table
      .delete(&format!(
        "file_path = '{}' AND project_id = '{}'",
        escape_sql(&file.file_path),
        file.project_id
      ))
      .await;

    let batch = indexed_file_to_batch(file)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], indexed_files_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Save multiple file metadata entries in batch using merge_insert
  ///
  /// This atomically:
  /// - Updates existing entries with matching file_path + project_id
  /// - Inserts new entries that don't exist
  /// - Deletes old entries for these file paths that aren't in the new set
  #[tracing::instrument(level = "trace", skip(self, files), fields(count = files.len()))]
  pub async fn save_indexed_files_batch(&self, files: &[IndexedFile]) -> Result<()> {
    if files.is_empty() {
      return Ok(());
    }

    let table = self.indexed_files_table();
    let project_id = &files[0].project_id;

    // Build batch for all files
    let file_paths: Vec<String> = files.iter().map(|f| f.file_path.clone()).collect();
    let project_ids: Vec<String> = files.iter().map(|f| f.project_id.clone()).collect();
    let mtimes: Vec<i64> = files.iter().map(|f| f.mtime).collect();
    let content_hashes: Vec<String> = files.iter().map(|f| f.content_hash.clone()).collect();
    let file_sizes: Vec<u64> = files.iter().map(|f| f.file_size).collect();
    let last_indexed_ats: Vec<i64> = files.iter().map(|f| f.last_indexed_at).collect();

    let batch = RecordBatch::try_new(
      indexed_files_schema(),
      vec![
        Arc::new(StringArray::from(file_paths.clone())),
        Arc::new(StringArray::from(project_ids)),
        Arc::new(Int64Array::from(mtimes)),
        Arc::new(StringArray::from(content_hashes)),
        Arc::new(UInt64Array::from(file_sizes)),
        Arc::new(Int64Array::from(last_indexed_ats)),
      ],
    )?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], indexed_files_schema());

    // Build IN clause for delete filter: file_path IN ('path1', 'path2', ...)
    let paths_filter = file_paths
      .iter()
      .map(|p| format!("'{}'", escape_sql(p)))
      .collect::<Vec<_>>()
      .join(", ");

    // Use merge_insert with file_path + project_id as the key
    let mut builder = table.merge_insert(&["file_path", "project_id"]);
    builder
      .when_matched_update_all(None)
      .when_not_matched_insert_all()
      .when_not_matched_by_source_delete(Some(format!(
        "project_id = '{}' AND file_path IN ({})",
        project_id, paths_filter
      )));
    builder.execute(Box::new(batches)).await?;

    Ok(())
  }

  /// Get metadata for a specific file
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_indexed_file(&self, project_id: &str, file_path: &str) -> Result<Option<IndexedFile>> {
    let table = self.indexed_files_table();

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!(
        "file_path = '{}' AND project_id = '{}'",
        escape_sql(file_path),
        project_id
      ))
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

    Ok(Some(batch_to_indexed_file(batch, 0)?))
  }

  /// List all indexed files for a project
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn list_indexed_files(&self, project_id: &str) -> Result<Vec<IndexedFile>> {
    let table = self.indexed_files_table();

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("project_id = '{}'", project_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut files = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        files.push(batch_to_indexed_file(&batch, i)?);
      }
    }

    Ok(files)
  }

  /// Delete metadata for a specific file
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn delete_indexed_file(&self, project_id: &str, file_path: &str) -> Result<()> {
    let table = self.indexed_files_table();
    table
      .delete(&format!(
        "file_path = '{}' AND project_id = '{}'",
        escape_sql(file_path),
        project_id
      ))
      .await?;
    Ok(())
  }

  /// Update file path (for rename operations)
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn rename_indexed_file(&self, project_id: &str, from: &str, to: &str) -> Result<()> {
    // Get existing entry
    if let Some(mut file) = self.get_indexed_file(project_id, from).await? {
      // Delete old entry
      self.delete_indexed_file(project_id, from).await?;

      // Save with new path
      file.file_path = to.to_string();
      file.last_indexed_at = Utc::now().timestamp_millis();
      self.save_indexed_file(&file).await?;
    }
    Ok(())
  }

  /// Get count of indexed files for a project
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn count_indexed_files(&self, project_id: &str) -> Result<usize> {
    let table = self.indexed_files_table();
    let count = table.count_rows(Some(format!("project_id = '{}'", project_id))).await?;
    Ok(count)
  }

  /// Check if any files have been indexed for this project
  /// (Used to determine if startup scan should run)
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn has_indexed_files(&self, project_id: &str) -> Result<bool> {
    let count = self.count_indexed_files(project_id).await?;
    Ok(count > 0)
  }

  /// Check if this project was manually indexed via CLI
  ///
  /// A project is considered "manually indexed" if the indexed_files table
  /// has entries for it. This flag is used to determine whether:
  /// - Startup scan should run (only for previously indexed projects)
  /// - Watcher should auto-start (only for previously indexed projects)
  ///
  /// This prevents auto-scanning every directory that Claude Code touches.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn is_manually_indexed(&self, project_id: &str) -> Result<bool> {
    self.has_indexed_files(project_id).await
  }
}

/// Escape single quotes in SQL strings
fn escape_sql(s: &str) -> String {
  s.replace('\'', "''")
}

/// Convert an IndexedFile to an Arrow RecordBatch
fn indexed_file_to_batch(file: &IndexedFile) -> Result<RecordBatch> {
  let file_path = StringArray::from(vec![file.file_path.clone()]);
  let project_id = StringArray::from(vec![file.project_id.clone()]);
  let mtime = Int64Array::from(vec![file.mtime]);
  let content_hash = StringArray::from(vec![file.content_hash.clone()]);
  let file_size = UInt64Array::from(vec![file.file_size]);
  let last_indexed_at = Int64Array::from(vec![file.last_indexed_at]);

  let batch = RecordBatch::try_new(
    indexed_files_schema(),
    vec![
      Arc::new(file_path),
      Arc::new(project_id),
      Arc::new(mtime),
      Arc::new(content_hash),
      Arc::new(file_size),
      Arc::new(last_indexed_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to an IndexedFile
fn batch_to_indexed_file(batch: &RecordBatch, row: usize) -> Result<IndexedFile> {
  let file_path = batch
    .column_by_name("file_path")
    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
    .map(|a| a.value(row).to_string())
    .ok_or_else(|| DbError::NotFound("file_path column".to_string()))?;

  let project_id = batch
    .column_by_name("project_id")
    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
    .map(|a| a.value(row).to_string())
    .ok_or_else(|| DbError::NotFound("project_id column".to_string()))?;

  let mtime = batch
    .column_by_name("mtime")
    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
    .map(|a| a.value(row))
    .ok_or_else(|| DbError::NotFound("mtime column".to_string()))?;

  let content_hash = batch
    .column_by_name("content_hash")
    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
    .map(|a| a.value(row).to_string())
    .ok_or_else(|| DbError::NotFound("content_hash column".to_string()))?;

  let file_size = batch
    .column_by_name("file_size")
    .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
    .map(|a| a.value(row))
    .ok_or_else(|| DbError::NotFound("file_size column".to_string()))?;

  let last_indexed_at = batch
    .column_by_name("last_indexed_at")
    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
    .map(|a| a.value(row))
    .ok_or_else(|| DbError::NotFound("last_indexed_at column".to_string()))?;

  Ok(IndexedFile {
    file_path,
    project_id,
    mtime,
    content_hash,
    file_size,
    last_indexed_at,
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

  #[tokio::test]
  async fn test_save_and_get_indexed_file() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project";

    let file = IndexedFile {
      file_path: "src/main.rs".to_string(),
      project_id: project_id.to_string(),
      mtime: 1234567890,
      content_hash: "abc123".to_string(),
      file_size: 1024,
      last_indexed_at: Utc::now().timestamp_millis(),
    };

    db.save_indexed_file(&file).await.unwrap();

    let retrieved = db.get_indexed_file(project_id, "src/main.rs").await.unwrap();
    assert!(retrieved.is_some(), "File should be retrievable after save");
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.file_path, "src/main.rs");
    assert_eq!(retrieved.content_hash, "abc123");
    assert_eq!(retrieved.file_size, 1024);
  }

  #[tokio::test]
  async fn test_save_indexed_files_batch() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project";
    let now = Utc::now().timestamp_millis();

    let files = vec![
      IndexedFile {
        file_path: "src/a.rs".to_string(),
        project_id: project_id.to_string(),
        mtime: 1000,
        content_hash: "hash_a".to_string(),
        file_size: 100,
        last_indexed_at: now,
      },
      IndexedFile {
        file_path: "src/b.rs".to_string(),
        project_id: project_id.to_string(),
        mtime: 2000,
        content_hash: "hash_b".to_string(),
        file_size: 200,
        last_indexed_at: now,
      },
    ];

    db.save_indexed_files_batch(&files).await.unwrap();

    let all = db.list_indexed_files(project_id).await.unwrap();
    assert_eq!(all.len(), 2, "Should have saved 2 files in batch");
  }

  #[tokio::test]
  async fn test_delete_indexed_file() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project";

    let file = IndexedFile {
      file_path: "to_delete.rs".to_string(),
      project_id: project_id.to_string(),
      mtime: 1000,
      content_hash: "hash".to_string(),
      file_size: 50,
      last_indexed_at: Utc::now().timestamp_millis(),
    };

    db.save_indexed_file(&file).await.unwrap();
    assert!(
      db.get_indexed_file(project_id, "to_delete.rs").await.unwrap().is_some(),
      "File should exist before deletion"
    );

    db.delete_indexed_file(project_id, "to_delete.rs").await.unwrap();
    assert!(
      db.get_indexed_file(project_id, "to_delete.rs").await.unwrap().is_none(),
      "File should not exist after deletion"
    );
  }

  #[tokio::test]
  async fn test_rename_indexed_file() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project";

    let file = IndexedFile {
      file_path: "old_name.rs".to_string(),
      project_id: project_id.to_string(),
      mtime: 1000,
      content_hash: "hash".to_string(),
      file_size: 100,
      last_indexed_at: Utc::now().timestamp_millis(),
    };

    db.save_indexed_file(&file).await.unwrap();
    db.rename_indexed_file(project_id, "old_name.rs", "new_name.rs")
      .await
      .unwrap();

    assert!(
      db.get_indexed_file(project_id, "old_name.rs").await.unwrap().is_none(),
      "Old path should not exist after rename"
    );
    let renamed = db.get_indexed_file(project_id, "new_name.rs").await.unwrap();
    assert!(renamed.is_some(), "New path should exist after rename");
    assert_eq!(
      renamed.unwrap().content_hash,
      "hash",
      "Content hash should be preserved after rename"
    );
  }

  #[tokio::test]
  async fn test_count_and_has_indexed_files() {
    let (_temp, db) = create_test_db().await;
    let project_id = "test_project";

    assert!(
      !db.has_indexed_files(project_id).await.unwrap(),
      "Should have no files initially"
    );
    assert_eq!(db.count_indexed_files(project_id).await.unwrap(), 0);

    let file = IndexedFile {
      file_path: "file.rs".to_string(),
      project_id: project_id.to_string(),
      mtime: 1000,
      content_hash: "hash".to_string(),
      file_size: 100,
      last_indexed_at: Utc::now().timestamp_millis(),
    };
    db.save_indexed_file(&file).await.unwrap();

    assert!(
      db.has_indexed_files(project_id).await.unwrap(),
      "Should have files after save"
    );
    assert_eq!(db.count_indexed_files(project_id).await.unwrap(), 1);
  }
}
