//! Database schema migration system for LanceDB
//!
//! Provides forward-only schema evolution with version tracking.
//! LanceDB supports adding columns to existing tables via schema evolution.

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::*;
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Current schema version - increment when schema changes
pub const CURRENT_SCHEMA_VERSION: i64 = 1;

/// Schema for the _migrations metadata table
fn migrations_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("version", DataType::Int64, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("applied_at", DataType::Int64, false), // Unix timestamp ms
  ]))
}

/// A migration definition
#[derive(Debug, Clone)]
pub struct Migration {
  pub version: i64,
  pub name: &'static str,
  pub description: &'static str,
}

/// All migrations in order
pub const MIGRATIONS: &[Migration] = &[Migration {
  version: 1,
  name: "initial_schema",
  description: "Initial schema with all core tables",
}];

/// Migration record from the database
#[derive(Debug, Clone)]
pub struct MigrationRecord {
  pub version: i64,
  pub name: String,
  pub applied_at: i64,
}

impl ProjectDb {
  /// Run all pending migrations
  pub async fn run_migrations(&self) -> Result<Vec<MigrationRecord>> {
    // Ensure _migrations table exists
    self.ensure_migrations_table().await?;

    // Get current version
    let current_version = self.get_current_version().await?;
    info!(
      "Current schema version: {}, target: {}",
      current_version, CURRENT_SCHEMA_VERSION
    );

    // Find and run pending migrations
    let pending: Vec<_> = MIGRATIONS.iter().filter(|m| m.version > current_version).collect();

    if pending.is_empty() {
      debug!("No pending migrations");
      return Ok(Vec::new());
    }

    let mut applied = Vec::new();

    for migration in pending {
      info!(
        "Running migration {}: {} - {}",
        migration.version, migration.name, migration.description
      );

      // Apply the migration
      self.apply_migration(migration).await?;

      // Record it
      let record = self.record_migration(migration).await?;
      applied.push(record);

      info!("Migration {} applied successfully", migration.version);
    }

    Ok(applied)
  }

  /// Ensure the _migrations table exists
  async fn ensure_migrations_table(&self) -> Result<()> {
    let table_names = self.connection.table_names().execute().await?;

    if !table_names.contains(&"_migrations".to_string()) {
      debug!("Creating _migrations table");
      self
        .connection
        .create_empty_table("_migrations", migrations_schema())
        .execute()
        .await?;
    }

    Ok(())
  }

  /// Get the current schema version (0 if no migrations applied)
  pub async fn get_current_version(&self) -> Result<i64> {
    let table = match self.connection.open_table("_migrations").execute().await {
      Ok(t) => t,
      Err(_) => return Ok(0),
    };

    let results: Vec<RecordBatch> = table.query().execute().await?.try_collect().await?;

    if results.is_empty() {
      return Ok(0);
    }

    let mut max_version = 0i64;
    for batch in results {
      if batch.num_rows() == 0 {
        continue;
      }

      let versions = batch
        .column_by_name("version")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| DbError::NotFound("version column".to_string()))?;

      for i in 0..versions.len() {
        let v = versions.value(i);
        if v > max_version {
          max_version = v;
        }
      }
    }

    Ok(max_version)
  }

  /// Apply a specific migration
  async fn apply_migration(&self, migration: &Migration) -> Result<()> {
    match migration.version {
      1 => self.migrate_v1_initial_schema().await,
      v => {
        warn!("Unknown migration version: {}", v);
        Ok(())
      }
    }
  }

  /// Record a migration as applied
  async fn record_migration(&self, migration: &Migration) -> Result<MigrationRecord> {
    let table = self.connection.open_table("_migrations").execute().await?;

    let applied_at = Utc::now().timestamp_millis();

    let versions = Int64Array::from(vec![migration.version]);
    let names = StringArray::from(vec![migration.name]);
    let applied_ats = Int64Array::from(vec![applied_at]);

    let batch = RecordBatch::try_new(
      migrations_schema(),
      vec![Arc::new(versions), Arc::new(names), Arc::new(applied_ats)],
    )?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], migrations_schema());
    table.add(Box::new(batches)).execute().await?;

    Ok(MigrationRecord {
      version: migration.version,
      name: migration.name.to_string(),
      applied_at,
    })
  }

  /// Get all applied migrations
  pub async fn get_migration_history(&self) -> Result<Vec<MigrationRecord>> {
    self.ensure_migrations_table().await?;

    let table = self.connection.open_table("_migrations").execute().await?;
    let results: Vec<RecordBatch> = table.query().execute().await?.try_collect().await?;

    let mut records = Vec::new();

    for batch in results {
      if batch.num_rows() == 0 {
        continue;
      }

      let versions = batch
        .column_by_name("version")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| DbError::NotFound("version column".to_string()))?;

      let names = batch
        .column_by_name("name")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| DbError::NotFound("name column".to_string()))?;

      let applied_ats = batch
        .column_by_name("applied_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| DbError::NotFound("applied_at column".to_string()))?;

      for i in 0..batch.num_rows() {
        records.push(MigrationRecord {
          version: versions.value(i),
          name: names.value(i).to_string(),
          applied_at: applied_ats.value(i),
        });
      }
    }

    // Sort by version
    records.sort_by_key(|r| r.version);

    Ok(records)
  }

  /// Migration v1: Initial schema
  /// Creates all core tables with their initial schema
  async fn migrate_v1_initial_schema(&self) -> Result<()> {
    let table_names = self.connection.table_names().execute().await?;

    // Create memories table
    if !table_names.contains(&"memories".to_string()) {
      debug!("Migration v1: Creating memories table");
      self
        .connection
        .create_empty_table("memories", memories_schema(self.vector_dim))
        .execute()
        .await?;
    }

    // Create code_chunks table
    if !table_names.contains(&"code_chunks".to_string()) {
      debug!("Migration v1: Creating code_chunks table");
      self
        .connection
        .create_empty_table("code_chunks", code_chunks_schema(self.vector_dim))
        .execute()
        .await?;
    }

    // Create sessions table
    if !table_names.contains(&"sessions".to_string()) {
      debug!("Migration v1: Creating sessions table");
      self
        .connection
        .create_empty_table("sessions", sessions_schema())
        .execute()
        .await?;
    }

    // Create events table
    if !table_names.contains(&"events".to_string()) {
      debug!("Migration v1: Creating events table");
      self
        .connection
        .create_empty_table("events", events_schema())
        .execute()
        .await?;
    }

    // Create documents table
    if !table_names.contains(&"documents".to_string()) {
      debug!("Migration v1: Creating documents table");
      self
        .connection
        .create_empty_table("documents", documents_schema(self.vector_dim))
        .execute()
        .await?;
    }

    // Create session_memories table
    if !table_names.contains(&"session_memories".to_string()) {
      debug!("Migration v1: Creating session_memories table");
      self
        .connection
        .create_empty_table("session_memories", session_memories_schema())
        .execute()
        .await?;
    }

    // Create memory_relationships table
    if !table_names.contains(&"memory_relationships".to_string()) {
      debug!("Migration v1: Creating memory_relationships table");
      self
        .connection
        .create_empty_table("memory_relationships", memory_relationships_schema())
        .execute()
        .await?;
    }

    // Create document_metadata table
    if !table_names.contains(&"document_metadata".to_string()) {
      debug!("Migration v1: Creating document_metadata table");
      self
        .connection
        .create_empty_table("document_metadata", document_metadata_schema())
        .execute()
        .await?;
    }

    // Create entities table
    if !table_names.contains(&"entities".to_string()) {
      debug!("Migration v1: Creating entities table");
      self
        .connection
        .create_empty_table("entities", entities_schema())
        .execute()
        .await?;
    }

    // Create memory_entities table
    if !table_names.contains(&"memory_entities".to_string()) {
      debug!("Migration v1: Creating memory_entities table");
      self
        .connection
        .create_empty_table("memory_entities", memory_entities_schema())
        .execute()
        .await?;
    }

    // Create index_checkpoints table
    if !table_names.contains(&"index_checkpoints".to_string()) {
      debug!("Migration v1: Creating index_checkpoints table");
      self
        .connection
        .create_empty_table("index_checkpoints", index_checkpoints_schema())
        .execute()
        .await?;
    }

    // Create segment_accumulators table
    if !table_names.contains(&"segment_accumulators".to_string()) {
      debug!("Migration v1: Creating segment_accumulators table");
      self
        .connection
        .create_empty_table("segment_accumulators", segment_accumulators_schema())
        .execute()
        .await?;
    }

    // Create extraction_segments table
    if !table_names.contains(&"extraction_segments".to_string()) {
      debug!("Migration v1: Creating extraction_segments table");
      self
        .connection
        .create_empty_table("extraction_segments", extraction_segments_schema())
        .execute()
        .await?;
    }

    Ok(())
  }

  /// Check if migrations need to be run
  pub async fn needs_migration(&self) -> Result<bool> {
    let current = self.get_current_version().await?;
    Ok(current < CURRENT_SCHEMA_VERSION)
  }

  /// Get a summary of pending migrations
  pub async fn pending_migrations(&self) -> Result<Vec<&'static Migration>> {
    let current = self.get_current_version().await?;
    Ok(MIGRATIONS.iter().filter(|m| m.version > current).collect())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::connection::ProjectDb;
  use engram_core::ProjectId;
  use std::path::Path;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_migrations_run_on_new_db() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));

    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    // Run migrations
    let applied = db.run_migrations().await.unwrap();
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].version, 1);
    assert_eq!(applied[0].name, "initial_schema");

    // Check version
    let version = db.get_current_version().await.unwrap();
    assert_eq!(version, 1);
  }

  #[tokio::test]
  async fn test_migrations_idempotent() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));

    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    // Run migrations twice
    let applied1 = db.run_migrations().await.unwrap();
    let applied2 = db.run_migrations().await.unwrap();

    // Second run should not apply anything
    assert_eq!(applied1.len(), 1);
    assert_eq!(applied2.len(), 0);
  }

  #[tokio::test]
  async fn test_migration_history() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));

    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    // Run migrations
    db.run_migrations().await.unwrap();

    // Check history
    let history = db.get_migration_history().await.unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].version, 1);
    assert!(history[0].applied_at > 0);
  }

  #[tokio::test]
  async fn test_needs_migration() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));

    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    // Before migrations, needs_migration should be true
    // Note: ensure_tables is called in open_at_path, so tables exist but migrations haven't been recorded
    assert!(db.needs_migration().await.unwrap());

    // After migrations, needs_migration should be false
    db.run_migrations().await.unwrap();
    assert!(!db.needs_migration().await.unwrap());
  }

  #[tokio::test]
  async fn test_pending_migrations() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));

    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    // Check pending before running
    let pending = db.pending_migrations().await.unwrap();
    assert_eq!(pending.len(), 1);

    // Run migrations
    db.run_migrations().await.unwrap();

    // Check pending after running
    let pending = db.pending_migrations().await.unwrap();
    assert_eq!(pending.len(), 0);
  }

  #[test]
  fn test_migrations_have_unique_versions() {
    let mut versions: Vec<i64> = MIGRATIONS.iter().map(|m| m.version).collect();
    let original_len = versions.len();
    versions.sort();
    versions.dedup();
    assert_eq!(versions.len(), original_len, "Migration versions must be unique");
  }

  #[test]
  fn test_migrations_are_ordered() {
    for i in 1..MIGRATIONS.len() {
      assert!(
        MIGRATIONS[i].version > MIGRATIONS[i - 1].version,
        "Migrations must be in ascending order"
      );
    }
  }

  #[test]
  fn test_migrations_have_names() {
    for m in MIGRATIONS {
      assert!(!m.name.is_empty(), "Migration {} must have a name", m.version);
      assert!(
        !m.description.is_empty(),
        "Migration {} must have a description",
        m.version
      );
    }
  }
}
