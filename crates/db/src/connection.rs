use engram_core::ProjectId;
use lancedb::index::Index;
use lancedb::table::OptimizeAction;
use lancedb::{Connection, connect};
use std::path::PathBuf;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::schema::{
  DEFAULT_VECTOR_DIM, code_chunks_schema, code_references_schema, document_metadata_schema, documents_schema,
  entities_schema, events_schema, extraction_segments_schema, index_checkpoints_schema, memories_schema,
  memory_entities_schema, memory_relationships_schema, segment_accumulators_schema, session_memories_schema,
  sessions_schema,
};

#[derive(Error, Debug)]
pub enum DbError {
  #[error("LanceDB error: {0}")]
  Lance(#[from] lancedb::Error),
  #[error("Arrow error: {0}")]
  Arrow(#[from] arrow::error::ArrowError),
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Not found: {0}")]
  NotFound(String),
  #[error("Serialization error: {0}")]
  Serialization(#[from] serde_json::Error),
  #[error("Invalid input: {0}")]
  InvalidInput(String),
  #[error("Ambiguous prefix '{prefix}' matches {count} items. Use more characters.")]
  AmbiguousPrefix { prefix: String, count: usize },
}

pub type Result<T> = std::result::Result<T, DbError>;

/// Database connection for a specific project
pub struct ProjectDb {
  pub project_id: ProjectId,
  pub path: PathBuf,
  pub connection: Connection,
  pub vector_dim: usize,
}

impl ProjectDb {
  /// Open or create a project database
  pub async fn open(project_id: ProjectId, base_path: &std::path::Path) -> Result<Self> {
    let db_path = project_id.data_dir(base_path).join("lancedb");
    Self::open_at_path(project_id, db_path, DEFAULT_VECTOR_DIM).await
  }

  /// Open database at a specific path
  pub async fn open_at_path(project_id: ProjectId, db_path: PathBuf, vector_dim: usize) -> Result<Self> {
    // Ensure directory exists
    if let Some(parent) = db_path.parent() {
      std::fs::create_dir_all(parent)?;
    }

    info!("Opening LanceDB at {:?}", db_path);
    let connection = connect(db_path.to_string_lossy().as_ref()).execute().await?;

    let db = Self {
      project_id,
      path: db_path,
      connection,
      vector_dim,
    };

    // Ensure tables exist
    db.ensure_tables().await?;

    Ok(db)
  }

  /// Ensure all required tables exist
  async fn ensure_tables(&self) -> Result<()> {
    let table_names = self.connection.table_names().execute().await?;

    if !table_names.contains(&"memories".to_string()) {
      debug!("Creating memories table");
      self
        .connection
        .create_empty_table("memories", memories_schema(self.vector_dim))
        .execute()
        .await?;
    }

    if !table_names.contains(&"code_chunks".to_string()) {
      debug!("Creating code_chunks table");
      self
        .connection
        .create_empty_table("code_chunks", code_chunks_schema(self.vector_dim))
        .execute()
        .await?;
    }

    if !table_names.contains(&"sessions".to_string()) {
      debug!("Creating sessions table");
      self
        .connection
        .create_empty_table("sessions", sessions_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"events".to_string()) {
      debug!("Creating events table");
      self
        .connection
        .create_empty_table("events", events_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"documents".to_string()) {
      debug!("Creating documents table");
      self
        .connection
        .create_empty_table("documents", documents_schema(self.vector_dim))
        .execute()
        .await?;
    }

    if !table_names.contains(&"session_memories".to_string()) {
      debug!("Creating session_memories table");
      self
        .connection
        .create_empty_table("session_memories", session_memories_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"memory_relationships".to_string()) {
      debug!("Creating memory_relationships table");
      self
        .connection
        .create_empty_table("memory_relationships", memory_relationships_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"document_metadata".to_string()) {
      debug!("Creating document_metadata table");
      self
        .connection
        .create_empty_table("document_metadata", document_metadata_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"entities".to_string()) {
      debug!("Creating entities table");
      self
        .connection
        .create_empty_table("entities", entities_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"memory_entities".to_string()) {
      debug!("Creating memory_entities table");
      self
        .connection
        .create_empty_table("memory_entities", memory_entities_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"index_checkpoints".to_string()) {
      debug!("Creating index_checkpoints table");
      self
        .connection
        .create_empty_table("index_checkpoints", index_checkpoints_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"segment_accumulators".to_string()) {
      debug!("Creating segment_accumulators table");
      self
        .connection
        .create_empty_table("segment_accumulators", segment_accumulators_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"extraction_segments".to_string()) {
      debug!("Creating extraction_segments table");
      self
        .connection
        .create_empty_table("extraction_segments", extraction_segments_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"code_references".to_string()) {
      debug!("Creating code_references table");
      self
        .connection
        .create_empty_table("code_references", code_references_schema())
        .execute()
        .await?;
    }

    Ok(())
  }

  /// Get the memories table
  pub async fn memories_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("memories").execute().await?)
  }

  /// Get the code_chunks table
  pub async fn code_chunks_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("code_chunks").execute().await?)
  }

  /// Get the sessions table
  pub async fn sessions_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("sessions").execute().await?)
  }

  /// Get the events table
  pub async fn events_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("events").execute().await?)
  }

  /// Get the documents table
  pub async fn documents_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("documents").execute().await?)
  }

  /// Get the session_memories table
  pub async fn session_memories_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("session_memories").execute().await?)
  }

  /// Get the memory_relationships table
  pub async fn memory_relationships_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("memory_relationships").execute().await?)
  }

  /// Get the document_metadata table
  pub async fn document_metadata_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("document_metadata").execute().await?)
  }

  /// Get the entities table
  pub async fn entities_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("entities").execute().await?)
  }

  /// Get the memory_entities table
  pub async fn memory_entities_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("memory_entities").execute().await?)
  }

  /// Get the index_checkpoints table
  pub async fn index_checkpoints_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("index_checkpoints").execute().await?)
  }

  /// Get the segment_accumulators table
  pub async fn segment_accumulators_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("segment_accumulators").execute().await?)
  }

  /// Get the extraction_segments table
  pub async fn extraction_segments_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("extraction_segments").execute().await?)
  }

  /// Get the code_references table
  pub async fn code_references_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("code_references").execute().await?)
  }

  /// Create vector indexes on tables that support vector search
  ///
  /// This should be called after initial indexing completes to enable
  /// efficient vector search. Without indexes, LanceDB does brute-force O(n) scans.
  ///
  /// Uses IVF_PQ index which is optimal for large datasets (10K+ vectors).
  /// For smaller datasets, the overhead of index creation may not be worth it.
  pub async fn create_vector_indexes(&self) -> Result<()> {
    self.create_vector_index_on_table("code_chunks").await?;
    self.create_vector_index_on_table("memories").await?;
    self.create_vector_index_on_table("documents").await?;
    Ok(())
  }

  /// Create a vector index on a single table
  async fn create_vector_index_on_table(&self, table_name: &str) -> Result<()> {
    let table = self.connection.open_table(table_name).execute().await?;

    // Check if we have enough rows to benefit from an index
    // IVF_PQ needs at least 256 vectors for meaningful partitions
    let row_count = table.count_rows(None).await?;
    if row_count < 256 {
      debug!(
        "Skipping index creation for {} - only {} rows (need 256+)",
        table_name, row_count
      );
      return Ok(());
    }

    info!("Creating vector index on {} ({} rows)", table_name, row_count);

    // Use Auto index which selects IVF_PQ for vector columns
    match table.create_index(&["vector"], Index::Auto).execute().await {
      Ok(_) => {
        info!("Vector index created on {}", table_name);
        Ok(())
      }
      Err(e) => {
        // Index may already exist or column may not support indexing
        warn!("Could not create index on {}: {}", table_name, e);
        Ok(()) // Don't fail the whole operation
      }
    }
  }

  /// Optimize indexes by incorporating new unindexed data
  ///
  /// Call this periodically after adding significant amounts of data
  /// to ensure the index includes recent additions.
  pub async fn optimize_indexes(&self) -> Result<()> {
    for table_name in ["code_chunks", "memories", "documents"] {
      let table = self.connection.open_table(table_name).execute().await?;
      if let Err(e) = table.optimize(OptimizeAction::All).await {
        warn!("Could not optimize {}: {}", table_name, e);
      }
    }
    Ok(())
  }
}

/// Get the default base path for CCEngram data
///
/// Respects the following environment variables (in order of precedence):
/// 1. DATA_DIR - explicit data directory override
/// 2. XDG_DATA_HOME - standard XDG data home directory
/// 3. dirs::data_local_dir() - platform default
pub fn default_data_dir() -> PathBuf {
  // Check explicit override first
  if let Ok(dir) = std::env::var("DATA_DIR") {
    return PathBuf::from(dir);
  }

  // Check XDG_DATA_HOME
  if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME") {
    return PathBuf::from(xdg_data).join("ccengram");
  }

  // Fall back to platform default
  dirs::data_local_dir()
    .unwrap_or_else(|| PathBuf::from("."))
    .join("ccengram")
}

/// Get the default config directory
///
/// Respects the following environment variables (in order of precedence):
/// 1. CONFIG_DIR - explicit config directory override
/// 2. XDG_CONFIG_HOME - standard XDG config home directory
/// 3. dirs::config_dir() - platform default
pub fn default_config_dir() -> PathBuf {
  // Check explicit override first
  if let Ok(dir) = std::env::var("CONFIG_DIR") {
    return PathBuf::from(dir);
  }

  // Check XDG_CONFIG_HOME
  if let Ok(xdg_config) = std::env::var("XDG_CONFIG_HOME") {
    return PathBuf::from(xdg_config).join("ccengram");
  }

  // Fall back to platform default
  dirs::config_dir()
    .unwrap_or_else(|| PathBuf::from("."))
    .join("ccengram")
}

/// Get the default cache directory
///
/// Respects the following environment variables (in order of precedence):
/// 1. XDG_CACHE_HOME - standard XDG cache home directory
/// 2. dirs::cache_dir() - platform default
pub fn default_cache_dir() -> PathBuf {
  // Check XDG_CACHE_HOME
  if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
    return PathBuf::from(xdg_cache).join("ccengram");
  }

  // Fall back to platform default
  dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".")).join("ccengram")
}

/// Get the daemon port
///
/// Respects PORT environment variable, defaults to 8642
pub fn default_port() -> u16 {
  std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8642)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::path::Path;
  use std::sync::Mutex;
  use tempfile::TempDir;

  // Mutex to serialize tests that modify environment variables
  static ENV_MUTEX: Mutex<()> = Mutex::new(());

  #[tokio::test]
  async fn test_open_database() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test/project"));

    let db = ProjectDb::open_at_path(project_id.clone(), temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();

    assert_eq!(db.project_id.as_str(), project_id.as_str());
  }

  #[tokio::test]
  async fn test_tables_created() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test/project"));

    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();

    let tables = db.connection.table_names().execute().await.unwrap();
    assert!(tables.contains(&"memories".to_string()));
    assert!(tables.contains(&"code_chunks".to_string()));
    assert!(tables.contains(&"sessions".to_string()));
    assert!(tables.contains(&"events".to_string()));
  }

  #[test]
  fn test_default_port() {
    let _guard = ENV_MUTEX.lock().unwrap();
    // Default value
    unsafe {
      std::env::remove_var("PORT");
    }
    assert_eq!(default_port(), 8642);
  }

  #[test]
  fn test_env_override_data_dir() {
    let _guard = ENV_MUTEX.lock().unwrap();
    // Store original value
    let original = std::env::var("DATA_DIR").ok();

    // Test with override
    unsafe {
      std::env::set_var("DATA_DIR", "/custom/data/path");
    }
    let dir = default_data_dir();
    assert_eq!(dir, PathBuf::from("/custom/data/path"));

    // Cleanup
    if let Some(orig) = original {
      unsafe {
        std::env::set_var("DATA_DIR", orig);
      }
    } else {
      unsafe {
        std::env::remove_var("DATA_DIR");
      }
    }
  }

  #[test]
  fn test_env_override_config_dir() {
    let _guard = ENV_MUTEX.lock().unwrap();
    // Store original value
    let original = std::env::var("CONFIG_DIR").ok();

    // Test with override
    unsafe {
      std::env::set_var("CONFIG_DIR", "/custom/config/path");
    }
    let dir = default_config_dir();
    assert_eq!(dir, PathBuf::from("/custom/config/path"));

    // Cleanup
    if let Some(orig) = original {
      unsafe {
        std::env::set_var("CONFIG_DIR", orig);
      }
    } else {
      unsafe {
        std::env::remove_var("CONFIG_DIR");
      }
    }
  }

  #[test]
  fn test_xdg_data_home() {
    let _guard = ENV_MUTEX.lock().unwrap();
    // Store original values
    let original_data_dir = std::env::var("DATA_DIR").ok();
    let original_xdg = std::env::var("XDG_DATA_HOME").ok();

    // Ensure DATA_DIR is not set (it takes precedence)
    unsafe {
      std::env::remove_var("DATA_DIR");
    }

    // Test with XDG_DATA_HOME
    unsafe {
      std::env::set_var("XDG_DATA_HOME", "/xdg/data");
    }
    let dir = default_data_dir();
    assert_eq!(dir, PathBuf::from("/xdg/data/ccengram"));

    // Cleanup
    if let Some(orig) = original_data_dir {
      unsafe {
        std::env::set_var("DATA_DIR", orig);
      }
    } else {
      unsafe {
        std::env::remove_var("DATA_DIR");
      }
    }
    if let Some(orig) = original_xdg {
      unsafe {
        std::env::set_var("XDG_DATA_HOME", orig);
      }
    } else {
      unsafe {
        std::env::remove_var("XDG_DATA_HOME");
      }
    }
  }
}
