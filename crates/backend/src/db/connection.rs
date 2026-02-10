use std::{path::PathBuf, sync::Arc};

use lancedb::{Connection, ObjectStoreRegistry, Session, Table, connect, index::Index};
use thiserror::Error;
use tracing::{debug, error, trace};

use crate::{
  config::Config,
  db::schema::{
    code_chunks_schema, document_metadata_schema, documents_schema, indexed_files_schema, memories_schema,
    memory_relationships_schema, session_memories_schema, sessions_schema,
  },
  domain::project::ProjectId,
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
  #[error("Database query error: {0}")]
  Query(String),
  #[error("Ambiguous prefix '{prefix}' matches {count} items. Use more characters.")]
  AmbiguousPrefix { prefix: String, count: usize },
}

pub type Result<T> = std::result::Result<T, DbError>;

/// Database connection for a specific project
///
/// Uses a shared Session with controlled cache sizes to avoid LanceDB's
/// default 7 GB per-table cache (6 GB index + 1 GB metadata).
/// All table handles are held permanently for zero per-operation overhead.
pub struct ProjectDb {
  pub project_id: ProjectId,
  #[allow(dead_code)] // idk i might need this later
  pub connection: Connection,
  pub vector_dim: usize,
  session: Arc<Session>,

  // Table handles held permanently - Table is Send + Sync
  // Dropping tables doesn't free cached memory (Session holds caches)
  memories: Table,
  code_chunks: Table,
  sessions_table: Table, // renamed to avoid confusion with Session
  documents: Table,
  session_memories: Table,
  memory_relationships: Table,
  document_metadata: Table,
  indexed_files: Table,
}

impl ProjectDb {
  /// Open or create a project database
  pub async fn open(project_id: ProjectId, base_path: &std::path::Path, config: Arc<Config>) -> Result<Self> {
    let db_path = project_id.data_dir(base_path).join("lancedb");
    Self::open_at_path(project_id, db_path, config).await
  }

  /// Open database at a specific path
  ///
  /// Creates a shared Session with controlled cache sizes (from config.database)
  /// and opens all table handles permanently.
  pub async fn open_at_path(project_id: ProjectId, db_path: PathBuf, config: Arc<Config>) -> Result<Self> {
    // Ensure directory exists
    if let Some(parent) = db_path.parent() {
      tokio::fs::create_dir_all(parent).await?;
    }

    // Create shared session with controlled cache sizes
    // Default LanceDB: 6 GB index + 1 GB metadata per table = 56 GB for 8 tables
    // Our config: 256 MB index + 64 MB metadata shared across ALL tables
    let index_cache_bytes = config.database.index_cache_mb * 1024 * 1024;
    let metadata_cache_bytes = config.database.metadata_cache_mb * 1024 * 1024;
    let registry = Arc::new(ObjectStoreRegistry::default());
    let session = Arc::new(Session::new(index_cache_bytes, metadata_cache_bytes, registry));

    debug!(
      path = %db_path.display(),
      project_id = %project_id.as_str(),
      vector_dim = config.embedding.dimensions,
      index_cache_mb = config.database.index_cache_mb,
      metadata_cache_mb = config.database.metadata_cache_mb,
      "Opening database connection with shared session"
    );

    let connection = match connect(db_path.to_string_lossy().as_ref())
      .session(session.clone())
      .execute()
      .await
    {
      Ok(conn) => {
        debug!(path = %db_path.display(), "Database connection established");
        conn
      }
      Err(e) => {
        error!(path = %db_path.display(), err = %e, "Failed to connect to database");
        return Err(e.into());
      }
    };

    // Ensure tables exist before opening handles
    debug!("Initializing database schema");
    Self::ensure_tables_static(&connection, config.embedding.dimensions).await?;

    // Open all table handles once, hold permanently
    // Table is Send + Sync, so concurrent access is safe
    debug!("Opening table handles");
    let memories = connection.open_table("memories").execute().await?;
    let code_chunks = connection.open_table("code_chunks").execute().await?;
    let sessions_table = connection.open_table("sessions").execute().await?;
    let documents = connection.open_table("documents").execute().await?;
    let session_memories = connection.open_table("session_memories").execute().await?;
    let memory_relationships = connection.open_table("memory_relationships").execute().await?;
    let document_metadata = connection.open_table("document_metadata").execute().await?;
    let indexed_files = connection.open_table("indexed_files").execute().await?;

    let db = Self {
      project_id,
      connection,
      vector_dim: config.embedding.dimensions,
      session,
      memories,
      code_chunks,
      sessions_table,
      documents,
      session_memories,
      memory_relationships,
      document_metadata,
      indexed_files,
    };

    // Create scalar indexes for improved query and merge_insert performance
    // This is idempotent - indexes that already exist are skipped
    db.create_scalar_indexes().await?;

    // Create FTS indexes for keyword search (idempotent)
    db.create_fts_indexes().await?;

    Ok(db)
  }

  /// Ensure all required tables exist (static version for use before struct creation)
  async fn ensure_tables_static(connection: &Connection, vector_dim: usize) -> Result<()> {
    let table_names = connection.table_names().execute().await?;
    debug!(existing_tables = table_names.len(), "Checking required tables");

    if !table_names.contains(&"memories".to_string()) {
      debug!("Creating memories table");
      connection
        .create_empty_table("memories", memories_schema(vector_dim))
        .execute()
        .await?;
    }

    if !table_names.contains(&"code_chunks".to_string()) {
      debug!("Creating code_chunks table");
      connection
        .create_empty_table("code_chunks", code_chunks_schema(vector_dim))
        .execute()
        .await?;
    }

    if !table_names.contains(&"sessions".to_string()) {
      debug!("Creating sessions table");
      connection
        .create_empty_table("sessions", sessions_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"documents".to_string()) {
      debug!("Creating documents table");
      connection
        .create_empty_table("documents", documents_schema(vector_dim))
        .execute()
        .await?;
    }

    if !table_names.contains(&"session_memories".to_string()) {
      debug!("Creating session_memories table");
      connection
        .create_empty_table("session_memories", session_memories_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"memory_relationships".to_string()) {
      debug!("Creating memory_relationships table");
      connection
        .create_empty_table("memory_relationships", memory_relationships_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"document_metadata".to_string()) {
      debug!("Creating document_metadata table");
      connection
        .create_empty_table("document_metadata", document_metadata_schema())
        .execute()
        .await?;
    }

    if !table_names.contains(&"indexed_files".to_string()) {
      debug!("Creating indexed_files table");
      connection
        .create_empty_table("indexed_files", indexed_files_schema())
        .execute()
        .await?;
    }

    Ok(())
  }

  // ============================================================================
  // Table Accessors - Simple field access, no async, no Result
  // Tables are held permanently; dropping them doesn't free cached memory.
  // ============================================================================

  /// Get the memories table
  pub fn memories_table(&self) -> &Table {
    &self.memories
  }

  /// Get the code_chunks table
  pub fn code_chunks_table(&self) -> &Table {
    &self.code_chunks
  }

  /// Get the sessions table
  pub fn sessions_table(&self) -> &Table {
    &self.sessions_table
  }

  /// Get the documents table
  pub fn documents_table(&self) -> &Table {
    &self.documents
  }

  /// Get the session_memories table
  pub fn session_memories_table(&self) -> &Table {
    &self.session_memories
  }

  /// Get the memory_relationships table
  pub fn memory_relationships_table(&self) -> &Table {
    &self.memory_relationships
  }

  /// Get the document_metadata table
  pub fn document_metadata_table(&self) -> &Table {
    &self.document_metadata
  }

  /// Get the indexed_files table
  pub fn indexed_files_table(&self) -> &Table {
    &self.indexed_files
  }

  // ============================================================================
  // Cache Statistics (for debugging memory usage)
  // ============================================================================

  /// Log current cache statistics
  ///
  /// Useful for debugging memory usage during indexing.
  /// Enable via `config.database.log_cache_stats = true`.
  pub async fn log_cache_stats(&self) {
    let index_stats = self.session.index_cache_stats().await;
    let meta_stats = self.session.metadata_cache_stats().await;
    #[cfg(feature = "statm")]
    let size = self.session.size_bytes() / (1024 * 1024);

    #[cfg(not(feature = "statm"))]
    let size = "enable statm feature";

    trace!(
      index_hits = index_stats.hits,
      index_misses = index_stats.misses,
      meta_hits = meta_stats.hits,
      meta_misses = meta_stats.misses,
      total_size_mb = size,
      "LanceDB cache stats"
    );
  }

  // ============================================================================
  // Index Management
  // ============================================================================

  /// Create scalar indexes for improved query and merge_insert performance
  ///
  /// Scalar indexes (BTREE) accelerate:
  /// - Filter queries with `only_if()` clauses
  /// - merge_insert operations (join columns should be indexed)
  /// - Equality lookups by ID
  ///
  /// Call this after initial schema creation. Indexes are idempotent - calling
  /// multiple times is safe (LanceDB skips if index already exists).
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn create_scalar_indexes(&self) -> Result<()> {
    debug!("Creating scalar indexes for improved query performance");

    // code_chunks: merge_insert uses (file_path, start_line), queries filter by file_path, id
    self
      .create_scalar_index_if_missing(&self.code_chunks, "file_path")
      .await?;
    self.create_scalar_index_if_missing(&self.code_chunks, "id").await?;

    // memories: merge_insert uses id, queries filter by id, is_deleted
    self.create_scalar_index_if_missing(&self.memories, "id").await?;
    self
      .create_scalar_index_if_missing(&self.memories, "is_deleted")
      .await?;

    // documents: merge_insert uses (source, chunk_index), queries filter by source, document_id
    self.create_scalar_index_if_missing(&self.documents, "source").await?;
    self
      .create_scalar_index_if_missing(&self.documents, "document_id")
      .await?;

    // indexed_files: merge_insert uses file_path, queries filter by project_id, file_path
    self
      .create_scalar_index_if_missing(&self.indexed_files, "file_path")
      .await?;
    self
      .create_scalar_index_if_missing(&self.indexed_files, "project_id")
      .await?;

    // document_metadata: queries filter by source, id
    self
      .create_scalar_index_if_missing(&self.document_metadata, "source")
      .await?;
    self
      .create_scalar_index_if_missing(&self.document_metadata, "id")
      .await?;

    // session_memories: junction table queries by session_id, memory_id
    self
      .create_scalar_index_if_missing(&self.session_memories, "session_id")
      .await?;
    self
      .create_scalar_index_if_missing(&self.session_memories, "memory_id")
      .await?;

    // memory_relationships: queries by from_memory_id, to_memory_id
    self
      .create_scalar_index_if_missing(&self.memory_relationships, "from_memory_id")
      .await?;
    self
      .create_scalar_index_if_missing(&self.memory_relationships, "to_memory_id")
      .await?;

    // sessions: queries by id
    self.create_scalar_index_if_missing(&self.sessions_table, "id").await?;

    debug!("Scalar index creation complete");
    Ok(())
  }

  /// Helper to create a scalar index if it doesn't already exist
  async fn create_scalar_index_if_missing(&self, table: &Table, column: &str) -> Result<()> {
    let indices = table.list_indices().await?;
    let index_exists = indices.iter().any(|idx| idx.columns.contains(&column.to_string()));

    if !index_exists {
      trace!(table = %table.name(), column = column, "Creating BTREE scalar index");
      table
        .create_index(&[column], Index::BTree(Default::default()))
        .execute()
        .await?;
    } else {
      trace!(table = %table.name(), column = column, "Scalar index already exists");
    }

    Ok(())
  }

  /// Create FTS indexes for keyword search on text columns.
  ///
  /// FTS indexes enable full-text search (BM25) on:
  /// - code_chunks.embedding_text: enriched text with tokenized identifiers
  /// - memories.content: natural language memory content
  /// - documents.content: document chunk content
  ///
  /// Idempotent - skips if indexes already exist.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn create_fts_indexes(&self) -> Result<()> {
    use lancedb::index::scalar::FtsIndexBuilder;

    debug!("Creating FTS indexes for keyword search");

    // code_chunks: FTS on embedding_text (contains enriched, tokenized text)
    self
      .create_fts_index_if_missing(&self.code_chunks, "embedding_text", FtsIndexBuilder::default())
      .await?;

    // memories: FTS on content (natural language)
    self
      .create_fts_index_if_missing(&self.memories, "content", FtsIndexBuilder::default())
      .await?;

    // documents: FTS on content (natural language)
    self
      .create_fts_index_if_missing(&self.documents, "content", FtsIndexBuilder::default())
      .await?;

    debug!("FTS index creation complete");
    Ok(())
  }

  /// Helper to create an FTS index if it doesn't already exist
  async fn create_fts_index_if_missing(
    &self,
    table: &Table,
    column: &str,
    builder: lancedb::index::scalar::FtsIndexBuilder,
  ) -> Result<()> {
    let indices = table.list_indices().await?;
    let fts_exists = indices
      .iter()
      .any(|idx| idx.columns.contains(&column.to_string()) && matches!(idx.index_type, lancedb::index::IndexType::FTS));

    if !fts_exists {
      trace!(table = %table.name(), column = column, "Creating FTS index");
      table.create_index(&[column], Index::FTS(builder)).execute().await?;
    } else {
      trace!(table = %table.name(), column = column, "FTS index already exists");
    }

    Ok(())
  }

  /// Rebuild FTS indexes (called after significant data changes).
  ///
  /// LanceDB FTS indexes may need rebuilding after soft deletes or many upserts.
  /// This drops and recreates the indexes.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn rebuild_fts_indexes(&self) -> Result<()> {
    use lancedb::index::scalar::FtsIndexBuilder;

    debug!("Rebuilding FTS indexes");

    // Recreate with replace semantics (create_index replaces existing)
    self
      .code_chunks
      .create_index(&["embedding_text"], Index::FTS(FtsIndexBuilder::default()))
      .replace(true)
      .execute()
      .await?;

    self
      .memories
      .create_index(&["content"], Index::FTS(FtsIndexBuilder::default()))
      .replace(true)
      .execute()
      .await?;

    self
      .documents
      .create_index(&["content"], Index::FTS(FtsIndexBuilder::default()))
      .replace(true)
      .execute()
      .await?;

    debug!("FTS index rebuild complete");
    Ok(())
  }

  /// Create vector indexes for improved similarity search performance
  ///
  /// Vector indexes (IVF_PQ) accelerate `vector_search()` queries from O(n) to O(log n).
  /// IVF_PQ is chosen over HNSW for lower memory usage (disk-based), which aligns
  /// with our bounded cache strategy (256 MB index cache).
  ///
  /// Requirements:
  /// - Tables should have at least 1,000+ rows for effective index training
  /// - Index training is a one-time operation per table
  ///
  /// Configuration:
  /// - num_partitions: 256 (good for 10K-1M rows, heuristic: sqrt(row_count))
  /// - num_sub_vectors: vector_dim / 16 (optimal compression vs accuracy)
  /// - distance_type: Cosine (match embedding model)
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn _create_vector_indexes(&self) -> Result<()> {
    use lancedb::{DistanceType, index::vector::IvfPqIndexBuilder};

    debug!("Creating vector indexes for similarity search acceleration");

    // Calculate optimal sub-vectors based on dimension
    // 4096 / 16 = 256 sub-vectors (sweet spot for compression vs accuracy)
    let num_sub_vectors = (self.vector_dim / 16).max(1) as u32;

    // code_chunks: primary table for code search
    let code_count = self.code_chunks.count_rows(None).await?;
    if code_count >= 1000 {
      debug!(table = "code_chunks", rows = code_count, "Creating IVF_PQ vector index");
      self
        .code_chunks
        .create_index(
          &["vector"],
          Index::IvfPq(
            IvfPqIndexBuilder::default()
              .distance_type(DistanceType::Cosine)
              .num_partitions(256)
              .num_sub_vectors(num_sub_vectors),
          ),
        )
        .execute()
        .await?;
    } else {
      debug!(
        table = "code_chunks",
        rows = code_count,
        "Skipping vector index (need >= 1000 rows)"
      );
    }

    // memories: semantic memory search
    let mem_count = self.memories.count_rows(None).await?;
    if mem_count >= 1000 {
      debug!(table = "memories", rows = mem_count, "Creating IVF_PQ vector index");
      self
        .memories
        .create_index(
          &["vector"],
          Index::IvfPq(
            IvfPqIndexBuilder::default()
              .distance_type(DistanceType::Cosine)
              .num_partitions(256)
              .num_sub_vectors(num_sub_vectors),
          ),
        )
        .execute()
        .await?;
    } else {
      debug!(
        table = "memories",
        rows = mem_count,
        "Skipping vector index (need >= 1000 rows)"
      );
    }

    // documents: document chunk search
    let doc_count = self.documents.count_rows(None).await?;
    if doc_count >= 1000 {
      debug!(table = "documents", rows = doc_count, "Creating IVF_PQ vector index");
      self
        .documents
        .create_index(
          &["vector"],
          Index::IvfPq(
            IvfPqIndexBuilder::default()
              .distance_type(DistanceType::Cosine)
              .num_partitions(256)
              .num_sub_vectors(num_sub_vectors),
          ),
        )
        .execute()
        .await?;
    } else {
      debug!(
        table = "documents",
        rows = doc_count,
        "Skipping vector index (need >= 1000 rows)"
      );
    }

    debug!("Vector index creation complete");
    Ok(())
  }

  /// Optimize all tables after batch write operations
  ///
  /// This updates scalar indexes to include newly written data. Without optimization,
  /// new rows are still searchable but require a flat scan of unindexed data.
  ///
  /// Call this:
  /// - After bulk indexing operations complete
  /// - Periodically during long-running indexing (e.g., every N flushes)
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn optimize_indexes(&self) -> Result<()> {
    use lancedb::table::OptimizeAction;

    debug!("Optimizing indexes after batch writes");

    // Optimize tables that receive frequent writes during indexing
    self.code_chunks.optimize(OptimizeAction::All).await?;
    self.indexed_files.optimize(OptimizeAction::All).await?;
    self.documents.optimize(OptimizeAction::All).await?;
    self.document_metadata.optimize(OptimizeAction::All).await?;

    // These tables have less frequent writes but still benefit from optimization
    self.memories.optimize(OptimizeAction::All).await?;
    self.sessions_table.optimize(OptimizeAction::All).await?;
    self.session_memories.optimize(OptimizeAction::All).await?;
    self.memory_relationships.optimize(OptimizeAction::All).await?;

    debug!("Index optimization complete");
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use tempfile::TempDir;

  use super::*;

  #[tokio::test]
  async fn test_open_database() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test/project")).await;

    let db = ProjectDb::open_at_path(
      project_id.clone(),
      temp_dir.path().join("test.lancedb"),
      Arc::new(Config::default()),
    )
    .await
    .unwrap();

    assert_eq!(db.project_id.as_str(), project_id.as_str());
  }

  #[tokio::test]
  async fn test_tables_created() {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test/project")).await;

    let db = ProjectDb::open_at_path(
      project_id,
      temp_dir.path().join("test.lancedb"),
      Arc::new(Config::default()),
    )
    .await
    .unwrap();

    let tables = db.connection.table_names().execute().await.unwrap();
    assert!(tables.contains(&"memories".to_string()), "memories table should exist");
    assert!(
      tables.contains(&"code_chunks".to_string()),
      "code_chunks table should exist"
    );
    assert!(tables.contains(&"sessions".to_string()), "sessions table should exist");
    assert!(
      tables.contains(&"documents".to_string()),
      "documents table should exist"
    );
  }
}
