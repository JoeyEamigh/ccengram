//! Project-level services.
//!
//! Provides operations for project management including:
//! - Project statistics
//! - Project cleanup

use std::path::Path;

use uuid::Uuid;

use crate::{
  db::ProjectDb,
  domain::project::ProjectId,
  ipc::project::{ProjectCleanResult, ProjectInfoResult, ProjectStatsResult},
  service::util::ServiceError,
};

/// Get project information.
///
/// # Arguments
/// * `db` - Project database
/// * `project_id` - Project ID
/// * `root` - Project root path
///
/// # Returns
/// * `Ok(ProjectInfoResult)` - Project information
/// * `Err(ServiceError)` - If query fails
pub async fn info(db: &ProjectDb, project_id: &ProjectId, root: &Path) -> Result<ProjectInfoResult, ServiceError> {
  // Run both queries in parallel - they read from different tables
  let (memory_result, code_result) = tokio::join!(db.list_memories(None, Some(1)), db.list_code_chunks(None, Some(1)));

  let memory_count = memory_result.map(|m| m.len()).unwrap_or(0);
  let code_chunk_count = code_result.map(|c| c.len()).unwrap_or(0);

  Ok(ProjectInfoResult {
    id: project_id.to_string(),
    path: root.to_string_lossy().to_string(),
    name: root
      .file_name()
      .map(|n| n.to_string_lossy().to_string())
      .unwrap_or_else(|| "unknown".to_string()),
    memory_count,
    code_chunk_count,
    document_count: 0,
    session_count: 0,
    db_path: String::new(), // Caller can fill this in if needed
  })
}

/// Get project statistics.
///
/// # Arguments
/// * `db` - Project database
/// * `project_id` - Project ID
/// * `project_uuid` - Project UUID for session counting
/// * `root` - Project root path
///
/// # Returns
/// * `Ok(ProjectStatsResult)` - Project statistics
/// * `Err(ServiceError)` - If query fails
pub async fn stats(
  db: &ProjectDb,
  project_id: &ProjectId,
  project_uuid: &Uuid,
  root: &Path,
) -> Result<ProjectStatsResult, ServiceError> {
  use std::collections::HashMap;

  // Run all four queries in parallel - they read from different tables
  let (memories_result, code_result, doc_result, sessions_result) = tokio::join!(
    db.list_memories(None, None),
    db.list_code_chunks(None, None),
    db.list_document_chunks(None, None),
    db.count_sessions(project_uuid)
  );

  let memories_list = memories_result.unwrap_or_default();
  let memories = memories_list.len();

  // Calculate memory stats
  let (memories_by_sector, average_salience) = if !memories_list.is_empty() {
    let mut by_sector: HashMap<String, usize> = HashMap::new();
    let mut total_salience = 0.0f32;

    for m in &memories_list {
      *by_sector.entry(m.sector.as_str().to_string()).or_default() += 1;
      total_salience += m.salience;
    }

    let avg = total_salience / memories_list.len() as f32;
    (Some(by_sector), Some(avg))
  } else {
    (None, None)
  };

  let code_chunks = code_result.map(|c| c.len()).unwrap_or(0);
  let documents = doc_result.map(|d| d.len()).unwrap_or(0);
  let sessions = sessions_result.unwrap_or(0);

  Ok(ProjectStatsResult {
    project_id: project_id.to_string(),
    path: root.to_string_lossy().to_string(),
    memories,
    code_chunks,
    documents,
    sessions,
    memories_by_sector,
    average_salience,
  })
}

/// Clean all data from a project.
///
/// Deletes all memories, code chunks, and documents.
///
/// # Arguments
/// * `db` - Project database
/// * `root` - Project root path
///
/// # Returns
/// * `Ok(ProjectCleanResult)` - Cleanup results with counts
/// * `Err(ServiceError)` - If cleanup fails
pub async fn clean(db: &ProjectDb, root: &Path) -> Result<ProjectCleanResult, ServiceError> {
  // List all data in parallel first
  let (memories_result, code_result, doc_result) = tokio::join!(
    db.list_memories(None, None),
    db.list_code_chunks(None, None),
    db.list_document_chunks(None, None)
  );

  let memories = memories_result.unwrap_or_default();
  let code_chunks = code_result.unwrap_or_default();
  let documents = doc_result.unwrap_or_default();

  let memories_deleted = memories.len();
  let code_chunks_deleted = code_chunks.len();
  let documents_deleted = documents.len();

  // Delete all data in parallel across different tables
  let memory_ids: Vec<_> = memories.iter().map(|m| m.id).collect();
  let code_ids: Vec<_> = code_chunks.iter().map(|c| c.id).collect();
  let doc_ids: Vec<_> = documents.iter().map(|d| d.id).collect();

  let delete_memories = async {
    for id in &memory_ids {
      let _ = db.delete_memory(id).await;
    }
  };

  let delete_code = async {
    for id in &code_ids {
      let _ = db.delete_code_chunk(id).await;
    }
  };

  let delete_docs = async {
    for id in &doc_ids {
      let _ = db.delete_document_chunk(id).await;
    }
  };

  // Run all three deletion loops in parallel
  tokio::join!(delete_memories, delete_code, delete_docs);

  Ok(ProjectCleanResult {
    path: root.to_string_lossy().to_string(),
    memories_deleted,
    code_chunks_deleted,
    documents_deleted,
  })
}
