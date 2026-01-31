//! Project IPC types - requests and responses
use serde::{Deserialize, Serialize};

// ============================================================================
// Request types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum ProjectRequest {
  List(ProjectListParams),
  Info(ProjectInfoParams),
  Clean(ProjectCleanParams),
  CleanAll(ProjectCleanAllParams),
  Sessions(SessionListParams),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectListParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectCleanAllParams;

/// Parameters for session list request
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionListParams {
  /// Maximum number of sessions to return
  pub limit: Option<usize>,
  /// Filter for active sessions only
  pub active_only: Option<bool>,
}

/// Parameters for project info request
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectInfoParams {
  /// Project path or ID prefix. If None, uses cwd from request.
  pub project: Option<String>,
}

/// Parameters for project clean request
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectCleanParams {
  /// Project path or ID prefix. If None, uses cwd from request.
  pub project: Option<String>,
}

// ============================================================================
// Response types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum ProjectResponse {
  List(Vec<ProjectListItem>),
  Info(ProjectInfoResult),
  Clean(ProjectCleanResult),
  CleanAll(ProjectCleanAllResult),
  Stats(ProjectStatsResult),
  Sessions(Vec<SessionItem>),
}

/// Lightweight project item for list responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectListItem {
  pub id: String,
  pub path: String,
  pub name: String,
}

/// Detailed project info response
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfoResult {
  pub id: String,
  pub path: String,
  pub name: String,
  pub memory_count: usize,
  pub code_chunk_count: usize,
  pub document_count: usize,
  pub session_count: usize,
  pub db_path: String,
}

/// Result from cleaning a single project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCleanResult {
  pub path: String,
  pub memories_deleted: usize,
  pub code_chunks_deleted: usize,
  pub documents_deleted: usize,
}

/// Result from cleaning all projects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCleanAllResult {
  pub projects_removed: usize,
}

/// Project statistics result
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatsResult {
  pub project_id: String,
  pub path: String,
  pub memories: usize,
  pub code_chunks: usize,
  pub documents: usize,
  pub sessions: usize,
  /// Memory count by sector (semantic, episodic, procedural, reflective)
  pub memories_by_sector: Option<std::collections::HashMap<String, usize>>,
  /// Average salience across all memories
  pub average_salience: Option<f32>,
}

/// Session item for list responses
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionItem {
  pub id: String,
  pub started_at: String,
  pub ended_at: Option<String>,
  pub summary: Option<String>,
  pub user_prompt: Option<String>,
}

// ============================================================================
// IpcRequest implementations
// ============================================================================

use crate::{
  impl_ipc_request,
  ipc::{RequestData, ResponseData},
};

impl_ipc_request!(
  ProjectListParams => Vec<ProjectListItem>,
  ResponseData::Project(ProjectResponse::List(v)) => v,
  v => RequestData::Project(ProjectRequest::List(v)),
  v => ResponseData::Project(ProjectResponse::List(v))
);
impl_ipc_request!(
  ProjectInfoParams => ProjectInfoResult,
  ResponseData::Project(ProjectResponse::Info(v)) => v,
  v => RequestData::Project(ProjectRequest::Info(v)),
  v => ResponseData::Project(ProjectResponse::Info(v))
);
impl_ipc_request!(
  ProjectCleanParams => ProjectCleanResult,
  ResponseData::Project(ProjectResponse::Clean(v)) => v,
  v => RequestData::Project(ProjectRequest::Clean(v)),
  v => ResponseData::Project(ProjectResponse::Clean(v))
);
impl_ipc_request!(
  ProjectCleanAllParams => ProjectCleanAllResult,
  ResponseData::Project(ProjectResponse::CleanAll(v)) => v,
  v => RequestData::Project(ProjectRequest::CleanAll(v)),
  v => ResponseData::Project(ProjectResponse::CleanAll(v))
);
impl_ipc_request!(
  SessionListParams => Vec<SessionItem>,
  ResponseData::Project(ProjectResponse::Sessions(v)) => v,
  v => RequestData::Project(ProjectRequest::Sessions(v)),
  v => ResponseData::Project(ProjectResponse::Sessions(v))
);
