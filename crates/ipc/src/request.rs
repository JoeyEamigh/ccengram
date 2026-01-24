use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PingParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatusParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemorySearchParams {
  pub query: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sector: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub tier: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
  pub memory_type: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub min_salience: Option<f32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub scope_path: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub scope_module: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub session_id: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
  #[serde(default)]
  pub include_superseded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAddParams {
  pub content: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sector: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
  pub memory_type: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub context: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub tags: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub categories: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub scope_path: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub scope_module: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub importance: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGetParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub include_related: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryListParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sector: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReinforceParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub amount: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDeemphasizeParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub amount: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDeleteParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRestoreParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryListDeletedParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySupersedeParams {
  pub old_memory_id: String,
  pub new_content: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimelineParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelatedParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeSearchParams {
  pub query: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub file_pattern: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub symbol_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextParams {
  pub chunk_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub before: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub after: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeIndexParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default)]
  pub force: bool,
  #[serde(default)]
  pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeListParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeStatsParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCallersParams {
  pub chunk_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleesParams {
  pub chunk_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExploreParams {
  pub query: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub scope: Option<String>, // "code" | "memory" | "docs" | "all"
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub expand_top: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub format: Option<String>, // "json" | "text"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub id: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub ids: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub depth: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocsSearchParams {
  pub query: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContextParams {
  pub doc_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub before: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub after: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocsIngestParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub directory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatchStartParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatchStopParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatchStatusParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityListParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityGetParams {
  pub entity_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityTopParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectsListParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfoParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCleanParams {
  pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectsCleanAllParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShutdownParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthCheckParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookParams {
  pub hook_name: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub session_id: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(flatten)]
  pub data: serde_json::Value, // Hook-specific data varies widely
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeImportChunkParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  pub file_path: String,
  pub content: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeMemoriesParams {
  pub chunk_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeRelatedParams {
  pub chunk_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeContextFullParams {
  pub chunk_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipAddParams {
  pub from_entity: String,
  pub to_entity: String,
  pub relationship_type: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipListParams {
  pub memory_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipDeleteParams {
  pub from_entity: String,
  pub to_entity: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub relationship_type: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipRelatedParams {
  pub entity_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectStatsParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MigrateEmbeddingParams {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cwd: Option<String>,
}
