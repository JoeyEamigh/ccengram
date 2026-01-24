use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResult(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResult {
  pub status: String,
  pub version: String,
  pub projects: usize,
  pub active_sessions: usize,
  pub idle_seconds: u64,
  pub uptime_seconds: u64,
  pub foreground: bool,
  pub auto_shutdown: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResult {
  pub daemon: DaemonMetrics,
  pub embedding_provider: EmbeddingProviderInfo,
  pub projects: Vec<ProjectMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonMetrics {
  pub version: String,
  pub uptime_seconds: u64,
  pub idle_seconds: u64,
  pub foreground: bool,
  pub auto_shutdown: bool,
  pub total_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProviderInfo {
  pub provider_type: String,
  pub model: String,
  pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetrics {
  pub path: String,
  pub memories: usize,
  pub code_chunks: usize,
  pub documents: usize,
  pub entities: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult(pub Vec<MemorySearchItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchItem {
  pub id: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  pub sector: String,
  pub tier: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub memory_type: Option<String>,
  pub salience: f32,
  pub importance: f32,
  pub similarity: f32,
  pub rank_score: f32,
  pub is_superseded: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub superseded_by: Option<String>,
  pub tags: Vec<String>,
  pub categories: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_path: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_module: Option<String>,
  pub created_at: String,
  pub last_accessed: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGetResult {
  pub memory: MemoryDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDetail {
  pub id: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  pub sector: String,
  pub tier: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub memory_type: Option<String>,
  pub salience: f32,
  pub importance: f32,
  pub access_count: u32,
  pub is_superseded: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub superseded_by: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub supersedes: Option<String>,
  pub tags: Vec<String>,
  pub categories: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_path: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_module: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context: Option<String>,
  pub created_at: String,
  pub last_accessed: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAddResult {
  pub id: String,
  pub message: String,
  #[serde(default)]
  pub is_duplicate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUpdateResult {
  pub id: String,
  pub new_salience: f32,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDeleteResult {
  pub id: String,
  pub message: String,
  #[serde(default)]
  pub hard_delete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryListResult(pub Vec<MemorySearchItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimelineResult {
  pub memory: MemoryDetail,
  pub timeline: Vec<TimelineEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
  pub id: String,
  pub content: String,
  pub relationship: String, // "supersedes" | "superseded_by"
  pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelatedResult(pub Vec<MemorySearchItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSearchResult {
  pub query: String,
  pub chunks: Vec<CodeChunkItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunkItem {
  pub id: String,
  pub file_path: String,
  pub start_line: u32,
  pub end_line: u32,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub symbol_name: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub symbol_type: Option<String>,
  pub similarity: f32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextResult {
  pub chunk: CodeChunkDetail,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub before: Option<Vec<CodeChunkItem>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub after: Option<Vec<CodeChunkItem>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunkDetail {
  pub id: String,
  pub file_path: String,
  pub start_line: u32,
  pub end_line: u32,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub symbol_name: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub symbol_type: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexResult {
  pub files_processed: u32,
  pub chunks_created: u32,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStatsResult {
  pub total_chunks: usize,
  pub total_files: usize,
  pub total_tokens_estimate: u64,
  pub total_lines: u64,
  pub average_chunks_per_file: f32,
  pub language_breakdown: std::collections::HashMap<String, usize>,
  pub chunk_type_breakdown: std::collections::HashMap<String, usize>,
  pub index_health_score: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
  pub language: String,
  pub files: usize,
  pub chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCallersResult(pub Vec<CodeChunkItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleesResult(pub Vec<CodeChunkItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreResult {
  pub query: String,
  pub results: Vec<ExploreResultItem>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub suggestions: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreResultItem {
  pub id: String,
  pub result_type: String, // "code" | "memory" | "doc"
  pub preview: String,
  pub similarity: f32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub file_path: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub line: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub hints: Option<ExploreHints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreHints {
  pub caller_count: usize,
  pub callee_count: usize,
  pub related_memory_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResult {
  pub items: Vec<ContextItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
  pub id: String,
  pub item_type: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callers: Option<Vec<CodeChunkItem>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub callees: Option<Vec<CodeChunkItem>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub related_memories: Option<Vec<MemorySearchItem>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsSearchResult(pub Vec<DocSearchItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSearchItem {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContextResult {
  pub doc: DocDetail,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub before: Option<Vec<DocSearchItem>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub after: Option<Vec<DocSearchItem>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocDetail {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsIngestResult {
  pub files_processed: usize,
  pub chunks_created: usize,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchStatusResult {
  pub running: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub root: Option<String>,
  pub pending_changes: usize,
  pub project_id: String,
  pub scanning: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scan_progress: Option<[usize; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchStartResult {
  pub status: String,
  pub path: String,
  pub project_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchStopResult {
  pub status: String,
  pub path: String,
  pub project_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityListResult(pub Vec<EntityItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityItem {
  pub id: String,
  pub name: String,
  pub entity_type: String,
  pub mention_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityGetResult {
  pub entity: EntityDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDetail {
  pub id: String,
  pub name: String,
  pub entity_type: String,
  pub mention_count: usize,
  pub relationships: Vec<EntityRelationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
  pub related_entity: String,
  pub relationship_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectsListResult(pub Vec<ProjectInfo>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
  pub path: String,
  pub id: String,
  pub memories: usize,
  pub code_chunks: usize,
  pub documents: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCleanResult {
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookResult {
  #[serde(flatten)]
  pub data: serde_json::Value, // Hook results vary
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownResult {
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
  pub healthy: bool,
  pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
  pub name: String,
  pub status: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeListResult(pub Vec<CodeChunkItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeImportChunkResult {
  pub chunk_id: String,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMemoriesResult(pub Vec<MemorySearchItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelatedResult(pub Vec<CodeChunkItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextFullResult {
  pub chunk: CodeChunkDetail,
  pub callers: Vec<CodeChunkItem>,
  pub callees: Vec<CodeChunkItem>,
  pub related_memories: Vec<MemorySearchItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipAddResult {
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipListResult(pub Vec<RelationshipItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipItem {
  pub from_entity: String,
  pub to_entity: String,
  pub relationship_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipDeleteResult {
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipRelatedResult(pub Vec<EntityItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatsResult {
  pub memories: usize,
  pub code_chunks: usize,
  pub documents: usize,
  pub entities: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrateEmbeddingResult {
  pub migrated: usize,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRestoreResult {
  pub id: String,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryListDeletedResult(pub Vec<MemorySearchItem>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySupersedeResult {
  pub old_id: String,
  pub new_id: String,
  pub message: String,
}

// ============================================================================
// Code tool response types (daemon/tools/code.rs)
// ============================================================================

/// Response for code_index dry_run mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexDryRunResult {
  pub status: String,
  pub files_found: usize,
  pub skipped: usize,
  pub total_bytes: u64,
  pub scan_duration_ms: u64,
}

/// Response for code_index streaming completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexStreamingResult {
  pub status: String,
  pub files_scanned: usize,
  pub files_indexed: u32,
  pub chunks_created: u32,
  pub failed_files: usize,
  pub resumed_from_checkpoint: bool,
  pub scan_duration_ms: u64,
  pub index_duration_ms: u64,
  pub total_duration_ms: u64,
  pub files_per_second: f64,
  pub bytes_processed: u64,
  pub total_bytes: u64,
}

/// Item in code_list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeListItem {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub language: String,
  pub chunk_type: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
  pub file_hash: String,
  pub tokens_estimate: u32,
}

/// Response for code_import_chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeImportResult {
  pub id: String,
  pub status: String,
}

/// Context section for code_context response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextSection {
  pub content: String,
  pub start_line: usize,
  pub end_line: usize,
}

/// Context sections for code_context response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextSections {
  pub before: CodeContextSection,
  pub target: CodeContextSection,
  pub after: CodeContextSection,
}

/// Response for code_context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextResponse {
  pub chunk_id: String,
  pub file_path: String,
  pub language: String,
  pub context: CodeContextSections,
  pub total_file_lines: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub warning: Option<String>,
}

/// Item in code_memories response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMemoryItem {
  pub id: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub memory_type: Option<String>,
  pub sector: String,
  pub salience: f32,
  pub score: f32,
  pub source: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_path: Option<String>,
  pub tags: Vec<String>,
  pub created_at: String,
}

/// Response for code_memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMemoriesResponse {
  pub file_path: String,
  pub memories: Vec<CodeMemoryItem>,
}

/// Item in code_callers response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCallerItem {
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
  pub language: String,
  pub chunk_type: String,
}

/// Response for code_callers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCallersResponse {
  pub symbol: String,
  pub callers: Vec<CodeCallerItem>,
  pub count: usize,
}

/// Item in code_callees response (with call reference)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleeItem {
  pub call: String,
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub end_line: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub language: Option<String>,
}

/// Response for code_callees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCalleesResponse {
  pub chunk_id: String,
  pub calls: Vec<String>,
  pub callees: Vec<CodeCalleeItem>,
  pub unresolved: Vec<String>,
}

/// Item in code_related response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelatedItem {
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
  pub language: String,
  pub chunk_type: String,
  pub score: f32,
  pub relationship: String,
}

/// Response for code_related
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelatedResponse {
  pub chunk_id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub related: Vec<CodeRelatedItem>,
  pub count: usize,
}

/// Chunk detail for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunkFullDetail {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub language: String,
  pub chunk_type: String,
  pub symbols: Vec<String>,
  pub imports: Vec<String>,
  pub calls: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
}

/// Caller item for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFullCaller {
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
  pub end_line: u32,
}

/// Callee item for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFullCallee {
  pub call: String,
  pub id: String,
  pub file_path: String,
  pub symbols: Vec<String>,
  pub start_line: u32,
}

/// Same-file sibling for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFullSibling {
  pub id: String,
  pub symbols: Vec<String>,
  pub chunk_type: String,
  pub start_line: u32,
  pub end_line: u32,
}

/// Memory item for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFullMemory {
  pub id: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub memory_type: Option<String>,
  pub salience: f32,
}

/// Documentation item for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFullDoc {
  pub id: String,
  pub title: String,
  pub content: String,
  pub similarity: f32,
}

/// Response for code_context_full
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContextFullResponse {
  pub chunk: CodeChunkFullDetail,
  pub callers: Vec<CodeFullCaller>,
  pub callees: Vec<CodeFullCallee>,
  pub unresolved_calls: Vec<String>,
  pub same_file: Vec<CodeFullSibling>,
  pub memories: Vec<CodeFullMemory>,
  pub documentation: Vec<CodeFullDoc>,
}

// ============================================================================
// Project metadata (daemon/projects.rs)
// ============================================================================

/// Project metadata stored in project.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadataJson {
  pub id: String,
  pub path: String,
  pub name: String,
}
