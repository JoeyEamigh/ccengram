use crate::activity_tracker::ActivityTracker;
use crate::hooks::{HookEvent, HookHandler};
use crate::projects::ProjectRegistry;
use crate::server::{ProgressSender, ShutdownHandle};
use crate::session_tracker::SessionTracker;
use crate::tools::ToolHandler;
use embedding::EmbeddingProvider;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// Import typed result/params from IPC for type-safe responses
pub use ipc::{
  CodeCalleesResult, CodeCallersResult, CodeChunkDetail, CodeChunkItem, CodeContextFullResult, CodeContextResult,
  CodeImportChunkResult, CodeIndexResult, CodeListResult, CodeMemoriesResult, CodeRelatedResult, CodeSearchResult,
  CodeStatsResult, ContextItem, ContextResult, DocContextResult, DocDetail, DocSearchItem, DocsIngestResult,
  DocsSearchResult, EntityDetail, EntityGetResult, EntityItem, EntityListResult, EntityRelationship, ExploreHints,
  ExploreResult, ExploreResultItem, HealthCheck, HealthCheckResult, HookResult, LanguageStats, MemoryAddResult,
  MemoryDeleteResult, MemoryDetail, MemoryGetResult, MemoryListDeletedResult, MemoryListResult, MemoryRelatedResult,
  MemoryRestoreResult, MemorySearchItem, MemorySearchResult, MemorySupersedeResult, MemoryTimelineResult,
  MemoryUpdateResult, MigrateEmbeddingResult, PingResult, ProjectCleanResult, ProjectInfo as IpcProjectInfo,
  ProjectStatsResult, ProjectsListResult, RelationshipAddResult, RelationshipDeleteResult, RelationshipItem,
  RelationshipListResult, RelationshipRelatedResult, ShutdownResult, StatusResult, TimelineEntry, WatchStartResult,
  WatchStatusResult, WatchStopResult,
};

// ============================================================================
// Local result types for daemon-specific responses
// These types match the current daemon output structure where IPC types differ
// ============================================================================

/// Metrics response - detailed daemon metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
  pub daemon: DaemonInfo,
  pub requests: RequestsInfo,
  pub sessions: SessionsInfo,
  pub projects: ProjectsInfo,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub embedding: Option<EmbeddingInfo>,
  pub memory: MemoryInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonInfo {
  pub version: String,
  pub uptime_seconds: u64,
  pub idle_seconds: u64,
  pub foreground: bool,
  pub auto_shutdown: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestsInfo {
  pub total: u64,
  pub per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionsInfo {
  pub active: usize,
  pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectsInfo {
  pub count: usize,
  pub names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingInfo {
  pub name: String,
  pub model: String,
  pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub rss_kb: Option<u64>,
}

/// Project list item for projects_list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectListItem {
  pub id: String,
  pub path: String,
  pub name: String,
}

/// Project info response for project_info
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

/// Project clean response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCleanResponse {
  pub path: String,
  pub memories_deleted: usize,
  pub code_chunks_deleted: usize,
  pub documents_deleted: usize,
}

/// Projects clean all response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectsCleanAllResult {
  pub projects_removed: usize,
}

// ============================================================================
// Hook result types - used by hooks.rs
// ============================================================================

/// Result from SessionStart hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStartHookResult {
  pub status: String,
  pub project_id: String,
  pub project_name: String,
  pub project_path: String,
  pub watcher_started: bool,
}

/// Result from SessionEnd hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEndHookResult {
  pub status: String,
  pub memories_created: Vec<String>,
  pub memories_promoted: usize,
}

/// Result from UserPromptSubmit hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPromptHookResult {
  pub status: String,
  pub memories_created: Vec<String>,
}

/// Result from PostToolUse hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostToolUseHookResult {
  pub status: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub observation_memory_id: Option<String>,
}

/// Result from PreCompact hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreCompactHookResult {
  pub status: String,
  pub background_extraction: bool,
  pub memories_created: Vec<String>,
}

/// Result from Stop hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopHookResult {
  pub status: String,
  pub background_extraction: bool,
  pub memories_created: Vec<String>,
}

/// Simple status-only hook result (SubagentStop, Notification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleHookResult {
  pub status: String,
}

// ============================================================================
// System tool result types
// ============================================================================

/// Database health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbHealthStatus {
  pub status: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub wal_mode: Option<bool>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub error: Option<String>,
}

/// Ollama health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaHealthStatus {
  pub available: bool,
  pub models_count: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub configured_model: Option<String>,
  pub configured_model_available: bool,
}

/// Embedding provider status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingHealthStatus {
  pub configured: bool,
  pub provider: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub model: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub dimensions: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub available: Option<bool>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context_length: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_batch_size: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub warning: Option<String>,
}

/// Full health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullHealthCheckResult {
  pub database: DbHealthStatus,
  pub ollama: OllamaHealthStatus,
  pub embedding: EmbeddingHealthStatus,
}

/// Migrate embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrateEmbeddingResponse {
  pub migrated_count: u64,
  pub skipped_count: u64,
  pub error_count: u64,
  pub duration_ms: u64,
  pub target_dimensions: usize,
}

// ============================================================================
// Document tool result types
// ============================================================================

/// Document search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSearchResultItem {
  pub id: String,
  pub document_id: String,
  pub title: String,
  pub source: String,
  pub content: String,
  pub chunk_index: usize,
  pub total_chunks: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub similarity: Option<f32>,
}

/// Document ingest response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocIngestResult {
  pub document_id: String,
  pub title: String,
  pub source: String,
  pub source_type: String,
  pub content_hash: String,
  pub char_count: usize,
  pub chunks_created: usize,
  pub total_chunks: usize,
}

/// Document context chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContextChunk {
  pub chunk_index: usize,
  pub content: String,
}

/// Document context sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContextSections {
  pub before: Vec<DocContextChunk>,
  pub target: DocContextChunk,
  pub after: Vec<DocContextChunk>,
}

/// Document context response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContextResponse {
  pub chunk_id: String,
  pub document_id: String,
  pub title: String,
  pub source: String,
  pub context: DocContextSections,
  pub total_chunks: usize,
}

// ============================================================================
// Entity tool result types
// ============================================================================

/// Entity list item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityListItem {
  pub id: String,
  pub name: String,
  #[serde(rename = "type")]
  pub entity_type: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  pub aliases: Vec<String>,
  pub mention_count: u32,
  pub first_seen_at: String,
  pub last_seen_at: String,
}

/// Entity memory link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMemoryLink {
  pub memory_id: String,
  pub role: String,
  pub confidence: f32,
}

/// Full entity with optional memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFullResult {
  pub id: String,
  pub name: String,
  #[serde(rename = "type")]
  pub entity_type: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  pub aliases: Vec<String>,
  pub mention_count: u32,
  pub first_seen_at: String,
  pub last_seen_at: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub memories: Option<Vec<EntityMemoryLink>>,
}

/// Top entity item (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopEntityItem {
  pub id: String,
  pub name: String,
  #[serde(rename = "type")]
  pub entity_type: String,
  pub mention_count: u32,
}

/// Relationship result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipResult {
  pub id: String,
  pub from_memory_id: String,
  pub to_memory_id: String,
  pub relationship_type: String,
  pub confidence: f32,
}

/// Relationship list item (with dates)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipListItem {
  pub id: String,
  pub from_memory_id: String,
  pub to_memory_id: String,
  pub relationship_type: String,
  pub confidence: f32,
  pub created_at: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub valid_until: Option<String>,
}

/// Delete result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletedResult {
  pub deleted: bool,
}

/// Memory summary (for related memories)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
  pub id: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  pub sector: String,
  pub salience: f32,
}

/// Relationship info (for related memories)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipInfo {
  #[serde(rename = "type")]
  pub relationship_type: String,
  pub confidence: f32,
  pub direction: String,
}

/// Related memory with relationship info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedMemoryItem {
  pub memory: MemorySummary,
  pub relationship: RelationshipInfo,
}

// ============================================================================
// Memory tool result types (additional)
// ============================================================================

/// Memory session info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySessionInfo {
  pub id: String,
  pub started_at: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub ended_at: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
}

/// Memory timeline item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimelineItem {
  pub id: String,
  pub content: String,
  pub sector: String,
  pub salience: f32,
  pub created_at: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub session_id: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub session: Option<MemorySessionInfo>,
}

/// Memory timeline response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimelineResponse {
  pub anchor: MemoryTimelineItem,
  pub before: Vec<MemoryTimelineItem>,
  pub after: Vec<MemoryTimelineItem>,
}

/// Memory relationship in get response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationshipItem {
  #[serde(rename = "type")]
  pub relationship_type: String,
  pub from_id: String,
  pub to_id: String,
  pub target_id: String,
  pub confidence: f32,
}

/// Full memory detail response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFullDetail {
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
  pub confidence: f32,
  pub access_count: u32,
  pub is_deleted: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub superseded_by: Option<String>,
  pub tags: Vec<String>,
  pub categories: Vec<String>,
  pub concepts: Vec<String>,
  pub files: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_path: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_module: Option<String>,
  pub created_at: String,
  pub updated_at: String,
  pub last_accessed: String,
  pub valid_from: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub valid_until: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub relationships: Option<Vec<MemoryRelationshipItem>>,
}

/// Related memory search item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelatedItem {
  pub id: String,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub summary: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub memory_type: Option<String>,
  pub sector: String,
  pub salience: f32,
  pub score: f32,
  pub relationship: String,
  pub created_at: String,
}

/// Memory related response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelatedResponse {
  pub memory_id: String,
  pub content: String,
  pub related: Vec<MemoryRelatedItem>,
  pub count: usize,
}

// ============================================================================
// Code tool result types (additional)
// ============================================================================

/// Code search query info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQueryInfo {
  pub original: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub expanded: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub intent: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub search_mode: Option<String>,
}

/// Code search response - generic over result type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSearchResponse<T: Serialize> {
  pub results: Vec<T>,
  pub query_info: CodeQueryInfo,
}

/// Code index response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexResponse {
  pub status: String,
  pub files_scanned: usize,
  pub files_indexed: usize,
  pub chunks_created: usize,
  pub failed_files: usize,
  pub resumed_from_checkpoint: bool,
  pub scan_duration_ms: u64,
  pub index_duration_ms: u64,
  pub total_duration_ms: u64,
  pub files_per_second: f64,
  pub bytes_processed: u64,
  pub total_bytes: u64,
}

/// Code index file info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexFileInfo {
  pub path: String,
  pub language: String,
  pub chunks: usize,
  pub skipped: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reason: Option<String>,
}

/// Code index dry run response (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexDryRunResponse {
  pub status: String,
  pub files_found: usize,
  pub skipped: usize,
  pub total_bytes: u64,
  pub scan_duration_ms: u64,
}

/// JSON-RPC style request (wire format - supports any JSON id and string method)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
  #[serde(default)]
  pub id: Option<serde_json::Value>,
  pub method: String,
  #[serde(default)]
  pub params: serde_json::Value,
}

/// JSON-RPC style response (wire format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub id: Option<serde_json::Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub result: Option<serde_json::Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub error: Option<RpcError>,
  /// Progress update for streaming responses.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub progress: Option<IndexProgress>,
}

/// Progress information for long-running operations like indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexProgress {
  pub phase: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub total_files: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub processed_files: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub chunks_created: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub current_file: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub bytes_processed: Option<u64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub total_bytes: Option<u64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
  pub code: i32,
  pub message: String,
}

impl Response {
  /// Create a success response with a typed result (serializes to JSON value)
  pub fn success<T: Serialize>(id: Option<serde_json::Value>, result: T) -> Self {
    Self {
      id,
      result: Some(serde_json::to_value(result).unwrap_or(serde_json::Value::Null)),
      error: None,
      progress: None,
    }
  }

  pub fn error(id: Option<serde_json::Value>, code: i32, message: &str) -> Self {
    Self {
      id,
      result: None,
      error: Some(RpcError {
        code,
        message: message.to_string(),
      }),
      progress: None,
    }
  }

  /// Create a progress event (intermediate response in a stream)
  pub fn progress(id: Option<serde_json::Value>, progress: IndexProgress) -> Self {
    Self {
      id,
      result: None,
      error: None,
      progress: Some(progress),
    }
  }
}

impl IndexProgress {
  pub fn scanning(scanned: u32, current_file: Option<String>) -> Self {
    Self {
      phase: "scanning".to_string(),
      total_files: None,
      processed_files: Some(scanned),
      chunks_created: None,
      current_file,
      bytes_processed: None,
      total_bytes: None,
      message: Some(format!("Scanning... {} files found", scanned)),
    }
  }

  pub fn indexing(
    processed: u32,
    total: u32,
    chunks: u32,
    current_file: Option<String>,
    bytes_processed: u64,
    total_bytes: u64,
  ) -> Self {
    let percent = if total > 0 { (processed * 100) / total } else { 0 };
    Self {
      phase: "indexing".to_string(),
      total_files: Some(total),
      processed_files: Some(processed),
      chunks_created: Some(chunks),
      current_file,
      bytes_processed: Some(bytes_processed),
      total_bytes: Some(total_bytes),
      message: Some(format!("Indexing... {}% ({}/{})", percent, processed, total)),
    }
  }

  pub fn complete(files: u32, chunks: u32) -> Self {
    Self {
      phase: "complete".to_string(),
      total_files: Some(files),
      processed_files: Some(files),
      chunks_created: Some(chunks),
      current_file: None,
      bytes_processed: None,
      total_bytes: None,
      message: Some(format!("Complete: {} files, {} chunks", files, chunks)),
    }
  }
}

/// Request router for the daemon
pub struct Router {
  registry: Arc<ProjectRegistry>,
  tool_handler: Arc<ToolHandler>,
  hook_handler: Arc<HookHandler>,
  shutdown_handle: Arc<Mutex<Option<ShutdownHandle>>>,
  /// Session tracker for lifecycle management
  session_tracker: Arc<Mutex<Option<Arc<SessionTracker>>>>,
  /// Activity tracker for idle detection
  activity_tracker: Arc<Mutex<Option<Arc<ActivityTracker>>>>,
  /// Whether daemon is in foreground mode
  foreground: Arc<Mutex<bool>>,
  /// Embedding provider reference for metrics
  embedding_provider: Arc<Mutex<Option<Arc<dyn EmbeddingProvider>>>>,
  /// Total requests handled (for metrics)
  request_count: AtomicU64,
}

impl Router {
  pub fn new() -> Self {
    let registry = Arc::new(ProjectRegistry::new());
    let tool_handler = Arc::new(ToolHandler::new(Arc::clone(&registry)));
    let hook_handler = Arc::new(HookHandler::new(Arc::clone(&registry)));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(None)),
      request_count: AtomicU64::new(0),
    }
  }

  pub fn with_registry(registry: Arc<ProjectRegistry>) -> Self {
    let tool_handler = Arc::new(ToolHandler::new(Arc::clone(&registry)));
    let hook_handler = Arc::new(HookHandler::new(Arc::clone(&registry)));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(None)),
      request_count: AtomicU64::new(0),
    }
  }

  pub fn with_embedding(registry: Arc<ProjectRegistry>, embedding: Arc<dyn EmbeddingProvider>) -> Self {
    let tool_handler = Arc::new(ToolHandler::with_embedding(
      Arc::clone(&registry),
      Arc::clone(&embedding),
    ));
    let hook_handler = Arc::new(HookHandler::with_embedding(
      Arc::clone(&registry),
      Arc::clone(&embedding),
    ));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(Some(embedding))),
      request_count: AtomicU64::new(0),
    }
  }

  /// Create a router with embedding provider and configuration
  pub fn with_embedding_and_config(
    registry: Arc<ProjectRegistry>,
    embedding: Arc<dyn EmbeddingProvider>,
    hooks_config: &engram_core::HooksConfig,
    embedding_config: &engram_core::EmbeddingConfig,
  ) -> Self {
    let tool_handler = Arc::new(ToolHandler::with_embedding_and_config(
      Arc::clone(&registry),
      Arc::clone(&embedding),
      embedding_config.clone(),
    ));
    let hook_handler =
      Arc::new(HookHandler::with_embedding(Arc::clone(&registry), Arc::clone(&embedding)).with_config(hooks_config));

    Self {
      registry,
      tool_handler,
      hook_handler,
      shutdown_handle: Arc::new(Mutex::new(None)),
      session_tracker: Arc::new(Mutex::new(None)),
      activity_tracker: Arc::new(Mutex::new(None)),
      foreground: Arc::new(Mutex::new(false)),
      embedding_provider: Arc::new(Mutex::new(Some(embedding))),
      request_count: AtomicU64::new(0),
    }
  }

  /// Set the shutdown handle (called after server is created)
  pub async fn set_shutdown_handle(&self, handle: ShutdownHandle) {
    let mut guard = self.shutdown_handle.lock().await;
    *guard = Some(handle);
  }

  /// Set the session tracker for lifecycle management
  pub async fn set_session_tracker(&self, tracker: Arc<SessionTracker>) {
    let mut guard = self.session_tracker.lock().await;
    *guard = Some(tracker.clone());
    // Also pass to hook handler
    self.hook_handler.set_session_tracker(tracker).await;
  }

  /// Set the activity tracker for idle detection
  pub async fn set_activity_tracker(&self, tracker: Arc<ActivityTracker>) {
    let mut guard = self.activity_tracker.lock().await;
    *guard = Some(tracker);
  }

  /// Set foreground mode flag
  pub async fn set_foreground(&self, foreground: bool) {
    let mut guard = self.foreground.lock().await;
    *guard = foreground;
  }

  /// Get the project registry (for testing/cleanup)
  pub fn registry(&self) -> &Arc<ProjectRegistry> {
    &self.registry
  }

  /// Handle an incoming request
  pub async fn handle(&self, request: Request) -> Response {
    let start = std::time::Instant::now();
    let method = request.method.clone();
    let id = request.id.clone();
    debug!(method = %method, id = ?id, "Handling request");

    // Increment request counter
    self.request_count.fetch_add(1, Ordering::Relaxed);

    // Touch activity tracker on every request
    {
      let guard = self.activity_tracker.lock().await;
      if let Some(ref tracker) = *guard {
        tracker.touch();
      }
    }

    let response = match request.method.as_str() {
      // Health/meta commands
      "ping" => Response::success(request.id, PingResult("pong".to_string())),
      "status" => self.handle_status(request).await,
      "metrics" => self.handle_metrics(request).await,
      "shutdown" => self.handle_shutdown(request).await,

      // Memory tools
      "memory_search" => self.tool_handler.memory_search(request).await,
      "memory_get" => self.tool_handler.memory_get(request).await,
      "memory_list" => self.tool_handler.memory_list(request).await,
      "memory_add" => self.tool_handler.memory_add(request).await,
      "memory_reinforce" => self.tool_handler.memory_reinforce(request).await,
      "memory_deemphasize" => self.tool_handler.memory_deemphasize(request).await,
      "memory_delete" => self.tool_handler.memory_delete(request).await,
      "memory_supersede" => self.tool_handler.memory_supersede(request).await,
      "memory_timeline" => self.tool_handler.memory_timeline(request).await,
      "memory_related" => self.tool_handler.memory_related(request).await,

      // Code tools
      "code_search" => self.tool_handler.code_search(request).await,
      "code_context" => self.tool_handler.code_context(request).await,
      "code_index" => self.tool_handler.code_index(request).await,
      "code_list" => self.tool_handler.code_list(request).await,
      "code_import_chunk" => self.tool_handler.code_import_chunk(request).await,
      "code_stats" => self.tool_handler.code_stats(request).await,
      "code_memories" => self.tool_handler.code_memories(request).await,
      "code_callers" => self.tool_handler.code_callers(request).await,
      "code_callees" => self.tool_handler.code_callees(request).await,
      "code_related" => self.tool_handler.code_related(request).await,
      "code_context_full" => self.tool_handler.code_context_full(request).await,

      // Watch tools
      "watch_start" => self.tool_handler.watch_start(request).await,
      "watch_stop" => self.tool_handler.watch_stop(request).await,
      "watch_status" => self.tool_handler.watch_status(request).await,

      // Document tools
      "docs_search" => self.tool_handler.docs_search(request).await,
      "doc_context" => self.tool_handler.doc_context(request).await,
      "docs_ingest" => self.tool_handler.docs_ingest(request).await,

      // Entity tools
      "entity_list" => self.tool_handler.entity_list(request).await,
      "entity_get" => self.tool_handler.entity_get(request).await,
      "entity_top" => self.tool_handler.entity_top(request).await,

      // Relationship tools
      "relationship_add" => self.tool_handler.relationship_add(request).await,
      "relationship_list" => self.tool_handler.relationship_list(request).await,
      "relationship_delete" => self.tool_handler.relationship_delete(request).await,
      "relationship_related" => self.tool_handler.relationship_related(request).await,

      // Statistics & Health
      "project_stats" => self.tool_handler.project_stats(request).await,
      "health_check" => self.tool_handler.health_check(request).await,

      // Unified exploration tools (new)
      "explore" => self.tool_handler.explore(request).await,
      "context" => self.tool_handler.context(request).await,

      // Migration
      "migrate_embedding" => self.tool_handler.migrate_embedding(request).await,

      // Memory restore/deleted
      "memory_restore" => self.tool_handler.memory_restore(request).await,
      "memory_list_deleted" => self.tool_handler.memory_list_deleted(request).await,

      // Project management
      "projects_list" => self.handle_projects_list(request).await,
      "project_info" => self.handle_project_info(request).await,
      "project_clean" => self.handle_project_clean(request).await,
      "projects_clean_all" => self.handle_projects_clean_all(request).await,

      // Hook events
      "hook" => self.handle_hook(request).await,

      // Unknown method
      _ => {
        warn!(method = %request.method, "Unknown method requested");
        Response::error(request.id, -32601, &format!("Method not found: {}", request.method))
      }
    };

    let elapsed = start.elapsed();
    debug!(
      method = %method,
      id = ?id,
      elapsed_ms = elapsed.as_millis() as u64,
      "Request completed"
    );

    response
  }

  /// Handle a streaming request that sends progress updates
  pub async fn handle_streaming(&self, request: Request, progress_tx: ProgressSender) {
    let start = std::time::Instant::now();
    let method = request.method.clone();
    let id = request.id.clone();
    debug!(method = %method, id = ?id, "Handling streaming request");

    // Increment request counter
    self.request_count.fetch_add(1, Ordering::Relaxed);

    // Touch activity tracker
    {
      let guard = self.activity_tracker.lock().await;
      if let Some(ref tracker) = *guard {
        tracker.touch();
      }
    }

    match request.method.as_str() {
      // Streaming-enabled methods
      "code_index" => {
        self.tool_handler.code_index_streaming(request, progress_tx).await;
      }

      // All other methods fall back to single response
      _ => {
        let response = self.handle(request).await;
        let _ = progress_tx.send(response).await;
      }
    }

    let elapsed = start.elapsed();
    debug!(
      method = %method,
      elapsed_ms = elapsed.as_millis() as u64,
      "Streaming request completed"
    );
  }

  async fn handle_status(&self, request: Request) -> Response {
    let projects = self.registry.list().await;

    // Get session count
    let active_sessions = {
      let guard = self.session_tracker.lock().await;
      match &*guard {
        Some(tracker) => tracker.active_count().await,
        None => 0,
      }
    };

    // Get activity info
    let (idle_seconds, uptime_seconds) = {
      let guard = self.activity_tracker.lock().await;
      match &*guard {
        Some(tracker) => (tracker.idle_duration().as_secs(), tracker.uptime().as_secs()),
        None => (0, 0),
      }
    };

    // Get foreground mode
    let foreground = *self.foreground.lock().await;

    let status = StatusResult {
      status: "running".to_string(),
      version: env!("CARGO_PKG_VERSION").to_string(),
      projects: projects.len(),
      active_sessions,
      idle_seconds,
      uptime_seconds,
      foreground,
      auto_shutdown: !foreground,
    };
    Response::success(request.id, status)
  }

  /// Handle metrics request - returns detailed daemon metrics for monitoring
  async fn handle_metrics(&self, request: Request) -> Response {
    let projects = self.registry.list().await;

    // Get session details
    let (active_sessions, session_list) = {
      let guard = self.session_tracker.lock().await;
      match &*guard {
        Some(tracker) => {
          let sessions = tracker.list_sessions().await;
          let session_ids: Vec<String> = sessions.iter().map(|s| s.0.clone()).collect();
          (sessions.len(), session_ids)
        }
        None => (0, vec![]),
      }
    };

    // Get activity info
    let (idle_seconds, uptime_seconds) = {
      let guard = self.activity_tracker.lock().await;
      match &*guard {
        Some(tracker) => (tracker.idle_duration().as_secs(), tracker.uptime().as_secs()),
        None => (0, 0),
      }
    };

    // Get request count
    let total_requests = self.request_count.load(Ordering::Relaxed);

    // Get foreground mode
    let foreground = *self.foreground.lock().await;

    // Get embedding provider info
    let embedding_info = {
      let guard = self.embedding_provider.lock().await;
      (*guard).as_ref().map(|provider| EmbeddingInfo {
        name: provider.name().to_string(),
        model: provider.model_id().to_string(),
        dimensions: provider.dimensions(),
      })
    };

    // Get process memory (if available on Linux)
    let memory_kb = Self::get_process_memory_kb();

    let metrics = MetricsResponse {
      daemon: DaemonInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds,
        idle_seconds,
        foreground,
        auto_shutdown: !foreground,
      },
      requests: RequestsInfo {
        total: total_requests,
        per_second: if uptime_seconds > 0 {
          total_requests as f64 / uptime_seconds as f64
        } else {
          0.0
        },
      },
      sessions: SessionsInfo {
        active: active_sessions,
        ids: session_list,
      },
      projects: ProjectsInfo {
        count: projects.len(),
        names: projects.iter().map(|p| p.name.clone()).collect(),
      },
      embedding: embedding_info,
      memory: MemoryInfo { rss_kb: memory_kb },
    };

    Response::success(request.id, metrics)
  }

  /// Get process RSS memory in KB (Linux only, returns None on other platforms)
  fn get_process_memory_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
      // Read /proc/self/statm
      if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        let parts: Vec<&str> = statm.split_whitespace().collect();
        if parts.len() >= 2 {
          // Second field is RSS in pages, page size is typically 4KB
          if let Ok(rss_pages) = parts[1].parse::<u64>() {
            return Some(rss_pages * 4); // Convert to KB
          }
        }
      }
      None
    }
    #[cfg(not(target_os = "linux"))]
    {
      None
    }
  }

  async fn handle_shutdown(&self, request: Request) -> Response {
    info!("Shutdown requested via RPC");
    let guard = self.shutdown_handle.lock().await;
    if let Some(ref handle) = *guard {
      handle.shutdown();
      Response::success(
        request.id,
        ShutdownResult {
          message: "shutting_down".to_string(),
        },
      )
    } else {
      Response::error(request.id, -32000, "Shutdown handle not available")
    }
  }

  /// List all projects in the registry
  async fn handle_projects_list(&self, request: Request) -> Response {
    let projects = self.registry.list().await;

    let project_list: Vec<ProjectListItem> = projects
      .iter()
      .map(|p| ProjectListItem {
        id: p.id.as_str().to_string(),
        path: p.path.to_string_lossy().to_string(),
        name: p.name.clone(),
      })
      .collect();

    Response::success(request.id, project_list)
  }

  /// Get detailed info for a specific project
  async fn handle_project_info(&self, request: Request) -> Response {
    let project_identifier = request.params.get("project").and_then(|v| v.as_str()).unwrap_or("");

    if project_identifier.is_empty() {
      return Response::error(request.id, -32602, "Missing project parameter");
    }

    // Try to find the project by path or ID
    let path = std::path::Path::new(project_identifier);
    let result = if path.exists() {
      self.registry.get_or_create(path).await
    } else {
      // Try to find by ID prefix in the registry
      let projects = self.registry.list().await;
      let found = projects.iter().find(|p| {
        p.id.as_str().starts_with(project_identifier) || p.path.to_string_lossy().contains(project_identifier)
      });
      match found {
        Some(p) => self.registry.get_or_create(&p.path).await,
        None => {
          return Response::error(
            request.id,
            -32000,
            &format!("Project not found: {}", project_identifier),
          );
        }
      }
    };

    match result {
      Ok((info, db)) => {
        // Get statistics
        let memory_count = db.count_memories(Some("is_deleted = false")).await.unwrap_or(0);
        let code_chunk_count = db.count_code_chunks(None).await.unwrap_or(0);
        let document_count = db.count_document_chunks(None).await.unwrap_or(0);

        // Get project UUID for session count
        let project_uuid = uuid::Uuid::parse_str(info.id.as_str()).unwrap_or_else(|_| uuid::Uuid::nil());
        let session_count = db.count_sessions(&project_uuid).await.unwrap_or(0);

        Response::success(
          request.id,
          ProjectInfoResult {
            id: info.id.as_str().to_string(),
            path: info.path.to_string_lossy().to_string(),
            name: info.name.clone(),
            memory_count,
            code_chunk_count,
            document_count,
            session_count,
            db_path: db.path.to_string_lossy().to_string(),
          },
        )
      }
      Err(e) => Response::error(request.id, -32000, &format!("Failed to get project info: {}", e)),
    }
  }

  /// Clean (remove) a specific project's data
  async fn handle_project_clean(&self, request: Request) -> Response {
    let project_identifier = request.params.get("project").and_then(|v| v.as_str()).unwrap_or("");

    if project_identifier.is_empty() {
      return Response::error(request.id, -32602, "Missing project parameter");
    }

    // Find the project
    let path = std::path::Path::new(project_identifier);
    let project_info = if path.exists() {
      match crate::projects::ProjectInfo::from_path(path) {
        Ok(info) => info,
        Err(e) => return Response::error(request.id, -32000, &format!("Invalid project path: {}", e)),
      }
    } else {
      // Try to find by ID prefix
      let projects = self.registry.list().await;
      match projects.iter().find(|p| {
        p.id.as_str().starts_with(project_identifier) || p.path.to_string_lossy().contains(project_identifier)
      }) {
        Some(p) => p.clone(),
        None => {
          return Response::error(
            request.id,
            -32000,
            &format!("Project not found: {}", project_identifier),
          );
        }
      }
    };

    // Get counts before deletion for reporting
    let counts = match self.registry.get_or_create(&project_info.path).await {
      Ok((_, db)) => {
        let memories = db.count_memories(None).await.unwrap_or(0);
        let code_chunks = db.count_code_chunks(None).await.unwrap_or(0);
        let documents = db.count_document_chunks(None).await.unwrap_or(0);
        (memories, code_chunks, documents)
      }
      Err(_) => (0, 0, 0),
    };

    // Close the project connection
    self.registry.close(project_info.id.as_str()).await;

    // Delete the project data directory
    let data_dir = project_info.id.data_dir(self.registry.data_dir());
    if data_dir.exists()
      && let Err(e) = std::fs::remove_dir_all(&data_dir)
    {
      return Response::error(request.id, -32000, &format!("Failed to remove project data: {}", e));
    }

    Response::success(
      request.id,
      ProjectCleanResponse {
        path: project_info.path.to_string_lossy().to_string(),
        memories_deleted: counts.0,
        code_chunks_deleted: counts.1,
        documents_deleted: counts.2,
      },
    )
  }

  /// Clean all projects
  async fn handle_projects_clean_all(&self, request: Request) -> Response {
    let projects = self.registry.list().await;
    let count = projects.len();

    // Close all connections first
    self.registry.close_all().await;

    // Remove all project data directories
    let data_dir = self.registry.data_dir();
    let projects_dir = data_dir.join("projects");
    if projects_dir.exists()
      && let Err(e) = std::fs::remove_dir_all(&projects_dir)
    {
      return Response::error(
        request.id,
        -32000,
        &format!("Failed to remove projects directory: {}", e),
      );
    }

    Response::success(
      request.id,
      ProjectsCleanAllResult {
        projects_removed: count,
      },
    )
  }

  async fn handle_hook(&self, request: Request) -> Response {
    let event_str = request
      .params
      .get("event")
      .and_then(|v| v.as_str())
      .unwrap_or("unknown");
    let session_id = request
      .params
      .get("params")
      .and_then(|p| p.get("session_id"))
      .and_then(|v| v.as_str())
      .unwrap_or("unknown");
    debug!(event = %event_str, session_id = %session_id, "Routing hook event");

    // Parse the event type
    let event: HookEvent = match event_str.parse() {
      Ok(e) => e,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid hook event: {}", e)),
    };

    // Get the params for the hook
    let params = request
      .params
      .get("params")
      .cloned()
      .unwrap_or_else(|| serde_json::Value::Object(Default::default()));

    // Delegate to hook handler
    match self.hook_handler.handle(event, params).await {
      Ok(result) => Response::success(request.id, result),
      Err(e) => Response::error(request.id, -32000, &format!("Hook error: {}", e)),
    }
  }
}

impl Default for Router {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ipc::{Method, MetricsParams, PingParams};

  /// Helper to create a wire-format Request from typed IPC params
  fn make_request<P: serde::Serialize>(id: u64, method: Method, params: P) -> Request {
    Request {
      id: Some(serde_json::Value::Number(id.into())),
      method: serde_json::to_value(method)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default(),
      params: serde_json::to_value(params).unwrap_or_default(),
    }
  }

  #[test]
  fn test_response_success() {
    let response = Response::success(Some(serde_json::Value::Number(1.into())), "test");
    assert!(response.result.is_some());
    assert!(response.error.is_none());
  }

  #[test]
  fn test_response_error() {
    let response = Response::error(Some(serde_json::Value::Number(1.into())), -1, "test error");
    assert!(response.result.is_none());
    assert!(response.error.is_some());
    assert_eq!(response.error.as_ref().unwrap().code, -1);
  }

  #[tokio::test]
  async fn test_ping() {
    let router = Router::new();
    let request = make_request(1, Method::Ping, PingParams);

    let response = router.handle(request).await;
    assert!(response.result.is_some());

    // Deserialize to typed result
    let result: ipc::PingResult = serde_json::from_value(response.result.unwrap()).unwrap();
    assert_eq!(result.0, "pong");
  }

  #[tokio::test]
  async fn test_unknown_method() {
    let router = Router::new();
    // Test with invalid method string (not using typed Method enum since we want to test error path)
    let request = Request {
      id: Some(serde_json::Value::Number(1.into())),
      method: "unknown_method".to_string(),
      params: serde_json::Value::Object(Default::default()),
    };

    let response = router.handle(request).await;
    assert!(response.error.is_some());
    assert_eq!(response.error.as_ref().unwrap().code, -32601);
  }

  #[tokio::test]
  async fn test_metrics() {
    let router = Router::new();

    // Make a few requests to increment the counter
    for _ in 0..3 {
      let request = make_request(1, Method::Ping, PingParams);
      router.handle(request).await;
    }

    // Now request metrics
    let request = make_request(1, Method::Metrics, MetricsParams);

    let response = router.handle(request).await;
    assert!(response.result.is_some());

    // Deserialize to typed MetricsResponse
    let result: MetricsResponse = serde_json::from_value(response.result.unwrap()).unwrap();

    // Check daemon info
    assert!(!result.daemon.version.is_empty());
    // uptime_seconds is u64 and always >= 0

    // Check requests info - 4 total requests: 3 pings + 1 metrics
    assert_eq!(result.requests.total, 4);

    // Check sessions info
    assert_eq!(result.sessions.active, 0);

    // Check projects info
    assert_eq!(result.projects.count, 0);
  }
}
