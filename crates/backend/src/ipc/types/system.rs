//! System IPC types - daemon status, metrics, health checks, and system operations
use serde::{Deserialize, Serialize};

// ============================================================================
// Request/Response enums (matches pattern of other domains)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "params")]
pub enum SystemRequest {
  Ping(PingParams),
  HealthCheck(HealthCheckParams),
  Metrics(MetricsParams),
  Shutdown(ShutdownParams),
  Status(StatusParams),
  ProjectStats(ProjectStatsParams),
  MigrateEmbedding(MigrateEmbeddingParams),
  Resolve(ResolveParams),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum SystemResponse {
  Ping(String),
  HealthCheck(HealthCheckResult),
  Metrics(MetricsResult),
  Shutdown { message: String },
  Status(StatusResult),
  ProjectStats(super::project::ProjectStatsResult),
  MigrateEmbedding(MigrateEmbeddingResult),
  Resolve(ResolveResult),
}

// ============================================================================
// Request param types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PingParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthCheckParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShutdownParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatusParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectStatsParams;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MigrateEmbeddingParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveParams {
  pub id: String,
}

// ============================================================================
// Status result
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResult {
  pub status: String,
  pub version: String,
  pub pid: u32,
  pub projects: usize,
  pub active_sessions: usize,
  pub idle_seconds: u64,
  pub uptime_seconds: u64,
  pub foreground: bool,
  pub auto_shutdown: bool,
}

// ============================================================================
// Metrics result
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResult {
  pub daemon: DaemonMetrics,
  pub requests: RequestsMetrics,
  pub sessions: SessionsMetrics,
  pub projects: ProjectsMetrics,
  pub embedding: Option<EmbeddingProviderInfo>,
  pub memory: MemoryUsageMetrics,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonMetrics {
  pub version: String,
  pub uptime_seconds: u64,
  pub idle_seconds: u64,
  pub foreground: bool,
  pub auto_shutdown: bool,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestsMetrics {
  pub total: u64,
  pub per_second: f64,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionsMetrics {
  pub active: usize,
  pub ids: Vec<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectsMetrics {
  pub count: usize,
  pub names: Vec<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProviderInfo {
  pub name: String,
  pub model: String,
  pub dimensions: usize,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageMetrics {
  pub rss_kb: Option<u64>,
}

// ============================================================================
// Health check result
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
  pub healthy: bool,
  pub checks: Vec<HealthCheck>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
  pub name: String,
  pub status: String,
  pub message: Option<String>,
}

// ============================================================================
// Migration result
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrateEmbeddingResult {
  pub migrated: usize,
  pub message: String,
}

// ============================================================================
// Resolve result
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveResult {
  pub id: String,
  pub entity_type: String,
}

// ============================================================================
// IpcRequest implementations
// ============================================================================

use crate::{
  impl_ipc_request,
  ipc::{RequestData, ResponseData},
};

impl_ipc_request!(
  PingParams => String,
  ResponseData::System(SystemResponse::Ping(v)) => v,
  v => RequestData::System(SystemRequest::Ping(v)),
  v => ResponseData::System(SystemResponse::Ping(v))
);
impl_ipc_request!(
  HealthCheckParams => HealthCheckResult,
  ResponseData::System(SystemResponse::HealthCheck(v)) => v,
  v => RequestData::System(SystemRequest::HealthCheck(v)),
  v => ResponseData::System(SystemResponse::HealthCheck(v))
);
impl_ipc_request!(
  MetricsParams => MetricsResult,
  ResponseData::System(SystemResponse::Metrics(v)) => v,
  v => RequestData::System(SystemRequest::Metrics(v)),
  v => ResponseData::System(SystemResponse::Metrics(v))
);
impl_ipc_request!(
  ShutdownParams => String,
  ResponseData::System(SystemResponse::Shutdown { message }) => message,
  v => RequestData::System(SystemRequest::Shutdown(v))
);
impl_ipc_request!(
  StatusParams => StatusResult,
  ResponseData::System(SystemResponse::Status(v)) => v,
  v => RequestData::System(SystemRequest::Status(v)),
  v => ResponseData::System(SystemResponse::Status(v))
);
impl_ipc_request!(
  ProjectStatsParams => super::project::ProjectStatsResult,
  ResponseData::System(SystemResponse::ProjectStats(v)) => v,
  v => RequestData::System(SystemRequest::ProjectStats(v)),
  v => ResponseData::System(SystemResponse::ProjectStats(v))
);
impl_ipc_request!(
  MigrateEmbeddingParams => MigrateEmbeddingResult,
  ResponseData::System(SystemResponse::MigrateEmbedding(v)) => v,
  v => RequestData::System(SystemRequest::MigrateEmbedding(v)),
  v => ResponseData::System(SystemResponse::MigrateEmbedding(v))
);
impl_ipc_request!(
  ResolveParams => ResolveResult,
  ResponseData::System(SystemResponse::Resolve(v)) => v,
  v => RequestData::System(SystemRequest::Resolve(v)),
  v => ResponseData::System(SystemResponse::Resolve(v))
);
