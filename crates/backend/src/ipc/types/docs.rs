//! Document IPC types - requests, responses, and conversions
use serde::{Deserialize, Serialize};

use crate::domain::document::DocumentChunk;

// ============================================================================
// Request types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum DocsRequest {
  Search(DocsSearchParams),
  Context(DocContextParams),
  Ingest(DocsIngestParams),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocsIngestParams {
  /// Directory to scan for documents (relative to project root)
  pub directory: Option<String>,
  /// Single file to ingest (can be absolute or relative to project root)
  pub file: Option<String>,
  /// Whether to stream progress updates
  #[serde(default)]
  pub stream: bool,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocsSearchParams {
  pub query: String,
  pub limit: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContextParams {
  pub doc_id: String,
  pub before: Option<usize>,
  pub after: Option<usize>,
}

// ============================================================================
// Response types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "data")]
pub enum DocsResponse {
  Search(Vec<DocSearchItem>),
  GetContext(DocContextResult),
  Ingest(DocsIngestResult),
  IngestFull(DocsIngestFullResult),
}

/// Document search result item
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSearchItem {
  pub id: String,
  pub document_id: String,
  pub title: String,
  pub source: String,
  /// Source type: "file", "url", or "content"
  pub source_type: String,
  pub content: String,
  pub chunk_index: usize,
  pub total_chunks: usize,
  /// Character offset within the original document
  #[serde(skip_serializing_if = "Option::is_none")]
  pub char_offset: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub similarity: Option<f32>,
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
pub struct DocContextResult {
  pub chunk_id: String,
  pub document_id: String,
  pub title: String,
  pub source: String,
  pub context: DocContextSections,
  pub total_chunks: usize,
}

/// Document ingest result (single file)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsIngestResult {
  pub document_id: String,
  pub title: String,
  pub source: String,
  pub source_type: String,
  pub content_hash: String,
  pub char_count: usize,
  pub chunks_created: usize,
  pub total_chunks: usize,
}

/// Full document ingest result (directory ingest with stats)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsIngestFullResult {
  pub status: String,
  pub files_scanned: usize,
  pub files_ingested: usize,
  pub chunks_created: usize,
  pub failed_files: usize,
  pub scan_duration_ms: u64,
  pub ingest_duration_ms: u64,
  pub total_duration_ms: u64,
  pub files_per_second: f64,
  pub bytes_processed: u64,
  pub total_bytes: u64,
  /// Individual file results (empty if too many files)
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub results: Vec<DocsIngestResult>,
}

// ============================================================================
// Conversions from domain types
// ============================================================================

impl From<&DocumentChunk> for DocSearchItem {
  fn from(d: &DocumentChunk) -> Self {
    Self::from_chunk(d, None)
  }
}

impl DocSearchItem {
  pub fn from_chunk(d: &DocumentChunk, similarity: Option<f32>) -> Self {
    Self {
      id: d.id.to_string(),
      document_id: d.document_id.to_string(),
      title: d.title.clone(),
      source: d.source.clone(),
      source_type: d.source_type.as_str().to_string(),
      content: d.content.clone(),
      chunk_index: d.chunk_index,
      total_chunks: d.total_chunks,
      char_offset: Some(d.char_offset),
      similarity,
    }
  }

  pub fn from_search(d: &DocumentChunk, similarity: f32) -> Self {
    Self::from_chunk(d, Some(similarity))
  }
}

impl From<&DocumentChunk> for DocContextChunk {
  fn from(d: &DocumentChunk) -> Self {
    Self {
      chunk_index: d.chunk_index,
      content: d.content.clone(),
    }
  }
}

// ============================================================================
// IpcRequest implementations
// ============================================================================

use crate::{
  impl_ipc_request,
  ipc::{RequestData, ResponseData},
};

impl_ipc_request!(
  DocsSearchParams => Vec<DocSearchItem>,
  ResponseData::Docs(DocsResponse::Search(v)) => v,
  v => RequestData::Docs(DocsRequest::Search(v)),
  v => ResponseData::Docs(DocsResponse::Search(v))
);
impl_ipc_request!(
  DocContextParams => DocContextResult,
  ResponseData::Docs(DocsResponse::GetContext(v)) => v,
  v => RequestData::Docs(DocsRequest::Context(v)),
  v => ResponseData::Docs(DocsResponse::GetContext(v))
);
impl_ipc_request!(
  DocsIngestParams => DocsIngestFullResult,
  ResponseData::Docs(DocsResponse::IngestFull(v)) => v,
  v => RequestData::Docs(DocsRequest::Ingest(v)),
  v => ResponseData::Docs(DocsResponse::IngestFull(v))
);
