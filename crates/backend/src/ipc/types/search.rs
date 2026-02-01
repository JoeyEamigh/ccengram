use serde::{Deserialize, Serialize};

use crate::{
  impl_ipc_request,
  ipc::{RequestData, ResponseData},
};

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExploreParams {
  pub query: String,
  pub scope: Option<String>, // "code" | "memory" | "docs" | "all"
  pub expand_top: Option<usize>,
  pub limit: Option<usize>,
  pub depth: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextParams {
  pub id: Option<String>,
  pub ids: Option<Vec<String>>,
  pub depth: Option<usize>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreResult {
  pub query: String,
  pub results: Vec<ExploreResultItem>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResult(pub Vec<ContextItem>);

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreResultItem {
  pub id: String,
  pub result_type: String, // "code" | "memory" | "doc"
  pub preview: String,
  pub similarity: f32,
  pub file_path: Option<String>,
  pub line: Option<u32>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub symbols: Vec<String>,
  pub hints: Option<ExploreHints>,
  pub context: Option<ExploreContext>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreHints {
  pub caller_count: usize,
  pub callee_count: usize,
  pub related_memory_count: usize,
}

/// Expanded context for explore results (callers, callees, siblings)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreContext {
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub callers: Vec<ExploreCallInfo>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub callees: Vec<ExploreCallInfo>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub siblings: Vec<ExploreSiblingInfo>,
}

/// Caller/callee info for expanded context
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreCallInfo {
  pub id: String,
  pub file: String,
  pub start_line: u32,
  pub end_line: u32,
  pub preview: String,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub symbols: Vec<String>,
  pub signature: Option<String>,
}

/// Sibling symbol info
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreSiblingInfo {
  pub symbol: String,
  pub kind: String,
  pub line: u32,
  pub file: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
  pub id: String,
  pub item_type: String,
  pub content: String,
  pub callers: Option<Vec<super::code::CodeItem>>,
  pub callees: Option<Vec<super::code::CodeItem>>,
  pub related_memories: Option<Vec<super::memory::MemoryItem>>,
}

impl_ipc_request!(
  ExploreParams => ExploreResult,
  ResponseData::Explore(v) => v,
  v => RequestData::Explore(v),
  v => ResponseData::Explore(v)
);
impl_ipc_request!(
  ContextParams => Vec<ContextItem>,
  ResponseData::Context(v) => v,
  v => RequestData::Context(v),
  v => ResponseData::Context(v)
);
