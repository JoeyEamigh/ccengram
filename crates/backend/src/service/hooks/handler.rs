//! Hook event handlers.
//!
//! This module provides the main entry points for processing hook events.
//! Handlers are thin adapters that:
//! 1. Parse request parameters
//! 2. Create service context
//! 3. Call service methods
//! 4. Build responses
//!
//! Business logic lives in the service modules (extraction, observation, etc.)

use std::collections::HashSet;

use llm::LlmProvider;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::{
  context::SegmentContext,
  event::HookEvent,
  extraction::{self, ExtractionContext},
  observation::{self, ObservationContext},
};
use crate::{
  db::ProjectDb,
  domain::config::HooksConfig,
  embedding::EmbeddingProvider,
  ipc::types::hook::{
    PostToolUseHookResult, PreCompactHookResult, SessionEndHookResult, SessionStartHookResult, SimpleHookResult,
    StopHookResult, UserPromptHookResult,
  },
  service::util::ServiceError,
};

/// Context for hook handling operations.
///
/// Bundles all dependencies needed for hook processing.
pub struct HookContext<'a> {
  /// Project database connection
  pub db: &'a ProjectDb,
  /// Optional embedding provider
  pub embedding: &'a dyn EmbeddingProvider,
  /// Optional LLM provider for intelligent extraction
  pub llm: Option<&'a dyn LlmProvider>,
  /// Project UUID
  pub project_id: Uuid,
  /// Hooks configuration
  pub config: &'a HooksConfig,
}

impl<'a> HookContext<'a> {
  /// Create a new hook context
  pub fn new(
    db: &'a ProjectDb,
    embedding: &'a dyn EmbeddingProvider,
    llm: Option<&'a dyn LlmProvider>,
    project_id: Uuid,
    config: &'a HooksConfig,
  ) -> Self {
    Self {
      db,
      embedding,
      llm,
      project_id,
      config,
    }
  }

  /// Create an extraction context from this hook context
  fn extraction_context(&self) -> ExtractionContext<'_> {
    ExtractionContext::new(self.db, self.embedding, self.llm, self.project_id)
  }

  /// Create an observation context from this hook context
  fn observation_context(&self) -> ObservationContext<'_> {
    ObservationContext::new(self.db, self.embedding, self.project_id)
  }

  /// Check if hooks are enabled
  fn is_enabled(&self) -> bool {
    self.config.enabled
  }

  /// Check if LLM extraction is enabled
  fn use_llm_extraction(&self) -> bool {
    self.config.llm_extraction && self.llm.is_some()
  }

  /// Check if background extraction is enabled
  fn use_background_extraction(&self) -> bool {
    self.config.background_extraction
  }

  /// Check if tool observations are enabled
  fn tool_observations_enabled(&self) -> bool {
    self.config.tool_observations
  }

  /// Check if high-priority signal detection is enabled
  fn high_priority_signals_enabled(&self) -> bool {
    self.config.high_priority_signals && self.llm.is_some()
  }
}

/// Mutable state passed through hook handlers.
///
/// This is separate from HookContext to allow mutation without
/// requiring &mut access to the entire context.
pub struct HookState {
  /// Session contexts keyed by Claude session ID
  pub session_contexts: std::collections::HashMap<String, SegmentContext>,
  /// Deduplication hash set
  pub seen_hashes: HashSet<String>,
}

impl HookState {
  /// Create new hook state
  pub fn new() -> Self {
    Self {
      session_contexts: std::collections::HashMap::new(),
      seen_hashes: HashSet::new(),
    }
  }

  /// Maximum number of hashes to keep before clearing
  const MAX_SEEN_HASHES: usize = 10_000;

  /// Clear seen hashes if over limit
  pub fn maybe_clear_seen_hashes(&mut self) {
    if self.seen_hashes.len() >= Self::MAX_SEEN_HASHES {
      debug!(
        "Clearing seen_hashes cache (reached {} entries)",
        self.seen_hashes.len()
      );
      self.seen_hashes.clear();
    }
  }
}

impl Default for HookState {
  fn default() -> Self {
    Self::new()
  }
}

// ============================================================================
// Hook Handlers
// ============================================================================

/// Handle SessionStart hook event.
pub async fn handle_session_start(
  ctx: &HookContext<'_>,
  _state: &mut HookState,
  params: &serde_json::Value,
  project_info: SessionStartInfo,
) -> Result<SessionStartHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  info!("Session started: {} in {:?}", session_id, project_info.project_path);

  // Create or resume session in database
  let _session = ctx
    .db
    .get_or_create_session(session_id, ctx.project_id)
    .await
    .map_err(|e| ServiceError::internal(format!("Failed to create session: {}", e)))?;

  Ok(SessionStartHookResult {
    status: "ok".to_string(),
    project_id: project_info.project_id.clone(),
    project_name: project_info.project_name.clone(),
    project_path: project_info.project_path.clone(),
    watcher_started: project_info.watcher_started,
  })
}

/// Info about the project for session start
pub struct SessionStartInfo {
  pub project_id: String,
  pub project_name: String,
  pub project_path: String,
  pub watcher_started: bool,
}

/// Handle SessionEnd hook event.
pub async fn handle_session_end(
  ctx: &HookContext<'_>,
  state: &mut HookState,
  params: &serde_json::Value,
) -> Result<SessionEndHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  let summary = params.get("summary").and_then(|v| v.as_str());

  info!("Session ended: {}", session_id);

  let mut memories_created = Vec::new();
  let mut memories_promoted = 0;

  // Extract memory from session summary if provided
  if ctx.is_enabled()
    && let Some(summary_text) = summary
  {
    let ext_ctx = ctx.extraction_context();
    if let Ok(res) = extraction::extract_memory(&ext_ctx, summary_text, &mut state.seen_hashes).await
      && let Some(id) = res.memory_id
    {
      memories_created.push(id);
    }
  }

  // End the session in the database
  if let Err(e) = ctx.db.end_session(session_id, summary.map(String::from)).await {
    warn!("Failed to end session in database: {}", e);
  }

  // Promote session memories based on usage count (threshold: 2 sessions)
  match ctx.db.promote_session_memories(session_id, 2).await {
    Ok(count) => memories_promoted += count,
    Err(e) => warn!("Failed to promote session memories: {}", e),
  }

  // Also promote high-salience memories (threshold: 0.8)
  match ctx.db.promote_high_salience_memories(session_id, 0.8).await {
    Ok(count) => memories_promoted += count,
    Err(e) => warn!("Failed to promote high-salience memories: {}", e),
  }

  if memories_promoted > 0 {
    debug!(
      session_id = %session_id,
      promoted = memories_promoted,
      "Promoted session memories to project tier"
    );
  }

  // Clean up session context
  state.session_contexts.remove(session_id);
  state.maybe_clear_seen_hashes();

  Ok(SessionEndHookResult {
    status: "ok".to_string(),
    memories_created,
    memories_promoted,
  })
}

/// Handle UserPromptSubmit hook event.
pub async fn handle_user_prompt_submit(
  ctx: &HookContext<'_>,
  state: &mut HookState,
  params: &serde_json::Value,
) -> Result<UserPromptHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  let prompt = params.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

  debug!(session_id = %session_id, prompt_len = prompt.len(), "User prompt received");

  let mut memories_created = Vec::new();

  // Extract from previous segment if meaningful work was done
  if let Some(segment_ctx) = state.session_contexts.get_mut(session_id) {
    if ctx.is_enabled() && segment_ctx.has_meaningful_work() {
      if ctx.use_llm_extraction() {
        let ext_ctx = ctx.extraction_context();
        if let Ok(ids) = extraction::extract_with_llm(&ext_ctx, segment_ctx, &mut state.seen_hashes).await {
          memories_created.extend(ids);
        }
      } else if let Some(summary) = segment_ctx.summary() {
        let ext_ctx = ctx.extraction_context();
        if let Ok(res) = extraction::extract_memory(&ext_ctx, &summary, &mut state.seen_hashes).await
          && let Some(id) = res.memory_id
        {
          memories_created.push(id);
        }
      }
    }
    // Reset for new segment
    segment_ctx.reset();
    segment_ctx.user_prompt = Some(prompt.to_string());
  } else {
    // New session context
    state
      .session_contexts
      .insert(session_id.to_string(), SegmentContext::with_prompt(prompt.to_string()));
  }

  // Check for high-priority signals (corrections/preferences)
  if ctx.is_enabled()
    && ctx.high_priority_signals_enabled()
    && !prompt.is_empty()
    && prompt.len() >= 20
    && let Some(llm) = ctx.llm
    && let Ok(classification) = extraction::classify_signal(llm, prompt).await
    && classification.category.is_high_priority()
    && classification.is_extractable
  {
    let ext_ctx = ctx.extraction_context();
    if let Ok(ids) = extraction::extract_high_priority(&ext_ctx, prompt, &classification, &mut state.seen_hashes).await
    {
      memories_created.extend(ids);
    }
  }

  state.maybe_clear_seen_hashes();

  Ok(UserPromptHookResult {
    status: "ok".to_string(),
    memories_created,
  })
}

/// Handle PostToolUse hook event.
pub async fn handle_post_tool_use(
  ctx: &HookContext<'_>,
  state: &mut HookState,
  params: &serde_json::Value,
) -> Result<PostToolUseHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  let tool_name = params.get("tool_name").and_then(|v| v.as_str()).unwrap_or("unknown");
  let tool_params = params
    .get("tool_input")
    .cloned()
    .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
  let tool_result = params.get("tool_response"); // Claude Code sends "tool_response", not "tool_result"

  debug!(session_id = %session_id, tool = %tool_name, "Tool use recorded");

  let mut observation_memory_id = None;

  // Create tool observation memory if enabled
  if ctx.is_enabled() && ctx.tool_observations_enabled() {
    let obs_ctx = ctx.observation_context();
    if let Ok(res) =
      observation::create_tool_observation(&obs_ctx, tool_name, &tool_params, tool_result, &mut state.seen_hashes).await
    {
      observation_memory_id = res.memory_id;
    }
  }

  // Accumulate tool use data in session context
  let segment_ctx = state.session_contexts.entry(session_id.to_string()).or_default();

  // Parse typed tool use from hook event
  let tool_use = llm::ToolUse::from_hook_event(tool_name, &tool_params, tool_result);

  // Track derived data for quick access (files, commands, searches, tasks)
  if let Some(path) = tool_use.file_path()
    && !path.is_empty()
  {
    if tool_use.is_file_modification() {
      segment_ctx.record_file_modified(path);
    } else if tool_use.is_file_read() {
      segment_ctx.record_file_read(path);
    }
  }

  if let Some((cmd, exit_code)) = tool_use.command_info()
    && !cmd.is_empty()
  {
    segment_ctx.record_command(cmd.to_string(), exit_code);
  }

  if let Some(pattern) = tool_use.search_pattern()
    && !pattern.is_empty()
  {
    segment_ctx.record_search(pattern);
  }

  if let Some(tasks) = tool_use.completed_tasks() {
    for task in tasks {
      segment_ctx.record_completed_task(task);
    }
  }

  segment_ctx.record_tool_use(tool_use);

  // Check for todo completion trigger: ≥3 tasks completed AND ≥5 tool calls
  let should_trigger = segment_ctx.completed_tasks.len() >= 3 && segment_ctx.tool_call_count() >= 5;

  if should_trigger && ctx.is_enabled() && ctx.use_llm_extraction() {
    debug!(
      "Todo completion trigger: extracting memories for session {}",
      session_id
    );
    let ext_ctx = ctx.extraction_context();
    if let Ok(_ids) = extraction::extract_with_llm(&ext_ctx, segment_ctx, &mut state.seen_hashes).await {
      // Memories stored from todo_completion trigger
    }
  }

  state.maybe_clear_seen_hashes();

  Ok(PostToolUseHookResult {
    status: "ok".to_string(),
    observation_memory_id,
  })
}

/// Handle PreCompact hook event.
pub async fn handle_pre_compact(
  ctx: &HookContext<'_>,
  state: &mut HookState,
  params: &serde_json::Value,
) -> Result<PreCompactHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  let summary = params.get("summary").and_then(|v| v.as_str());

  debug!(session_id = %session_id, has_summary = summary.is_some(), "Pre-compact trigger");

  let mut memories_created = Vec::new();

  // Extract from current segment before compaction
  if let Some(segment_ctx) = state.session_contexts.get_mut(session_id) {
    if ctx.is_enabled() && segment_ctx.has_meaningful_work() {
      if ctx.use_llm_extraction() {
        let ext_ctx = ctx.extraction_context();
        match extraction::extract_with_llm(&ext_ctx, segment_ctx, &mut state.seen_hashes).await {
          Ok(ids) => memories_created.extend(ids),
          Err(e) => {
            warn!("LLM extraction failed in pre-compact: {}", e);
            // Fallback to basic summary
            if let Some(ctx_summary) = segment_ctx.summary() {
              let ext_ctx = ctx.extraction_context();
              if let Ok(res) = extraction::extract_memory(&ext_ctx, &ctx_summary, &mut state.seen_hashes).await
                && let Some(id) = res.memory_id
              {
                memories_created.push(id);
              }
            }
          }
        }
      } else if let Some(ctx_summary) = segment_ctx.summary() {
        let ext_ctx = ctx.extraction_context();
        if let Ok(res) = extraction::extract_memory(&ext_ctx, &ctx_summary, &mut state.seen_hashes).await
          && let Some(id) = res.memory_id
        {
          memories_created.push(id);
        }
      }
    }
    segment_ctx.reset();
  }

  // Also extract from provided summary if any
  if ctx.is_enabled()
    && let Some(summary_text) = summary
  {
    let ext_ctx = ctx.extraction_context();
    if let Ok(res) = extraction::extract_memory(&ext_ctx, summary_text, &mut state.seen_hashes).await
      && let Some(id) = res.memory_id
    {
      memories_created.push(id);
    }
  }

  state.maybe_clear_seen_hashes();

  Ok(PreCompactHookResult {
    status: "ok".to_string(),
    background_extraction: ctx.use_background_extraction(),
    memories_created,
  })
}

/// Handle Stop hook event.
pub async fn handle_stop(
  ctx: &HookContext<'_>,
  state: &mut HookState,
  params: &serde_json::Value,
) -> Result<StopHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  let summary = params.get("summary").and_then(|v| v.as_str());

  debug!(session_id = %session_id, has_summary = summary.is_some(), "Stop event");

  let mut memories_created = Vec::new();

  // Final extraction from accumulated context
  if let Some(segment_ctx) = state.session_contexts.remove(session_id)
    && ctx.is_enabled()
    && segment_ctx.has_meaningful_work()
  {
    if ctx.use_llm_extraction() {
      let ext_ctx = ctx.extraction_context();
      match extraction::extract_with_llm(&ext_ctx, &segment_ctx, &mut state.seen_hashes).await {
        Ok(ids) => memories_created.extend(ids),
        Err(e) => {
          warn!("LLM extraction failed: {}", e);
          // Fallback to basic summary
          if let Some(ctx_summary) = segment_ctx.summary() {
            let ext_ctx = ctx.extraction_context();
            if let Ok(res) = extraction::extract_memory(&ext_ctx, &ctx_summary, &mut state.seen_hashes).await
              && let Some(id) = res.memory_id
            {
              memories_created.push(id);
            }
          }
        }
      }
    } else if let Some(ctx_summary) = segment_ctx.summary() {
      let ext_ctx = ctx.extraction_context();
      if let Ok(res) = extraction::extract_memory(&ext_ctx, &ctx_summary, &mut state.seen_hashes).await
        && let Some(id) = res.memory_id
      {
        memories_created.push(id);
      }
    }
  }

  // Extract from provided summary
  if ctx.is_enabled()
    && let Some(summary_text) = summary
  {
    let ext_ctx = ctx.extraction_context();
    if let Ok(res) = extraction::extract_memory(&ext_ctx, summary_text, &mut state.seen_hashes).await
      && let Some(id) = res.memory_id
    {
      memories_created.push(id);
    }
  }

  state.maybe_clear_seen_hashes();

  Ok(StopHookResult {
    status: "ok".to_string(),
    background_extraction: ctx.use_background_extraction(),
    memories_created,
  })
}

/// Handle SubagentStop hook event.
pub async fn handle_subagent_stop(
  _ctx: &HookContext<'_>,
  _state: &mut HookState,
  params: &serde_json::Value,
) -> Result<SimpleHookResult, ServiceError> {
  let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
  debug!(session_id = %session_id, "Subagent stop");

  Ok(SimpleHookResult {
    status: "ok".to_string(),
  })
}

/// Handle Notification hook event.
pub async fn handle_notification(
  _ctx: &HookContext<'_>,
  _state: &mut HookState,
  params: &serde_json::Value,
) -> Result<SimpleHookResult, ServiceError> {
  let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
  debug!("Notification: {}", message);

  Ok(SimpleHookResult {
    status: "ok".to_string(),
  })
}

/// Dispatch a hook event to the appropriate handler.
///
/// This is the main entry point for processing hook events.
///
/// # Arguments
/// * `ctx` - Hook context with dependencies
/// * `state` - Mutable hook state
/// * `event` - The hook event type
/// * `params` - Event parameters as JSON
/// * `session_info` - Optional session start info (only for SessionStart)
///
/// # Returns
/// * `Ok(serde_json::Value)` - The serialized result
/// * `Err(ServiceError)` - If handling fails
pub async fn dispatch(
  ctx: &HookContext<'_>,
  state: &mut HookState,
  event: HookEvent,
  params: &serde_json::Value,
  session_info: Option<SessionStartInfo>,
) -> Result<serde_json::Value, ServiceError> {
  match event {
    HookEvent::SessionStart => {
      let info = session_info.ok_or_else(|| ServiceError::validation("SessionStart requires session_info"))?;
      let result = handle_session_start(ctx, state, params, info).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::SessionEnd => {
      let result = handle_session_end(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::UserPromptSubmit => {
      let result = handle_user_prompt_submit(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::PostToolUse => {
      let result = handle_post_tool_use(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::PreCompact => {
      let result = handle_pre_compact(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::Stop => {
      let result = handle_stop(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::SubagentStop => {
      let result = handle_subagent_stop(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
    HookEvent::Notification => {
      let result = handle_notification(ctx, state, params).await?;
      serde_json::to_value(result).map_err(|e| ServiceError::validation(e.to_string()))
    }
  }
}
