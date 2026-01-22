use crate::projects::ProjectRegistry;
use embedding::EmbeddingProvider;
use engram_core::{Memory, MemoryType, Sector, resolve_project_path};
use extract::{classify_sector, compute_hashes, extract_concepts, extract_files};
use llm::{ExtractedMemory, classify_signal, extract_high_priority, extract_memories};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum HookError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("JSON error: {0}")]
  Json(#[from] serde_json::Error),
  #[error("Unknown hook: {0}")]
  UnknownHook(String),
  #[error("Project error: {0}")]
  Project(#[from] crate::projects::ProjectError),
}

/// Hook event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum HookEvent {
  SessionStart,
  SessionEnd,
  UserPromptSubmit,
  PostToolUse,
  PreCompact,
  Stop,
  SubagentStop,
  Notification,
}

impl std::str::FromStr for HookEvent {
  type Err = HookError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "SessionStart" => Ok(Self::SessionStart),
      "SessionEnd" => Ok(Self::SessionEnd),
      "UserPromptSubmit" => Ok(Self::UserPromptSubmit),
      "PostToolUse" => Ok(Self::PostToolUse),
      "PreCompact" => Ok(Self::PreCompact),
      "Stop" => Ok(Self::Stop),
      "SubagentStop" => Ok(Self::SubagentStop),
      "Notification" => Ok(Self::Notification),
      _ => Err(HookError::UnknownHook(s.to_string())),
    }
  }
}

/// Extract file paths from tool parameters
fn extract_tool_file_paths(params: &serde_json::Value) -> Vec<String> {
  let mut files = Vec::new();

  // Common file path keys
  for key in ["file_path", "notebook_path", "path", "source"] {
    if let Some(path) = params.get(key).and_then(|v| v.as_str())
      && !path.is_empty()
      && !files.contains(&path.to_string())
    {
      files.push(path.to_string());
    }
  }

  files
}

/// Accumulated context from a session segment
#[derive(Debug, Default, Clone)]
struct SegmentContext {
  /// All tool uses in this segment
  tool_uses: Vec<ToolUseRecord>,
  /// The user's original prompt
  user_prompt: Option<String>,
  /// Files that were read (paths)
  files_read: Vec<String>,
  /// Files that were modified (paths)
  files_modified: Vec<String>,
  /// Commands run with exit codes
  commands_run: Vec<(String, i32)>,
  /// Errors encountered
  errors_encountered: Vec<String>,
  /// Search patterns executed
  searches_performed: Vec<String>,
  /// Tasks completed (from TodoWrite)
  completed_tasks: Vec<String>,
  /// Last assistant message (if captured)
  last_assistant_message: Option<String>,
}

impl SegmentContext {
  /// Total tool call count in this segment
  fn tool_call_count(&self) -> usize {
    self.tool_uses.len()
  }

  /// Check if this segment has meaningful work to extract
  fn has_meaningful_work(&self) -> bool {
    // At least 3 tool calls OR file modifications OR completed tasks
    self.tool_call_count() >= 3
      || !self.files_modified.is_empty()
      || !self.completed_tasks.is_empty()
      || !self.errors_encountered.is_empty()
  }

  /// Convert to LLM extraction context
  fn to_extraction_context(&self) -> llm::ExtractionContext {
    llm::ExtractionContext {
      user_prompt: self.user_prompt.clone(),
      files_read: self.files_read.clone(),
      files_modified: self.files_modified.clone(),
      commands_run: self.commands_run.clone(),
      errors_encountered: self.errors_encountered.clone(),
      searches_performed: self.searches_performed.clone(),
      completed_tasks: self.completed_tasks.clone(),
      last_assistant_message: self.last_assistant_message.clone(),
      tool_call_count: self.tool_call_count(),
    }
  }

  /// Generate summary text from accumulated context (fallback for non-LLM extraction)
  fn summary(&self) -> Option<String> {
    if self.tool_uses.is_empty() && self.files_read.is_empty() && self.files_modified.is_empty() {
      return None;
    }

    let mut parts = Vec::new();

    if let Some(ref prompt) = self.user_prompt {
      parts.push(format!("User request: {}", prompt));
    }

    if !self.files_modified.is_empty() {
      parts.push(format!("Modified: {}", self.files_modified.join(", ")));
    }

    if !self.files_read.is_empty() {
      let read_count = self.files_read.len();
      if read_count <= 3 {
        parts.push(format!("Read: {}", self.files_read.join(", ")));
      } else {
        parts.push(format!("Read {} files", read_count));
      }
    }

    if !self.commands_run.is_empty() {
      let cmds: Vec<_> = self
        .commands_run
        .iter()
        .take(3)
        .map(|(cmd, code)| format!("{} (exit {})", cmd, code))
        .collect();
      parts.push(format!("Commands: {}", cmds.join(", ")));
    }

    if !self.completed_tasks.is_empty() {
      parts.push(format!("Completed: {}", self.completed_tasks.join(", ")));
    }

    if !self.errors_encountered.is_empty() {
      parts.push(format!("Errors: {}", self.errors_encountered.join("; ")));
    }

    let tools_used: Vec<_> = self
      .tool_uses
      .iter()
      .map(|t| t.tool_name.as_str())
      .collect::<std::collections::HashSet<_>>()
      .into_iter()
      .collect();
    if !tools_used.is_empty() {
      parts.push(format!("Tools: {}", tools_used.join(", ")));
    }

    if parts.is_empty() { None } else { Some(parts.join(". ")) }
  }

  /// Reset the context for a new segment
  fn reset(&mut self) {
    self.tool_uses.clear();
    self.user_prompt = None;
    self.files_read.clear();
    self.files_modified.clear();
    self.commands_run.clear();
    self.errors_encountered.clear();
    self.searches_performed.clear();
    self.completed_tasks.clear();
    self.last_assistant_message = None;
  }
}

/// Record of a tool use for context accumulation
#[derive(Debug, Clone)]
struct ToolUseRecord {
  tool_name: String,
  #[allow(dead_code)] // Preserved for future detailed extraction
  params: serde_json::Value,
}

/// Maximum number of hashes to keep in the dedup cache before clearing
const MAX_SEEN_HASHES: usize = 10_000;

/// Hook handler for processing Claude Code hook events
pub struct HookHandler {
  registry: Arc<ProjectRegistry>,
  embedding: Option<Arc<dyn EmbeddingProvider>>,
  /// Current session context, keyed by session_id
  session_contexts: RwLock<HashMap<String, SegmentContext>>,
  /// Simple hash-based deduplication (content hashes seen recently)
  /// Cleared when it exceeds MAX_SEEN_HASHES to prevent unbounded growth
  seen_hashes: RwLock<HashSet<String>>,
  /// Whether to use LLM extraction (true) or basic summary fallback (false)
  use_llm_extraction: bool,
  /// Session to project path binding - ensures directory changes don't switch projects
  session_projects: RwLock<HashMap<String, PathBuf>>,
  /// Whether to use background extraction (non-blocking) for PreCompact/Stop triggers
  use_background_extraction: bool,
}

impl HookHandler {
  pub fn new(registry: Arc<ProjectRegistry>) -> Self {
    Self {
      registry,
      embedding: None,
      session_contexts: RwLock::new(HashMap::new()),
      seen_hashes: RwLock::new(HashSet::new()),
      use_llm_extraction: true,
      session_projects: RwLock::new(HashMap::new()),
      use_background_extraction: true, // Default to background extraction
    }
  }

  pub fn with_embedding(registry: Arc<ProjectRegistry>, embedding: Arc<dyn EmbeddingProvider>) -> Self {
    Self {
      registry,
      embedding: Some(embedding),
      session_contexts: RwLock::new(HashMap::new()),
      seen_hashes: RwLock::new(HashSet::new()),
      use_llm_extraction: true,
      session_projects: RwLock::new(HashMap::new()),
      use_background_extraction: true, // Default to background extraction
    }
  }

  /// Get the project path for a session, using stored binding if available
  async fn get_session_project_path(&self, session_id: &str, cwd: &str) -> PathBuf {
    // Check if we already have a bound project path for this session
    {
      let bindings = self.session_projects.read().await;
      if let Some(path) = bindings.get(session_id) {
        return path.clone();
      }
    }

    // No binding yet - resolve the project path (with git root detection)
    resolve_project_path(&PathBuf::from(cwd))
  }

  /// Bind a session to a project path (called on session start)
  async fn bind_session_project(&self, session_id: &str, cwd: &str) -> PathBuf {
    let project_path = resolve_project_path(&PathBuf::from(cwd));

    {
      let mut bindings = self.session_projects.write().await;
      bindings.insert(session_id.to_string(), project_path.clone());
    }

    debug!("Bound session {} to project path {:?}", session_id, project_path);
    project_path
  }

  /// Unbind a session from its project (called on session end)
  async fn unbind_session_project(&self, session_id: &str) {
    let mut bindings = self.session_projects.write().await;
    bindings.remove(session_id);
  }

  /// Set whether to use LLM extraction (true) or basic summary fallback (false)
  pub fn with_llm_extraction(mut self, enabled: bool) -> Self {
    self.use_llm_extraction = enabled;
    self
  }

  /// Check if a watcher is running for a project, and start one if not
  ///
  /// This implements the auto-start behavior from TypeScript's `maybeAutoStartWatcher()`
  async fn maybe_auto_start_watcher(&self, project_id: &str, project_path: &std::path::Path) -> bool {
    // Check if watcher is already running
    let status = self.registry.watcher_status(project_id).await;
    if status.running {
      debug!("Watcher already running for project {}", project_id);
      return false;
    }

    // Start the watcher (with embedding if available)
    match self
      .registry
      .start_watcher(project_id, project_path, self.embedding.clone())
      .await
    {
      Ok(()) => {
        info!("Auto-started watcher for project {} at {:?}", project_id, project_path);
        true
      }
      Err(e) => {
        warn!("Failed to auto-start watcher for project {}: {}", project_id, e);
        false
      }
    }
  }

  /// Create an episodic memory from a tool observation
  ///
  /// This captures significant tool uses as immediate episodic memories for the "tool trail"
  async fn create_tool_observation_memory(
    &self,
    tool_name: &str,
    tool_params: &serde_json::Value,
    tool_result: Option<&serde_json::Value>,
    cwd: &str,
  ) -> Result<Option<String>, HookError> {
    // Format the observation based on tool type
    let observation = match tool_name {
      "Read" => {
        let Some(path) = tool_params.get("file_path").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        format!("Read file: {}", path)
      }
      "Edit" => {
        let Some(path) = tool_params.get("file_path").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        let old_str = tool_params.get("old_string").and_then(|v| v.as_str()).unwrap_or("");
        let new_str = tool_params.get("new_string").and_then(|v| v.as_str()).unwrap_or("");
        let old_preview = if old_str.len() > 50 { &old_str[..50] } else { old_str };
        let new_preview = if new_str.len() > 50 { &new_str[..50] } else { new_str };
        format!("Edited {}: '{}...' -> '{}...'", path, old_preview, new_preview)
      }
      "Write" => {
        let Some(path) = tool_params.get("file_path").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        format!("Created/wrote file: {}", path)
      }
      "Bash" => {
        let Some(cmd) = tool_params.get("command").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        let exit_code = tool_result
          .and_then(|r| r.get("exit_code"))
          .and_then(|v| v.as_i64())
          .unwrap_or(0);
        let cmd_preview = if cmd.len() > 80 {
          format!("{}...", &cmd[..80])
        } else {
          cmd.to_string()
        };
        if exit_code == 0 {
          format!("Ran command: {}", cmd_preview)
        } else {
          format!("Command failed (exit {}): {}", exit_code, cmd_preview)
        }
      }
      "Grep" | "Glob" => {
        let Some(pattern) = tool_params.get("pattern").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        let path = tool_params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        format!("Searched '{}' in {}", pattern, path)
      }
      "WebFetch" => {
        let Some(url) = tool_params.get("url").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        format!("Fetched URL: {}", url)
      }
      "WebSearch" => {
        let Some(query) = tool_params.get("query").and_then(|v| v.as_str()) else {
          return Ok(None);
        };
        format!("Web search: {}", query)
      }
      // Skip tools that don't produce meaningful observations
      "TodoWrite" | "AskUserQuestion" | "Task" | "Skill" | "EnterPlanMode" | "ExitPlanMode" => {
        return Ok(None);
      }
      _ => {
        // Generic observation for other tools
        format!("Used tool: {}", tool_name)
      }
    };

    // Skip if observation is too short
    if observation.len() < 15 {
      return Ok(None);
    }

    // Compute hashes for dedup
    let (content_hash, simhash) = compute_hashes(&observation);

    // Check for duplicates using simple hash set
    {
      let seen = self.seen_hashes.read().await;
      if seen.contains(&content_hash) {
        debug!(
          "Skipping duplicate tool observation: {}",
          &observation[..observation.len().min(50)]
        );
        return Ok(None);
      }
    }

    // Get or create project
    let project_path = PathBuf::from(cwd);
    let (_info, db) = self.registry.get_or_create(&project_path).await?;

    // Create episodic memory (tool trail memories are always episodic)
    let project_uuid = uuid::Uuid::new_v4();
    let mut memory = Memory::new(project_uuid, observation, Sector::Episodic);
    memory.memory_type = None; // Tool observations don't have a specific memory type
    memory.importance = 0.3; // Lower importance for tool observations
    memory.salience = 0.4; // Medium salience - they decay faster
    memory.content_hash = content_hash.clone();
    memory.simhash = simhash;

    // Extract file paths from tool params
    let files = extract_tool_file_paths(tool_params);
    if !files.is_empty() {
      memory.files = files;
    }

    // Get embedding
    let embedding = self.get_embedding(&memory.content).await;

    // Store the memory
    db.add_memory(&memory, embedding.as_deref())
      .await
      .map_err(|e| HookError::Io(std::io::Error::other(e.to_string())))?;

    // Add to seen hashes, clearing if over limit to prevent unbounded growth
    {
      let mut seen = self.seen_hashes.write().await;
      if seen.len() >= MAX_SEEN_HASHES {
        debug!("Clearing seen_hashes cache (reached {} entries)", seen.len());
        seen.clear();
      }
      seen.insert(content_hash);
    }

    debug!(
      "Created tool observation memory: {} ({})",
      memory.id,
      &memory.content[..memory.content.len().min(50)]
    );
    Ok(Some(memory.id.to_string()))
  }

  /// Get embedding for content
  async fn get_embedding(&self, text: &str) -> Option<Vec<f32>> {
    if let Some(ref provider) = self.embedding {
      match provider.embed(text).await {
        Ok(vec) => Some(vec),
        Err(e) => {
          warn!("Embedding failed: {}", e);
          None
        }
      }
    } else {
      None
    }
  }

  /// Extract and store a memory from content
  async fn extract_memory(&self, content: &str, cwd: &str, _session_id: &str) -> Result<Option<String>, HookError> {
    // Skip if content is too short
    if content.len() < 20 {
      return Ok(None);
    }

    // Compute hashes for dedup
    let (content_hash, simhash) = compute_hashes(content);

    // Check for duplicates using simple hash set
    {
      let seen = self.seen_hashes.read().await;
      if seen.contains(&content_hash) {
        debug!("Skipping duplicate memory (exact hash match)");
        return Ok(None);
      }
    }

    // Get project
    let project_path = PathBuf::from(cwd);
    let (info, db) = self.registry.get_or_create(&project_path).await?;
    let project_uuid = uuid::Uuid::parse_str(info.id.as_str()).unwrap_or_else(|_| uuid::Uuid::new_v4());

    // Classify sector
    let sector = classify_sector(content);

    // Create memory
    let mut memory = Memory::new(project_uuid, content.to_string(), sector);
    memory.content_hash = content_hash.clone();
    memory.simhash = simhash;
    memory.concepts = extract_concepts(content);
    memory.files = extract_files(content);

    // Generate embedding
    let vector = match self.get_embedding(content).await {
      Some(v) => v,
      None => vec![0.0f32; db.vector_dim],
    };

    // Store memory
    db.add_memory(&memory, Some(&vector))
      .await
      .map_err(|e| HookError::Io(std::io::Error::other(e.to_string())))?;

    // Add to seen hashes, clearing if over limit to prevent unbounded growth
    {
      let mut seen = self.seen_hashes.write().await;
      if seen.len() >= MAX_SEEN_HASHES {
        debug!("Clearing seen_hashes cache (reached {} entries)", seen.len());
        seen.clear();
      }
      seen.insert(content_hash);
    }

    info!("Extracted memory: {} ({:?})", memory.id, sector);
    Ok(Some(memory.id.to_string()))
  }

  /// Store an extracted memory from LLM extraction
  async fn store_extracted_memory(&self, extracted: &ExtractedMemory, cwd: &str) -> Result<Option<String>, HookError> {
    // Skip if content is too short
    if extracted.content.len() < 20 {
      return Ok(None);
    }

    // Compute hashes for dedup
    let (content_hash, simhash) = compute_hashes(&extracted.content);

    // Check for duplicates
    {
      let seen = self.seen_hashes.read().await;
      if seen.contains(&content_hash) {
        debug!("Skipping duplicate extracted memory (exact hash match)");
        return Ok(None);
      }
    }

    // Get project
    let project_path = PathBuf::from(cwd);
    let (info, db) = self.registry.get_or_create(&project_path).await?;
    let project_uuid = uuid::Uuid::parse_str(info.id.as_str()).unwrap_or_else(|_| uuid::Uuid::new_v4());

    // Parse sector from extracted memory
    let sector = extracted
      .sector
      .as_ref()
      .and_then(|s| s.parse::<Sector>().ok())
      .unwrap_or_else(|| classify_sector(&extracted.content));

    // Parse memory type
    let memory_type = extracted.memory_type.parse::<MemoryType>().ok();

    // Create memory
    let mut memory = Memory::new(project_uuid, extracted.content.clone(), sector);
    memory.content_hash = content_hash.clone();
    memory.simhash = simhash;
    memory.concepts = extract_concepts(&extracted.content);
    memory.files = extract_files(&extracted.content);
    memory.tags = extracted.tags.clone();
    memory.salience = extracted.confidence;
    memory.memory_type = memory_type;
    if let Some(ref summary) = extracted.summary {
      memory.summary = Some(summary.clone());
    }

    // Generate embedding
    let vector = match self.get_embedding(&extracted.content).await {
      Some(v) => v,
      None => vec![0.0f32; db.vector_dim],
    };

    // Store memory
    db.add_memory(&memory, Some(&vector))
      .await
      .map_err(|e| HookError::Io(std::io::Error::other(e.to_string())))?;

    // Add to seen hashes, clearing if over limit to prevent unbounded growth
    {
      let mut seen = self.seen_hashes.write().await;
      if seen.len() >= MAX_SEEN_HASHES {
        debug!("Clearing seen_hashes cache (reached {} entries)", seen.len());
        seen.clear();
      }
      seen.insert(content_hash);
    }

    info!(
      "Stored LLM-extracted memory: {} ({:?}, {:?}, confidence: {:.2})",
      memory.id, sector, memory.memory_type, extracted.confidence
    );
    Ok(Some(memory.id.to_string()))
  }

  /// Extract memories using LLM from segment context
  async fn extract_with_llm(&self, ctx: &SegmentContext, cwd: &str) -> Result<Vec<String>, HookError> {
    if !ctx.has_meaningful_work() {
      return Ok(Vec::new());
    }

    let extraction_context = ctx.to_extraction_context();
    let mut memories_created = Vec::new();

    match extract_memories(&extraction_context).await {
      Ok(result) => {
        for extracted in &result.memories {
          if let Ok(Some(id)) = self.store_extracted_memory(extracted, cwd).await {
            memories_created.push(id);
          }
        }
        info!(
          "LLM extraction completed: {} memories created from {} candidates",
          memories_created.len(),
          result.memories.len()
        );
      }
      Err(e) => {
        warn!("LLM extraction failed, falling back to basic summary: {}", e);
        // Fall back to basic summary extraction
        if let Some(summary) = ctx.summary()
          && let Ok(Some(id)) = self.extract_memory(&summary, cwd, "").await
        {
          memories_created.push(id);
        }
      }
    }

    Ok(memories_created)
  }

  /// Spawn background extraction task (non-blocking)
  ///
  /// This allows the hook to return immediately while extraction runs in the background.
  /// Used for triggers like PreCompact and Stop where we don't need the result synchronously.
  fn spawn_background_extraction(&self, ctx: SegmentContext, cwd: String, trigger: &str) {
    let registry = Arc::clone(&self.registry);
    let embedding = self.embedding.clone();
    let use_llm = self.use_llm_extraction;
    let trigger_name = trigger.to_string();

    tokio::spawn(async move {
      info!("Background extraction started for trigger: {}", trigger_name);

      if !ctx.has_meaningful_work() {
        debug!("Background extraction skipped: no meaningful work");
        return;
      }

      let extraction_context = ctx.to_extraction_context();

      if use_llm {
        match extract_memories(&extraction_context).await {
          Ok(result) => {
            if result.memories.is_empty() {
              debug!("Background extraction: no memories extracted");
              return;
            }

            // Get project database
            let project_path = PathBuf::from(&cwd);
            let (_info, db) = match registry.get_or_create(&project_path).await {
              Ok(r) => r,
              Err(e) => {
                warn!("Background extraction failed to get project: {}", e);
                return;
              }
            };

            for extracted in &result.memories {
              if extracted.content.len() < 20 {
                continue;
              }

              // Compute hashes for dedup
              let (content_hash, simhash) = compute_hashes(&extracted.content);
              let project_uuid = uuid::Uuid::new_v4();

              // Parse sector
              let sector = extracted
                .sector
                .as_ref()
                .and_then(|s| s.parse::<Sector>().ok())
                .unwrap_or_else(|| classify_sector(&extracted.content));

              // Parse memory type
              let memory_type = extracted.memory_type.parse::<MemoryType>().ok();

              // Create memory
              let mut memory = Memory::new(project_uuid, extracted.content.clone(), sector);
              memory.content_hash = content_hash;
              memory.simhash = simhash;
              memory.concepts = extract_concepts(&extracted.content);
              memory.files = extract_files(&extracted.content);
              memory.tags = extracted.tags.clone();
              memory.salience = extracted.confidence;
              memory.memory_type = memory_type;
              if let Some(ref summary) = extracted.summary {
                memory.summary = Some(summary.clone());
              }

              // Generate embedding
              let vector = match &embedding {
                Some(provider) => provider.embed(&extracted.content).await.ok(),
                None => None,
              }
              .unwrap_or_else(|| vec![0.0f32; db.vector_dim]);

              // Store memory
              if let Err(e) = db.add_memory(&memory, Some(&vector)).await {
                warn!("Background extraction failed to store memory: {}", e);
              } else {
                info!("Background extraction stored memory: {} ({:?})", memory.id, sector);
              }
            }

            info!(
              "Background extraction completed: {} memories from trigger {}",
              result.memories.len(),
              trigger_name
            );
          }
          Err(e) => {
            warn!("Background LLM extraction failed for {}: {}", trigger_name, e);
          }
        }
      } else if let Some(summary) = ctx.summary() {
        // Non-LLM fallback
        info!("Background extraction using summary fallback: {}", summary);
      }
    });
  }

  /// Handle a hook event
  pub async fn handle(&self, event: HookEvent, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    debug!("Processing hook event: {:?}", event);

    match event {
      HookEvent::SessionStart => self.on_session_start(params).await,
      HookEvent::SessionEnd => self.on_session_end(params).await,
      HookEvent::UserPromptSubmit => self.on_user_prompt_submit(params).await,
      HookEvent::PostToolUse => self.on_post_tool_use(params).await,
      HookEvent::PreCompact => self.on_pre_compact(params).await,
      HookEvent::Stop => self.on_stop(params).await,
      HookEvent::SubagentStop => self.on_subagent_stop(params).await,
      HookEvent::Notification => self.on_notification(params).await,
    }
  }

  async fn on_session_start(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");

    info!("Session started: {} in {}", session_id, cwd);

    // Bind session to project path (with git root detection)
    // This ensures directory changes within the session don't switch projects
    let project_path = self.bind_session_project(session_id, cwd).await;

    // Get or create project
    let (info, _db) = self.registry.get_or_create(&project_path).await?;

    debug!(
      "Session {} bound to project {} at {:?}",
      session_id, info.name, project_path
    );

    // Auto-start watcher if not already running
    let watcher_started = self.maybe_auto_start_watcher(info.id.as_str(), &project_path).await;

    Ok(serde_json::json!({
        "status": "ok",
        "project_id": info.id.as_str(),
        "project_name": info.name,
        "project_path": project_path.to_string_lossy(),
        "watcher_started": watcher_started,
    }))
  }

  async fn on_session_end(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
    let summary = params.get("summary").and_then(|v| v.as_str());

    info!("Session ended: {}", session_id);

    // Get the bound project path (or resolve from cwd as fallback)
    let project_path = self.get_session_project_path(session_id, cwd).await;
    let cwd_str = project_path.to_string_lossy().to_string();

    // Extract memory from session summary if provided
    let mut memories_created = Vec::new();
    if let Some(summary_text) = summary
      && let Ok(Some(id)) = self.extract_memory(summary_text, &cwd_str, session_id).await
    {
      memories_created.push(id);
    }

    // Promote session memories to project tier based on:
    // 1. Usage count (memories used 2+ times are valuable)
    // 2. High salience (memories with salience >= 0.8 are important)
    let mut memories_promoted = 0;
    if let Ok(session_uuid) = uuid::Uuid::parse_str(session_id)
      && let Ok((_info, db)) = self.registry.get_or_create(&project_path).await
    {
      // Promote memories used at least twice
      match db.promote_session_memories(&session_uuid, 2).await {
        Ok(count) => {
          memories_promoted += count;
          if count > 0 {
            info!("Promoted {} memories based on usage count", count);
          }
        }
        Err(e) => {
          warn!("Failed to promote session memories by usage: {}", e);
        }
      }

      // Also promote high-salience memories (corrections, preferences)
      match db.promote_high_salience_memories(&session_uuid, 0.8).await {
        Ok(count) => {
          memories_promoted += count;
          if count > 0 {
            info!("Promoted {} high-salience memories", count);
          }
        }
        Err(e) => {
          warn!("Failed to promote high-salience memories: {}", e);
        }
      }
    }

    // Clean up session context and project binding
    {
      let mut contexts = self.session_contexts.write().await;
      contexts.remove(session_id);
    }
    self.unbind_session_project(session_id).await;

    Ok(serde_json::json!({
        "status": "ok",
        "memories_created": memories_created,
        "memories_promoted": memories_promoted,
    }))
  }

  async fn on_user_prompt_submit(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
    let prompt = params.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    debug!("User prompt in session {}: {} chars", session_id, prompt.len());

    // Get the bound project path (or resolve from cwd as fallback)
    let project_path = self.get_session_project_path(session_id, cwd).await;
    let cwd_str = project_path.to_string_lossy().to_string();

    let mut memories_created = Vec::new();

    // Extract from previous segment if meaningful work was done
    {
      let mut contexts = self.session_contexts.write().await;
      if let Some(ctx) = contexts.get_mut(session_id) {
        if ctx.has_meaningful_work()
          && let Some(summary) = ctx.summary()
          && let Ok(Some(id)) = self.extract_memory(&summary, &cwd_str, session_id).await
        {
          memories_created.push(id);
        }
        // Reset for new segment
        ctx.reset();
        ctx.user_prompt = Some(prompt.to_string());
      } else {
        // New session context
        let ctx = SegmentContext {
          user_prompt: Some(prompt.to_string()),
          ..Default::default()
        };
        contexts.insert(session_id.to_string(), ctx);
      }
    }

    // Check for high-priority signals (corrections/preferences) for immediate extraction
    if self.use_llm_extraction && !prompt.is_empty() && prompt.len() >= 20 {
      match classify_signal(prompt).await {
        Ok(classification) if classification.category.is_high_priority() && classification.is_extractable => {
          info!("High-priority signal detected: {:?}", classification.category);
          // Immediate extraction for corrections/preferences
          match extract_high_priority(prompt, &classification).await {
            Ok(result) => {
              for extracted in &result.memories {
                if let Ok(Some(id)) = self.store_extracted_memory(extracted, &cwd_str).await {
                  memories_created.push(id);
                }
              }
              if !result.memories.is_empty() {
                info!("High-priority extraction: {} memories", result.memories.len());
              }
            }
            Err(e) => {
              debug!("High-priority extraction failed: {}", e);
            }
          }
        }
        Ok(_) => {
          // Not high-priority, will be extracted at Stop/PreCompact
        }
        Err(e) => {
          debug!("Signal classification failed: {}", e);
        }
      }
    }

    Ok(serde_json::json!({
        "status": "ok",
        "memories_created": memories_created,
    }))
  }

  async fn on_post_tool_use(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let tool_name = params.get("tool_name").and_then(|v| v.as_str()).unwrap_or("unknown");
    let tool_params = params.get("tool_input").cloned().unwrap_or(serde_json::json!({}));
    let tool_result = params.get("tool_result");

    debug!("Tool used in session {}: {}", session_id, tool_name);

    let cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
    // Get the bound project path (or resolve from cwd as fallback)
    let project_path = self.get_session_project_path(session_id, cwd).await;
    let cwd_str = project_path.to_string_lossy().to_string();

    // Create tool observation memory (immediate episodic memory for tool trail)
    let mut observation_memory_id = None;
    if let Err(e) = self
      .create_tool_observation_memory(tool_name, &tool_params, tool_result, &cwd_str)
      .await
      .map(|id| observation_memory_id = id)
    {
      debug!("Failed to create tool observation memory: {}", e);
    }

    // Accumulate tool use data for extraction
    let should_trigger_extraction = {
      let mut contexts = self.session_contexts.write().await;
      let ctx = contexts.entry(session_id.to_string()).or_default();

      // Record the tool use
      ctx.tool_uses.push(ToolUseRecord {
        tool_name: tool_name.to_string(),
        params: tool_params.clone(),
      });

      // Track files and commands based on tool type
      match tool_name {
        "Read" => {
          if let Some(path) = tool_params.get("file_path").and_then(|v| v.as_str())
            && !ctx.files_read.contains(&path.to_string())
          {
            ctx.files_read.push(path.to_string());
          }
        }
        "Edit" | "Write" | "NotebookEdit" => {
          if let Some(path) = tool_params.get("file_path").and_then(|v| v.as_str())
            && !ctx.files_modified.contains(&path.to_string())
          {
            ctx.files_modified.push(path.to_string());
          }
          if let Some(path) = tool_params.get("notebook_path").and_then(|v| v.as_str())
            && !ctx.files_modified.contains(&path.to_string())
          {
            ctx.files_modified.push(path.to_string());
          }
        }
        "Bash" => {
          if let Some(cmd) = tool_params.get("command").and_then(|v| v.as_str()) {
            // Try to get exit code from tool result (if available)
            let exit_code = tool_params.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            // Truncate long commands
            let cmd_display = if cmd.len() > 100 {
              format!("{}...", &cmd[..100])
            } else {
              cmd.to_string()
            };
            // Check for errors in the exit code
            if exit_code != 0 {
              ctx.errors_encountered.push(format!(
                "Command '{}' failed with exit code {}",
                &cmd_display, exit_code
              ));
            }
            ctx.commands_run.push((cmd_display, exit_code));
          }
        }
        "Glob" | "Grep" => {
          // Track search patterns
          if let Some(pattern) = tool_params.get("pattern").and_then(|v| v.as_str())
            && !ctx.searches_performed.contains(&pattern.to_string())
          {
            ctx.searches_performed.push(pattern.to_string());
          }
        }
        "TodoWrite" => {
          // Track completed tasks
          if let Some(todos) = tool_params.get("todos").and_then(|v| v.as_array()) {
            for todo in todos {
              if todo.get("status").and_then(|v| v.as_str()) == Some("completed")
                && let Some(content) = todo.get("content").and_then(|v| v.as_str())
                && !ctx.completed_tasks.contains(&content.to_string())
              {
                ctx.completed_tasks.push(content.to_string());
              }
            }
          }
        }
        _ => {}
      }

      // Check for todo_completion trigger: ≥3 tasks completed AND ≥5 tool calls
      ctx.completed_tasks.len() >= 3 && ctx.tool_call_count() >= 5
    };

    // If todo_completion threshold met, trigger extraction
    if should_trigger_extraction && self.use_llm_extraction {
      info!(
        "Todo completion trigger: extracting memories for session {}",
        session_id
      );
      let ctx_snapshot = {
        let contexts = self.session_contexts.read().await;
        contexts.get(session_id).map(|c| c.to_extraction_context())
      };

      if let Some(extraction_ctx) = ctx_snapshot {
        match extract_memories(&extraction_ctx).await {
          Ok(result) => {
            for extracted in &result.memories {
              if let Ok(Some(_id)) = self.store_extracted_memory(extracted, &cwd_str).await {
                // Memory stored from todo_completion trigger
              }
            }
            if !result.memories.is_empty() {
              info!(
                "Todo completion extraction: {} memories from {} completed tasks",
                result.memories.len(),
                extraction_ctx.completed_tasks.len()
              );
            }
          }
          Err(e) => {
            debug!("Todo completion extraction failed: {}", e);
          }
        }
      }
    }

    Ok(serde_json::json!({
        "status": "ok",
        "observation_memory_id": observation_memory_id,
    }))
  }

  async fn on_pre_compact(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
    let summary = params.get("summary").and_then(|v| v.as_str());

    info!("Pre-compact trigger for session {}", session_id);

    // Get the bound project path (or resolve from cwd as fallback)
    let project_path = self.get_session_project_path(session_id, cwd).await;
    let cwd_str = project_path.to_string_lossy().to_string();

    let mut memories_created = Vec::new();

    // Extract from current segment before compaction
    {
      let mut contexts = self.session_contexts.write().await;
      if let Some(ctx) = contexts.get_mut(session_id) {
        if self.use_background_extraction && self.use_llm_extraction && ctx.has_meaningful_work() {
          // Use background extraction to avoid blocking the hook response
          self.spawn_background_extraction(ctx.clone(), cwd_str.clone(), "pre_compact");
          // memories_created will be empty since extraction is async
        } else if self.use_llm_extraction {
          // Synchronous LLM extraction
          match self.extract_with_llm(ctx, &cwd_str).await {
            Ok(ids) => memories_created.extend(ids),
            Err(e) => {
              warn!("LLM extraction failed in pre-compact: {}", e);
              // Fallback to basic summary
              if ctx.has_meaningful_work()
                && let Some(ctx_summary) = ctx.summary()
                && let Ok(Some(id)) = self.extract_memory(&ctx_summary, &cwd_str, session_id).await
              {
                memories_created.push(id);
              }
            }
          }
        } else {
          // Use basic summary extraction
          if ctx.has_meaningful_work()
            && let Some(ctx_summary) = ctx.summary()
            && let Ok(Some(id)) = self.extract_memory(&ctx_summary, &cwd_str, session_id).await
          {
            memories_created.push(id);
          }
        }
        ctx.reset();
      }
    }

    // Also extract from provided summary if any (always synchronous)
    if let Some(summary_text) = summary
      && let Ok(Some(id)) = self.extract_memory(summary_text, &cwd_str, session_id).await
    {
      memories_created.push(id);
    }

    Ok(serde_json::json!({
        "status": "ok",
        "background_extraction": self.use_background_extraction,
        "memories_created": memories_created,
    }))
  }

  async fn on_stop(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
    let summary = params.get("summary").and_then(|v| v.as_str());

    info!("Stop event for session {}", session_id);

    // Get the bound project path (or resolve from cwd as fallback)
    let project_path = self.get_session_project_path(session_id, cwd).await;
    let cwd_str = project_path.to_string_lossy().to_string();

    let mut memories_created = Vec::new();

    // Final extraction from accumulated context
    {
      let mut contexts = self.session_contexts.write().await;
      if let Some(ctx) = contexts.remove(session_id) {
        if self.use_background_extraction && self.use_llm_extraction && ctx.has_meaningful_work() {
          // Use background extraction to avoid blocking the hook response
          self.spawn_background_extraction(ctx, cwd_str.clone(), "stop");
          // memories_created will be empty since extraction is async
        } else if self.use_llm_extraction {
          // Synchronous LLM extraction
          match self.extract_with_llm(&ctx, &cwd_str).await {
            Ok(ids) => memories_created.extend(ids),
            Err(e) => {
              warn!("LLM extraction failed: {}", e);
              // Fallback to basic summary
              if ctx.has_meaningful_work()
                && let Some(ctx_summary) = ctx.summary()
                && let Ok(Some(id)) = self.extract_memory(&ctx_summary, &cwd_str, session_id).await
              {
                memories_created.push(id);
              }
            }
          }
        } else {
          // Use basic summary extraction
          if ctx.has_meaningful_work()
            && let Some(ctx_summary) = ctx.summary()
            && let Ok(Some(id)) = self.extract_memory(&ctx_summary, &cwd_str, session_id).await
          {
            memories_created.push(id);
          }
        }
      }
    }

    // Extract from provided summary (always use basic extraction for raw summaries)
    if let Some(summary_text) = summary
      && let Ok(Some(id)) = self.extract_memory(summary_text, &cwd_str, session_id).await
    {
      memories_created.push(id);
    }

    Ok(serde_json::json!({
        "status": "ok",
        "background_extraction": self.use_background_extraction,
        "memories_created": memories_created,
    }))
  }

  async fn on_subagent_stop(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let session_id = params.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown");

    debug!("Subagent stop for session {}", session_id);

    Ok(serde_json::json!({
        "status": "ok",
    }))
  }

  async fn on_notification(&self, params: serde_json::Value) -> Result<serde_json::Value, HookError> {
    let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");

    debug!("Notification: {}", message);

    Ok(serde_json::json!({
        "status": "ok",
    }))
  }
}

/// Read hook input from stdin
pub fn read_hook_input() -> Result<serde_json::Value, HookError> {
  use std::io::Read;

  let mut input = String::new();
  std::io::stdin().read_to_string(&mut input)?;

  if input.trim().is_empty() {
    return Ok(serde_json::json!({}));
  }

  Ok(serde_json::from_str(&input)?)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_hook_event_from_str() {
    assert_eq!("SessionStart".parse::<HookEvent>().unwrap(), HookEvent::SessionStart);
    assert_eq!("PostToolUse".parse::<HookEvent>().unwrap(), HookEvent::PostToolUse);
    assert!("Invalid".parse::<HookEvent>().is_err());
  }

  #[tokio::test]
  async fn test_hook_handler_session_start() {
    let registry = Arc::new(ProjectRegistry::new());
    let handler = HookHandler::new(Arc::clone(&registry));

    let params = serde_json::json!({
        "session_id": "test-123",
        "cwd": "/tmp"
    });

    let result = handler.on_session_start(params).await;
    assert!(result.is_ok());

    // Clean up watcher
    registry.stop_all_watchers().await;
  }

  #[tokio::test]
  async fn test_session_project_binding_stable_across_directory_changes() {
    use std::fs;

    // Create a temporary directory structure with git
    let temp = std::env::temp_dir().join(format!("hook_test_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    fs::create_dir_all(temp.join(".git")).unwrap();
    fs::create_dir_all(temp.join("src/deep")).unwrap();

    let registry = Arc::new(ProjectRegistry::new());
    let handler = HookHandler::new(Arc::clone(&registry));
    let session_id = "test-session-stable";

    // Start session from the root
    let start_result = handler
      .on_session_start(serde_json::json!({
        "session_id": session_id,
        "cwd": temp.to_string_lossy()
      }))
      .await
      .unwrap();

    let project_path_at_start = start_result["project_path"].as_str().unwrap();

    // Simulate directory change to subdir - should still use same project
    let subdir_project_path = handler
      .get_session_project_path(session_id, &temp.join("src/deep").to_string_lossy())
      .await;

    // Project path should still be the git root
    assert_eq!(
      subdir_project_path.to_string_lossy(),
      project_path_at_start,
      "Session should be bound to original project path"
    );

    // Clean up - stop watcher first
    let project_id = start_result["project_id"].as_str().unwrap();
    let _ = registry.stop_watcher(project_id).await;
    handler.unbind_session_project(session_id).await;
    let _ = fs::remove_dir_all(&temp);
  }

  #[tokio::test]
  async fn test_session_project_binding_uses_git_root() {
    use std::fs;

    // Create a temporary directory structure with git
    let temp = std::env::temp_dir().join(format!("git_test_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    fs::create_dir_all(temp.join(".git")).unwrap();
    fs::create_dir_all(temp.join("src")).unwrap();

    let registry = Arc::new(ProjectRegistry::new());
    let handler = HookHandler::new(Arc::clone(&registry));
    let session_id = "test-session-git";

    // Start session from a subdir
    let start_result = handler
      .on_session_start(serde_json::json!({
        "session_id": session_id,
        "cwd": temp.join("src").to_string_lossy()
      }))
      .await
      .unwrap();

    let project_path = PathBuf::from(start_result["project_path"].as_str().unwrap());

    // Project path should be the git root, not the subdir
    assert_eq!(
      project_path.canonicalize().unwrap(),
      temp.canonicalize().unwrap(),
      "Session should be bound to git root, not cwd"
    );

    // Clean up - stop watcher first
    let project_id = start_result["project_id"].as_str().unwrap();
    let _ = registry.stop_watcher(project_id).await;
    handler.unbind_session_project(session_id).await;
    let _ = fs::remove_dir_all(&temp);
  }

  #[tokio::test]
  async fn test_session_start_auto_starts_watcher() {
    use std::fs;

    // Create a temporary directory structure
    let temp = std::env::temp_dir().join(format!("watcher_test_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    fs::create_dir_all(temp.join(".git")).unwrap();

    let registry = Arc::new(ProjectRegistry::new());
    let handler = HookHandler::new(registry.clone());
    let session_id = "test-session-watcher";

    // Start session - should auto-start watcher
    let start_result = handler
      .on_session_start(serde_json::json!({
        "session_id": session_id,
        "cwd": temp.to_string_lossy()
      }))
      .await
      .unwrap();

    // Verify watcher was started
    assert_eq!(
      start_result["watcher_started"].as_bool(),
      Some(true),
      "Watcher should have been auto-started"
    );

    let project_id = start_result["project_id"].as_str().unwrap();
    let status = registry.watcher_status(project_id).await;
    assert!(status.running, "Watcher should be running");

    // Second session start for same project should NOT start a new watcher
    let session_id_2 = "test-session-watcher-2";
    let start_result_2 = handler
      .on_session_start(serde_json::json!({
        "session_id": session_id_2,
        "cwd": temp.to_string_lossy()
      }))
      .await
      .unwrap();

    assert_eq!(
      start_result_2["watcher_started"].as_bool(),
      Some(false),
      "Watcher should NOT be started again (already running)"
    );

    // Clean up - stop watcher before unbinding
    let _ = registry.stop_watcher(project_id).await;
    handler.unbind_session_project(session_id).await;
    handler.unbind_session_project(session_id_2).await;
    let _ = fs::remove_dir_all(&temp);
  }
}
