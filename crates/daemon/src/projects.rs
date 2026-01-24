use db::{CodeReference, ProjectDb, default_data_dir};
use embedding::EmbeddingProvider;
use engram_core::{ChunkParams, Config, DocumentChunk, DocumentId, DocumentSource, ProjectId, chunk_text};
use index::{ChangeKind, Chunker, DebounceConfig, DebouncedWatcher, GITIGNORE_CACHE, Scanner, WatcherCoordinator};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum ProjectError {
  #[error("Database error: {0}")]
  Database(#[from] db::DbError),
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Project not found: {0}")]
  NotFound(String),
  #[error("Serialization error: {0}")]
  Serialization(String),
}

/// Project metadata for the registry
#[derive(Debug, Clone)]
pub struct ProjectInfo {
  pub id: ProjectId,
  pub path: PathBuf,
  pub name: String,
}

impl ProjectInfo {
  /// Create ProjectInfo from a path, applying workspace aliasing rules.
  ///
  /// Resolution order:
  /// 1. Explicit alias in config (`[workspace] alias = "/path"`)
  /// 2. Git worktree detection (auto-resolve to main repo)
  /// 3. Standard git root detection
  /// 4. Use the path as-is
  pub fn from_path(path: &Path) -> Result<Self, ProjectError> {
    let canonical = path.canonicalize()?;

    // Load config to check for workspace alias
    let config = Config::load_for_project(&canonical);

    Self::from_path_with_config(&canonical, &config)
  }

  /// Create ProjectInfo using an explicit config.
  pub fn from_path_with_config(path: &Path, config: &Config) -> Result<Self, ProjectError> {
    let canonical = path.canonicalize()?;

    // Check for explicit workspace alias first
    if let Some(ref alias) = config.workspace.alias {
      let alias_path = PathBuf::from(alias);
      if alias_path.exists() {
        let alias_canonical = alias_path.canonicalize()?;
        let id = ProjectId::from_path_exact(&alias_canonical);
        let name = alias_canonical
          .file_name()
          .map(|n| n.to_string_lossy().to_string())
          .unwrap_or_else(|| "unnamed".to_string());

        info!("Using workspace alias: {:?} -> {:?}", canonical, alias_canonical);

        return Ok(Self {
          id,
          path: alias_canonical,
          name,
        });
      } else {
        warn!(
          "Workspace alias path does not exist: {:?}, falling back to auto-detection",
          alias
        );
      }
    }

    // Check if worktree detection is disabled
    let id = if config.workspace.disable_worktree_detection {
      // Use local git root only, don't resolve worktrees
      let local_root = engram_core::find_git_root_local(&canonical).unwrap_or_else(|| canonical.clone());
      ProjectId::from_path_exact(&local_root)
    } else {
      // Normal resolution (includes worktree detection)
      ProjectId::from_path(&canonical)
    };

    let name = canonical
      .file_name()
      .map(|n| n.to_string_lossy().to_string())
      .unwrap_or_else(|| "unnamed".to_string());

    Ok(Self {
      id,
      path: canonical,
      name,
    })
  }
}

/// Status of a file watcher for a project
#[derive(Debug, Clone, Default)]
pub struct WatcherStatus {
  pub running: bool,
  pub root: Option<PathBuf>,
  pub pending_changes: usize,
  pub files_indexed: usize,
  pub last_error: Option<String>,
  /// Whether a startup scan is in progress
  pub scanning: bool,
  /// Startup scan progress (processed, total)
  pub scan_progress: Option<(usize, usize)>,
}

/// Active watcher task handle with cancellation
struct WatcherTask {
  handle: JoinHandle<()>,
  cancel: Arc<AtomicBool>,
}

/// Registry of active projects and their database connections
pub struct ProjectRegistry {
  data_dir: PathBuf,
  projects: RwLock<HashMap<String, Arc<ProjectDb>>>,
  metadata: RwLock<HashMap<String, ProjectInfo>>,
  watcher_status: Arc<RwLock<HashMap<String, WatcherStatus>>>,
  watcher_tasks: RwLock<HashMap<String, WatcherTask>>,
  /// Coordinator for multi-instance watcher locking
  coordinator: WatcherCoordinator,
  /// Cache for file contents to enable incremental parsing
  file_content_cache: Arc<crate::cache::FileContentCache>,
  /// Scan states for each project (for blocking searches during startup scan)
  scan_states: Arc<RwLock<HashMap<String, Arc<crate::startup_scan::ScanState>>>>,
}

impl ProjectRegistry {
  pub fn new() -> Self {
    Self::with_data_dir(default_data_dir())
  }

  pub fn with_data_dir(data_dir: PathBuf) -> Self {
    Self {
      data_dir: data_dir.clone(),
      projects: RwLock::new(HashMap::new()),
      metadata: RwLock::new(HashMap::new()),
      watcher_status: Arc::new(RwLock::new(HashMap::new())),
      watcher_tasks: RwLock::new(HashMap::new()),
      coordinator: WatcherCoordinator::with_locks_dir(data_dir.join("watchers")),
      file_content_cache: Arc::new(crate::cache::FileContentCache::new()),
      scan_states: Arc::new(RwLock::new(HashMap::new())),
    }
  }

  /// Get the file content cache (for testing/diagnostics)
  pub fn file_content_cache(&self) -> &Arc<crate::cache::FileContentCache> {
    &self.file_content_cache
  }

  /// Get the data directory
  pub fn data_dir(&self) -> &Path {
    &self.data_dir
  }

  /// Get or create a project database connection
  pub async fn get_or_create(&self, path: &Path) -> Result<(ProjectInfo, Arc<ProjectDb>), ProjectError> {
    let info = ProjectInfo::from_path(path)?;
    let id_str = info.id.as_str().to_string();

    // Check if already loaded
    {
      let projects = self.projects.read().await;
      if let Some(db) = projects.get(&id_str) {
        debug!("Using cached project: {}", info.name);
        return Ok((info, Arc::clone(db)));
      }
    }

    // Create new connection
    info!("Opening project database: {} at {:?}", info.name, self.data_dir);

    let db = ProjectDb::open(info.id.clone(), &self.data_dir).await?;
    let db = Arc::new(db);

    // Store metadata
    let project_dir = info.id.data_dir(&self.data_dir);
    let metadata_path = project_dir.join("project.json");
    self.save_metadata(&info, &metadata_path).await?;

    // Cache the connection
    {
      let mut projects = self.projects.write().await;
      let mut metadata = self.metadata.write().await;
      projects.insert(id_str.clone(), Arc::clone(&db));
      metadata.insert(id_str, info.clone());
    }

    Ok((info, db))
  }

  /// Get an existing project by ID string
  pub async fn get(&self, id: &str) -> Option<(ProjectInfo, Arc<ProjectDb>)> {
    let projects = self.projects.read().await;
    let metadata = self.metadata.read().await;

    if let (Some(db), Some(info)) = (projects.get(id), metadata.get(id)) {
      Some((info.clone(), Arc::clone(db)))
    } else {
      None
    }
  }

  /// List all active projects
  pub async fn list(&self) -> Vec<ProjectInfo> {
    let metadata = self.metadata.read().await;
    metadata.values().cloned().collect()
  }

  /// Close a project connection
  pub async fn close(&self, id: &str) {
    let mut projects = self.projects.write().await;
    let mut metadata = self.metadata.write().await;

    if projects.remove(id).is_some() {
      metadata.remove(id);
      info!("Closed project: {}", id);
    }
  }

  /// Close all project connections
  pub async fn close_all(&self) {
    let mut projects = self.projects.write().await;
    let mut metadata = self.metadata.write().await;

    let count = projects.len();
    projects.clear();
    metadata.clear();

    info!("Closed {} projects", count);
  }

  async fn save_metadata(&self, info: &ProjectInfo, path: &Path) -> Result<(), ProjectError> {
    use ipc::ProjectMetadataJson;

    if let Some(parent) = path.parent() {
      tokio::fs::create_dir_all(parent).await?;
    }

    let metadata = ProjectMetadataJson {
      id: info.id.as_str().to_string(),
      path: info.path.to_string_lossy().to_string(),
      name: info.name.clone(),
    };

    let json = serde_json::to_string_pretty(&metadata).map_err(|e| ProjectError::Serialization(e.to_string()))?;
    tokio::fs::write(path, json).await?;
    Ok(())
  }

  // Watcher management

  /// Start a file watcher for a project with actual file system watching
  ///
  /// If the project was previously indexed and startup_scan is enabled in config,
  /// this will run a startup scan to reconcile the database with the filesystem.
  pub async fn start_watcher(
    &self,
    id: &str,
    root: &Path,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
  ) -> Result<(), ProjectError> {
    self.start_watcher_with_scan_config(id, root, embedding, None).await
  }

  /// Start a file watcher with explicit startup scan configuration
  ///
  /// If `scan_config` is None, settings from the project config will be used.
  pub async fn start_watcher_with_scan_config(
    &self,
    id: &str,
    root: &Path,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    scan_config: Option<crate::startup_scan::StartupScanConfig>,
  ) -> Result<(), ProjectError> {
    // Check if already running in this process
    {
      let tasks = self.watcher_tasks.read().await;
      if tasks.contains_key(id) {
        return Ok(()); // Already running
      }
    }

    // Try to acquire cross-process lock to prevent multiple instances
    match self.coordinator.try_acquire(id, root) {
      Ok(true) => {
        debug!("Acquired watcher lock for project {}", id);
      }
      Ok(false) => {
        warn!("Watcher already running for project {} in another process", id);
        return Ok(()); // Another process has the lock
      }
      Err(e) => {
        warn!("Failed to acquire watcher lock: {}", e);
        // Continue anyway - coordination is best-effort
      }
    }

    // Get the database for this project
    let db = {
      let projects = self.projects.read().await;
      projects.get(id).cloned()
    };

    let db = match db {
      Some(d) => d,
      None => return Err(ProjectError::NotFound(id.to_string())),
    };

    // Load config
    let config = Config::load_for_project(root);
    let debounce_ms = config.index.watcher_debounce_ms;

    // Use provided scan config or load from project config
    let scan_config = scan_config.unwrap_or_else(|| crate::startup_scan::StartupScanConfig::from_config(&config));

    // Check if project was previously indexed (only run startup scan if so)
    let chunk_count = db.count_code_chunks(None).await.unwrap_or(0);
    let was_previously_indexed = chunk_count > 0;

    // Determine if we should run startup scan
    let should_scan = was_previously_indexed && scan_config.enabled;

    // Create scan state if scanning
    let scan_state = if should_scan {
      let state = Arc::new(crate::startup_scan::ScanState::new());
      {
        let mut scan_states = self.scan_states.write().await;
        scan_states.insert(id.to_string(), Arc::clone(&state));
      }
      Some(state)
    } else {
      None
    };

    // Update initial status
    {
      let mut status_map = self.watcher_status.write().await;
      status_map.insert(
        id.to_string(),
        WatcherStatus {
          running: true,
          root: Some(root.to_path_buf()),
          pending_changes: 0,
          files_indexed: 0,
          last_error: None,
          scanning: should_scan,
          scan_progress: None,
        },
      );
    }

    // Run startup scan if needed
    if should_scan {
      let scanner = crate::startup_scan::StartupScanner::new(scan_config.clone());

      if scan_config.blocking {
        // Blocking mode: run scan before starting watcher
        info!(
          "Running blocking startup scan for project {} ({} indexed chunks)",
          id, chunk_count
        );

        match scanner.scan(&db, root).await {
          Ok(result) => {
            if !result.is_empty() {
              info!(
                "Startup scan found {} deleted, {} added, {} modified files",
                result.deleted.len(),
                result.added.len(),
                result.modified.len()
              );

              // Apply changes
              if let Err(e) = scanner.apply(&result, &db, root, embedding.clone(), &config).await {
                warn!("Failed to apply startup scan results: {}", e);
              }
            } else {
              info!("Startup scan complete: no changes detected");
            }
          }
          Err(e) => {
            warn!("Startup scan failed: {}", e);
          }
        }

        // Clear scanning state
        if let Some(state) = &scan_state {
          state.finish();
        }
        {
          let mut status_map = self.watcher_status.write().await;
          if let Some(status) = status_map.get_mut(id) {
            status.scanning = false;
          }
        }
      } else {
        // Non-blocking mode: start scan in background
        info!(
          "Starting background startup scan for project {} ({} indexed chunks)",
          id, chunk_count
        );

        let db_clone = Arc::clone(&db);
        let root_clone = root.to_path_buf();
        let config_clone = config.clone();
        let embedding_clone = embedding.clone();
        let id_clone = id.to_string();
        let scan_states_clone = Arc::clone(&self.scan_states);
        let watcher_status_clone = Arc::clone(&self.watcher_status);

        tokio::spawn(async move {
          match scanner.scan(&db_clone, &root_clone).await {
            Ok(result) => {
              if !result.is_empty() {
                info!(
                  "Background startup scan found {} deleted, {} added, {} modified files",
                  result.deleted.len(),
                  result.added.len(),
                  result.modified.len()
                );

                if let Err(e) = scanner
                  .apply(&result, &db_clone, &root_clone, embedding_clone, &config_clone)
                  .await
                {
                  warn!("Failed to apply startup scan results: {}", e);
                }
              } else {
                info!("Background startup scan complete: no changes detected");
              }
            }
            Err(e) => {
              warn!("Background startup scan failed: {}", e);
            }
          }

          // Clear scanning state
          if let Some(state) = scan_states_clone.write().await.get(&id_clone) {
            state.finish();
          }
          if let Some(status) = watcher_status_clone.write().await.get_mut(&id_clone) {
            status.scanning = false;
          }
        });
      }
    }

    // Create the debounced watcher
    let debounce_config = DebounceConfig {
      file_debounce_ms: debounce_ms,
      ..Default::default()
    };

    let watcher = match DebouncedWatcher::new(root, debounce_config) {
      Ok(w) => w,
      Err(e) => return Err(ProjectError::Io(std::io::Error::other(e.to_string()))),
    };

    // Set up cancellation
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel);

    // Clone data for the task
    let id_owned = id.to_string();
    let root_owned = root.to_path_buf();
    let registry_data_dir = self.data_dir.clone();
    let content_cache = Arc::clone(&self.file_content_cache);

    // Spawn the watcher task
    let handle = tokio::task::spawn_blocking(move || {
      run_watcher_loop(
        watcher,
        db,
        embedding,
        cancel_clone,
        &id_owned,
        &root_owned,
        registry_data_dir,
        content_cache,
      );
    });

    // Store the task
    {
      let mut tasks = self.watcher_tasks.write().await;
      tasks.insert(id.to_string(), WatcherTask { handle, cancel });
    }

    info!("Started file watcher for project {} at {:?}", id, root);
    Ok(())
  }

  /// Stop a file watcher for a project
  pub async fn stop_watcher(&self, id: &str) -> Result<(), ProjectError> {
    // Get the root path for lock release
    let root = {
      let status_map = self.watcher_status.read().await;
      status_map.get(id).and_then(|s| s.root.clone())
    };

    // Signal cancellation
    {
      let tasks = self.watcher_tasks.read().await;
      if let Some(task) = tasks.get(id) {
        task.cancel.store(true, Ordering::SeqCst);
      }
    }

    // Remove and await the task
    let task = {
      let mut tasks = self.watcher_tasks.write().await;
      tasks.remove(id)
    };

    if let Some(task) = task {
      // Wait for the task to finish (with timeout)
      let _ = tokio::time::timeout(Duration::from_secs(5), task.handle).await;
    }

    // Release the coordination lock
    if let Some(root) = root
      && let Err(e) = self.coordinator.release(&root)
    {
      warn!("Failed to release watcher lock: {}", e);
    }

    // Update status
    {
      let mut status_map = self.watcher_status.write().await;
      if let Some(status) = status_map.get_mut(id) {
        status.running = false;
      }
    }

    info!("Stopped watcher for project {}", id);
    Ok(())
  }

  /// Get watcher status for a project
  pub async fn watcher_status(&self, id: &str) -> WatcherStatus {
    let status_map = self.watcher_status.read().await;
    status_map.get(id).cloned().unwrap_or_default()
  }

  /// Update watcher status (called from watcher loop)
  pub async fn update_watcher_status(&self, id: &str, pending: usize, indexed: usize, error: Option<String>) {
    let mut status_map = self.watcher_status.write().await;
    if let Some(status) = status_map.get_mut(id) {
      status.pending_changes = pending;
      status.files_indexed = indexed;
      status.last_error = error;
    }
  }

  /// Stop all watchers (for cleanup/shutdown)
  pub async fn stop_all_watchers(&self) {
    let ids: Vec<String> = {
      let tasks = self.watcher_tasks.read().await;
      tasks.keys().cloned().collect()
    };

    for id in ids {
      let _ = self.stop_watcher(&id).await;
    }
  }

  /// Check if a startup scan is in progress for a project
  pub async fn is_scanning(&self, id: &str) -> bool {
    let scan_states = self.scan_states.read().await;
    scan_states.get(id).map(|state| state.is_in_progress()).unwrap_or(false)
  }

  /// Wait for the startup scan to complete for a project
  ///
  /// Returns immediately if no scan is in progress.
  /// This should be called before executing searches to ensure results are consistent.
  pub async fn wait_for_scan(&self, id: &str, timeout: Duration) -> bool {
    let state = {
      let scan_states = self.scan_states.read().await;
      scan_states.get(id).cloned()
    };

    let Some(state) = state else {
      return true; // No scan state means no scan in progress
    };

    if !state.is_in_progress() {
      return true;
    }

    // Poll until scan completes or timeout
    let start = std::time::Instant::now();
    let poll_interval = Duration::from_millis(100);

    while start.elapsed() < timeout {
      if !state.is_in_progress() {
        return true;
      }
      tokio::time::sleep(poll_interval).await;
    }

    // Timeout reached
    warn!("Timed out waiting for scan to complete for project {}", id);
    false
  }

  /// Get scan progress for a project
  pub async fn scan_progress(&self, id: &str) -> Option<(usize, usize)> {
    let scan_states = self.scan_states.read().await;
    scan_states.get(id).map(|state| state.progress())
  }
}

/// Holds context for processing a file change
pub(crate) struct FileChangeContext {
  pub change_path: PathBuf,
  pub relative_path: String,
  pub is_doc_file: bool,
  pub is_delete: bool,
  /// Cached old content for incremental parsing (None if not available or delete)
  pub old_content: Option<Arc<String>>,
}

/// Prepared code file data (parsed, ready for embedding)
struct PreparedCodeFile {
  relative_path: String,
  chunks: Vec<engram_core::CodeChunk>,
  /// Existing embeddings that can be reused (content_hash -> embedding)
  existing_embeddings: HashMap<String, Vec<f32>>,
  /// Indices of chunks that need new embeddings
  chunks_needing_embeddings: Vec<usize>,
  /// New file content (for cache update)
  content: String,
}

/// Prepared document file data (parsed, ready for embedding)
struct PreparedDocFile {
  relative_path: String,
  chunks: Vec<DocumentChunk>,
}

/// Result of preparing a file for indexing
enum PreparedFile {
  Code(PreparedCodeFile),
  Doc(PreparedDocFile),
  Delete { relative_path: String, is_doc: bool },
  Skip,
}

/// Text needing embedding with source info for routing back
struct EmbeddingRequest {
  /// Index into the PreparedFile list
  file_idx: usize,
  /// For code: chunk index; for doc: chunk index
  chunk_idx: usize,
  /// The text to embed
  text: String,
}

/// Phase 1: Prepare a file for indexing (read, parse, identify what needs embedding)
/// This runs in parallel for all files and doesn't do any embedding.
async fn prepare_file_change(
  ctx: FileChangeContext,
  db: Arc<ProjectDb>,
  root: PathBuf,
  docs_config: engram_core::DocsConfig,
  project_id: String,
) -> PreparedFile {
  if ctx.is_delete {
    return PreparedFile::Delete {
      relative_path: ctx.relative_path,
      is_doc: ctx.is_doc_file,
    };
  }

  // Handle create/modify
  if ctx.is_doc_file {
    // Prepare document file
    let content = match tokio::fs::read_to_string(&ctx.change_path).await {
      Ok(c) => c,
      Err(_) => return PreparedFile::Skip,
    };

    if content.len() > docs_config.max_file_size {
      debug!("Skipping large doc file: {:?}", ctx.change_path);
      return PreparedFile::Skip;
    }

    let title = ctx
      .change_path
      .file_name()
      .and_then(|n| n.to_str())
      .unwrap_or("Untitled")
      .to_string();

    // Delete existing document if any
    let filter = format!("source = '{}'", ctx.relative_path.replace('\'', "''"));
    if let Ok(chunks) = db.list_document_chunks(Some(&filter), Some(1)).await
      && let Some(chunk) = chunks.first()
    {
      let doc_id = chunk.document_id;
      let _ = db.delete_document(&doc_id).await;
    }

    // Chunk the document
    let params = ChunkParams::default();
    let text_chunks = chunk_text(&content, &params);
    let total_chunks = text_chunks.len();

    if total_chunks == 0 {
      return PreparedFile::Skip;
    }

    let document_id = DocumentId::new();
    let project_uuid = uuid::Uuid::parse_str(&project_id).unwrap_or_else(|_| uuid::Uuid::new_v4());

    // Create all chunks
    let chunks: Vec<DocumentChunk> = text_chunks
      .into_iter()
      .enumerate()
      .map(|(i, (chunk_content, char_offset))| {
        DocumentChunk::new(
          document_id,
          project_uuid,
          chunk_content,
          title.clone(),
          ctx.relative_path.clone(),
          DocumentSource::File,
          i,
          total_chunks,
          char_offset,
        )
      })
      .collect();

    PreparedFile::Doc(PreparedDocFile {
      relative_path: ctx.relative_path,
      chunks,
    })
  } else {
    // Prepare code file
    let scanner = Scanner::new();
    let scanned = match scanner.scan_file(&ctx.change_path, &root) {
      Some(s) => s,
      None => return PreparedFile::Skip,
    };

    let content = match tokio::fs::read_to_string(&ctx.change_path).await {
      Ok(c) => c,
      Err(_) => return PreparedFile::Skip,
    };

    let mut chunker = Chunker::default();

    // Use incremental parsing if old content is available (much faster for small edits)
    let old_content_ref = ctx.old_content.as_deref().map(|s| s.as_str());
    let new_chunks = chunker.chunk_incremental(
      &content,
      &ctx.relative_path,
      scanned.language,
      &scanned.checksum,
      old_content_ref,
    );

    if new_chunks.is_empty() {
      return PreparedFile::Skip;
    }

    // Get existing chunks with their embeddings for differential re-indexing
    let existing_chunks = db
      .get_chunks_with_embeddings_for_file(&ctx.relative_path)
      .await
      .unwrap_or_default();

    // Build a map of content_hash -> embedding from existing chunks
    let existing_embeddings: HashMap<String, Vec<f32>> = existing_chunks
      .into_iter()
      .filter_map(|(chunk, embedding)| {
        let hash = chunk.content_hash?;
        let emb = embedding?;
        Some((hash, emb))
      })
      .collect();

    // Delete old chunks (after we've captured their embeddings)
    if let Err(e) = db.delete_chunks_for_file(&ctx.relative_path).await {
      warn!("Failed to delete old chunks for {}: {}", ctx.relative_path, e);
    }

    // Determine which chunks need new embeddings
    let mut chunks_needing_embeddings: Vec<usize> = Vec::new();
    let mut reused_count = 0;

    for (i, chunk) in new_chunks.iter().enumerate() {
      if let Some(ref hash) = chunk.content_hash {
        if !existing_embeddings.contains_key(hash) {
          chunks_needing_embeddings.push(i);
        } else {
          reused_count += 1;
        }
      } else {
        chunks_needing_embeddings.push(i);
      }
    }

    if reused_count > 0 {
      debug!(
        "Reusing {} embeddings for {} (need {} new)",
        reused_count,
        ctx.relative_path,
        chunks_needing_embeddings.len()
      );
    }

    PreparedFile::Code(PreparedCodeFile {
      relative_path: ctx.relative_path,
      chunks: new_chunks,
      existing_embeddings,
      chunks_needing_embeddings,
      content,
    })
  }
}

#[allow(clippy::too_many_arguments)]
/// Phase 3: Finalize a prepared file by inserting with embeddings
async fn finalize_file_change(
  prepared: PreparedFile,
  db: Arc<ProjectDb>,
  root: PathBuf,
  project_id: String,
  content_cache: Arc<crate::cache::FileContentCache>,
  embeddings: &HashMap<(usize, usize), Vec<f32>>,
  file_idx: usize,
  dim: usize,
) -> (bool, bool) {
  match prepared {
    PreparedFile::Delete { relative_path, is_doc } => {
      content_cache.remove(&root, &relative_path);

      if is_doc {
        let filter = format!("source = '{}'", relative_path.replace('\'', "''"));
        if let Ok(chunks) = db.list_document_chunks(Some(&filter), Some(1)).await
          && let Some(chunk) = chunks.first()
        {
          let doc_id = chunk.document_id;
          if let Err(e) = db.delete_document(&doc_id).await {
            warn!("Failed to delete document {}: {}", relative_path, e);
          }
        }
      } else if let Err(e) = db.delete_chunks_for_file(&relative_path).await {
        warn!("Failed to delete chunks for {}: {}", relative_path, e);
      }
      (false, false)
    }

    PreparedFile::Doc(doc) => {
      // Build vectors from cross-file embeddings
      let vectors: Vec<Option<Vec<f32>>> = doc
        .chunks
        .iter()
        .enumerate()
        .map(|(chunk_idx, _)| embeddings.get(&(file_idx, chunk_idx)).cloned())
        .collect();

      if let Err(e) = db.add_document_chunks(&doc.chunks, &vectors).await {
        warn!(
          "Failed to batch insert document chunks for {}: {}",
          doc.relative_path, e
        );
        (false, false)
      } else {
        debug!(
          "Batch inserted {} document chunks for {}",
          doc.chunks.len(),
          doc.relative_path
        );
        (false, true)
      }
    }

    PreparedFile::Code(code) => {
      // Update content cache for future incremental parsing
      content_cache.insert(&root, &code.relative_path, code.content);

      // Build vectors - reuse existing or use new from cross-file batch
      let chunks_with_vectors: Vec<_> = code
        .chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
          let vector = if let Some(ref hash) = chunk.content_hash {
            // Try to reuse existing embedding
            code
              .existing_embeddings
              .get(hash)
              .cloned()
              .or_else(|| embeddings.get(&(file_idx, i)).cloned())
              .unwrap_or_else(|| vec![0.0f32; dim])
          } else {
            embeddings
              .get(&(file_idx, i))
              .cloned()
              .unwrap_or_else(|| vec![0.0f32; dim])
          };
          (chunk, vector)
        })
        .collect();

      if let Err(e) = db.add_code_chunks(&chunks_with_vectors).await {
        warn!("Failed to batch insert chunks for {}: {}", code.relative_path, e);
        return (false, false);
      }

      debug!(
        "Batch inserted {} chunks for {}",
        chunks_with_vectors.len(),
        code.relative_path
      );

      // Extract and store references for efficient caller/callee lookups
      let references: Vec<CodeReference> = chunks_with_vectors
        .iter()
        .flat_map(|(chunk, _)| {
          chunk
            .calls
            .iter()
            .map(|call| CodeReference::from_call(&project_id, &chunk.id.to_string(), call))
        })
        .collect();

      if !references.is_empty() {
        let chunk_ids: Vec<String> = chunks_with_vectors.iter().map(|(c, _)| c.id.to_string()).collect();
        if let Err(e) = db.delete_references_for_chunks(&chunk_ids).await {
          warn!("Failed to delete old references for {}: {}", code.relative_path, e);
        }

        if let Err(e) = db.insert_references(&references).await {
          warn!("Failed to insert references for {}: {}", code.relative_path, e);
        } else {
          debug!("Inserted {} references for {}", references.len(), code.relative_path);
        }
      }

      (true, false)
    }

    PreparedFile::Skip => (false, false),
  }
}

/// Process multiple file changes with cross-file embedding batching.
///
/// This is the main optimization: instead of each file making its own embedding API call,
/// we collect all texts that need embedding across ALL files, make ONE batch call,
/// then distribute the embeddings back.
///
/// For example: 10 files with 3 chunks each = 1 API call instead of 10.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn process_file_changes_batched(
  file_contexts: Vec<FileChangeContext>,
  db: Arc<ProjectDb>,
  embedding: Option<Arc<dyn EmbeddingProvider>>,
  project_id: String,
  root: PathBuf,
  docs_config: engram_core::DocsConfig,
  content_cache: Arc<crate::cache::FileContentCache>,
  parallel_files: usize,
) -> (usize, usize) {
  use tokio::sync::Semaphore;

  let num_files = file_contexts.len();

  // Phase 1: Prepare all files in parallel (read, parse, identify what needs embedding)
  let semaphore = Arc::new(Semaphore::new(parallel_files));
  let mut prepare_tasks = Vec::with_capacity(num_files);

  for ctx in file_contexts {
    let db_clone = Arc::clone(&db);
    let root_clone = root.clone();
    let docs_config_clone = docs_config.clone();
    let project_id_clone = project_id.clone();
    let sem_clone = Arc::clone(&semaphore);

    prepare_tasks.push(async move {
      let _permit = sem_clone.acquire().await.ok()?;
      Some(prepare_file_change(ctx, db_clone, root_clone, docs_config_clone, project_id_clone).await)
    });
  }

  let prepared_files: Vec<PreparedFile> = futures::future::join_all(prepare_tasks)
    .await
    .into_iter()
    .flatten()
    .collect();

  if prepared_files.is_empty() {
    return (0, 0);
  }

  // Phase 2: Collect all texts needing embedding across ALL files
  let mut embedding_requests: Vec<EmbeddingRequest> = Vec::new();

  for (file_idx, prepared) in prepared_files.iter().enumerate() {
    match prepared {
      PreparedFile::Code(code) => {
        for &chunk_idx in &code.chunks_needing_embeddings {
          if let Some(chunk) = code.chunks.get(chunk_idx) {
            embedding_requests.push(EmbeddingRequest {
              file_idx,
              chunk_idx,
              text: chunk.content.clone(),
            });
          }
        }
      }
      PreparedFile::Doc(doc) => {
        // All doc chunks need embedding (no reuse mechanism for docs currently)
        for (chunk_idx, chunk) in doc.chunks.iter().enumerate() {
          embedding_requests.push(EmbeddingRequest {
            file_idx,
            chunk_idx,
            text: chunk.content.clone(),
          });
        }
      }
      _ => {}
    }
  }

  // Phase 2b: Single batch embedding call for ALL texts
  let dim = embedding.as_ref().map(|e| e.dimensions()).unwrap_or(4096);
  let embeddings: HashMap<(usize, usize), Vec<f32>> = if !embedding_requests.is_empty() {
    if let Some(ref emb) = embedding {
      let texts: Vec<&str> = embedding_requests.iter().map(|r| r.text.as_str()).collect();

      debug!(
        "Cross-file batch embedding: {} texts from {} files",
        texts.len(),
        prepared_files.len()
      );

      match emb.embed_batch(&texts).await {
        Ok(vectors) => {
          // Map vectors back to (file_idx, chunk_idx)
          embedding_requests
            .iter()
            .zip(vectors.into_iter())
            .map(|(req, vec)| ((req.file_idx, req.chunk_idx), vec))
            .collect()
        }
        Err(e) => {
          warn!("Cross-file batch embedding failed: {}, using zero vectors", e);
          embedding_requests
            .iter()
            .map(|req| ((req.file_idx, req.chunk_idx), vec![0.0f32; dim]))
            .collect()
        }
      }
    } else {
      HashMap::new()
    }
  } else {
    HashMap::new()
  };

  // Phase 3: Finalize all files in parallel (insert with embeddings)
  let embeddings = Arc::new(embeddings);
  let mut finalize_tasks = Vec::with_capacity(prepared_files.len());

  for (file_idx, prepared) in prepared_files.into_iter().enumerate() {
    let db_clone = Arc::clone(&db);
    let root_clone = root.clone();
    let project_id_clone = project_id.clone();
    let cache_clone = Arc::clone(&content_cache);
    let embeddings_clone = Arc::clone(&embeddings);
    let sem_clone = Arc::clone(&semaphore);

    finalize_tasks.push(async move {
      let _permit = sem_clone.acquire().await.ok()?;
      Some(
        finalize_file_change(
          prepared,
          db_clone,
          root_clone,
          project_id_clone,
          cache_clone,
          &embeddings_clone,
          file_idx,
          dim,
        )
        .await,
      )
    });
  }

  let results: Vec<(bool, bool)> = futures::future::join_all(finalize_tasks)
    .await
    .into_iter()
    .flatten()
    .collect();

  // Count indexed files
  let mut code_count = 0;
  let mut doc_count = 0;
  for (is_code, is_doc) in results {
    if is_code {
      code_count += 1;
    }
    if is_doc {
      doc_count += 1;
    }
  }

  if code_count > 0 || doc_count > 0 {
    debug!(
      "Batch processed {} files: {} code, {} docs",
      num_files, code_count, doc_count
    );
  }

  (code_count, doc_count)
}

#[allow(clippy::too_many_arguments)]
/// Run the file watcher loop (blocking, runs in spawn_blocking)
fn run_watcher_loop(
  mut watcher: DebouncedWatcher,
  db: Arc<ProjectDb>,
  embedding: Option<Arc<dyn EmbeddingProvider>>,
  cancel: Arc<AtomicBool>,
  project_id: &str,
  root: &Path,
  _data_dir: PathBuf,
  content_cache: Arc<crate::cache::FileContentCache>,
) {
  let mut config = Config::load_for_project(root);
  let mut files_indexed = 0;
  let mut docs_indexed = 0;

  // Track config file for reloading
  let config_path = Config::project_config_path(root);

  // Set up docs directory watching if configured
  let mut docs_dir = config.docs.directory.as_ref().map(|d| root.join(d));
  let mut doc_extensions: HashSet<String> = config.docs.extensions.iter().cloned().collect();
  let mut parallel_files = config.index.parallel_files.max(1);

  info!(
    "Watcher loop started for {} (parallel_files={})",
    project_id, parallel_files
  );
  if let Some(ref dir) = docs_dir {
    info!("Watching docs directory: {:?}", dir);
  }

  // Maximum time to wait for events before checking cancellation
  let max_wait = Duration::from_secs(5);

  loop {
    // Check for cancellation
    if cancel.load(Ordering::SeqCst) {
      debug!("Watcher cancelled for {}", project_id);
      break;
    }

    // Wait for changes (blocks efficiently, no busy-polling)
    let changes = match watcher.wait_ready(max_wait) {
      Ok(c) => c,
      Err(e) => {
        warn!("Watcher error: {}", e);
        continue;
      }
    };

    // No changes (timeout) - loop to check cancellation
    if changes.is_empty() {
      continue;
    }

    // Check for gitignore changes
    if watcher.check_gitignore_change() {
      info!(
        "Gitignore changed for {}, triggering full re-index would be needed",
        project_id
      );
    }

    // Check for config file changes and reload
    let config_changed = changes.iter().any(|c| c.path == config_path);
    if config_changed {
      info!("Config file changed, reloading configuration");
      let old_dimensions = config.embedding.dimensions;
      let old_model = config.embedding.model.clone();

      config = Config::load_for_project(root);
      docs_dir = config.docs.directory.as_ref().map(|d| root.join(d));
      doc_extensions = config.docs.extensions.iter().cloned().collect();
      parallel_files = config.index.parallel_files.max(1);

      if let Some(ref dir) = docs_dir {
        info!("Updated docs directory: {:?}", dir);
      }

      // Warn about settings that require daemon restart
      if config.embedding.dimensions != old_dimensions || config.embedding.model != old_model {
        warn!("Embedding configuration changed - restart daemon to apply (ccengram daemon)");
      }
    }

    // Separate rename events from other changes
    // Renames are processed first to preserve embeddings
    let mut rename_events: Vec<(String, String)> = Vec::new();
    let mut other_changes: Vec<index::FileChange> = Vec::new();

    for change in changes {
      // Skip config file (already handled)
      if change.path == config_path {
        continue;
      }

      // Skip if should be ignored (uses cached gitignore patterns)
      if GITIGNORE_CACHE.should_ignore(root, &change.path) {
        debug!("Ignoring change to {:?}", change.path);
        continue;
      }

      // Handle rename events specially
      if change.kind == ChangeKind::Renamed {
        if let Some(ref old_path) = change.old_path {
          // Get relative paths for both old and new
          let old_relative = match old_path.strip_prefix(root) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => continue,
          };
          let new_relative = match change.path.strip_prefix(root) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => continue,
          };

          debug!("Detected rename: {} -> {}", old_relative, new_relative);
          rename_events.push((old_relative, new_relative));
        } else {
          // No old_path, treat as regular modified event
          other_changes.push(index::FileChange {
            path: change.path,
            kind: ChangeKind::Modified,
            old_path: None,
          });
        }
      } else {
        other_changes.push(change);
      }
    }

    // Process renames first (update paths in DB, preserve embeddings)
    if !rename_events.is_empty()
      && let Ok(rt) = tokio::runtime::Handle::try_current()
    {
      let db_clone = Arc::clone(&db);
      let content_cache_clone = Arc::clone(&content_cache);
      let root_clone = root.to_path_buf();

      for (old_path, new_path) in rename_events {
        let renamed = rt.block_on(async { db_clone.rename_file(&old_path, &new_path).await });

        match renamed {
          Ok(count) => {
            if count > 0 {
              info!("Renamed {} -> {} ({} chunks)", old_path, new_path, count);
              files_indexed += 1; // Count as an indexed operation

              // Update content cache: remove old key, file will be added with new key on next read
              content_cache_clone.remove(&root_clone, &old_path);
            }
          }
          Err(e) => {
            warn!("Failed to rename {} -> {}: {}", old_path, new_path, e);
          }
        }
      }
    }

    // Collect and filter remaining changes into FileChangeContext
    let mut file_contexts: Vec<FileChangeContext> = Vec::new();

    for change in other_changes {
      // Get relative path
      let relative_path = match change.path.strip_prefix(root) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => continue,
      };

      debug!("Processing change: {:?} {:?}", change.kind, relative_path);

      // Check if this is a document file in the docs directory
      let is_doc_file = if let Some(ref docs_dir) = docs_dir {
        change.path.starts_with(docs_dir)
          && change
            .path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| doc_extensions.contains(ext))
      } else {
        false
      };

      let is_delete = matches!(change.kind, ChangeKind::Deleted);

      // Get cached old content for incremental parsing (skip for deletes and docs)
      let old_content = if !is_delete && !is_doc_file {
        let root_buf = root.to_path_buf();
        content_cache.get(&root_buf, &relative_path).map(|c| c.content)
      } else {
        None
      };

      file_contexts.push(FileChangeContext {
        change_path: change.path,
        relative_path,
        is_doc_file,
        is_delete,
        old_content,
      });
    }

    // Process file changes with cross-file embedding batching
    if !file_contexts.is_empty()
      && let Ok(rt) = tokio::runtime::Handle::try_current()
    {
      let results = rt.block_on(async {
        process_file_changes_batched(
          file_contexts,
          Arc::clone(&db),
          embedding.clone(),
          project_id.to_string(),
          root.to_path_buf(),
          config.docs.clone(),
          Arc::clone(&content_cache),
          parallel_files,
        )
        .await
      });

      files_indexed += results.0;
      docs_indexed += results.1;
    }
  }

  info!(
    "Watcher loop ended for {}, indexed {} code files and {} docs",
    project_id, files_indexed, docs_indexed
  );
}

impl Default for ProjectRegistry {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_project_id_consistency() {
    let dir = TempDir::new().unwrap();
    let id1 = ProjectId::from_path(dir.path());
    let id2 = ProjectId::from_path(dir.path());
    assert_eq!(id1, id2);
  }

  #[test]
  fn test_project_from_path() {
    let dir = TempDir::new().unwrap();
    let info = ProjectInfo::from_path(dir.path()).unwrap();
    assert!(!info.id.as_str().is_empty());
    assert!(!info.name.is_empty());
  }

  #[tokio::test]
  async fn test_registry_get_or_create() {
    let data_dir = TempDir::new().unwrap();
    let project_dir = TempDir::new().unwrap();

    let registry = ProjectRegistry::with_data_dir(data_dir.path().to_path_buf());
    let (info, _db) = registry.get_or_create(project_dir.path()).await.unwrap();

    // Second call should return cached connection
    let (info2, _db2) = registry.get_or_create(project_dir.path()).await.unwrap();
    assert_eq!(info.id, info2.id);

    // Should be in the list
    let projects = registry.list().await;
    assert_eq!(projects.len(), 1);
  }
}
