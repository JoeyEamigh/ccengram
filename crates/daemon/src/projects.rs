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
  watcher_status: RwLock<HashMap<String, WatcherStatus>>,
  watcher_tasks: RwLock<HashMap<String, WatcherTask>>,
  /// Coordinator for multi-instance watcher locking
  coordinator: WatcherCoordinator,
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
      watcher_status: RwLock::new(HashMap::new()),
      watcher_tasks: RwLock::new(HashMap::new()),
      coordinator: WatcherCoordinator::with_locks_dir(data_dir.join("watchers")),
    }
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
    use serde_json::json;

    if let Some(parent) = path.parent() {
      tokio::fs::create_dir_all(parent).await?;
    }

    let metadata = json!({
        "id": info.id.as_str(),
        "path": info.path.to_string_lossy(),
        "name": info.name,
    });

    let json = serde_json::to_string_pretty(&metadata).map_err(|e| ProjectError::Serialization(e.to_string()))?;
    tokio::fs::write(path, json).await?;
    Ok(())
  }

  // Watcher management

  /// Start a file watcher for a project with actual file system watching
  pub async fn start_watcher(
    &self,
    id: &str,
    root: &Path,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
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

    // Load config for debounce settings
    let config = Config::load_for_project(root);
    let debounce_ms = config.index.watcher_debounce_ms;

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
      );
    });

    // Store the task
    {
      let mut tasks = self.watcher_tasks.write().await;
      tasks.insert(id.to_string(), WatcherTask { handle, cancel });
    }

    // Update status
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
        },
      );
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
}

/// Holds context for processing a file change
struct FileChangeContext {
  change_path: PathBuf,
  relative_path: String,
  is_doc_file: bool,
  is_delete: bool,
}

/// Process a single file change (async, for parallel processing)
async fn process_file_change(
  ctx: FileChangeContext,
  db: Arc<ProjectDb>,
  embedding: Option<Arc<dyn EmbeddingProvider>>,
  project_id: String,
  root: PathBuf,
  docs_config: engram_core::DocsConfig,
) -> Result<(bool, bool), ()> {
  // Returns (is_code_file_indexed, is_doc_indexed)
  let mut code_indexed = false;
  let mut doc_indexed = false;

  if ctx.is_delete {
    // Handle deletes
    if ctx.is_doc_file {
      let filter = format!("source = '{}'", ctx.relative_path.replace('\'', "''"));
      if let Ok(chunks) = db.list_document_chunks(Some(&filter), Some(1)).await
        && let Some(chunk) = chunks.first()
      {
        let doc_id = chunk.document_id;
        if let Err(e) = db.delete_document(&doc_id).await {
          warn!("Failed to delete document {}: {}", ctx.relative_path, e);
        }
      }
    } else if let Err(e) = db.delete_chunks_for_file(&ctx.relative_path).await {
      warn!("Failed to delete chunks for {}: {}", ctx.relative_path, e);
    }
    return Ok((false, false));
  }

  // Handle create/modify
  if ctx.is_doc_file {
    // Index as document
    if let Ok(content) = tokio::fs::read_to_string(&ctx.change_path).await {
      if content.len() > docs_config.max_file_size {
        debug!("Skipping large doc file: {:?}", ctx.change_path);
        return Ok((false, false));
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
        return Ok((false, false));
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

      // Batch embed all chunks
      let vectors: Vec<Option<Vec<f32>>> = if let Some(ref emb) = embedding {
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        match emb.embed_batch(&texts).await {
          Ok(vecs) => vecs.into_iter().map(Some).collect(),
          Err(e) => {
            warn!(
              "Batch embedding failed for doc {}: {}, using no vectors",
              ctx.relative_path, e
            );
            chunks.iter().map(|_| None).collect()
          }
        }
      } else {
        chunks.iter().map(|_| None).collect()
      };

      // Batch insert all chunks
      if let Err(e) = db.add_document_chunks(&chunks, &vectors).await {
        warn!(
          "Failed to batch insert document chunks for {}: {}",
          ctx.relative_path, e
        );
      } else {
        debug!(
          "Batch inserted {} document chunks for {}",
          chunks.len(),
          ctx.relative_path
        );
        doc_indexed = true;
      }
    }
  } else {
    // Index as code
    let scanner = Scanner::new();
    if let Some(scanned) = scanner.scan_file(&ctx.change_path, &root) {
      // Read and chunk the file
      if let Ok(content) = tokio::fs::read_to_string(&ctx.change_path).await {
        let mut chunker = Chunker::default();
        let new_chunks = chunker.chunk(&content, &ctx.relative_path, scanned.language, &scanned.checksum);

        if new_chunks.is_empty() {
          return Ok((false, false));
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
        let mut chunks_needing_embeddings: Vec<(usize, &str)> = Vec::new();
        let mut reused_count = 0;

        for (i, chunk) in new_chunks.iter().enumerate() {
          if let Some(ref hash) = chunk.content_hash {
            if !existing_embeddings.contains_key(hash) {
              // Need new embedding
              chunks_needing_embeddings.push((i, chunk.content.as_str()));
            } else {
              reused_count += 1;
            }
          } else {
            // No hash, need new embedding
            chunks_needing_embeddings.push((i, chunk.content.as_str()));
          }
        }

        if reused_count > 0 {
          debug!(
            "Reusing {} embeddings for {} (generating {} new)",
            reused_count,
            ctx.relative_path,
            chunks_needing_embeddings.len()
          );
        }

        // Generate embeddings only for chunks that need them
        let new_embeddings: HashMap<usize, Vec<f32>> = if !chunks_needing_embeddings.is_empty() {
          if let Some(ref emb) = embedding {
            let texts: Vec<&str> = chunks_needing_embeddings.iter().map(|(_, t)| *t).collect();
            match emb.embed_batch(&texts).await {
              Ok(vecs) => chunks_needing_embeddings
                .iter()
                .map(|(i, _)| *i)
                .zip(vecs.into_iter())
                .collect(),
              Err(e) => {
                warn!(
                  "Batch embedding failed for {}: {}, using zero vectors",
                  ctx.relative_path, e
                );
                let dim = emb.dimensions();
                chunks_needing_embeddings
                  .iter()
                  .map(|(i, _)| (*i, vec![0.0f32; dim]))
                  .collect()
              }
            }
          } else {
            HashMap::new()
          }
        } else {
          HashMap::new()
        };

        // Prepare batch data for insert - reuse existing embeddings where possible
        let dim = embedding.as_ref().map(|e| e.dimensions()).unwrap_or(4096);
        let chunks_with_vectors: Vec<_> = new_chunks
          .into_iter()
          .enumerate()
          .map(|(i, chunk)| {
            let vector = if let Some(ref hash) = chunk.content_hash {
              // Try to reuse existing embedding
              existing_embeddings
                .get(hash)
                .cloned()
                .or_else(|| new_embeddings.get(&i).cloned())
                .unwrap_or_else(|| vec![0.0f32; dim])
            } else {
              new_embeddings.get(&i).cloned().unwrap_or_else(|| vec![0.0f32; dim])
            };
            (chunk, vector)
          })
          .collect();

        // Batch insert all chunks
        if let Err(e) = db.add_code_chunks(&chunks_with_vectors).await {
          warn!("Failed to batch insert chunks for {}: {}", ctx.relative_path, e);
        } else {
          debug!(
            "Batch inserted {} chunks for {}",
            chunks_with_vectors.len(),
            ctx.relative_path
          );
          code_indexed = true;

          // Extract and store references for efficient caller/callee lookups
          // This replaces the expensive LIKE queries on JSON columns
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
            // Delete old references for chunks in this file (by chunk IDs)
            let chunk_ids: Vec<String> = chunks_with_vectors.iter().map(|(c, _)| c.id.to_string()).collect();
            if let Err(e) = db.delete_references_for_chunks(&chunk_ids).await {
              warn!("Failed to delete old references for {}: {}", ctx.relative_path, e);
            }

            // Insert new references
            if let Err(e) = db.insert_references(&references).await {
              warn!("Failed to insert references for {}: {}", ctx.relative_path, e);
            } else {
              debug!("Inserted {} references for {}", references.len(), ctx.relative_path);
            }
          }
        }
      }
    }
  }

  Ok((code_indexed, doc_indexed))
}

/// Run the file watcher loop (blocking, runs in spawn_blocking)
fn run_watcher_loop(
  mut watcher: DebouncedWatcher,
  db: Arc<ProjectDb>,
  embedding: Option<Arc<dyn EmbeddingProvider>>,
  cancel: Arc<AtomicBool>,
  project_id: &str,
  root: &Path,
  _data_dir: PathBuf,
) {
  let mut config = Config::load_for_project(root);

  let poll_interval = Duration::from_millis(config.index.watcher_debounce_ms.max(100));
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

  loop {
    // Check for cancellation
    if cancel.load(Ordering::SeqCst) {
      debug!("Watcher cancelled for {}", project_id);
      break;
    }

    // Poll for changes
    let changes = watcher.collect_ready();

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

    // Collect and filter changes
    let mut file_contexts: Vec<FileChangeContext> = Vec::new();

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

      file_contexts.push(FileChangeContext {
        change_path: change.path,
        relative_path,
        is_doc_file,
        is_delete,
      });
    }

    // Process file changes in parallel (P1)
    if !file_contexts.is_empty()
      && let Ok(rt) = tokio::runtime::Handle::try_current()
    {
      let results: Vec<Option<(bool, bool)>> = rt.block_on(async {
        use std::sync::Arc as StdArc;
        use tokio::sync::Semaphore;

        let semaphore = StdArc::new(Semaphore::new(parallel_files));
        let mut tasks = Vec::with_capacity(file_contexts.len());

        for ctx in file_contexts {
          let db_clone = Arc::clone(&db);
          let emb_clone = embedding.clone();
          let project_id_clone = project_id.to_string();
          let root_clone = root.to_path_buf();
          let docs_config_clone = config.docs.clone();
          let sem_clone = StdArc::clone(&semaphore);

          tasks.push(async move {
            // Acquire semaphore permit to limit concurrency
            let _permit = sem_clone.acquire().await.ok()?;
            process_file_change(
              ctx,
              db_clone,
              emb_clone,
              project_id_clone,
              root_clone,
              docs_config_clone,
            )
            .await
            .ok()
          });
        }

        futures::future::join_all(tasks).await
      });

      // Count indexed files
      for result in results.into_iter().flatten() {
        if result.0 {
          files_indexed += 1;
        }
        if result.1 {
          docs_indexed += 1;
        }
      }
    }

    // Sleep before next poll
    std::thread::sleep(poll_interval);
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
