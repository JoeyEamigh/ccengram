use db::{ProjectDb, default_data_dir};
use embedding::EmbeddingProvider;
use engram_core::{Config, ProjectId};
use index::{ChangeKind, Chunker, DebounceConfig, DebouncedWatcher, Scanner, WatcherCoordinator, should_ignore};
use std::collections::HashMap;
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
  pub fn from_path(path: &Path) -> Result<Self, ProjectError> {
    let canonical = path.canonicalize()?;
    let id = ProjectId::from_path(&canonical);
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
  let chunker = Chunker::default();
  let scanner = Scanner::new();
  let config = Config::load_for_project(root);

  let poll_interval = Duration::from_millis(config.index.watcher_debounce_ms.max(100));
  let mut files_indexed = 0;

  info!("Watcher loop started for {}", project_id);

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
      // Could trigger a full re-index here, but for now just log
    }

    // Process changes
    for change in changes {
      // Skip if should be ignored
      if should_ignore(&change.path) {
        debug!("Ignoring change to {:?}", change.path);
        continue;
      }

      // Get relative path
      let relative_path = match change.path.strip_prefix(root) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => continue,
      };

      debug!("Processing change: {:?} {:?}", change.kind, relative_path);

      match change.kind {
        ChangeKind::Deleted => {
          // Delete chunks for this file
          let db_clone = Arc::clone(&db);
          let path_clone = relative_path.clone();
          if let Ok(rt) = tokio::runtime::Handle::try_current()
            && let Err(e) = rt.block_on(async { db_clone.delete_chunks_for_file(&path_clone).await })
          {
            warn!("Failed to delete chunks for {}: {}", path_clone, e);
          }
        }
        ChangeKind::Created | ChangeKind::Modified | ChangeKind::Renamed => {
          // Re-index the file
          if let Some(scanned) = scanner.scan_file(&change.path, root) {
            // Delete old chunks
            let db_clone = Arc::clone(&db);
            let path_clone = relative_path.clone();
            if let Ok(rt) = tokio::runtime::Handle::try_current()
              && let Err(e) = rt.block_on(async { db_clone.delete_chunks_for_file(&path_clone).await })
            {
              warn!("Failed to delete old chunks for {}: {}", path_clone, e);
            }

            // Read and chunk the file
            if let Ok(content) = std::fs::read_to_string(&change.path) {
              let chunks = chunker.chunk(&content, &relative_path, scanned.language, &scanned.checksum);

              // Store chunks
              for chunk in chunks {
                let vector = if let Some(ref emb) = embedding {
                  if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    rt.block_on(async { emb.embed(&chunk.content).await.ok() })
                  } else {
                    None
                  }
                } else {
                  None
                };

                let vector_slice: Option<Vec<f32>> = vector.map(|v| v.into_iter().collect());

                let db_clone = Arc::clone(&db);
                let chunk_clone = chunk.clone();
                if let Ok(rt) = tokio::runtime::Handle::try_current()
                  && let Err(e) =
                    rt.block_on(async { db_clone.add_code_chunk(&chunk_clone, vector_slice.as_deref()).await })
                {
                  warn!("Failed to add chunk for {}: {}", relative_path, e);
                }
              }

              files_indexed += 1;
            }
          }
        }
      }
    }

    // Sleep before next poll
    std::thread::sleep(poll_interval);
  }

  info!("Watcher loop ended for {}, indexed {} files", project_id, files_indexed);
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
