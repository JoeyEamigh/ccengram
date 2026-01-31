//! ProjectRouter - Routes requests to ProjectActors, spawning them on demand
//!
//! The router is a thin layer that maps project paths to their corresponding
//! `ProjectActor` instances. It uses lock-free concurrent access via `DashMap`
//! and handles race conditions when multiple requests try to spawn the same project.
//!
//! # Design Principles
//!
//! - **No god objects**: Router only routes, actors own their state
//! - **Lock-free**: `DashMap` instead of `RwLock<HashMap<...>>`
//! - **Idempotent**: `get_or_create` handles race conditions safely
//! - **Immutable sharing**: Embedding provider shared via `Arc`
//!
//! # Usage
//!
//! ```ignore
//! let router = ProjectRouter::new(data_dir, Some(embedding), cancel_token);
//! let handle = router.get_or_create(Path::new("/my/project")).await?;
//! let response = handle.request("req-1".to_string(), payload).await?;
//! ```

use std::{
  path::{Path, PathBuf},
  sync::Arc,
};

use dashmap::DashMap;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::{
  handle::ProjectHandle,
  message::{ProjectActorMessage, ProjectActorPayload},
  project::{ProjectActor, ProjectActorConfig, ProjectActorError},
};
use crate::{domain::project::ProjectId, embedding::EmbeddingProvider};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur in the ProjectRouter
#[derive(Debug, thiserror::Error)]
pub enum ProjectRouterError {
  #[error("Failed to spawn ProjectActor: {0}")]
  SpawnFailed(#[source] ProjectActorError),
}

// ============================================================================
// ProjectRouter
// ============================================================================

/// Routes requests to ProjectActors, spawning them on demand
///
/// The router maintains a map of active projects and spawns new `ProjectActor`
/// instances as needed. It uses `DashMap` for lock-free concurrent access,
/// which is important for high-throughput scenarios.
///
/// # Thread Safety
///
/// The router is safe to share across tasks via `Arc<ProjectRouter>`. All
/// operations are either atomic (via `DashMap`) or use async coordination.
///
/// # Lifecycle
///
/// - Projects are spawned lazily on first access via `get_or_create`
/// - Each project gets a child `CancellationToken` for coordinated shutdown
/// - `shutdown_project` gracefully terminates a single project
/// - `shutdown_all` terminates all projects (used during daemon shutdown)
pub struct ProjectRouter {
  /// Active project actors, keyed by ProjectId
  ///
  /// Using DashMap for lock-free concurrent access instead of RwLock<HashMap>
  projects: DashMap<ProjectId, ProjectHandle>,

  /// Path -> ProjectId cache to avoid repeated git root lookups
  ///
  /// Computing ProjectId requires finding the git root, which involves
  /// filesystem operations. This cache stores resolved mappings to make
  /// subsequent lookups for the same path instant.
  path_cache: DashMap<PathBuf, ProjectId>,

  /// Base data directory for project databases
  ///
  /// Each project gets its own subdirectory: `{data_dir}/projects/{project_id}/`
  data_dir: PathBuf,

  /// Shared embedding provider (immutable, just needs Arc)
  ///
  /// All projects share the same embedding provider. Since it's immutable
  /// and thread-safe, we just clone the Arc for each project.
  embedding: Arc<dyn EmbeddingProvider>,

  /// Parent cancellation token
  ///
  /// Each spawned ProjectActor gets a child token. When this token is
  /// cancelled, all project actors will shut down.
  cancel: CancellationToken,
}

impl ProjectRouter {
  /// Create a new ProjectRouter
  ///
  /// # Arguments
  ///
  /// * `data_dir` - Base directory for project databases
  /// * `embedding` - Optional shared embedding provider
  /// * `cancel` - Parent cancellation token for coordinated shutdown
  pub fn new(data_dir: PathBuf, embedding: Arc<dyn EmbeddingProvider>, cancel: CancellationToken) -> Self {
    Self {
      projects: DashMap::new(),
      path_cache: DashMap::new(),
      data_dir,
      embedding,
      cancel,
    }
  }

  /// Get or create a ProjectActor for the given path
  ///
  /// This method is idempotent - calling it multiple times with the same
  /// path will return the same handle. If multiple tasks call this concurrently
  /// for a new project, only one actor will be spawned (via DashMap's entry API).
  ///
  /// The path is resolved to a git root if possible, ensuring that requests
  /// from any subdirectory of a project map to the same ProjectActor.
  ///
  /// # Arguments
  ///
  /// * `path` - Path to the project (can be any subdirectory)
  ///
  /// # Returns
  ///
  /// A cloned handle to the ProjectActor. Handles are cheap to clone.
  pub async fn get_or_create(&self, path: &Path) -> Result<ProjectHandle, ProjectRouterError> {
    // Canonicalize path for consistent cache keys
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Check path cache first to avoid repeated git root lookups
    let id = if let Some(cached_id) = self.path_cache.get(&canonical) {
      cached_id.value().clone()
    } else {
      // Compute project ID (this resolves git root if available)
      let id = ProjectId::from_path(path).await;
      self.path_cache.insert(canonical, id.clone());
      id
    };

    // Fast path: project already exists
    if let Some(handle) = self.projects.get(&id) {
      debug!(project_id = %id, "Reusing existing ProjectActor");
      return Ok(handle.value().clone());
    }

    // Slow path: need to create the actor
    // We use entry().or_try_insert_with() to handle the race condition
    // where multiple tasks might try to create the same project
    self.spawn_project(id, path).await
  }

  /// Get an existing ProjectActor handle without spawning
  ///
  /// Returns `None` if the project hasn't been accessed yet.
  pub fn get(&self, id: &ProjectId) -> Option<ProjectHandle> {
    self.projects.get(id).map(|h| h.value().clone())
  }

  /// Spawn a new project actor (internal helper)
  ///
  /// This handles the race condition where multiple tasks might try to
  /// spawn the same project concurrently. Only one will succeed in inserting
  /// into the DashMap.
  async fn spawn_project(&self, id: ProjectId, path: &Path) -> Result<ProjectHandle, ProjectRouterError> {
    // Resolve the actual project root (git root or the path itself)
    let root = crate::domain::project::resolve_project_path(path).await;

    // Check again after resolving - another task might have inserted
    if let Some(handle) = self.projects.get(&id) {
      debug!(project_id = %id, "ProjectActor created by another task");
      return Ok(handle.value().clone());
    }

    // Create config for the actor
    let config = ProjectActorConfig {
      id: id.clone(),
      root: root.clone(),
      data_dir: self.data_dir.clone(),
    };

    // Spawn the actor with a child cancellation token
    let handle = ProjectActor::spawn(config, self.embedding.clone(), self.cancel.child_token())
      .await
      .map_err(ProjectRouterError::SpawnFailed)?;

    info!(project_id = %id, root = %root.display(), "Spawned new ProjectActor");

    // Insert into the map
    // Using entry API to handle race condition - if another task inserted
    // while we were spawning, use their handle instead
    let entry = self.projects.entry(id.clone());
    let final_handle = match entry {
      dashmap::mapref::entry::Entry::Occupied(existing) => {
        // Another task won the race - use their handle
        // Our spawned actor will shut down due to having no handle refs
        warn!(project_id = %id, "Race condition: using existing ProjectActor");
        existing.get().clone()
      }
      dashmap::mapref::entry::Entry::Vacant(vacant) => {
        // We won - insert our handle
        vacant.insert(handle.clone());
        handle
      }
    };

    Ok(final_handle)
  }

  /// List all active project IDs
  ///
  /// Returns a snapshot of active projects. The actual set may change
  /// immediately after this call returns.
  pub fn list(&self) -> Vec<ProjectId> {
    self.projects.iter().map(|entry| entry.key().clone()).collect()
  }

  /// Get embedding provider info for metrics.
  pub fn embedding_info(&self) -> (String, String, usize) {
    (
      self.embedding.name().to_string(),
      self.embedding.model_id().to_string(),
      self.embedding.dimensions(),
    )
  }

  /// Shutdown a specific project
  ///
  /// Sends a shutdown message to the project actor and removes it from
  /// the active projects map. The actor will clean up its resources
  /// (database, indexer, watcher) before terminating.
  ///
  /// This is a graceful shutdown - we send a message and let the actor
  /// handle cleanup. The actor may take some time to fully stop.
  pub async fn shutdown_project(&self, id: &ProjectId) {
    if let Some((_, handle)) = self.projects.remove(id) {
      info!(project_id = %id, "Shutting down ProjectActor");

      // Create a one-shot reply channel (we don't need the response)
      let (reply_tx, _reply_rx) = tokio::sync::mpsc::channel(1);

      // Send shutdown message
      let shutdown_msg = ProjectActorMessage {
        id: format!("shutdown-{}", id),
        reply: reply_tx,
        payload: ProjectActorPayload::Shutdown,
      };

      // Best-effort send - actor might already be dead
      if let Err(e) = handle.tx.send(shutdown_msg).await {
        debug!(project_id = %id, error = %e, "Failed to send shutdown (actor may already be stopped)");
      }
    }
  }

  /// Shutdown all active projects
  ///
  /// Iterates through all projects and sends shutdown messages. This is
  /// called during daemon shutdown to gracefully terminate all actors.
  ///
  /// Projects are shut down in parallel for faster overall shutdown.
  pub async fn shutdown_all(&self) {
    let ids: Vec<ProjectId> = self.list();

    if ids.is_empty() {
      return;
    }

    info!(count = ids.len(), "Shutting down all ProjectActors");

    // Shutdown all projects concurrently
    let futures: Vec<_> = ids.iter().map(|id| self.shutdown_project(id)).collect();

    futures::future::join_all(futures).await;

    info!("All ProjectActors shut down");
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::Config;

  #[tokio::test]
  async fn test_project_id_consistency() {
    // ProjectId should be stable for the same path
    let path = Path::new("/tmp/test-project");
    let id1 = ProjectId::from_path(path).await;
    let id2 = ProjectId::from_path(path).await;
    assert_eq!(id1, id2);
  }

  #[tokio::test]
  async fn test_router_shutdown_nonexistent() {
    let config = Config::default();
    let embedding = <dyn EmbeddingProvider>::from_config(&config.embedding).expect("embedding provider required");
    let cancel = CancellationToken::new();
    let router = ProjectRouter::new(PathBuf::from("/tmp/data"), embedding, cancel);

    // Should not panic when shutting down nonexistent project
    let fake_id = ProjectId::from_path_exact(Path::new("/fake/project"));
    router.shutdown_project(&fake_id).await;
  }

  #[tokio::test]
  async fn test_router_shutdown_all_empty() {
    let config = Config::default();
    let embedding = <dyn EmbeddingProvider>::from_config(&config.embedding).expect("embedding provider required");
    let cancel = CancellationToken::new();
    let router = ProjectRouter::new(PathBuf::from("/tmp/data"), embedding, cancel);

    // Should not panic when no projects exist
    router.shutdown_all().await;
  }
}
