//! Test helpers for actor integration tests.
//!
//! Provides `ActorTestContext` which manages temporary directories, database setup,
//! and ProjectActor spawning for E2E testing of the actor indexing system.

use std::{sync::Arc, time::Duration};

use filetime::FileTime;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use crate::{
  actor::{
    handle::ProjectHandle,
    message::{ProjectActorPayload, ProjectActorResponse},
    project::{ProjectActor, ProjectActorConfig},
  },
  domain::{
    config::{Config, DaemonSettings},
    project::ProjectId,
  },
  embedding::EmbeddingProvider,
  ipc::{
    RequestData, ResponseData,
    types::{
      code::{CodeIndexParams, CodeRequest, CodeResponse, CodeSearchParams},
      watch::{WatchRequest, WatchStartParams, WatchStatusParams, WatchStopParams},
    },
  },
};

/// Test context for actor integration tests.
///
/// Manages temporary directories and provides helpers for spawning ProjectActors
/// and manipulating test files.
pub struct ActorTestContext {
  /// Temporary directory for the project (source files)
  pub project_dir: TempDir,
  /// Temporary directory for data (database)
  pub data_dir: TempDir,
  /// Project ID derived from project_dir
  pub project_id: ProjectId,
  #[allow(dead_code)]
  /// Configuration for tests (with short debounce)
  pub config: Arc<Config>,
  /// Embedding provider for tests
  pub embedding: Arc<dyn EmbeddingProvider>,
}

impl ActorTestContext {
  /// Create a new test context with the user's configured embedding provider.
  pub async fn new() -> Self {
    let project_dir = TempDir::new().expect("create project temp dir");
    let data_dir = TempDir::new().expect("create data temp dir");
    let project_id = ProjectId::from_path(project_dir.path()).await;

    // Load global config so test uses the same config that ProjectActor::spawn
    // will see (it calls Config::load_for_project which falls back to load_global)
    let mut config = Config::load_global().await;
    config.index.watcher_debounce_ms = 50; // Fast debounce for tests

    // Use the configured embedding provider - must match the config the actor will use
    let embedding = <dyn EmbeddingProvider>::from_config(&config.embedding)
      .await
      .expect("Embedding provider should be available for tests");

    Self {
      project_dir,
      data_dir,
      project_id,
      config: Arc::new(config),
      embedding,
    }
  }

  /// Spawn a ProjectActor and return a handle for communication.
  ///
  /// Returns the handle and a cancellation token to stop the actor.
  pub async fn spawn_project_actor(
    &self,
  ) -> Result<(ProjectHandle, CancellationToken), crate::actor::project::ProjectActorError> {
    let cancel = CancellationToken::new();

    let config = ProjectActorConfig {
      id: self.project_id.clone(),
      root: self.project_dir.path().to_path_buf(),
      data_dir: self.data_dir.path().to_path_buf(),
    };

    // Create daemon settings from the test config
    let daemon_settings = Arc::new(DaemonSettings::from_config(&self.config));

    let handle = ProjectActor::spawn(config, self.embedding.clone(), None, daemon_settings, cancel.clone()).await?;

    Ok((handle, cancel))
  }

  /// Write a source file to the project directory.
  pub async fn write_source_file(&self, path: &str, content: &str) {
    let full_path = self.project_dir.path().join(path);
    if let Some(parent) = full_path.parent() {
      tokio::fs::create_dir_all(parent).await.expect("create parent dirs");
    }
    tokio::fs::write(&full_path, content).await.expect("write file");
  }

  /// Delete a source file from the project directory.
  pub async fn delete_source_file(&self, path: &str) {
    let full_path = self.project_dir.path().join(path);
    let _ = tokio::fs::remove_file(full_path).await;
  }

  /// Rename a source file in the project directory.
  pub async fn rename_source_file(&self, from: &str, to: &str) {
    let from_path = self.project_dir.path().join(from);
    let to_path = self.project_dir.path().join(to);
    if let Some(parent) = to_path.parent() {
      tokio::fs::create_dir_all(parent).await.expect("create parent dirs");
    }
    tokio::fs::rename(from_path, to_path).await.expect("rename file");
  }

  /// Update file mtime without changing content.
  pub fn touch_file(&self, path: &str) {
    let full_path = self.project_dir.path().join(path);
    let now = FileTime::now();
    filetime::set_file_mtime(&full_path, now).expect("set mtime");
  }

  /// Create a .gitignore file in the project directory.
  ///
  /// Also creates a minimal .git directory so the ignore crate recognizes
  /// this as a git repository and respects the .gitignore.
  pub async fn write_gitignore(&self, content: &str) {
    // Create .git directory so the ignore crate treats this as a git repo
    let git_dir = self.project_dir.path().join(".git");
    tokio::fs::create_dir_all(&git_dir).await.expect("create .git dir");

    // Create minimal git config
    tokio::fs::write(git_dir.join("config"), "[core]\n\trepositoryformatversion = 0\n")
      .await
      .expect("write git config");

    // Create HEAD file
    tokio::fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n")
      .await
      .expect("write HEAD");

    // Now write the .gitignore
    self.write_source_file(".gitignore", content).await;
  }
}

/// Trigger code indexing via the ProjectHandle.
pub async fn trigger_index(handle: &ProjectHandle) -> Result<crate::ipc::code::CodeIndexResult, String> {
  let response = handle
    .request(
      "test-index".to_string(),
      ProjectActorPayload::Request(RequestData::Code(CodeRequest::Index(CodeIndexParams {
        force: false,
        stream: false,
      }))),
    )
    .await
    .map_err(|e| e.to_string())?;

  match response {
    ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Index(result))) => Ok(result),
    ProjectActorResponse::Error { code, message, .. } => Err(format!("Index error {}: {}", code, message)),
    other => Err(format!("Unexpected response: {:?}", other)),
  }
}

/// Search code via the ProjectHandle.
pub async fn search_code(
  handle: &ProjectHandle,
  query: &str,
) -> Result<crate::ipc::types::code::CodeSearchResult, String> {
  let response = handle
    .request(
      "test-search".to_string(),
      ProjectActorPayload::Request(RequestData::Code(CodeRequest::Search(CodeSearchParams {
        query: query.to_string(),
        limit: Some(10),
        file_pattern: None,
        symbol_type: None,
        language: None,
        visibility: vec![],
        chunk_type: vec![],
        min_caller_count: None,
      }))),
    )
    .await
    .map_err(|e| e.to_string())?;

  match response {
    ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Search(result))) => Ok(result),
    ProjectActorResponse::Error { code, message, .. } => Err(format!("Search error {}: {}", code, message)),
    other => Err(format!("Unexpected response: {:?}", other)),
  }
}

/// Start the file watcher via the ProjectHandle.
pub async fn start_watcher(handle: &ProjectHandle) -> Result<(), String> {
  let payload = ProjectActorPayload::Request(RequestData::Watch(WatchRequest::Start(WatchStartParams)));
  let response = handle
    .request("test-start-watcher".to_string(), payload)
    .await
    .map_err(|e| e.to_string())?;

  match response {
    ProjectActorResponse::Done(ResponseData::Watch(_)) => Ok(()),
    ProjectActorResponse::Error { code, message, .. } => Err(format!("Start watcher error {}: {}", code, message)),
    other => Err(format!("Unexpected response: {:?}", other)),
  }
}

/// Stop the file watcher via the ProjectHandle.
pub async fn stop_watcher(handle: &ProjectHandle) -> Result<(), String> {
  let payload = ProjectActorPayload::Request(RequestData::Watch(WatchRequest::Stop(WatchStopParams)));
  let response = handle
    .request("test-stop-watcher".to_string(), payload)
    .await
    .map_err(|e| e.to_string())?;

  match response {
    ProjectActorResponse::Done(ResponseData::Watch(_)) => Ok(()),
    ProjectActorResponse::Error { code, message, .. } => Err(format!("Stop watcher error {}: {}", code, message)),
    other => Err(format!("Unexpected response: {:?}", other)),
  }
}

/// Get watcher status via the ProjectHandle.
pub async fn get_watcher_status(handle: &ProjectHandle) -> Result<crate::ipc::types::watch::WatchStatusResult, String> {
  let payload = ProjectActorPayload::Request(RequestData::Watch(WatchRequest::Status(WatchStatusParams)));
  let response = handle
    .request("test-watcher-status".to_string(), payload)
    .await
    .map_err(|e| e.to_string())?;

  match response {
    ProjectActorResponse::Done(ResponseData::Watch(crate::ipc::types::watch::WatchResponse::Status(status))) => {
      Ok(status)
    }
    ProjectActorResponse::Error { code, message, .. } => Err(format!("Watcher status error {}: {}", code, message)),
    other => Err(format!("Unexpected response: {:?}", other)),
  }
}

/// Wait for a condition to become true, with timeout.
pub async fn wait_for<F, Fut>(timeout: Duration, mut check: F) -> bool
where
  F: FnMut() -> Fut,
  Fut: std::future::Future<Output = bool>,
{
  let start = std::time::Instant::now();
  let poll_interval = Duration::from_millis(50);

  while start.elapsed() < timeout {
    if check().await {
      return true;
    }
    tokio::time::sleep(poll_interval).await;
  }

  false
}

/// Wait for scan to complete (scanning = false).
pub async fn wait_for_scan_complete(handle: &ProjectHandle, timeout: Duration) -> bool {
  wait_for(timeout, || async {
    if let Ok(status) = get_watcher_status(handle).await {
      !status.scanning
    } else {
      false
    }
  })
  .await
}
