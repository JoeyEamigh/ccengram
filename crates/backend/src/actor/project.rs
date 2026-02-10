//! ProjectActor - Per-project actor that owns database, indexer, and watcher
//!
//! This actor is the central coordinator for all project-specific operations:
//! - Owns the project database connection (via Arc for sharing with Indexer)
//! - Owns the IndexerActor handle for sending indexing jobs
//! - Manages the WatcherTask lifecycle (start/stop via messages)
//! - Processes all project-level requests (memory, code, entity, etc.)
//!
//! # Lifecycle
//!
//! The actor runs until one of:
//! - The CancellationToken is triggered
//! - A ProjectPayload::Shutdown message is received
//! - The request channel is closed
//!
//! # Message Flow
//!
//! ```text
//! IPC Server -> ProjectRouter -> ProjectActor -> [IndexerActor, WatcherTask, DB]
//!                                     |
//!                                     v
//!                            Response Channel (mpsc, supports streaming)
//! ```

use std::{path::PathBuf, sync::Arc, time::Duration};

use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::{
  handle::{IndexerHandle, ProjectHandle},
  indexer::{IndexerActor, IndexerConfig},
  message::{ProjectActorMessage, ProjectActorPayload, ProjectActorResponse},
  watcher::{WatcherConfig, WatcherTask},
};
use crate::{
  db::{DbError, ProjectDb},
  domain::{
    code::Language,
    config::{Config, DaemonSettings},
    project::ProjectId,
  },
  embedding::EmbeddingProvider,
  ipc::{
    RequestData, ResponseData,
    code::{CodeIndexResult, CodeItem, CodeMemoriesResponse},
    hook::{HookParams, HookResult},
    memory::{
      MemoryDeleteParams, MemoryDeleteResult, MemoryHardDeleteParams, MemoryItem, MemoryListDeletedParams,
      MemoryReinforceParams, MemoryRestoreParams, MemorySetSalienceParams, MemorySummary, MemoryTimelineParams,
    },
    project::ProjectResponse,
    relationship::{RelatedMemoryItem, RelationshipInfo, RelationshipListParams, RelationshipResponse},
    search::{ContextParams, ExploreParams},
    types::{
      code::{
        CodeCalleesParams, CodeCallersParams, CodeContextFullParams, CodeContextParams, CodeIndexParams,
        CodeListParams, CodeMemoriesParams, CodeRelatedParams, CodeRequest, CodeResponse, CodeSearchParams,
        CodeStatsParams,
      },
      docs::{DocContextParams, DocsIngestParams, DocsRequest, DocsResponse},
      memory::{
        MemoryDeemphasizeParams, MemoryRelatedParams, MemoryRequest, MemoryResponse, MemoryRestoreResult,
        MemorySupersedeParams,
      },
      project::ProjectRequest,
      relationship::RelationshipRequest,
      watch::{StartupScanInfo, WatchRequest, WatchResponse, WatchStartResult, WatchStatusResult, WatchStopResult},
    },
  },
  rerank::RerankerProvider,
  service::{
    self,
    explore::ExploreScope,
    util::{Resolver, ServiceError},
  },
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a ProjectActor
#[derive(Debug, Clone)]
pub struct ProjectActorConfig {
  /// Unique project identifier
  pub id: ProjectId,
  /// Project root directory
  pub root: PathBuf,
  /// Base data directory for databases
  pub data_dir: PathBuf,
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur in the ProjectActor
#[derive(Debug, thiserror::Error)]
pub enum ProjectActorError {
  #[error("Database error: {0}")]
  Database(#[source] DbError),
  #[error("Watcher error: {0}")]
  Watcher(String),
  #[error("Embedding error: {0}")]
  Embedding(#[from] crate::embedding::EmbeddingError),
  #[error("Internal error: {0}")]
  Internal(String),
}

// ============================================================================
// ProjectActor
// ============================================================================

/// The per-project actor - owns all project-specific state
///
/// This actor coordinates all operations for a single project:
/// - Database access (memories, code, entities, documents)
/// - Indexing (via IndexerActor)
/// - File watching (via WatcherTask)
///
/// # Ownership Model
///
/// - `db` is wrapped in `Arc` because it's shared with `IndexerActor`
/// - `indexer` is a handle (cheap to clone) for sending jobs
/// - `watcher_handle` is owned and managed by this actor
/// - `embedding_cache` stores recently used query embeddings to avoid redundant API calls
/// - No `Mutex` or `RwLock` - state is owned, not shared
pub struct ProjectActor {
  config: ProjectActorConfig,
  db: Arc<ProjectDb>,
  /// Project-level config (tools, decay, search, index, docs, workspace, hooks)
  project_config: Arc<Config>,
  embedding: Arc<dyn EmbeddingProvider>,
  /// Reranker provider for cross-encoder reranking (None if disabled)
  reranker: Option<Arc<dyn RerankerProvider>>,
  /// LLM provider for memory extraction (None if unavailable)
  llm_provider: Option<Box<dyn llm::LlmProvider>>,
  /// Deterministic UUID for this project (used in memory creation)
  project_uuid: Uuid,
  /// Hook state for session tracking and deduplication
  hook_state: service::hooks::HookState,
  indexer: IndexerHandle,
  watcher_handle: Option<JoinHandle<()>>,
  watcher_cancel: Option<CancellationToken>,
  /// Whether a code scan/index operation is in progress
  scan_in_progress: bool,
  /// Latest scan progress [processed, total] if scan is in progress
  scan_progress: Option<(usize, usize)>,
  request_rx: mpsc::Receiver<ProjectActorMessage>,
  cancel: CancellationToken,
}

impl ProjectActor {
  /// Spawn a new ProjectActor and return a handle for communication
  ///
  /// This opens the database, spawns the IndexerActor, and starts the
  /// actor's event loop. The returned handle can be used to send requests.
  ///
  /// # Arguments
  ///
  /// * `config` - Project-specific actor config (id, root, data_dir)
  /// * `embedding` - Shared embedding provider
  /// * `reranker` - Optional reranker provider for cross-encoder reranking
  /// * `daemon_settings` - Daemon-level settings (embedding batch size, hooks, etc.)
  /// * `cancel` - Cancellation token for coordinated shutdown
  pub async fn spawn(
    config: ProjectActorConfig,
    embedding: Arc<dyn EmbeddingProvider>,
    reranker: Option<Arc<dyn RerankerProvider>>,
    daemon_settings: Arc<DaemonSettings>,
    cancel: CancellationToken,
  ) -> Result<ProjectHandle, ProjectActorError> {
    info!(
        project_id = %config.id,
        root = %config.root.display(),
        "Spawning ProjectActor"
    );
    // Load project-specific config (tools, decay, search, index, docs, workspace)
    let project_config = Config::load_for_project(&config.root).await;
    let project_config = Arc::new(project_config);

    // Open database
    let db = ProjectDb::open(config.id.clone(), &config.data_dir, project_config.clone())
      .await
      .map_err(ProjectActorError::Database)?;
    let db = Arc::new(db);

    // Spawn indexer actor with a child cancellation token
    // Use daemon-level embedding settings (from global config, not project config)
    let embedding_batch_size = daemon_settings.embedding_batch_size.unwrap_or(512);

    let indexer_config = IndexerConfig {
      root: config.root.clone(),
      index: project_config.index.clone(),
      embedding_batch_size,
      embedding_context_length: daemon_settings.embedding_context_length,
      log_cache_stats: daemon_settings.log_cache_stats,
    };
    let indexer = IndexerActor::spawn(indexer_config, Arc::clone(&db), embedding.clone(), cancel.child_token());

    // Create message channel
    let (tx, rx) = mpsc::channel(256);

    // Generate deterministic project UUID from project ID (for memory creation)
    let project_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, config.id.as_str().as_bytes());

    // Create LLM provider for memory extraction (if available)
    let llm_provider = match llm::create_provider() {
      Ok(provider) => {
        debug!("LLM provider available: {}", provider.name());
        Some(provider)
      }
      Err(e) => {
        debug!("LLM provider not available: {}", e);
        None
      }
    };

    let actor = Self {
      config,
      db,
      project_config,
      embedding,
      reranker,
      llm_provider,
      project_uuid,
      hook_state: service::hooks::HookState::new(),
      indexer,
      watcher_handle: None,
      watcher_cancel: None,
      scan_in_progress: false,
      scan_progress: None,
      request_rx: rx,
      cancel,
    };

    // Spawn the actor task
    tokio::spawn(actor.run());

    Ok(ProjectHandle::new(tx))
  }

  /// Main actor event loop
  ///
  /// Processes messages until shutdown is requested via:
  /// - CancellationToken being cancelled
  /// - ProjectPayload::Shutdown message
  /// - Request channel being closed
  async fn run(mut self) {
    info!(
      project_id = %self.config.id,
      root = %self.config.root.display(),
      "ProjectActor started"
    );

    // Auto-start watcher for previously indexed projects
    match self.db.is_manually_indexed(self.config.id.as_str()).await {
      Ok(true) => {
        info!(project_id = %self.config.id, "Project was previously indexed, auto-starting watcher");
        if let Err(e) = self.start_watcher().await {
          warn!(project_id = %self.config.id, error = %e, "Failed to auto-start watcher");
        }
      }
      Ok(false) => {
        debug!(project_id = %self.config.id, "Project not previously indexed, watcher will not auto-start");
      }
      Err(e) => {
        warn!(project_id = %self.config.id, error = %e, "Failed to check if project was indexed");
      }
    }

    loop {
      tokio::select! {
        // Check cancellation first (biased)
        biased;

        _ = self.cancel.cancelled() => {
          info!(project_id = %self.config.id, "ProjectActor shutting down (cancelled)");
          break;
        }

        msg = self.request_rx.recv() => {
          match msg {
            Some(msg) => {
              self.handle_message(msg).await;
            }
            None => {
              info!(project_id = %self.config.id, "ProjectActor shutting down (channel closed)");
              break;
            }
          }
        }
      }
    }

    // Cleanup
    self.cleanup().await;

    info!(
        project_id = %self.config.id,
        "ProjectActor stopped"
    );
  }

  /// Clean up resources on shutdown
  async fn cleanup(&mut self) {
    // Stop watcher if running
    self.stop_watcher().await;

    // Shutdown indexer
    if let Err(e) = self.indexer.shutdown().await {
      debug!(error = %e, "Failed to send shutdown to indexer"); // this is fine
    }
  }

  /// Handle an incoming message
  async fn handle_message(&mut self, msg: ProjectActorMessage) {
    let ProjectActorMessage { id, reply, payload } = msg;

    match payload {
      ProjectActorPayload::Request(req) => {
        self.handle_request(&id, req, reply).await;
      }
      ProjectActorPayload::ApplyDecay => {
        let result = self.apply_decay().await;
        let response = match result {
          Ok((processed, changed)) => {
            ProjectActorResponse::Done(ResponseData::System(crate::ipc::system::SystemResponse::Ping(format!(
              "Decay applied: {}/{} memories changed",
              changed, processed
            ))))
          }
          Err(e) => ProjectActorResponse::error(-32000, e.to_string()),
        };
        let _ = reply.send(response).await;
      }
      ProjectActorPayload::CleanupSessions { max_age_hours } => {
        let result = self.cleanup_sessions(max_age_hours).await;
        let response = match result {
          Ok(count) => ProjectActorResponse::Done(ResponseData::System(crate::ipc::system::SystemResponse::Ping(
            format!("{} stale sessions cleaned", count),
          ))),
          Err(e) => ProjectActorResponse::error(-32000, e.to_string()),
        };
        let _ = reply.send(response).await;
      }
      ProjectActorPayload::Shutdown => {
        let _ = reply
          .send(ProjectActorResponse::Done(ResponseData::System(
            crate::ipc::system::SystemResponse::Shutdown {
              message: "Project actor shutting down".to_string(),
            },
          )))
          .await;
        self.cancel.cancel();
      }
    }
  }

  /// Route a request to the appropriate handler
  async fn handle_request(&mut self, id: &str, request: RequestData, reply: mpsc::Sender<ProjectActorResponse>) {
    debug!(request_id = id, request_type = ?std::mem::discriminant(&request), "Handling request");

    match request {
      RequestData::Memory(mem_req) => {
        self.handle_memory(id, mem_req, reply).await;
      }
      RequestData::Code(code_req) => {
        self.handle_code(id, code_req, reply).await;
      }
      RequestData::Watch(watch_req) => {
        self.handle_watch(id, watch_req, reply).await;
      }
      RequestData::Docs(docs_req) => {
        self.handle_docs(id, docs_req, reply).await;
      }
      RequestData::Relationship(rel_req) => {
        self.handle_relationship(id, rel_req, reply).await;
      }
      RequestData::Project(proj_req) => {
        self.handle_project(id, proj_req, reply).await;
      }
      RequestData::System(sys_req) => {
        self.handle_system(id, sys_req, reply).await;
      }
      RequestData::Explore(params) => {
        self.handle_explore(id, params, reply).await;
      }
      RequestData::Context(params) => {
        self.handle_context(id, params, reply).await;
      }
      RequestData::Hook(params) => {
        self.handle_hook(id, params, reply).await;
      }
    }
  }

  // ========================================================================
  // Helper Methods for Service Contexts
  // ========================================================================

  /// Create a memory service context
  fn memory_context(&self) -> service::memory::MemoryContext<'_> {
    service::memory::MemoryContext::new(&self.db, self.embedding.as_ref(), self.project_id())
  }

  /// Create a code service context
  fn code_context(&self) -> service::code::CodeContext<'_> {
    service::code::CodeContext::new(&self.db, self.embedding.as_ref())
  }

  /// Create an explore service context
  fn explore_context(&self) -> service::explore::ExploreContext<'_> {
    service::explore::ExploreContext::new(&self.db, self.embedding.as_ref())
  }

  /// Get the project UUID
  fn project_id(&self) -> Uuid {
    // Create a deterministic UUID from the project ID string
    Uuid::new_v5(&Uuid::NAMESPACE_OID, self.config.id.to_string().as_bytes())
  }

  /// Convert a ServiceError to a ProjectActorResponse
  fn service_error_response(e: ServiceError) -> ProjectActorResponse {
    ProjectActorResponse::error(e.code(), e.to_string())
  }

  // ========================================================================
  // Watcher Management
  // ========================================================================

  /// Start the file watcher for this project
  ///
  /// If the project was previously indexed, performs a startup scan to detect
  /// file changes that occurred while the daemon was down.
  ///
  /// Returns scan info if a startup scan was performed.
  async fn start_watcher(&mut self) -> Result<Option<StartupScanInfo>, ProjectActorError> {
    if self.watcher_cancel.is_some() {
      debug!(project_id = %self.config.id, "Watcher already running");
      return Ok(None);
    }

    // Perform startup scan if project was previously indexed
    let scan_info =
      if let Some(scan_result) = service::code::startup_scan::startup_scan(&self.db, &self.config.root).await {
        let files_queued = if scan_result.was_indexed && scan_result.has_changes() {
          info!(
            project_id = %self.config.id,
            added = scan_result.added.len(),
            modified = scan_result.modified.len(),
            deleted = scan_result.deleted.len(),
            moved = scan_result.moved.len(),
            "Startup scan detected changes, queueing reindex"
          );

          // Handle deleted files - remove from DB (both code and document tables)
          for deleted_path in &scan_result.deleted {
            // Delete code chunks
            if let Err(e) = self.db.delete_chunks_for_file(deleted_path).await {
              warn!(path = %deleted_path, error = %e, "Failed to delete code chunks for removed file");
            }
            // Delete document chunks and metadata (no-op for code files)
            if let Err(e) = self.db.delete_document_chunks_by_source(deleted_path).await {
              warn!(path = %deleted_path, error = %e, "Failed to delete document chunks for removed file");
            }
            if let Err(e) = self.db.delete_document_by_source(deleted_path).await {
              warn!(path = %deleted_path, error = %e, "Failed to delete document metadata for removed file");
            }
            // Delete indexed_files entry
            if let Err(e) = self.db.delete_indexed_file(self.config.id.as_str(), deleted_path).await {
              warn!(path = %deleted_path, error = %e, "Failed to delete indexed_file entry");
            }
          }

          // Optimize indexes after deletes to ensure deleted rows are compacted
          // and no longer appear in vector search results
          if !scan_result.deleted.is_empty()
            && let Err(e) = self.db.optimize_indexes().await
          {
            warn!(error = %e, "Failed to optimize indexes after startup scan deletes");
          }

          // Handle moved files - update paths in DB
          for (old_path, new_path) in &scan_result.moved {
            let new_relative = new_path
              .strip_prefix(&self.config.root)
              .map(|p| p.to_string_lossy().to_string())
              .unwrap_or_else(|_| new_path.to_string_lossy().to_string());

            // Handle both code and document files - one will be a no-op depending on file type
            if let Err(e) = self.db.rename_file(old_path, &new_relative).await {
              warn!(from = %old_path, to = %new_relative, error = %e, "Failed to rename code chunks");
            }
            if let Err(e) = self.db.rename_document(old_path, &new_relative).await {
              warn!(from = %old_path, to = %new_relative, error = %e, "Failed to rename document chunks");
            }
            if let Err(e) = self
              .db
              .rename_indexed_file(self.config.id.as_str(), old_path, &new_relative)
              .await
            {
              warn!(from = %old_path, to = %new_relative, error = %e, "Failed to rename indexed_file entry");
            }
          }

          // Queue added and modified files for reindexing
          let files_to_index = scan_result.files_to_index();
          let queued = files_to_index.len();
          if !files_to_index.is_empty() {
            debug!(
              project_id = %self.config.id,
              file_count = queued,
              "Queueing files for reindex"
            );
            if let Err(e) = self.indexer.index_batch(files_to_index, None).await {
              warn!(error = %e, "Failed to queue startup scan files for reindex");
            }
          }
          queued
        } else if !scan_result.was_indexed {
          debug!(project_id = %self.config.id, "Project not previously indexed, skipping startup scan");
          0
        } else {
          debug!(project_id = %self.config.id, "No changes detected during startup scan");
          0
        };

        Some(StartupScanInfo {
          was_indexed: scan_result.was_indexed,
          files_added: scan_result.added.len(),
          files_modified: scan_result.modified.len(),
          files_deleted: scan_result.deleted.len(),
          files_moved: scan_result.moved.len(),
          files_queued,
        })
      } else {
        None
      };

    let cancel = self.cancel.child_token();
    let watcher_config = WatcherConfig {
      root: self.config.root.clone(),
      index: self.project_config.index.clone(),
    };

    let handle = WatcherTask::spawn(watcher_config, self.indexer.clone(), cancel.clone())
      .map_err(|e| ProjectActorError::Watcher(e.to_string()))?;

    self.watcher_handle = Some(handle);
    self.watcher_cancel = Some(cancel);

    info!(project_id = %self.config.id, "Started watcher for {:?}", self.config.root);
    Ok(scan_info)
  }

  /// Stop the file watcher for this project
  async fn stop_watcher(&mut self) {
    if let Some(cancel) = self.watcher_cancel.take() {
      cancel.cancel();
      info!(project_id = %self.config.id, "Stopped watcher for {:?}", self.config.root);
    }

    if let Some(handle) = self.watcher_handle.take() {
      // Give the watcher a moment to clean up
      let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    }
  }

  // ========================================================================
  // Scheduler-Triggered Operations
  // ========================================================================

  /// Apply memory decay for this project.
  ///
  /// Returns (total_processed, changed_count).
  async fn apply_decay(&self) -> Result<(usize, usize), ProjectActorError> {
    let decay_config = service::memory::MemoryDecay {
      archive_threshold: self.project_config.decay.archive_threshold as f32,
      max_idle_days: self.project_config.decay.max_idle_days,
    };

    let ctx = self.memory_context();
    let stats = service::memory::apply_decay(&ctx, &decay_config)
      .await
      .map_err(|e| ProjectActorError::Internal(e.to_string()))?;

    debug!(
      project_id = %self.config.id,
      processed = stats.total_processed,
      decayed = stats.decayed_count,
      archive_candidates = stats.archive_candidates,
      "Decay applied"
    );

    Ok((stats.total_processed, stats.decayed_count))
  }

  /// Cleanup stale sessions for this project.
  ///
  /// Returns the number of sessions cleaned up.
  async fn cleanup_sessions(&self, max_age_hours: u64) -> Result<usize, ProjectActorError> {
    let cleaned = self
      .db
      .cleanup_stale_sessions(max_age_hours)
      .await
      .map_err(ProjectActorError::Database)?;

    debug!(
      project_id = %self.config.id,
      cleaned = cleaned,
      max_age_hours = max_age_hours,
      "Session cleanup complete"
    );

    Ok(cleaned)
  }

  // ========================================================================
  // Memory Handler
  // ========================================================================

  async fn handle_memory(&self, _id: &str, req: MemoryRequest, reply: mpsc::Sender<ProjectActorResponse>) {
    let ctx = self.memory_context();

    let response = match req {
      MemoryRequest::Search(params) => {
        match service::memory::search(&ctx, params, &self.project_config, self.reranker.as_deref()).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Search(
            crate::ipc::types::memory::MemorySearchResult {
              items: result.items,
              search_quality: Some(result.search_quality),
            },
          ))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::Get(params) => match service::memory::get(&ctx, params).await {
        Ok(detail) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Get(detail))),
        Err(e) => Self::service_error_response(e),
      },
      MemoryRequest::Add(params) => match service::memory::add(&ctx, params).await {
        Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Add(result))),
        Err(e) => Self::service_error_response(e),
      },
      MemoryRequest::List(params) => match service::memory::list(&ctx, params).await {
        Ok(items) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::List(items))),
        Err(e) => Self::service_error_response(e),
      },
      MemoryRequest::Reinforce(MemoryReinforceParams { memory_id, amount }) => {
        match service::memory::reinforce(&ctx, &memory_id, amount).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Update(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::Deemphasize(MemoryDeemphasizeParams { memory_id, amount }) => {
        match service::memory::deemphasize(&ctx, &memory_id, amount).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Update(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::Delete(MemoryDeleteParams { memory_id }) => {
        match service::memory::delete(&ctx, &memory_id).await {
          Ok(memory) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Delete(MemoryDeleteResult {
            id: memory.id.to_string(),
            message: "Memory deleted".to_string(),
            hard_delete: false,
          }))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::HardDelete(MemoryHardDeleteParams { memory_id }) => {
        match service::memory::hard_delete(&ctx, &memory_id).await {
          Ok(id) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Delete(MemoryDeleteResult {
            id,
            message: "Memory permanently deleted".to_string(),
            hard_delete: true,
          }))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::SetSalience(MemorySetSalienceParams { memory_id, salience }) => {
        match service::memory::set_salience(&ctx, &memory_id, salience).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Update(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::Restore(MemoryRestoreParams { memory_id }) => {
        match service::memory::restore(&ctx, &memory_id).await {
          Ok(memory) => {
            ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Restore(MemoryRestoreResult {
              id: memory.id.to_string(),
              message: "Memory restored".to_string(),
            })))
          }
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::ListDeleted(MemoryListDeletedParams { limit }) => {
        match service::memory::list_deleted(&ctx, limit).await {
          Ok(items) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::ListDeleted(items))),
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::Supersede(MemorySupersedeParams {
        old_memory_id,
        new_content,
      }) => {
        // Supersede involves: add new memory, then link old -> new
        match service::memory::add(
          &ctx,
          crate::ipc::types::memory::MemoryAddParams {
            content: new_content,
            sector: None,
            memory_type: None,
            context: None,
            tags: None,
            categories: None,
            scope_path: None,
            scope_module: None,
            importance: None,
          },
        )
        .await
        {
          Ok(add_result) => match service::memory::supersede(&ctx, &old_memory_id, &add_result.id).await {
            Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Supersede(result))),
            Err(e) => Self::service_error_response(e),
          },
          Err(e) => Self::service_error_response(e),
        }
      }
      MemoryRequest::Related(params) => match service::memory::related(&ctx, params).await {
        Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Related(result))),
        Err(e) => Self::service_error_response(e),
      },
      MemoryRequest::Timeline(MemoryTimelineParams { memory_id }) => {
        match service::memory::timeline(&ctx, &memory_id, 5, 5).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Memory(MemoryResponse::Timeline(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
    };

    let _ = reply.send(response).await;
  }

  // ========================================================================
  // Code Handler
  // ========================================================================

  async fn handle_code(&mut self, _id: &str, req: CodeRequest, reply: mpsc::Sender<ProjectActorResponse>) {
    let ctx = self.code_context();
    let is_streaming_index = matches!(&req, CodeRequest::Index(CodeIndexParams { stream: true, .. }));

    let response = match req {
      CodeRequest::Search(CodeSearchParams {
        query,
        limit,
        file_pattern,
        symbol_type: _,
        language,
        visibility,
        chunk_type,
        min_caller_count,
      }) => {
        // Language can come from either explicit param or file_pattern (e.g., "*.rs")
        let resolved_language = language.or_else(|| {
          file_pattern
            .as_deref()
            .and_then(Language::from_file_pattern)
            .map(|l| l.as_db_str().to_string())
        });

        let params = service::code::SearchParams {
          query,
          language: resolved_language,
          limit,
          include_context: false,
          visibility,
          chunk_type,
          min_caller_count,
          adaptive_limit: false,
        };
        let config = service::code::RankingConfig::default();

        match service::code::search(
          &ctx,
          params,
          &config,
          Some(&self.project_config.search),
          self.reranker.as_deref(),
        )
        .await
        {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Search(
            crate::ipc::types::code::CodeSearchResult {
              query: result.query,
              chunks: result.results,
              search_quality: Some(result.search_quality),
            },
          ))),
          Err(e) => Self::service_error_response(e),
        }
      }
      CodeRequest::Callers(CodeCallersParams { chunk_id, limit }) => {
        let params = service::code::CallersParams {
          chunk_id: Some(chunk_id),
          symbol: None,
          limit,
        };
        match service::code::get_callers_response(&self.db, params).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Callers(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      CodeRequest::Callees(CodeCalleesParams { chunk_id, limit }) => {
        let params = service::code::CalleesParams { chunk_id, limit };
        match service::code::get_callees_response(&self.db, params).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Callees(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      CodeRequest::Related(CodeRelatedParams { chunk_id, limit }) => {
        let params = service::code::RelatedParams {
          chunk_id,
          methods: None,
          limit,
        };
        match service::code::get_related(&ctx, params).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Related(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      CodeRequest::ContextFull(CodeContextFullParams { chunk_id, depth }) => {
        let params = service::code::ContextFullParams { chunk_id, depth };
        match service::code::get_full_context(&ctx, params).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::ContextFull(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      CodeRequest::Stats(CodeStatsParams {}) => match service::code::get_stats(&self.db).await {
        Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Stats(result))),
        Err(e) => Self::service_error_response(e),
      },
      CodeRequest::List(CodeListParams { limit }) => match self.db.list_code_chunks(None, limit).await {
        Ok(chunks) => {
          let items: Vec<CodeItem> = chunks.into_iter().map(|c| CodeItem::from_list(&c)).collect();
          ProjectActorResponse::Done(ResponseData::Code(CodeResponse::List(items)))
        }
        Err(e) => Self::service_error_response(ServiceError::from(e)),
      },
      CodeRequest::Context(CodeContextParams {
        chunk_id,
        before,
        after,
      }) => {
        // Code context: get file context around a chunk
        let params = service::code::context::FileContextParams {
          chunk_id,
          before,
          after,
        };
        match service::code::context::get_file_context(&self.db, &self.config.root, params).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Context(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      CodeRequest::Memories(CodeMemoriesParams { chunk_id, limit }) => {
        // Get memories related to a code chunk
        self.handle_code_memories(&chunk_id, limit).await
      }
      CodeRequest::Index(CodeIndexParams { force, stream }) => {
        // Indexing goes through the IndexerActor
        self.handle_code_index(force, stream, reply.clone()).await
      }
    };

    // For Index with streaming, response is already sent
    if !is_streaming_index {
      let _ = reply.send(response).await;
    }
  }

  /// Handle code memories request
  async fn handle_code_memories(&self, chunk_id: &str, limit: Option<usize>) -> ProjectActorResponse {
    let chunk = match Resolver::code_chunk(&self.db, chunk_id).await {
      Ok(c) => c,
      Err(e) => return Self::service_error_response(e.into()),
    };

    let limit = limit.unwrap_or(10);
    let memories = service::code::get_related_memories(&self.db, &chunk.file_path, &chunk.symbols, limit)
      .await
      .unwrap_or_default();

    let items: Vec<MemoryItem> = memories.into_iter().map(|m| MemoryItem::from_list(&m)).collect();

    ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Memories(CodeMemoriesResponse {
      file_path: chunk.file_path,
      memories: items,
    })))
  }

  /// Handle code index request
  async fn handle_code_index(
    &mut self,
    _force: bool,
    stream: bool,
    reply: mpsc::Sender<ProjectActorResponse>,
  ) -> ProjectActorResponse {
    // Mark scan as in progress
    self.scan_in_progress = true;
    self.scan_progress = None;

    // Send initial progress if streaming
    if stream {
      let _ = reply
        .send(ProjectActorResponse::progress("Scanning files...", Some(0)))
        .await;
    }

    // Scan for files
    let scan_params = service::code::index::ScanParams {
      max_file_size: self.project_config.index.max_file_size as u64,
    };
    let scan_result = service::code::index::scan_directory(&self.config.root, &scan_params);
    let total_files = scan_result.files.len();

    debug!(
      files_scanned = total_files,
      scan_ms = scan_result.duration.as_millis() as u64,
      "File scan complete"
    );

    // Update scan progress with total
    self.scan_progress = Some((0, total_files));

    if stream && !scan_result.files.is_empty() {
      let _ = reply
        .send(ProjectActorResponse::progress(
          format!("Indexing {} files...", scan_result.files.len()),
          Some(10),
        ))
        .await;
    }

    // Create progress channel and spawn forwarder only if streaming
    // IMPORTANT: If progress_tx is passed but progress_rx is not consumed, the channel
    // will fill up and block the sender, causing a deadlock. Only create when needed.
    let progress_tx = if stream {
      let (progress_tx, mut progress_rx) = mpsc::channel::<super::message::IndexProgress>(64);
      tokio::spawn({
        let reply = reply.clone();
        async move {
          while let Some(progress) = progress_rx.recv().await {
            // Send rich progress info with stage details
            let _ = reply.send(ProjectActorResponse::from_index_progress(&progress)).await;
          }
        }
      });
      Some(progress_tx)
    } else {
      None
    };

    // Run indexing via service
    let result = service::code::index::run_indexing(&self.indexer, scan_result, progress_tx).await;

    // Mark scan as complete
    self.scan_in_progress = false;
    self.scan_progress = None;

    // Auto-start watcher after successful indexing
    if result.status == "complete" && result.files_indexed > 0 && self.watcher_cancel.is_none() {
      info!(project_id = %self.config.id, "Auto-starting watcher after initial indexing");
      if let Err(e) = self.start_watcher().await {
        warn!(project_id = %self.config.id, error = %e, "Failed to auto-start watcher after indexing");
      }
    }

    // Convert service result to IPC response
    let response = ProjectActorResponse::Done(ResponseData::Code(CodeResponse::Index(CodeIndexResult {
      status: result.status,
      files_scanned: result.files_scanned,
      files_indexed: result.files_indexed,
      chunks_created: result.chunks_created,
      failed_files: result.failed_files,
      resumed_from_checkpoint: result.resumed_from_checkpoint,
      scan_duration_ms: result.scan_duration.as_millis() as u64,
      index_duration_ms: result.index_duration.as_millis() as u64,
      total_duration_ms: result.total_duration.as_millis() as u64,
      files_per_second: result.files_per_second,
      bytes_processed: result.bytes_processed,
      total_bytes: result.total_bytes,
    })));

    let _ = reply.send(response).await;
    ProjectActorResponse::Done(ResponseData::System(crate::ipc::system::SystemResponse::Ping(
      "indexed".to_string(),
    )))
  }

  // ========================================================================
  // Explore Handler
  // ========================================================================

  async fn handle_explore(&self, _id: &str, params: ExploreParams, reply: mpsc::Sender<ProjectActorResponse>) {
    let ctx = self.explore_context();

    let scope = params
      .scope
      .as_deref()
      .and_then(ExploreScope::from_str)
      .unwrap_or_default();

    let search_params = service::explore::SearchParams {
      query: params.query.clone(),
      scope,
      expand_top: params.expand_top.unwrap_or(3),
      limit: params.limit.unwrap_or(10),
      depth: params.depth.unwrap_or(5),
    };

    let response = match service::explore::search(&ctx, &search_params).await {
      Ok(explore_response) => {
        // Convert service response to IPC response
        let items: Vec<crate::ipc::search::ExploreResultItem> = explore_response
          .results
          .into_iter()
          .map(|r| {
            // Convert context if present
            let context = r.context.map(|ctx| crate::ipc::search::ExploreContext {
              callers: ctx
                .callers
                .into_iter()
                .map(|c| crate::ipc::search::ExploreCallInfo {
                  id: c.id,
                  file: c.file,
                  start_line: c.lines.0,
                  end_line: c.lines.1,
                  preview: c.preview,
                  symbols: c.symbols.unwrap_or_default(),
                  signature: c.signature,
                })
                .collect(),
              callees: ctx
                .callees
                .into_iter()
                .map(|c| crate::ipc::search::ExploreCallInfo {
                  id: c.id,
                  file: c.file,
                  start_line: c.lines.0,
                  end_line: c.lines.1,
                  preview: c.preview,
                  symbols: c.symbols.unwrap_or_default(),
                  signature: c.signature,
                })
                .collect(),
              siblings: ctx
                .siblings
                .into_iter()
                .map(|s| crate::ipc::search::ExploreSiblingInfo {
                  symbol: s.symbol,
                  kind: s.kind,
                  line: s.line,
                  file: None,
                })
                .collect(),
            });

            crate::ipc::search::ExploreResultItem {
              id: r.id,
              result_type: r.result_type,
              preview: r.preview,
              similarity: r.score,
              file_path: r.file,
              line: r.lines.map(|(start, _)| start),
              symbols: r.symbols,
              hints: Some(crate::ipc::search::ExploreHints {
                caller_count: r.hints.callers.unwrap_or(0),
                callee_count: r.hints.callees.unwrap_or(0),
                related_memory_count: r.hints.related_memories.unwrap_or(0),
              }),
              context,
            }
          })
          .collect();

        ProjectActorResponse::Done(ResponseData::Explore(crate::ipc::search::ExploreResult {
          query: params.query,
          results: items,
        }))
      }
      Err(e) => Self::service_error_response(e),
    };

    let _ = reply.send(response).await;
  }

  async fn handle_context(&self, _id: &str, params: ContextParams, reply: mpsc::Sender<ProjectActorResponse>) {
    let ctx = self.explore_context();

    // Collect IDs from both `id` and `ids` parameters
    let ids: Vec<String> = match (params.id, params.ids) {
      (Some(id), None) => vec![id],
      (None, Some(ids)) => ids,
      (Some(id), Some(mut ids)) => {
        ids.insert(0, id);
        ids
      }
      (None, None) => {
        let _ = reply
          .send(ProjectActorResponse::error(-32602, "Must provide id or ids parameter"))
          .await;
        return;
      }
    };

    let depth = params.depth.unwrap_or(5);

    let response = match service::explore::get_context(&ctx, &ids, depth).await {
      Ok(context_response) => {
        // Convert service response to IPC response
        let items: Vec<crate::ipc::search::ContextItem> = match context_response {
          service::explore::ContextResponse::Code { items } => items
            .into_iter()
            .map(|c| crate::ipc::search::ContextItem {
              id: c.id,
              item_type: "code".to_string(),
              content: c.content,
              callers: Some(
                c.callers
                  .into_iter()
                  .map(|caller| crate::ipc::types::code::CodeItem {
                    id: caller.id,
                    file_path: caller.file,
                    content: caller.preview,
                    start_line: caller.lines.0,
                    end_line: caller.lines.1,
                    language: None,
                    chunk_type: None,
                    symbol_name: None,
                    symbols: caller.symbols.unwrap_or_default(),
                    definition_kind: None,
                    visibility: None,
                    signature: caller.signature,
                    docstring: None,
                    parent_definition: None,
                    similarity: None,
                    confidence: None,
                    file_hash: None,
                    tokens_estimate: None,
                    imports: vec![],
                    calls: vec![],
                    caller_count: None,
                    callee_count: None,
                  })
                  .collect(),
              ),
              callees: Some(
                c.callees
                  .into_iter()
                  .map(|callee| crate::ipc::types::code::CodeItem {
                    id: callee.id,
                    file_path: callee.file,
                    content: callee.preview,
                    start_line: callee.lines.0,
                    end_line: callee.lines.1,
                    language: None,
                    chunk_type: None,
                    symbol_name: None,
                    symbols: callee.symbols.unwrap_or_default(),
                    definition_kind: None,
                    visibility: None,
                    signature: callee.signature,
                    docstring: None,
                    parent_definition: None,
                    similarity: None,
                    confidence: None,
                    file_hash: None,
                    tokens_estimate: None,
                    imports: vec![],
                    calls: vec![],
                    caller_count: None,
                    callee_count: None,
                  })
                  .collect(),
              ),
              related_memories: None,
            })
            .collect(),
          service::explore::ContextResponse::Memory { items } => items
            .into_iter()
            .map(|m| crate::ipc::search::ContextItem {
              id: m.id,
              item_type: "memory".to_string(),
              content: m.content,
              callers: None,
              callees: None,
              related_memories: None,
            })
            .collect(),
          service::explore::ContextResponse::Doc { items } => items
            .into_iter()
            .map(|d| crate::ipc::search::ContextItem {
              id: d.id,
              item_type: "doc".to_string(),
              content: d.content,
              callers: None,
              callees: None,
              related_memories: None,
            })
            .collect(),
          service::explore::ContextResponse::Mixed { code, memories, docs } => {
            let mut all = Vec::new();
            all.extend(code.into_iter().map(|c| crate::ipc::search::ContextItem {
              id: c.id,
              item_type: "code".to_string(),
              content: c.content,
              callers: None,
              callees: None,
              related_memories: None,
            }));
            all.extend(memories.into_iter().map(|m| crate::ipc::search::ContextItem {
              id: m.id,
              item_type: "memory".to_string(),
              content: m.content,
              callers: None,
              callees: None,
              related_memories: None,
            }));
            all.extend(docs.into_iter().map(|d| crate::ipc::search::ContextItem {
              id: d.id,
              item_type: "doc".to_string(),
              content: d.content,
              callers: None,
              callees: None,
              related_memories: None,
            }));
            all
          }
        };

        ProjectActorResponse::Done(ResponseData::Context(items))
      }
      Err(e) => Self::service_error_response(e),
    };

    let _ = reply.send(response).await;
  }

  // ========================================================================
  // Watch Handler
  // ========================================================================

  async fn handle_watch(&mut self, _id: &str, req: WatchRequest, reply: mpsc::Sender<ProjectActorResponse>) {
    let response = match req {
      WatchRequest::Start(_) => match self.start_watcher().await {
        Ok(scan_info) => ProjectActorResponse::Done(ResponseData::Watch(WatchResponse::Start(WatchStartResult {
          status: "started".to_string(),
          path: self.config.root.to_string_lossy().to_string(),
          project_id: self.config.id.to_string(),
          startup_scan: scan_info,
        }))),
        Err(e) => ProjectActorResponse::error(-32000, e.to_string()),
      },
      WatchRequest::Stop(_) => {
        self.stop_watcher().await;
        ProjectActorResponse::Done(ResponseData::Watch(WatchResponse::Stop(WatchStopResult {
          status: "stopped".to_string(),
          path: self.config.root.to_string_lossy().to_string(),
          project_id: self.config.id.to_string(),
        })))
      }
      WatchRequest::Status(_) => {
        let running = self.watcher_cancel.is_some();
        ProjectActorResponse::Done(ResponseData::Watch(WatchResponse::Status(WatchStatusResult {
          running,
          root: Some(self.config.root.to_string_lossy().to_string()),
          pending_changes: self.indexer.pending_count(),
          project_id: self.config.id.to_string(),
          scanning: self.scan_in_progress,
          scan_progress: self.scan_progress.map(|(current, total)| [current, total]),
        })))
      }
    };
    let _ = reply.send(response).await;
  }

  // ========================================================================
  // Docs Handler
  // ========================================================================

  async fn handle_docs(&self, _id: &str, req: DocsRequest, reply: mpsc::Sender<ProjectActorResponse>) {
    let is_streaming_ingest = matches!(&req, DocsRequest::Ingest(DocsIngestParams { stream: true, .. }));

    let response = match req {
      DocsRequest::Search(params) => {
        let ctx = service::docs::DocsContext::new(&self.db, self.embedding.as_ref());
        let search_params = service::docs::SearchParams::from(params);
        match service::docs::search(
          &ctx,
          search_params,
          Some(&self.project_config.search),
          self.reranker.as_deref(),
        )
        .await
        {
          Ok(items) => ProjectActorResponse::Done(ResponseData::Docs(DocsResponse::Search(items))),
          Err(e) => Self::service_error_response(e),
        }
      }
      DocsRequest::Context(DocContextParams { doc_id, before, after }) => {
        let params = service::docs::ContextParams { doc_id, before, after };
        match service::docs::get_context(&self.db, params).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::Docs(DocsResponse::GetContext(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      DocsRequest::Ingest(DocsIngestParams {
        directory,
        file,
        stream,
      }) => self.handle_docs_ingest(directory, file, stream, reply.clone()).await,
    };

    // For Ingest with streaming, response is already sent
    if !is_streaming_ingest {
      let _ = reply.send(response).await;
    }
  }

  /// Handle document ingest request with optional streaming
  async fn handle_docs_ingest(
    &self,
    directory: Option<String>,
    file: Option<String>,
    stream: bool,
    reply: mpsc::Sender<ProjectActorResponse>,
  ) -> ProjectActorResponse {
    let ctx = service::docs::IngestContext::new(self.db.clone(), self.embedding.clone());
    let params = service::docs::IngestParams {
      directory,
      file,
      project_id: self.project_uuid,
      root: self.config.root.clone(),
    };

    // Send initial progress if streaming
    if stream {
      let _ = reply
        .send(ProjectActorResponse::progress("Scanning for documents...", Some(0)))
        .await;
    }

    // Create progress channel if streaming
    let (progress_tx, mut progress_rx) = mpsc::channel::<service::docs::IngestProgress>(64);
    let progress_tx_opt = if stream { Some(progress_tx) } else { None };

    // Spawn progress forwarder if streaming
    if stream {
      tokio::spawn({
        let reply = reply.clone();
        async move {
          while let Some(progress) = progress_rx.recv().await {
            let percent = progress.percent().min(99);
            let msg = format!("Ingested {}/{} documents", progress.processed, progress.total);
            let _ = reply.send(ProjectActorResponse::progress(&msg, Some(percent))).await;
          }
        }
      });
    }

    // Run ingestion
    match service::docs::ingest(&ctx, params, progress_tx_opt).await {
      Ok(result) => {
        // Always return full result for typed API consistency
        let full_result = crate::ipc::types::docs::DocsIngestFullResult {
          status: result.status,
          files_scanned: result.files_scanned,
          files_ingested: result.files_ingested,
          chunks_created: result.chunks_created,
          failed_files: result.failed_files,
          scan_duration_ms: result.scan_duration.as_millis() as u64,
          ingest_duration_ms: result.ingest_duration.as_millis() as u64,
          total_duration_ms: result.total_duration.as_millis() as u64,
          files_per_second: result.files_per_second,
          bytes_processed: result.bytes_processed,
          total_bytes: result.total_bytes,
          // Include individual results only if reasonable number
          results: if result.results.len() <= 50 {
            result.results
          } else {
            Vec::new()
          },
        };
        let response = ProjectActorResponse::Done(ResponseData::Docs(DocsResponse::IngestFull(full_result)));

        let _ = reply.send(response).await;
        ProjectActorResponse::Done(ResponseData::System(crate::ipc::system::SystemResponse::Ping(
          "ingested".to_string(),
        )))
      }
      Err(e) => {
        let error_response = Self::service_error_response(e);
        let _ = reply.send(error_response.clone()).await;
        error_response
      }
    }
  }

  // ========================================================================
  // Relationship Handler
  // ========================================================================

  async fn handle_relationship(&self, _id: &str, req: RelationshipRequest, reply: mpsc::Sender<ProjectActorResponse>) {
    let response = match req {
      RelationshipRequest::List(RelationshipListParams { memory_id }) => {
        match service::memory::relationship::list(&self.db, &memory_id).await {
          Ok(items) => ProjectActorResponse::Done(ResponseData::Relationship(RelationshipResponse::List(items))),
          Err(e) => Self::service_error_response(e),
        }
      }
      RelationshipRequest::Add(params) => match service::memory::relationship::add(&self.db, params).await {
        Ok(result) => ProjectActorResponse::Done(ResponseData::Relationship(RelationshipResponse::Add(result))),
        Err(e) => Self::service_error_response(e),
      },
      RelationshipRequest::Delete(params) => match service::memory::relationship::delete(&self.db, params).await {
        Ok(result) => ProjectActorResponse::Done(ResponseData::Relationship(RelationshipResponse::Delete(result))),
        Err(e) => Self::service_error_response(e),
      },
      RelationshipRequest::Related(params) => {
        // This is essentially memory_related, delegate to memory service
        let ctx = self.memory_context();
        match service::memory::related(
          &ctx,
          MemoryRelatedParams {
            memory_id: params.memory_id,
            limit: params.limit,
          },
        )
        .await
        {
          Ok(result) => {
            let items: Vec<RelatedMemoryItem> = result
              .related
              .into_iter()
              .map(|r| RelatedMemoryItem {
                memory: MemorySummary {
                  id: r.id,
                  content: r.content,
                  summary: r.summary,
                  sector: r.sector,
                  salience: r.salience,
                },
                relationship: RelationshipInfo {
                  relationship_type: r.relationship.clone(),
                  confidence: r.score,
                  direction: "outgoing".to_string(),
                },
              })
              .collect();
            ProjectActorResponse::Done(ResponseData::Relationship(RelationshipResponse::Related(items)))
          }
          Err(e) => Self::service_error_response(e),
        }
      }
    };

    let _ = reply.send(response).await;
  }

  // ========================================================================
  // Project Handler
  // ========================================================================

  async fn handle_project(&self, _id: &str, req: ProjectRequest, reply: mpsc::Sender<ProjectActorResponse>) {
    let response = match req {
      ProjectRequest::Info(_params) => {
        match service::project::info(&self.db, &self.config.id, &self.config.root).await {
          Ok(mut result) => {
            result.db_path = self.config.data_dir.to_string_lossy().to_string();
            ProjectActorResponse::Done(ResponseData::Project(ProjectResponse::Info(result)))
          }
          Err(e) => Self::service_error_response(e),
        }
      }
      ProjectRequest::List(_) => {
        // List is handled at the router level, not per-project
        ProjectActorResponse::internal_error("Project list should be handled by router")
      }
      ProjectRequest::Clean(_params) => match service::project::clean(&self.db, &self.config.root).await {
        Ok(result) => ProjectActorResponse::Done(ResponseData::Project(ProjectResponse::Clean(result))),
        Err(e) => Self::service_error_response(e),
      },
      ProjectRequest::CleanAll(_) => {
        // CleanAll is handled at the router level
        ProjectActorResponse::internal_error("Project clean-all should be handled by router")
      }
      ProjectRequest::Sessions(params) => {
        // Build filter based on params
        let filter = if params.active_only.unwrap_or(false) {
          Some("ended_at IS NULL")
        } else {
          None
        };

        match self.db.list_sessions(filter, params.limit).await {
          Ok(sessions) => {
            use crate::ipc::project::SessionItem;
            let items: Vec<SessionItem> = sessions
              .into_iter()
              .map(|s| SessionItem {
                id: s.id,
                started_at: s.started_at.to_rfc3339(),
                ended_at: s.ended_at.map(|e| e.to_rfc3339()),
                summary: s.summary,
                user_prompt: s.user_prompt,
              })
              .collect();
            ProjectActorResponse::Done(ResponseData::Project(ProjectResponse::Sessions(items)))
          }
          Err(e) => Self::service_error_response(ServiceError::from(e)),
        }
      }
    };

    let _ = reply.send(response).await;
  }

  async fn handle_system(
    &self,
    _id: &str,
    request: crate::ipc::system::SystemRequest,
    reply: mpsc::Sender<ProjectActorResponse>,
  ) {
    use crate::ipc::system::{SystemRequest, SystemResponse};

    let response = match request {
      SystemRequest::Ping(_) => {
        ProjectActorResponse::Done(ResponseData::System(SystemResponse::Ping("pong".to_string())))
      }
      SystemRequest::HealthCheck(_) => ProjectActorResponse::Done(ResponseData::System(SystemResponse::HealthCheck(
        crate::ipc::system::HealthCheckResult {
          healthy: true,
          checks: vec![crate::ipc::system::HealthCheck {
            name: "database".to_string(),
            status: "ok".to_string(),
            message: None,
          }],
        },
      ))),
      SystemRequest::ProjectStats(_) => {
        match service::project::stats(&self.db, &self.config.id, &self.project_uuid, &self.config.root).await {
          Ok(result) => ProjectActorResponse::Done(ResponseData::System(SystemResponse::ProjectStats(result))),
          Err(e) => Self::service_error_response(e),
        }
      }
      SystemRequest::Resolve(params) => {
        use crate::service::util::Resolver;
        match Resolver::any(&self.db, &params.id).await {
          Ok(resolved) => ProjectActorResponse::Done(ResponseData::System(SystemResponse::Resolve(
            crate::ipc::system::ResolveResult {
              id: resolved.id(),
              entity_type: resolved.entity_type().to_string(),
            },
          ))),
          Err(e) => Self::service_error_response(service::util::ServiceError::from(e)),
        }
      }
      // These are handled at the daemon level, not here
      SystemRequest::Metrics(_) | SystemRequest::Shutdown(_) | SystemRequest::Status(_) => {
        ProjectActorResponse::method_not_found(&format!("{:?}", request))
      }
    };

    let _ = reply.send(response).await;
  }

  async fn handle_hook(&mut self, _id: &str, params: HookParams, reply: mpsc::Sender<ProjectActorResponse>) {
    // Parse hook event from hook_name
    let event = match params.hook_name.parse::<service::hooks::HookEvent>() {
      Ok(e) => e,
      Err(e) => {
        let response = Self::service_error_response(e);
        let _ = reply.send(response).await;
        return;
      }
    };

    // Build hook context (use project-level hooks config, merged with global defaults)
    let hook_ctx = service::hooks::HookContext::new(
      &self.db,
      self.embedding.as_ref(),
      self.llm_provider.as_deref(),
      self.project_uuid,
      &self.project_config.hooks,
    );

    // For SessionStart, provide project info
    let session_info = if event == service::hooks::HookEvent::SessionStart {
      Some(service::hooks::SessionStartInfo {
        project_id: self.config.id.to_string(),
        project_name: self
          .config
          .root
          .file_name()
          .map(|n| n.to_string_lossy().to_string())
          .unwrap_or_else(|| "unknown".to_string()),
        project_path: self.config.root.to_string_lossy().to_string(),
        watcher_started: self.watcher_cancel.is_some(),
      })
    } else {
      None
    };

    // Merge session_id into data for handlers (session_id comes from HookParams, not data)
    let mut hook_data = params.data.clone();
    if let Some(session_id) = &params.session_id
      && let Some(obj) = hook_data.as_object_mut()
    {
      obj.insert("session_id".to_string(), serde_json::Value::String(session_id.clone()));
    }

    // Dispatch to hook service
    let result = service::hooks::dispatch(&hook_ctx, &mut self.hook_state, event, &hook_data, session_info).await;

    let response = match result {
      Ok(data) => ProjectActorResponse::Done(ResponseData::Hook(HookResult { data })),
      Err(e) => Self::service_error_response(e),
    };

    let _ = reply.send(response).await;
  }
}
