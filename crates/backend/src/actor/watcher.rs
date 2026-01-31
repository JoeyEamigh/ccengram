//! WatcherTask - Async file watcher that feeds the indexing pipeline
//!
//! This task watches a directory for file changes, debounces them, and sends
//! `IndexJob` messages to an `IndexerActor`.
//!
//! # Design
//!
//! The watcher bridges the sync notify callbacks with our async actor system:
//! 1. notify's sync callback uses `blocking_send` to forward events to a channel
//! 2. The async task consumes events from that channel
//! 3. Events are debounced to avoid processing rapid changes multiple times
//! 4. Settled events are converted to `IndexJob` messages
//!
//! # Content Cache
//!
//! The watcher maintains a cache of recent file contents for incremental parsing.
//! When a file is modified, the old content is passed along with the new content,
//! allowing the parser to reuse AST nodes that haven't changed.
//!
//! # Gitignore Integration
//!
//! Uses the `ignore` crate's `Gitignore` struct for efficient filtering.
//! Files matching .gitignore patterns are silently skipped.
//!
//! # Lifecycle
//!
//! The watcher runs until:
//! - The `CancellationToken` is triggered
//! - The event channel closes (notify watcher dropped)

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
  sync::Arc,
  time::{Duration, Instant},
};

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};

use super::{handle::IndexerHandle, message::IndexJob};
use crate::{
  context::files::is_document_extension,
  domain::{code::Language, config::IndexConfig},
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the WatcherTask
///
/// Stores a reference to IndexConfig and derives watcher settings from it.
#[derive(Debug, Clone)]
pub struct WatcherConfig {
  /// Root directory to watch
  pub root: PathBuf,
  /// Index configuration (contains watcher settings)
  pub index: IndexConfig,
}

impl WatcherConfig {
  /// Get the debounce duration from IndexConfig
  pub fn debounce(&self) -> Duration {
    Duration::from_millis(self.index.watcher_debounce_ms)
  }

  /// Get the poll interval from IndexConfig
  pub fn poll_interval(&self) -> Duration {
    Duration::from_secs(self.index.watcher_poll_secs)
  }

  /// Get the content cache size from IndexConfig
  pub fn content_cache_size(&self) -> usize {
    self.index.content_cache_size
  }

  /// Get the max cached file size from IndexConfig
  pub fn max_cached_file_size(&self) -> usize {
    self.index.max_cached_file_size
  }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur in the watcher
#[derive(Debug, thiserror::Error)]
pub enum WatcherError {
  #[error("Failed to initialize watcher: {0}")]
  Init(#[source] notify::Error),

  #[error("Failed to watch path: {0}")]
  Watch(#[source] notify::Error),

  #[error("Failed to build gitignore: {0}")]
  Gitignore(#[source] ignore::Error),
}

// ============================================================================
// Internal Types
// ============================================================================

/// The kind of pending change
#[derive(Debug, Clone)]
enum ChangeKind {
  Created,
  Modified,
  Deleted,
  Renamed { from: PathBuf },
}

/// A pending change that is being debounced
#[derive(Debug)]
struct PendingChange {
  kind: ChangeKind,
  last_event: Instant,
}

impl PendingChange {
  fn new(kind: ChangeKind) -> Self {
    Self {
      kind,
      last_event: Instant::now(),
    }
  }

  /// Update the pending change with a new event, coalescing where appropriate
  fn update(&mut self, kind: ChangeKind) {
    self.last_event = Instant::now();

    // Coalesce event types
    match (&self.kind, &kind) {
      // Create followed by modify is still a create
      (ChangeKind::Created, ChangeKind::Modified) => {
        trace!("Coalescing create+modify -> create");
      }
      // Delete followed by create is a modify
      (ChangeKind::Deleted, ChangeKind::Created) => {
        self.kind = ChangeKind::Modified;
        trace!("Coalescing delete+create -> modified");
      }
      // Create followed by delete cancels out (we'll emit delete to clean up)
      (ChangeKind::Created, ChangeKind::Deleted) => {
        self.kind = ChangeKind::Deleted;
        trace!("Coalescing create+delete -> delete");
      }
      // Rename followed by modify is still a rename
      (ChangeKind::Renamed { .. }, ChangeKind::Modified) => {
        trace!("Coalescing rename+modify -> rename");
      }
      // Otherwise take the latest
      _ => {
        self.kind = kind;
      }
    }
  }
}

/// LRU-style content cache for incremental parsing
struct ContentCache {
  /// file_path -> (content, last_access)
  cache: HashMap<PathBuf, (Arc<String>, Instant)>,
  max_size: usize,
  max_file_size: usize,
}

impl ContentCache {
  fn new(max_size: usize, max_file_size: usize) -> Self {
    Self {
      cache: HashMap::new(),
      max_size,
      max_file_size,
    }
  }

  /// Get cached content for a file
  fn get(&mut self, path: &PathBuf) -> Option<Arc<String>> {
    if let Some((content, last_access)) = self.cache.get_mut(path) {
      *last_access = Instant::now();
      Some(content.clone())
    } else {
      None
    }
  }

  /// Cache content for a file
  fn put(&mut self, path: PathBuf, content: String) {
    // Don't cache if content is too large
    if content.len() > self.max_file_size {
      trace!(
        path = %path.display(),
        size = content.len(),
        max = self.max_file_size,
        "File too large to cache"
      );
      return;
    }

    // Evict oldest entries if at capacity
    while self.cache.len() >= self.max_size {
      if let Some(oldest_key) = self
        .cache
        .iter()
        .min_by_key(|(_, (_, last_access))| *last_access)
        .map(|(k, _)| k.clone())
      {
        self.cache.remove(&oldest_key);
      } else {
        break;
      }
    }

    self.cache.insert(path, (Arc::new(content), Instant::now()));
  }

  /// Remove a file from the cache
  fn remove(&mut self, path: &PathBuf) {
    self.cache.remove(path);
  }
}

// ============================================================================
// WatcherTask
// ============================================================================

/// Async file watcher that sends IndexJobs to an IndexerActor
///
/// The watcher debounces file events and converts them into appropriate
/// `IndexJob` messages for the indexer.
///
/// # Example
///
/// ```ignore
/// let config = WatcherConfig {
///     root: project_root,
///     index: index_config,
/// };
/// let watcher = WatcherTask::new(config, indexer_handle, cancel_token)?;
/// tokio::spawn(watcher.run());
/// ```
pub struct WatcherTask {
  config: WatcherConfig,
  indexer: IndexerHandle,
  cancel: CancellationToken,
  // The notify watcher must be held to keep it alive
  _watcher: RecommendedWatcher,
  // Channel receiving events from notify's sync callback
  event_rx: mpsc::Receiver<Result<Event, notify::Error>>,
  // Gitignore matcher
  gitignore: Option<Gitignore>,
  // Content cache for incremental parsing
  content_cache: ContentCache,
}

impl WatcherTask {
  /// Create a new WatcherTask
  ///
  /// This initializes the file watcher and starts watching the configured root.
  /// The task is not started until `run()` is called.
  pub fn new(config: WatcherConfig, indexer: IndexerHandle, cancel: CancellationToken) -> Result<Self, WatcherError> {
    info!(root = %config.root.display(), "Initializing file watcher");

    // Build gitignore matcher
    let gitignore = build_gitignore(&config.root)?;

    // Create a channel for notify events
    // The sync callback will use blocking_send, so we need a reasonable buffer
    let (event_tx, event_rx) = mpsc::channel::<Result<Event, notify::Error>>(256);

    // Create the watcher with a sync callback that forwards to our channel
    let notify_config = Config::default().with_poll_interval(config.poll_interval());

    let mut watcher = RecommendedWatcher::new(
      move |res| {
        // This runs on notify's thread - use blocking_send
        // If the channel is full or closed, we drop the event
        let _ = event_tx.blocking_send(res);
      },
      notify_config,
    )
    .map_err(WatcherError::Init)?;

    // Start watching
    watcher
      .watch(&config.root, RecursiveMode::Recursive)
      .map_err(WatcherError::Watch)?;

    // Create content cache using config values
    let content_cache = ContentCache::new(config.content_cache_size(), config.max_cached_file_size());

    info!(root = %config.root.display(), "File watcher initialized");

    Ok(Self {
      config,
      indexer,
      cancel,
      _watcher: watcher,
      event_rx,
      gitignore,
      content_cache,
    })
  }

  /// Spawn the watcher task and return a handle to cancel it
  ///
  /// This is a convenience method that spawns the task and returns
  /// a `CancellationToken` that can be used to stop it.
  pub fn spawn(
    config: WatcherConfig,
    indexer: IndexerHandle,
    cancel: CancellationToken,
  ) -> Result<tokio::task::JoinHandle<()>, WatcherError> {
    let task = Self::new(config, indexer, cancel)?;
    Ok(tokio::spawn(task.run()))
  }

  /// Run the watcher task
  ///
  /// This consumes the task and runs until:
  /// - The `CancellationToken` is triggered
  /// - The event channel closes
  pub async fn run(mut self) {
    info!(root = %self.config.root.display(), "WatcherTask started");

    // Pending changes being debounced (keyed by path)
    let mut pending: HashMap<PathBuf, PendingChange> = HashMap::new();

    // Timer for checking debounced events
    let mut debounce_interval = tokio::time::interval(self.config.debounce());

    loop {
      tokio::select! {
        // Check cancellation first (biased)
        biased;

        _ = self.cancel.cancelled() => {
          info!("WatcherTask shutting down (cancelled)");
          break;
        }

        // Process incoming file events
        event = self.event_rx.recv() => {
          match event {
            Some(Ok(event)) => {
              self.process_event(&mut pending, event);
            }
            Some(Err(e)) => {
              warn!(error = %e, "Watcher error");
            }
            None => {
              info!("WatcherTask shutting down (channel closed)");
              break;
            }
          }
      }

        // Check for settled (debounced) events
        _ = debounce_interval.tick() => {
          self.flush_settled(&mut pending).await;
        }
      }
    }

    // Flush any remaining pending events before shutdown
    if !pending.is_empty() {
      debug!(pending = pending.len(), "Flushing remaining pending events on shutdown");
      self.flush_all(&mut pending).await;
    }

    info!(root = %self.config.root.display(), "WatcherTask stopped");
  }

  /// Check if a file should be ignored (gitignore match)
  fn is_ignored(&self, path: &PathBuf) -> bool {
    if let Some(ref gitignore) = self.gitignore {
      let is_dir = path.is_dir();
      gitignore.matched(path, is_dir).is_ignore()
    } else {
      false
    }
  }

  /// Check if a file is a supported type for indexing (code or document)
  fn is_indexable(&self, path: &Path) -> bool {
    // Skip directories
    if path.is_dir() {
      return false;
    }

    // Check if we support this file type (code or document)
    path
      .extension()
      .and_then(|ext| ext.to_str())
      .is_some_and(|ext| Language::from_extension(ext).is_some() || is_document_extension(ext))
  }

  /// Process a single notify event into pending changes
  fn process_event(&mut self, pending: &mut HashMap<PathBuf, PendingChange>, event: Event) {
    for path in &event.paths {
      // Skip directories
      if path.is_dir() {
        // trace!(path = %path.display(), "Skipping directory event");
        continue;
      }

      // Check gitignore
      if self.is_ignored(path) {
        // trace!(path = %path.display(), "Skipping ignored file");
        continue;
      }

      let kind = match event.kind {
        EventKind::Create(_) => {
          // Check if file is indexable
          if !self.is_indexable(path) {
            trace!(path = %path.display(), "Skipping unsupported file type");
            continue;
          }
          debug!(file = %path.display(), "File created");
          ChangeKind::Created
        }
        EventKind::Modify(notify::event::ModifyKind::Name(rename_mode)) => {
          use notify::event::RenameMode;

          match rename_mode {
            RenameMode::Both => {
              // Both paths in a single event: paths[0] = from, paths[1] = to
              // This event comes with both paths, so handle specially
              if event.paths.len() >= 2 {
                let from = &event.paths[0];
                let to = &event.paths[1];

                if to.is_dir() {
                  continue;
                }

                // Skip if ignored
                if self.is_ignored(to) {
                  // But we still need to handle the "from" as a delete
                  if !self.is_ignored(from) {
                    pending.insert(from.clone(), PendingChange::new(ChangeKind::Deleted));
                  }
                  continue;
                }

                debug!(
                    from = %from.display(),
                    to = %to.display(),
                    "File renamed (Both mode)"
                );

                // Remove any pending for the old path
                pending.remove(from);
                // Remove from content cache and re-associate with new path
                if let Some(content) = self.content_cache.get(&from.clone()) {
                  self.content_cache.remove(&from.clone());
                  self.content_cache.put(to.clone(), (*content).clone());
                }

                // Add rename for the new path (key is `to`, but we store `from` for the rename job)
                pending.insert(
                  to.clone(),
                  PendingChange::new(ChangeKind::Renamed { from: from.clone() }),
                );

                // We've handled this specially, skip the normal flow
                // Note: we need to return early since we've processed both paths
                return;
              }
              // Fallback if somehow only one path
              ChangeKind::Modified
            }
            RenameMode::From => {
              // "From" path only - treat as delete (will coalesce with "To")
              debug!(file = %path.display(), "File renamed from (treating as delete)");
              // Remove from content cache
              self.content_cache.remove(&path.clone());
              ChangeKind::Deleted
            }
            RenameMode::To => {
              // "To" path only - treat as create (will coalesce with "From")
              if !self.is_indexable(path) {
                trace!(path = %path.display(), "Skipping unsupported file type");
                continue;
              }
              debug!(file = %path.display(), "File renamed to (treating as create)");
              ChangeKind::Created
            }
            RenameMode::Any | RenameMode::Other => {
              // Generic rename - treat as modified
              if !self.is_indexable(path) {
                continue;
              }
              ChangeKind::Modified
            }
          }
        }
        EventKind::Modify(_) => {
          if !self.is_indexable(path) {
            trace!(path = %path.display(), "Skipping unsupported file type");
            continue;
          }
          debug!(file = %path.display(), "File modified");
          ChangeKind::Modified
        }
        EventKind::Remove(_) => {
          debug!(file = %path.display(), "File deleted");
          // Remove from content cache
          self.content_cache.remove(&path.clone());
          ChangeKind::Deleted
        }
        EventKind::Access(_) | EventKind::Any | EventKind::Other => {
          // Ignore access and other events
          // trace!(file = %path.display(), kind = ?event.kind, "Ignoring event");
          continue;
        }
      };

      // Update or insert pending change
      if let Some(existing) = pending.get_mut(path) {
        existing.update(kind);
      } else {
        pending.insert(path.clone(), PendingChange::new(kind));
      }
    }
  }

  /// Flush pending changes that have settled (debounce period has passed)
  async fn flush_settled(&mut self, pending: &mut HashMap<PathBuf, PendingChange>) {
    let now = Instant::now();
    let debounce = self.config.debounce();

    // Collect paths that have settled
    let settled: Vec<PathBuf> = pending
      .iter()
      .filter(|(_, change)| now.duration_since(change.last_event) >= debounce)
      .map(|(path, _)| path.clone())
      .collect();

    if settled.is_empty() {
      return;
    }

    debug!(count = settled.len(), "Flushing settled changes");

    for path in settled {
      if let Some(change) = pending.remove(&path) {
        self.send_change(path, change).await;
      }
    }
  }

  /// Flush all pending changes (for shutdown)
  async fn flush_all(&mut self, pending: &mut HashMap<PathBuf, PendingChange>) {
    let changes: Vec<(PathBuf, PendingChange)> = pending.drain().collect();

    for (path, change) in changes {
      self.send_change(path, change).await;
    }
  }

  /// Send a change to the indexer
  async fn send_change(&mut self, path: PathBuf, change: PendingChange) {
    // Get old content from cache for incremental parsing
    let old_content = match change.kind {
      ChangeKind::Modified => self.content_cache.get(&path).map(|arc| (*arc).clone()),
      _ => None,
    };

    // Update cache with new content for creates and modifies
    if matches!(change.kind, ChangeKind::Created | ChangeKind::Modified)
      && let Ok(content) = std::fs::read_to_string(&path)
    {
      self.content_cache.put(path.clone(), content);
    }

    let job = match change.kind {
      ChangeKind::Created | ChangeKind::Modified => IndexJob::File { path, old_content },
      ChangeKind::Deleted => IndexJob::Delete { path },
      // path is the key (new location), from is stored in ChangeKind
      ChangeKind::Renamed { from } => IndexJob::Rename { from, to: path },
    };

    if let Err(e) = self.indexer.send(job).await {
      warn!(error = %e, "Failed to send index job");
    }
  }
}

// ============================================================================
// Gitignore Helper
// ============================================================================

/// Build a gitignore matcher for the given root directory
fn build_gitignore(root: &PathBuf) -> Result<Option<Gitignore>, WatcherError> {
  let gitignore_path = root.join(".gitignore");

  if !gitignore_path.exists() {
    debug!(root = %root.display(), "No .gitignore found, all files will be processed");
    return Ok(None);
  }

  let mut builder = GitignoreBuilder::new(root);

  // Add .gitignore rules
  if let Some(err) = builder.add(&gitignore_path) {
    warn!(error = %err, "Error parsing .gitignore, continuing with partial rules");
  }

  // Also add .ccengramignore if present
  let ccengramignore_path = root.join(".ccengramignore");
  if ccengramignore_path.exists()
    && let Some(err) = builder.add(&ccengramignore_path)
  {
    warn!(error = %err, "Error parsing .ccengramignore");
  }

  // Add common patterns that should always be ignored
  let _ = builder.add_line(None, ".git/");
  let _ = builder.add_line(None, "node_modules/");
  let _ = builder.add_line(None, "target/");
  let _ = builder.add_line(None, "__pycache__/");
  let _ = builder.add_line(None, ".venv/");
  let _ = builder.add_line(None, "*.pyc");

  let gitignore = builder.build().map_err(WatcherError::Gitignore)?;

  debug!(
    root = %root.display(),
    gitignore_path = %gitignore_path.display(),
    "Gitignore matcher built"
  );

  Ok(Some(gitignore))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_pending_change_coalescing() {
    // Create + Modify = Create
    let mut pending = PendingChange::new(ChangeKind::Created);
    pending.update(ChangeKind::Modified);
    assert!(matches!(pending.kind, ChangeKind::Created));

    // Delete + Create = Modified
    let mut pending = PendingChange::new(ChangeKind::Deleted);
    pending.update(ChangeKind::Created);
    assert!(matches!(pending.kind, ChangeKind::Modified));

    // Create + Delete = Deleted
    let mut pending = PendingChange::new(ChangeKind::Created);
    pending.update(ChangeKind::Deleted);
    assert!(matches!(pending.kind, ChangeKind::Deleted));

    // Rename + Modify = Rename (preserves rename info)
    let mut pending = PendingChange::new(ChangeKind::Renamed {
      from: PathBuf::from("/old"),
    });
    pending.update(ChangeKind::Modified);
    assert!(matches!(pending.kind, ChangeKind::Renamed { .. }));
  }

  #[test]
  fn test_content_cache() {
    let mut cache = ContentCache::new(3, 1024);

    // Test put and get
    cache.put(PathBuf::from("/a"), "content a".to_string());
    cache.put(PathBuf::from("/b"), "content b".to_string());

    assert_eq!(
      cache.get(&PathBuf::from("/a")).map(|s| s.as_str().to_string()),
      Some("content a".to_string())
    );
    assert_eq!(
      cache.get(&PathBuf::from("/b")).map(|s| s.as_str().to_string()),
      Some("content b".to_string())
    );
    assert!(cache.get(&PathBuf::from("/c")).is_none());

    // Test LRU eviction
    // At this point: /a accessed at T3, /b accessed at T4 (more recent)
    cache.put(PathBuf::from("/c"), "content c".to_string()); // /c at T5
    cache.put(PathBuf::from("/d"), "content d".to_string()); // Evicts /a (oldest at T3)

    // /b was accessed more recently than /a, so /a gets evicted
    assert!(cache.cache.contains_key(&PathBuf::from("/b")));
    assert!(cache.cache.contains_key(&PathBuf::from("/c")));
    assert!(cache.cache.contains_key(&PathBuf::from("/d")));
    assert_eq!(cache.cache.len(), 3);
  }

  #[test]
  fn test_content_cache_size_limit() {
    let mut cache = ContentCache::new(10, 100); // Max 100 bytes per file

    // Small file should be cached
    cache.put(PathBuf::from("/small"), "small".to_string());
    assert!(cache.get(&PathBuf::from("/small")).is_some());

    // Large file should not be cached
    let large_content = "x".repeat(200);
    cache.put(PathBuf::from("/large"), large_content);
    assert!(cache.get(&PathBuf::from("/large")).is_none());
  }

  #[tokio::test]
  async fn test_watcher_indexer_handle_integration() {
    // Verify we can create an IndexerHandle and use it with our job types
    let (tx, mut rx) = mpsc::channel::<IndexJob>(10);
    let handle = IndexerHandle::new(tx);

    // Send a file job
    handle
      .send(IndexJob::File {
        path: PathBuf::from("/test/file.rs"),
        old_content: None,
      })
      .await
      .expect("send should succeed");

    match rx.recv().await {
      Some(IndexJob::File { path, .. }) => {
        assert_eq!(path, PathBuf::from("/test/file.rs"));
      }
      other => panic!("expected IndexJob::File, got {:?}", other),
    }

    // Send a delete job
    handle
      .send(IndexJob::Delete {
        path: PathBuf::from("/test/deleted.rs"),
      })
      .await
      .expect("send should succeed");

    match rx.recv().await {
      Some(IndexJob::Delete { path }) => {
        assert_eq!(path, PathBuf::from("/test/deleted.rs"));
      }
      other => panic!("expected IndexJob::Delete, got {:?}", other),
    }

    // Send a rename job
    handle
      .send(IndexJob::Rename {
        from: PathBuf::from("/test/old.rs"),
        to: PathBuf::from("/test/new.rs"),
      })
      .await
      .expect("send should succeed");

    match rx.recv().await {
      Some(IndexJob::Rename { from, to }) => {
        assert_eq!(from, PathBuf::from("/test/old.rs"));
        assert_eq!(to, PathBuf::from("/test/new.rs"));
      }
      other => panic!("expected IndexJob::Rename, got {:?}", other),
    }
  }
}
