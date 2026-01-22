// Debounced file watcher - batches events and deduplicates changes
//
// Features:
// - 500ms debounce for file changes
// - 1000ms debounce for gitignore changes
// - Deduplication of events by file path
// - Coalescing of create+modify into single event

use crate::gitignore::GitignoreState;
use crate::watcher::{ChangeKind, FileChange, FileWatcher, WatchError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::debug;

/// Configuration for debounced watcher
#[derive(Debug, Clone)]
pub struct DebounceConfig {
  /// Debounce delay for file changes (default 500ms)
  pub file_debounce_ms: u64,
  /// Debounce delay for gitignore changes (default 1000ms)
  pub gitignore_debounce_ms: u64,
  /// Maximum events to collect before forcing a flush
  pub max_pending_events: usize,
}

impl Default for DebounceConfig {
  fn default() -> Self {
    Self {
      file_debounce_ms: 500,
      gitignore_debounce_ms: 1000,
      max_pending_events: 100,
    }
  }
}

/// Accumulated change state for a single file
#[derive(Debug, Clone)]
struct PendingChange {
  kind: ChangeKind,
  last_seen: Instant,
}

impl PendingChange {
  fn new(kind: ChangeKind) -> Self {
    Self {
      kind,
      last_seen: Instant::now(),
    }
  }

  fn update(&mut self, kind: ChangeKind) {
    self.last_seen = Instant::now();
    // Coalesce event types
    match (&self.kind, &kind) {
      // Create followed by modify is still a create
      (ChangeKind::Created, ChangeKind::Modified) => {}
      // Delete followed by create is a modify
      (ChangeKind::Deleted, ChangeKind::Created) => self.kind = ChangeKind::Modified,
      // Create followed by delete cancels out
      (ChangeKind::Created, ChangeKind::Deleted) => self.kind = ChangeKind::Deleted,
      // Otherwise take the latest
      _ => self.kind = kind,
    }
  }
}

/// A debounced file watcher that batches and deduplicates events
pub struct DebouncedWatcher {
  watcher: FileWatcher,
  config: DebounceConfig,
  pending: HashMap<PathBuf, PendingChange>,
  gitignore_state: Option<GitignoreState>,
  gitignore_last_change: Option<Instant>,
}

impl DebouncedWatcher {
  /// Create a new debounced watcher
  pub fn new(root: &Path, config: DebounceConfig) -> Result<Self, WatchError> {
    let watcher = FileWatcher::new(root)?;
    let gitignore_state = GitignoreState::load(root).ok();

    Ok(Self {
      watcher,
      config,
      pending: HashMap::new(),
      gitignore_state,
      gitignore_last_change: None,
    })
  }

  /// Create with default config
  pub fn with_defaults(root: &Path) -> Result<Self, WatchError> {
    Self::new(root, DebounceConfig::default())
  }

  /// Get the root directory being watched
  pub fn root(&self) -> &Path {
    self.watcher.root()
  }

  /// Poll for raw events and accumulate them
  pub fn poll_raw(&mut self) {
    while let Some(change) = self.watcher.poll() {
      self.handle_change(change);
    }
  }

  /// Collect ready changes (debounce period has passed)
  pub fn collect_ready(&mut self) -> Vec<FileChange> {
    self.poll_raw();

    let now = Instant::now();
    let debounce_duration = Duration::from_millis(self.config.file_debounce_ms);

    let mut ready = Vec::new();
    let mut to_remove = Vec::new();

    for (path, pending) in &self.pending {
      if now.duration_since(pending.last_seen) >= debounce_duration {
        ready.push(FileChange {
          path: path.clone(),
          kind: pending.kind.clone(),
        });
        to_remove.push(path.clone());
      }
    }

    for path in to_remove {
      self.pending.remove(&path);
    }

    ready
  }

  /// Force collect all pending changes regardless of debounce time
  pub fn collect_all(&mut self) -> Vec<FileChange> {
    self.poll_raw();

    let changes: Vec<FileChange> = self
      .pending
      .drain()
      .map(|(path, pending)| FileChange {
        path,
        kind: pending.kind,
      })
      .collect();

    changes
  }

  /// Check if gitignore has changed (with debouncing)
  pub fn check_gitignore_change(&mut self) -> bool {
    let now = Instant::now();

    // Only check if enough time has passed since last change
    if let Some(last_change) = self.gitignore_last_change
      && now.duration_since(last_change) < Duration::from_millis(self.config.gitignore_debounce_ms)
    {
      return false;
    }

    // Reload gitignore and check if changed
    let root = self.watcher.root().to_path_buf();
    if let Ok(new_state) = GitignoreState::load(&root) {
      if let Some(ref old_state) = self.gitignore_state {
        if new_state.hash != old_state.hash {
          debug!("Gitignore changed: {} -> {}", old_state.hash, new_state.hash);
          self.gitignore_state = Some(new_state);
          self.gitignore_last_change = Some(now);
          return true;
        }
      } else {
        self.gitignore_state = Some(new_state);
      }
    }

    false
  }

  /// Get current gitignore state
  pub fn gitignore_state(&self) -> Option<&GitignoreState> {
    self.gitignore_state.as_ref()
  }

  /// Number of pending events
  pub fn pending_count(&self) -> usize {
    self.pending.len()
  }

  /// Check if we should force a flush due to too many pending events
  pub fn should_force_flush(&self) -> bool {
    self.pending.len() >= self.config.max_pending_events
  }

  fn handle_change(&mut self, change: FileChange) {
    // Ignore changes to gitignore itself (handled separately)
    if change
      .path
      .file_name()
      .is_some_and(|n| n == ".gitignore" || n == ".ccengramignore")
    {
      self.gitignore_last_change = None; // Reset to force check on next call
      return;
    }

    // Accumulate the change
    if let Some(pending) = self.pending.get_mut(&change.path) {
      pending.update(change.kind);
    } else {
      self.pending.insert(change.path, PendingChange::new(change.kind));
    }
  }
}

/// Batch processor for debounced changes
pub struct BatchProcessor {
  watcher: DebouncedWatcher,
  batch_interval: Duration,
  last_batch: Instant,
}

impl BatchProcessor {
  pub fn new(watcher: DebouncedWatcher) -> Self {
    Self {
      watcher,
      batch_interval: Duration::from_secs(1),
      last_batch: Instant::now(),
    }
  }

  pub fn with_interval(watcher: DebouncedWatcher, interval: Duration) -> Self {
    Self {
      watcher,
      batch_interval: interval,
      last_batch: Instant::now(),
    }
  }

  /// Process a batch of changes, calling the handler for each
  pub fn process_batch<F>(&mut self, handler: F) -> Result<usize, WatchError>
  where
    F: FnMut(FileChange),
  {
    let now = Instant::now();

    // Check if it's time to process
    if now.duration_since(self.last_batch) < self.batch_interval && !self.watcher.should_force_flush() {
      return Ok(0);
    }

    // Collect ready changes
    let changes = if self.watcher.should_force_flush() {
      self.watcher.collect_all()
    } else {
      self.watcher.collect_ready()
    };

    let count = changes.len();

    // Process each change
    changes.into_iter().for_each(handler);

    self.last_batch = now;
    Ok(count)
  }

  /// Check if gitignore has changed
  pub fn check_gitignore_change(&mut self) -> bool {
    self.watcher.check_gitignore_change()
  }

  /// Get the underlying watcher for direct access
  pub fn watcher(&self) -> &DebouncedWatcher {
    &self.watcher
  }

  /// Get mutable access to the underlying watcher
  pub fn watcher_mut(&mut self) -> &mut DebouncedWatcher {
    &mut self.watcher
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;
  use tempfile::TempDir;

  #[test]
  fn test_debounce_config_defaults() {
    let config = DebounceConfig::default();
    assert_eq!(config.file_debounce_ms, 500);
    assert_eq!(config.gitignore_debounce_ms, 1000);
    assert_eq!(config.max_pending_events, 100);
  }

  #[test]
  fn test_pending_change_coalescing() {
    // Create + Modify = Create
    let mut pending = PendingChange::new(ChangeKind::Created);
    pending.update(ChangeKind::Modified);
    assert_eq!(pending.kind, ChangeKind::Created);

    // Delete + Create = Modified
    let mut pending = PendingChange::new(ChangeKind::Deleted);
    pending.update(ChangeKind::Created);
    assert_eq!(pending.kind, ChangeKind::Modified);

    // Create + Delete = Delete
    let mut pending = PendingChange::new(ChangeKind::Created);
    pending.update(ChangeKind::Deleted);
    assert_eq!(pending.kind, ChangeKind::Deleted);
  }

  #[test]
  fn test_debounced_watcher_creation() {
    let dir = TempDir::new().unwrap();
    let watcher = DebouncedWatcher::with_defaults(dir.path());
    assert!(watcher.is_ok());
  }

  #[test]
  fn test_debounced_watcher_collect_ready() {
    let dir = TempDir::new().unwrap();

    // Create a source file
    fs::write(dir.path().join("test.rs"), "fn main() {}").unwrap();

    let mut watcher = DebouncedWatcher::new(
      dir.path(),
      DebounceConfig {
        file_debounce_ms: 50, // Short but reliable for testing
        ..Default::default()
      },
    )
    .unwrap();

    // Initial state
    assert_eq!(watcher.pending_count(), 0);

    // Simulate some time passing and check ready
    std::thread::sleep(Duration::from_millis(100));
    let ready = watcher.collect_ready();
    assert!(ready.is_empty()); // No changes yet
  }

  #[test]
  fn test_gitignore_detection() {
    let dir = TempDir::new().unwrap();

    // Create initial gitignore
    fs::write(dir.path().join(".gitignore"), "*.log").unwrap();

    let mut watcher = DebouncedWatcher::new(
      dir.path(),
      DebounceConfig {
        gitignore_debounce_ms: 50, // Short but reliable for testing
        ..Default::default()
      },
    )
    .unwrap();

    // Initial state - no change
    assert!(!watcher.check_gitignore_change());

    // Modify gitignore
    fs::write(dir.path().join(".gitignore"), "*.log\n*.tmp").unwrap();

    // Wait for debounce with margin
    std::thread::sleep(Duration::from_millis(100));

    // Should detect change
    assert!(watcher.check_gitignore_change());
  }

  #[test]
  fn test_should_force_flush() {
    let dir = TempDir::new().unwrap();
    let mut watcher = DebouncedWatcher::new(
      dir.path(),
      DebounceConfig {
        max_pending_events: 5,
        ..Default::default()
      },
    )
    .unwrap();

    // Add events directly
    for i in 0..5 {
      watcher.pending.insert(
        PathBuf::from(format!("/test/{}.rs", i)),
        PendingChange::new(ChangeKind::Modified),
      );
    }

    assert!(watcher.should_force_flush());
  }

  #[test]
  fn test_batch_processor() {
    let dir = TempDir::new().unwrap();
    let watcher = DebouncedWatcher::new(
      dir.path(),
      DebounceConfig {
        file_debounce_ms: 50, // Short but reliable for testing
        ..Default::default()
      },
    )
    .unwrap();

    let mut processor = BatchProcessor::with_interval(watcher, Duration::from_millis(50));

    // Add some pending changes manually
    for i in 0..3 {
      processor.watcher_mut().pending.insert(
        PathBuf::from(format!("/test/{}.rs", i)),
        PendingChange::new(ChangeKind::Modified),
      );
    }

    // Wait for debounce with margin
    std::thread::sleep(Duration::from_millis(100));

    let mut processed = Vec::new();
    let count = processor.process_batch(|change| processed.push(change)).unwrap();

    assert_eq!(count, 3);
    assert_eq!(processed.len(), 3);
  }
}
