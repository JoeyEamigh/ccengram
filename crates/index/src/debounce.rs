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
use tracing::{debug, trace};

/// Configuration for debounced watcher
#[derive(Debug, Clone)]
pub struct DebounceConfig {
  /// Debounce delay for file changes (default 1000ms)
  pub file_debounce_ms: u64,
  /// Debounce delay for gitignore changes (default 1000ms)
  pub gitignore_debounce_ms: u64,
  /// Maximum age of oldest pending event before forcing flush (default 5000ms)
  /// This ensures events don't queue forever while allowing debouncing during bursts
  pub max_pending_age_ms: u64,
}

impl Default for DebounceConfig {
  fn default() -> Self {
    Self {
      file_debounce_ms: 1000,
      gitignore_debounce_ms: 1000,
      max_pending_age_ms: 5000,
    }
  }
}

/// Accumulated change state for a single file
#[derive(Debug, Clone)]
struct PendingChange {
  kind: ChangeKind,
  last_seen: Instant,
  /// Original path for rename operations (None for other operations)
  old_path: Option<PathBuf>,
}

impl PendingChange {
  fn new(kind: ChangeKind) -> Self {
    Self {
      kind,
      last_seen: Instant::now(),
      old_path: None,
    }
  }

  fn with_old_path(kind: ChangeKind, old_path: PathBuf) -> Self {
    Self {
      kind,
      last_seen: Instant::now(),
      old_path: Some(old_path),
    }
  }

  fn update(&mut self, kind: ChangeKind, old_path: Option<PathBuf>) {
    let old_kind = self.kind.clone();
    self.last_seen = Instant::now();
    // Coalesce event types
    match (&self.kind, &kind) {
      // Create followed by modify is still a create
      (ChangeKind::Created, ChangeKind::Modified) => {
        trace!(from = ?old_kind, to = ?kind, result = ?self.kind, "Event coalesced: create+modify -> create");
      }
      // Delete followed by create is a modify
      (ChangeKind::Deleted, ChangeKind::Created) => {
        self.kind = ChangeKind::Modified;
        trace!(from = ?old_kind, to = ?kind, result = ?self.kind, "Event coalesced: delete+create -> modified");
      }
      // Create followed by delete cancels out
      (ChangeKind::Created, ChangeKind::Deleted) => {
        self.kind = ChangeKind::Deleted;
        trace!(from = ?old_kind, to = ?kind, result = ?self.kind, "Event coalesced: create+delete -> deleted");
      }
      // Rename followed by modify is still a rename
      (ChangeKind::Renamed, ChangeKind::Modified) => {
        trace!(from = ?old_kind, to = ?kind, result = ?self.kind, "Event coalesced: rename+modify -> rename");
      }
      // Modified followed by rename becomes rename (preserve old_path from rename)
      (ChangeKind::Modified, ChangeKind::Renamed) => {
        self.kind = ChangeKind::Renamed;
        if old_path.is_some() {
          self.old_path = old_path;
        }
        trace!(from = ?old_kind, to = ?kind, result = ?self.kind, "Event coalesced: modified+rename -> rename");
      }
      // Otherwise take the latest
      _ => {
        self.kind = kind.clone();
        // Update old_path if provided
        if old_path.is_some() {
          self.old_path = old_path;
        }
        trace!(from = ?old_kind, to = ?kind, result = ?self.kind, "Event updated to latest");
      }
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

impl Drop for DebouncedWatcher {
  fn drop(&mut self) {
    if !self.pending.is_empty() {
      debug!(
        pending = self.pending.len(),
        "Debounced watcher stopping with pending events"
      );
    }
  }
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
          old_path: pending.old_path.clone(),
        });
        to_remove.push(path.clone());
      }
    }

    for path in to_remove {
      self.pending.remove(&path);
    }

    if !ready.is_empty() {
      debug!(
        count = ready.len(),
        pending_remaining = self.pending.len(),
        "Flushing debounced batch"
      );
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
        old_path: pending.old_path,
      })
      .collect();

    if !changes.is_empty() {
      debug!(count = changes.len(), "Force flushing all pending changes");
    }

    changes
  }

  /// Block until ready events are available or max_wait timeout expires.
  ///
  /// This is more efficient than poll+sleep because it:
  /// - Uses zero CPU when idle (blocks on channel recv)
  /// - Responds immediately when events arrive
  /// - Still respects debounce timing via calculated timeouts
  pub fn wait_ready(&mut self, max_wait: Duration) -> Result<Vec<FileChange>, WatchError> {
    let start = Instant::now();

    loop {
      // Check if we've exceeded max_wait
      if start.elapsed() >= max_wait {
        return Ok(Vec::new());
      }

      // Check if any events are ready now
      let ready = self.collect_ready();
      if !ready.is_empty() {
        return Ok(ready);
      }

      // Check for force flush condition
      if self.should_force_flush() {
        return Ok(self.collect_all());
      }

      // Calculate timeout: min of (next debounce expiry, remaining max_wait)
      let remaining = max_wait.saturating_sub(start.elapsed());
      let timeout = if self.pending.is_empty() {
        remaining // No pending events, just wait for new ones
      } else {
        let debounce = Duration::from_millis(self.config.file_debounce_ms);
        let now = Instant::now();
        self
          .pending
          .values()
          .map(|p| debounce.saturating_sub(now.duration_since(p.last_seen)))
          .min()
          .unwrap_or(remaining)
          .min(remaining)
      };

      // Block until event or timeout
      match self.watcher.wait_timeout(timeout) {
        Ok(Some(change)) => {
          self.handle_change(change);
          self.poll_raw(); // Drain any additional buffered events
        }
        Ok(None) => {
          // Timeout - loop to check if debounced events are ready
        }
        Err(e) => return Err(e),
      }
    }
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

  /// Check if we should force a flush due to oldest event exceeding max age
  /// This ensures events don't queue forever while allowing debouncing during bursts
  pub fn should_force_flush(&self) -> bool {
    if self.pending.is_empty() {
      return false;
    }
    let max_age = Duration::from_millis(self.config.max_pending_age_ms);
    self
      .pending
      .values()
      .map(|p| p.last_seen.elapsed())
      .max()
      .map(|oldest| oldest >= max_age)
      .unwrap_or(false)
  }

  fn handle_change(&mut self, change: FileChange) {
    // Ignore changes to gitignore itself (handled separately)
    if change
      .path
      .file_name()
      .is_some_and(|n| n == ".gitignore" || n == ".ccengramignore")
    {
      trace!(file = %change.path.display(), "Ignore file change detected, will check on next cycle");
      self.gitignore_last_change = None; // Reset to force check on next call
      return;
    }

    trace!(
      file = %change.path.display(),
      kind = ?change.kind,
      "Accumulating change event"
    );

    // For rename events, remove any pending changes for the old path
    // since those are now stale (the file has been renamed)
    if change.kind == ChangeKind::Renamed
      && let Some(ref old_path) = change.old_path
    {
      self.pending.remove(old_path);
      debug!(
        "Rename detected: {:?} -> {:?}, removed stale pending for old path",
        old_path, change.path
      );
    }

    // Accumulate the change
    if let Some(pending) = self.pending.get_mut(&change.path) {
      pending.update(change.kind, change.old_path);
    } else if change.kind == ChangeKind::Renamed {
      if let Some(old_path) = change.old_path {
        self
          .pending
          .insert(change.path, PendingChange::with_old_path(change.kind, old_path));
      } else {
        self.pending.insert(change.path, PendingChange::new(change.kind));
      }
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
    assert_eq!(config.file_debounce_ms, 1000);
    assert_eq!(config.gitignore_debounce_ms, 1000);
    assert_eq!(config.max_pending_age_ms, 5000);
  }

  #[test]
  fn test_pending_change_coalescing() {
    // Create + Modify = Create
    let mut pending = PendingChange::new(ChangeKind::Created);
    pending.update(ChangeKind::Modified, None);
    assert_eq!(pending.kind, ChangeKind::Created);

    // Delete + Create = Modified
    let mut pending = PendingChange::new(ChangeKind::Deleted);
    pending.update(ChangeKind::Created, None);
    assert_eq!(pending.kind, ChangeKind::Modified);

    // Create + Delete = Delete
    let mut pending = PendingChange::new(ChangeKind::Created);
    pending.update(ChangeKind::Deleted, None);
    assert_eq!(pending.kind, ChangeKind::Deleted);

    // Rename + Modify = Rename (preserve rename info)
    let mut pending = PendingChange::with_old_path(ChangeKind::Renamed, PathBuf::from("/old/path"));
    pending.update(ChangeKind::Modified, None);
    assert_eq!(pending.kind, ChangeKind::Renamed);
    assert!(pending.old_path.is_some());

    // Modified + Rename = Rename (acquires rename info)
    let mut pending = PendingChange::new(ChangeKind::Modified);
    pending.update(ChangeKind::Renamed, Some(PathBuf::from("/old/path")));
    assert_eq!(pending.kind, ChangeKind::Renamed);
    assert_eq!(pending.old_path.unwrap(), PathBuf::from("/old/path"));
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
        max_pending_age_ms: 50, // Very short for testing
        ..Default::default()
      },
    )
    .unwrap();

    // Add an event
    watcher
      .pending
      .insert(PathBuf::from("/test/old.rs"), PendingChange::new(ChangeKind::Modified));

    // Should not flush immediately
    assert!(!watcher.should_force_flush());

    // Wait for max age to pass
    std::thread::sleep(Duration::from_millis(100));

    // Now should force flush due to age
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

  #[test]
  fn test_wait_ready_timeout() {
    let dir = TempDir::new().unwrap();
    let mut watcher = DebouncedWatcher::new(
      dir.path(),
      DebounceConfig {
        file_debounce_ms: 50,
        ..Default::default()
      },
    )
    .unwrap();

    // wait_ready should return empty after timeout when no events
    let start = std::time::Instant::now();
    let result = watcher.wait_ready(Duration::from_millis(100));
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
    // Should have waited approximately the timeout duration
    assert!(elapsed >= Duration::from_millis(90));
  }

  #[test]
  fn test_wait_ready_with_pending_events() {
    let dir = TempDir::new().unwrap();
    let mut watcher = DebouncedWatcher::new(
      dir.path(),
      DebounceConfig {
        file_debounce_ms: 50,
        ..Default::default()
      },
    )
    .unwrap();

    // Add pending events that are already past debounce time
    watcher
      .pending
      .insert(PathBuf::from("/test/file.rs"), PendingChange::new(ChangeKind::Modified));

    // Wait for debounce period
    std::thread::sleep(Duration::from_millis(60));

    // wait_ready should return immediately with the ready event
    let start = std::time::Instant::now();
    let result = watcher.wait_ready(Duration::from_secs(5));
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
    // Should return quickly, not wait the full timeout
    assert!(elapsed < Duration::from_millis(100));
  }
}
