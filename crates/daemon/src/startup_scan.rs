//! Startup scan for filesystem reconciliation.
//!
//! When the file watcher starts, this module reconciles the database with the
//! actual filesystem state. It detects:
//! - **Deleted files**: Removed from filesystem, but chunks remain in database
//! - **Added files**: New files created while watcher wasn't running
//! - **Modified files**: File contents changed since last indexing
//!
//! This ensures the index accurately reflects the current project state.

use crate::projects::{FileChangeContext, process_file_changes_batched};
use db::ProjectDb;
use embedding::EmbeddingProvider;
use engram_core::Config;
pub use engram_core::ScanMode;
use index::{GITIGNORE_CACHE, Scanner};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Errors that can occur during startup scan
#[derive(Error, Debug)]
pub enum ScanError {
  #[error("Database error: {0}")]
  Database(#[from] db::DbError),
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Scan timed out after {0:?}")]
  Timeout(Duration),
  #[error("Scan cancelled")]
  Cancelled,
}

/// Configuration for startup scan behavior
#[derive(Debug, Clone)]
pub struct StartupScanConfig {
  /// Whether to run a scan on watcher startup
  pub enabled: bool,

  /// Scan mode
  pub mode: ScanMode,

  /// Whether to block watcher startup until scan completes
  pub blocking: bool,

  /// Maximum files to scan (0 = unlimited)
  pub max_files: usize,

  /// Timeout for the scan operation
  pub timeout: Duration,

  /// Number of parallel file operations
  pub parallelism: usize,
}

impl Default for StartupScanConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      mode: ScanMode::Full,
      blocking: false,
      max_files: 0,
      timeout: Duration::from_secs(300),
      parallelism: 8,
    }
  }
}

impl StartupScanConfig {
  /// Create config from the main Config
  pub fn from_config(config: &Config) -> Self {
    Self {
      enabled: config.index.startup_scan,
      mode: config.index.startup_scan_mode,
      blocking: config.index.startup_scan_blocking,
      max_files: 0, // Could add to config later
      timeout: Duration::from_secs(config.index.startup_scan_timeout_secs),
      parallelism: config.index.parallel_files.max(1),
    }
  }
}

/// Summary of a file in the database
#[derive(Debug, Clone)]
pub struct IndexedFile {
  pub file_path: String,
  pub file_hash: String,
  pub indexed_at_ms: i64,
  pub chunk_count: usize,
}

/// Summary of a file on the filesystem
#[derive(Debug, Clone)]
pub struct FilesystemFile {
  pub file_path: String,
  pub current_hash: String,
  pub mtime: u64,
  pub size: u64,
}

/// Classification of a file change
#[derive(Debug, Clone)]
pub enum FileChange {
  Deleted {
    path: String,
  },
  Added {
    path: String,
    hash: String,
  },
  Modified {
    path: String,
    old_hash: String,
    new_hash: String,
  },
  Unchanged {
    path: String,
  },
  /// File was renamed (same content hash, different path)
  Renamed {
    old_path: String,
    new_path: String,
    hash: String,
  },
}

/// Results from scanning
#[derive(Debug, Clone, Default)]
pub struct ScanResult {
  pub deleted: Vec<String>,
  pub added: Vec<String>,
  pub modified: Vec<String>,
  /// Files that were renamed (old_path, new_path)
  pub renamed: Vec<(String, String)>,
  pub unchanged_count: usize,
  pub scan_duration: Duration,
  pub errors: Vec<String>,
}

impl ScanResult {
  pub fn total_changes(&self) -> usize {
    self.deleted.len() + self.added.len() + self.modified.len() + self.renamed.len()
  }

  pub fn is_empty(&self) -> bool {
    self.deleted.is_empty() && self.added.is_empty() && self.modified.is_empty() && self.renamed.is_empty()
  }
}

/// Results from applying scan changes
#[derive(Debug, Clone, Default)]
pub struct ApplyResult {
  pub files_deleted: usize,
  pub files_indexed: usize,
  pub files_reindexed: usize,
  /// Files renamed (path updated, embeddings preserved)
  pub files_renamed: usize,
  pub apply_duration: Duration,
  pub errors: Vec<String>,
}

/// Scan state that can be checked by other components
#[derive(Debug)]
pub struct ScanState {
  /// Whether a scan is currently in progress
  pub in_progress: AtomicBool,
  /// Total files to process
  pub total_files: AtomicUsize,
  /// Files processed so far
  pub processed_files: AtomicUsize,
  /// Current phase description
  phase: RwLock<String>,
}

impl Default for ScanState {
  fn default() -> Self {
    Self::new()
  }
}

impl ScanState {
  pub fn new() -> Self {
    Self {
      in_progress: AtomicBool::new(false),
      total_files: AtomicUsize::new(0),
      processed_files: AtomicUsize::new(0),
      phase: RwLock::new(String::new()),
    }
  }

  pub fn is_in_progress(&self) -> bool {
    self.in_progress.load(Ordering::SeqCst)
  }

  pub fn progress(&self) -> (usize, usize) {
    (
      self.processed_files.load(Ordering::Relaxed),
      self.total_files.load(Ordering::Relaxed),
    )
  }

  pub async fn phase(&self) -> String {
    self.phase.read().await.clone()
  }

  pub(crate) fn start(&self) {
    self.in_progress.store(true, Ordering::SeqCst);
    self.total_files.store(0, Ordering::Relaxed);
    self.processed_files.store(0, Ordering::Relaxed);
  }

  pub(crate) fn finish(&self) {
    self.in_progress.store(false, Ordering::SeqCst);
  }

  pub(crate) async fn set_phase(&self, phase: &str) {
    *self.phase.write().await = phase.to_string();
  }
}

/// Startup scanner for filesystem reconciliation
pub struct StartupScanner {
  config: StartupScanConfig,
  state: Arc<ScanState>,
  cancel: Arc<AtomicBool>,
}

impl StartupScanner {
  pub fn new(config: StartupScanConfig) -> Self {
    Self {
      config,
      state: Arc::new(ScanState::new()),
      cancel: Arc::new(AtomicBool::new(false)),
    }
  }

  /// Get the scan state for external monitoring
  pub fn state(&self) -> Arc<ScanState> {
    Arc::clone(&self.state)
  }

  /// Request cancellation of the scan
  pub fn cancel(&self) {
    self.cancel.store(true, Ordering::SeqCst);
  }

  /// Check if scan is cancelled
  fn is_cancelled(&self) -> bool {
    self.cancel.load(Ordering::SeqCst)
  }

  /// Perform a startup scan and return results
  pub async fn scan(&self, db: &ProjectDb, root: &Path) -> Result<ScanResult, ScanError> {
    let start = Instant::now();
    self.state.start();
    self.state.set_phase("Loading indexed files").await;

    info!("Starting startup scan for {:?}", root);
    debug!(
      "Scan config: mode={:?}, blocking={}, parallelism={}",
      self.config.mode, self.config.blocking, self.config.parallelism
    );

    // Step 1: Load indexed files from database
    let indexed_files = self.load_indexed_files(db).await?;
    let indexed_count = indexed_files.len();
    debug!("Loaded {} indexed files from database", indexed_count);

    if self.is_cancelled() {
      self.state.finish();
      return Err(ScanError::Cancelled);
    }

    // Step 2: Scan filesystem
    self.state.set_phase("Scanning filesystem").await;
    let filesystem_files = self.scan_filesystem(root)?;
    let fs_count = filesystem_files.len();
    debug!("Found {} files on filesystem", fs_count);

    if self.is_cancelled() {
      self.state.finish();
      return Err(ScanError::Cancelled);
    }

    // Step 3: Compare and classify changes
    self.state.set_phase("Comparing files").await;
    let changes = self.classify_changes(&indexed_files, &filesystem_files);

    let mut result = ScanResult {
      scan_duration: start.elapsed(),
      ..Default::default()
    };

    for change in changes {
      match change {
        FileChange::Deleted { path } => result.deleted.push(path),
        FileChange::Added { path, .. } => result.added.push(path),
        FileChange::Modified { path, .. } => result.modified.push(path),
        FileChange::Unchanged { .. } => result.unchanged_count += 1,
        FileChange::Renamed { old_path, new_path, .. } => result.renamed.push((old_path, new_path)),
      }
    }

    self.state.finish();

    info!(
      "Scan complete: {} new, {} deleted, {} modified, {} renamed, {} unchanged ({:.2}s)",
      result.added.len(),
      result.deleted.len(),
      result.modified.len(),
      result.renamed.len(),
      result.unchanged_count,
      result.scan_duration.as_secs_f64()
    );

    Ok(result)
  }

  /// Load all indexed files from the database
  async fn load_indexed_files(&self, db: &ProjectDb) -> Result<HashMap<String, IndexedFile>, ScanError> {
    // Get all distinct files with their hashes and timestamps
    // We query all chunks and group by file_path
    let chunks = db.list_code_chunks(None, None).await?;

    let mut files: HashMap<String, IndexedFile> = HashMap::new();

    for chunk in chunks {
      let entry = files.entry(chunk.file_path.clone()).or_insert_with(|| IndexedFile {
        file_path: chunk.file_path.clone(),
        file_hash: chunk.file_hash.clone(),
        indexed_at_ms: chunk.indexed_at.timestamp_millis(),
        chunk_count: 0,
      });
      entry.chunk_count += 1;
      // Update hash if this chunk has a more recent timestamp
      if chunk.indexed_at.timestamp_millis() > entry.indexed_at_ms {
        entry.file_hash = chunk.file_hash;
        entry.indexed_at_ms = chunk.indexed_at.timestamp_millis();
      }
    }

    Ok(files)
  }

  /// Scan the filesystem for current files
  fn scan_filesystem(&self, root: &Path) -> Result<HashMap<String, FilesystemFile>, ScanError> {
    let scanner = Scanner::new();
    let scan_result = scanner.scan(root, |_progress| {});

    let mut files = HashMap::with_capacity(scan_result.files.len());

    for scanned in scan_result.files {
      // Apply gitignore filtering
      if GITIGNORE_CACHE.should_ignore(root, &scanned.path) {
        continue;
      }

      files.insert(
        scanned.relative_path.clone(),
        FilesystemFile {
          file_path: scanned.relative_path,
          current_hash: scanned.checksum,
          mtime: scanned.mtime,
          size: scanned.size,
        },
      );
    }

    Ok(files)
  }

  /// Classify changes between indexed and filesystem files
  fn classify_changes(
    &self,
    indexed: &HashMap<String, IndexedFile>,
    filesystem: &HashMap<String, FilesystemFile>,
  ) -> Vec<FileChange> {
    let mut changes = Vec::new();

    // Phase 1: Build maps of deleted and added files by hash for rename detection
    // A rename is detected when exactly one deleted file and one added file share the same hash
    let mut deleted_by_hash: HashMap<String, Vec<&String>> = HashMap::new();
    let mut added_by_hash: HashMap<String, Vec<(&String, &FilesystemFile)>> = HashMap::new();

    // Collect deleted files (in DB but not on filesystem)
    for (path, db_file) in indexed {
      if !filesystem.contains_key(path) {
        deleted_by_hash.entry(db_file.file_hash.clone()).or_default().push(path);
      }
    }

    // Collect added files (on filesystem but not in DB)
    if self.config.mode != ScanMode::DeletedOnly {
      for (path, fs_file) in filesystem {
        if !indexed.contains_key(path) {
          added_by_hash
            .entry(fs_file.current_hash.clone())
            .or_default()
            .push((path, fs_file));
        }
      }
    }

    // Phase 2: Match 1:1 pairs as renames (same hash, different path)
    let mut renamed_old_paths = std::collections::HashSet::new();
    let mut renamed_new_paths = std::collections::HashSet::new();

    for (hash, deleted_paths) in &deleted_by_hash {
      if let Some(added_files) = added_by_hash.get(hash) {
        // Only match as rename if exactly one deleted and one added file share this hash
        // Multiple files with same hash are ambiguous - treat as delete + add
        if deleted_paths.len() == 1 && added_files.len() == 1 {
          let old_path = deleted_paths[0];
          let (new_path, _) = added_files[0];

          changes.push(FileChange::Renamed {
            old_path: old_path.clone(),
            new_path: new_path.clone(),
            hash: hash.clone(),
          });

          renamed_old_paths.insert(old_path.clone());
          renamed_new_paths.insert(new_path.clone());
        }
      }
    }

    // Phase 3: Emit remaining deletes (excluding renames)
    for (path, db_file) in indexed {
      if !filesystem.contains_key(path) && !renamed_old_paths.contains(path) {
        changes.push(FileChange::Deleted { path: path.clone() });
        debug!("File deleted: {} (hash={})", path, db_file.file_hash);
      }
    }

    // Phase 4: Find new and modified files (excluding renames)
    if self.config.mode != ScanMode::DeletedOnly {
      for (path, fs_file) in filesystem {
        if renamed_new_paths.contains(path) {
          continue; // Already handled as rename
        }

        match indexed.get(path) {
          None => {
            // New file
            changes.push(FileChange::Added {
              path: path.clone(),
              hash: fs_file.current_hash.clone(),
            });
          }
          Some(db_file) => {
            if self.config.mode == ScanMode::Full {
              // Check for modifications using hybrid strategy
              if self.is_file_modified(db_file, fs_file) {
                changes.push(FileChange::Modified {
                  path: path.clone(),
                  old_hash: db_file.file_hash.clone(),
                  new_hash: fs_file.current_hash.clone(),
                });
              } else {
                changes.push(FileChange::Unchanged { path: path.clone() });
              }
            } else {
              changes.push(FileChange::Unchanged { path: path.clone() });
            }
          }
        }
      }
    }

    changes
  }

  /// Check if a file has been modified using hybrid mtime + hash strategy
  ///
  /// Uses the following logic:
  /// 1. If hash matches: definitely unchanged
  /// 2. If mtime is old (< indexed_at - 1s buffer): probably unchanged despite hash difference
  ///    (hash difference could be due to different hash algorithm between runs)
  /// 3. If mtime is recent: compare hashes to detect actual changes
  fn is_file_modified(&self, indexed: &IndexedFile, filesystem: &FilesystemFile) -> bool {
    // Quick check: if hash matches, definitely unchanged
    if indexed.file_hash == filesystem.current_hash {
      return false;
    }

    // Hybrid mode: check mtime before trusting hash difference
    // mtime is in milliseconds since UNIX_EPOCH (from scan), indexed_at_ms is also in ms
    let mtime_ms = (filesystem.mtime as i64) * 1000;
    let indexed_at_with_buffer = indexed.indexed_at_ms - 1000; // 1 second buffer for clock skew

    // If file's mtime is clearly before we indexed it, the hash difference
    // is likely due to different hash computation, not actual content change.
    // However, this is a heuristic - some filesystems preserve mtime across copies.
    // For safety, we still report as modified if hash differs but log at debug level.
    if mtime_ms < indexed_at_with_buffer {
      // File wasn't touched since indexing, but hash differs.
      // This could happen if:
      // 1. Hash algorithm changed between runs
      // 2. File was restored from backup with preserved mtime
      // 3. Clock skew issues
      //
      // In Full mode, we trust the hash comparison and report as modified.
      // Future optimization: add a "trust_mtime" mode that skips re-indexing here.
      tracing::debug!(
        "File {} has old mtime but different hash (mtime={}ms, indexed_at={}ms)",
        filesystem.file_path,
        mtime_ms,
        indexed.indexed_at_ms
      );
    }

    // Hash differs - file was modified
    true
  }

  /// Apply scan results to the database
  pub async fn apply(
    &self,
    result: &ScanResult,
    db: &ProjectDb,
    root: &Path,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    config: &Config,
  ) -> Result<ApplyResult, ScanError> {
    let start = Instant::now();
    self.state.start();

    let total_changes = result.total_changes();
    self.state.total_files.store(total_changes, Ordering::Relaxed);

    let mut apply_result = ApplyResult::default();

    // Step 1: Process renames (update paths, preserve embeddings)
    // Do this first before deletes to avoid accidentally deleting renamed files
    if !result.renamed.is_empty() {
      self.state.set_phase("Processing renamed files").await;
      info!("Processing {} renamed files", result.renamed.len());

      for (old_path, new_path) in &result.renamed {
        if self.is_cancelled() {
          self.state.finish();
          return Err(ScanError::Cancelled);
        }

        match db.rename_file(old_path, new_path).await {
          Ok(count) => {
            if count > 0 {
              debug!("Renamed {} -> {} ({} chunks)", old_path, new_path, count);
              apply_result.files_renamed += 1;
            }
          }
          Err(e) => {
            warn!("Failed to rename {} -> {}: {}", old_path, new_path, e);
            apply_result.errors.push(format!("Rename error: {}", e));
          }
        }
        self.state.processed_files.fetch_add(1, Ordering::Relaxed);
      }
    }

    // Step 2: Delete chunks for removed files
    if !result.deleted.is_empty() {
      self.state.set_phase("Deleting stale chunks").await;
      info!("Deleting chunks for {} removed files", result.deleted.len());

      let paths: Vec<&str> = result.deleted.iter().map(|s| s.as_str()).collect();

      // Delete in batches to avoid very long SQL statements
      const BATCH_SIZE: usize = 100;
      for batch in paths.chunks(BATCH_SIZE) {
        if self.is_cancelled() {
          self.state.finish();
          return Err(ScanError::Cancelled);
        }

        if let Err(e) = db.delete_chunks_for_files(batch).await {
          warn!("Failed to delete chunks: {}", e);
          apply_result.errors.push(format!("Delete error: {}", e));
        } else {
          apply_result.files_deleted += batch.len();
          self.state.processed_files.fetch_add(batch.len(), Ordering::Relaxed);
        }
      }
    }

    // Step 3: Index new and modified files
    let files_to_index: Vec<&String> = result.added.iter().chain(result.modified.iter()).collect();

    if !files_to_index.is_empty() {
      self.state.set_phase("Indexing new/modified files").await;
      info!("Indexing {} new/modified files", files_to_index.len());

      // Delete chunks for modified files first
      if !result.modified.is_empty() {
        let paths: Vec<&str> = result.modified.iter().map(|s| s.as_str()).collect();
        for batch in paths.chunks(BATCH_SIZE) {
          if let Err(e) = db.delete_chunks_for_files(batch).await {
            warn!("Failed to delete chunks for modified files: {}", e);
            apply_result.errors.push(format!("Delete modified error: {}", e));
          }
        }
      }

      // Build FileChangeContext for each file
      let file_contexts: Vec<FileChangeContext> = files_to_index
        .iter()
        .map(|path| FileChangeContext {
          change_path: root.join(path),
          relative_path: (*path).clone(),
          is_doc_file: false, // Startup scan focuses on code files
          is_delete: false,
          old_content: None, // No cached content on startup
        })
        .collect();

      // Use the existing batch processing function
      let content_cache = Arc::new(crate::cache::FileContentCache::new());
      let project_id = db.project_id().as_str().to_string();

      let (indexed_code, indexed_docs) = process_file_changes_batched(
        file_contexts,
        Arc::new(db.clone_connection().await?),
        embedding,
        project_id,
        root.to_path_buf(),
        config.docs.clone(),
        content_cache,
        self.config.parallelism,
      )
      .await;

      apply_result.files_indexed = result.added.len().min(indexed_code + indexed_docs);
      apply_result.files_reindexed = result.modified.len().min(indexed_code + indexed_docs);
      self
        .state
        .processed_files
        .fetch_add(indexed_code + indexed_docs, Ordering::Relaxed);
    }

    apply_result.apply_duration = start.elapsed();
    self.state.finish();

    info!(
      "Reconciliation complete: {} deleted, {} indexed, {} re-indexed, {} renamed ({:.2}s)",
      apply_result.files_deleted,
      apply_result.files_indexed,
      apply_result.files_reindexed,
      apply_result.files_renamed,
      apply_result.apply_duration.as_secs_f64()
    );

    Ok(apply_result)
  }
}

/// Batch size for delete operations
const BATCH_SIZE: usize = 100;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_scan_mode_from_str() {
    assert_eq!("deleted_only".parse::<ScanMode>().unwrap(), ScanMode::DeletedOnly);
    assert_eq!("deletedonly".parse::<ScanMode>().unwrap(), ScanMode::DeletedOnly);
    assert_eq!("deleted_and_new".parse::<ScanMode>().unwrap(), ScanMode::DeletedAndNew);
    assert_eq!("full".parse::<ScanMode>().unwrap(), ScanMode::Full);
    assert!("invalid".parse::<ScanMode>().is_err());
  }

  #[test]
  fn test_scan_config_defaults() {
    let config = StartupScanConfig::default();
    assert!(config.enabled);
    assert_eq!(config.mode, ScanMode::Full);
    assert!(!config.blocking);
    assert_eq!(config.timeout, Duration::from_secs(300));
  }

  #[test]
  fn test_scan_state() {
    let state = ScanState::new();
    assert!(!state.is_in_progress());

    state.start();
    assert!(state.is_in_progress());

    state.total_files.store(100, Ordering::Relaxed);
    state.processed_files.store(50, Ordering::Relaxed);
    assert_eq!(state.progress(), (50, 100));

    state.finish();
    assert!(!state.is_in_progress());
  }

  #[test]
  fn test_classify_deleted_files() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let mut indexed = HashMap::new();
    indexed.insert(
      "a.rs".to_string(),
      IndexedFile {
        file_path: "a.rs".to_string(),
        file_hash: "abc123".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );
    indexed.insert(
      "b.rs".to_string(),
      IndexedFile {
        file_path: "b.rs".to_string(),
        file_hash: "def456".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    // Only a.rs exists on filesystem
    let mut filesystem = HashMap::new();
    filesystem.insert(
      "a.rs".to_string(),
      FilesystemFile {
        file_path: "a.rs".to_string(),
        current_hash: "abc123".to_string(),
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    let deleted: Vec<_> = changes
      .iter()
      .filter_map(|c| match c {
        FileChange::Deleted { path } => Some(path.clone()),
        _ => None,
      })
      .collect();

    assert_eq!(deleted, vec!["b.rs"]);
  }

  #[test]
  fn test_classify_new_files() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let mut indexed = HashMap::new();
    indexed.insert(
      "a.rs".to_string(),
      IndexedFile {
        file_path: "a.rs".to_string(),
        file_hash: "abc123".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    // Both a.rs and b.rs exist on filesystem
    let mut filesystem = HashMap::new();
    filesystem.insert(
      "a.rs".to_string(),
      FilesystemFile {
        file_path: "a.rs".to_string(),
        current_hash: "abc123".to_string(),
        mtime: 2000,
        size: 100,
      },
    );
    filesystem.insert(
      "b.rs".to_string(),
      FilesystemFile {
        file_path: "b.rs".to_string(),
        current_hash: "def456".to_string(),
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    let added: Vec<_> = changes
      .iter()
      .filter_map(|c| match c {
        FileChange::Added { path, .. } => Some(path.clone()),
        _ => None,
      })
      .collect();

    assert_eq!(added, vec!["b.rs"]);
  }

  #[test]
  fn test_classify_modified_files() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let mut indexed = HashMap::new();
    indexed.insert(
      "a.rs".to_string(),
      IndexedFile {
        file_path: "a.rs".to_string(),
        file_hash: "old_hash".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    let mut filesystem = HashMap::new();
    filesystem.insert(
      "a.rs".to_string(),
      FilesystemFile {
        file_path: "a.rs".to_string(),
        current_hash: "new_hash".to_string(), // Different hash
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    let modified: Vec<_> = changes
      .iter()
      .filter_map(|c| match c {
        FileChange::Modified { path, .. } => Some(path.clone()),
        _ => None,
      })
      .collect();

    assert_eq!(modified, vec!["a.rs"]);
  }

  #[test]
  fn test_deleted_only_mode() {
    let scanner = StartupScanner::new(StartupScanConfig {
      mode: ScanMode::DeletedOnly,
      ..Default::default()
    });

    let mut indexed = HashMap::new();
    indexed.insert(
      "deleted.rs".to_string(),
      IndexedFile {
        file_path: "deleted.rs".to_string(),
        file_hash: "abc".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    let mut filesystem = HashMap::new();
    filesystem.insert(
      "new.rs".to_string(),
      FilesystemFile {
        file_path: "new.rs".to_string(),
        current_hash: "def".to_string(),
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    // Should only detect deleted files in DeletedOnly mode
    let deleted_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Deleted { .. }))
      .count();
    let added_count = changes.iter().filter(|c| matches!(c, FileChange::Added { .. })).count();

    assert_eq!(deleted_count, 1);
    assert_eq!(added_count, 0); // New files not detected in DeletedOnly mode
  }

  #[test]
  fn test_scan_result_helpers() {
    let result = ScanResult {
      deleted: vec!["a.rs".to_string()],
      added: vec!["b.rs".to_string(), "c.rs".to_string()],
      modified: vec!["d.rs".to_string()],
      renamed: vec![],
      unchanged_count: 10,
      scan_duration: Duration::from_secs(1),
      errors: vec![],
    };

    assert_eq!(result.total_changes(), 4); // 1 deleted + 2 added + 1 modified + 0 renamed
    assert!(!result.is_empty());

    let empty_result = ScanResult::default();
    assert!(empty_result.is_empty());
    assert_eq!(empty_result.total_changes(), 0);
  }

  #[test]
  fn test_deleted_and_new_mode() {
    let scanner = StartupScanner::new(StartupScanConfig {
      mode: ScanMode::DeletedAndNew,
      ..Default::default()
    });

    let mut indexed = HashMap::new();
    indexed.insert(
      "existing.rs".to_string(),
      IndexedFile {
        file_path: "existing.rs".to_string(),
        file_hash: "old_hash".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );
    indexed.insert(
      "deleted.rs".to_string(),
      IndexedFile {
        file_path: "deleted.rs".to_string(),
        file_hash: "abc".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    let mut filesystem = HashMap::new();
    filesystem.insert(
      "existing.rs".to_string(),
      FilesystemFile {
        file_path: "existing.rs".to_string(),
        current_hash: "new_hash".to_string(), // Modified, but should be ignored
        mtime: 2000,
        size: 100,
      },
    );
    filesystem.insert(
      "new.rs".to_string(),
      FilesystemFile {
        file_path: "new.rs".to_string(),
        current_hash: "def".to_string(),
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    let deleted_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Deleted { .. }))
      .count();
    let added_count = changes.iter().filter(|c| matches!(c, FileChange::Added { .. })).count();
    let modified_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Modified { .. }))
      .count();

    assert_eq!(deleted_count, 1); // deleted.rs
    assert_eq!(added_count, 1); // new.rs
    assert_eq!(modified_count, 0); // existing.rs marked as unchanged, not modified
  }

  #[test]
  fn test_unchanged_files() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let mut indexed = HashMap::new();
    indexed.insert(
      "a.rs".to_string(),
      IndexedFile {
        file_path: "a.rs".to_string(),
        file_hash: "same_hash".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    let mut filesystem = HashMap::new();
    filesystem.insert(
      "a.rs".to_string(),
      FilesystemFile {
        file_path: "a.rs".to_string(),
        current_hash: "same_hash".to_string(), // Same hash
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    let unchanged_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Unchanged { .. }))
      .count();

    assert_eq!(unchanged_count, 1);
  }

  #[test]
  fn test_is_file_modified_same_hash() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let indexed = IndexedFile {
      file_path: "a.rs".to_string(),
      file_hash: "abc123".to_string(),
      indexed_at_ms: 1000,
      chunk_count: 1,
    };

    let filesystem = FilesystemFile {
      file_path: "a.rs".to_string(),
      current_hash: "abc123".to_string(), // Same hash
      mtime: 2000,
      size: 100,
    };

    assert!(!scanner.is_file_modified(&indexed, &filesystem));
  }

  #[test]
  fn test_is_file_modified_different_hash() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let indexed = IndexedFile {
      file_path: "a.rs".to_string(),
      file_hash: "old_hash".to_string(),
      indexed_at_ms: 1000000, // 1000 seconds since epoch in ms
      chunk_count: 1,
    };

    let filesystem = FilesystemFile {
      file_path: "a.rs".to_string(),
      current_hash: "new_hash".to_string(), // Different hash
      mtime: 2000,                          // 2000 seconds since epoch (more recent than indexed_at)
      size: 100,
    };

    assert!(scanner.is_file_modified(&indexed, &filesystem));
  }

  #[test]
  fn test_apply_result_default() {
    let result = ApplyResult::default();
    assert_eq!(result.files_deleted, 0);
    assert_eq!(result.files_indexed, 0);
    assert_eq!(result.files_reindexed, 0);
    assert_eq!(result.files_renamed, 0);
    assert!(result.errors.is_empty());
  }

  #[test]
  fn test_classify_renames() {
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let mut indexed = HashMap::new();
    indexed.insert(
      "old/path/file.rs".to_string(),
      IndexedFile {
        file_path: "old/path/file.rs".to_string(),
        file_hash: "same_hash".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );
    indexed.insert(
      "other.rs".to_string(),
      IndexedFile {
        file_path: "other.rs".to_string(),
        file_hash: "other_hash".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    // old/path/file.rs renamed to new/path/file.rs (same hash, different path)
    let mut filesystem = HashMap::new();
    filesystem.insert(
      "new/path/file.rs".to_string(),
      FilesystemFile {
        file_path: "new/path/file.rs".to_string(),
        current_hash: "same_hash".to_string(),
        mtime: 2000,
        size: 100,
      },
    );
    filesystem.insert(
      "other.rs".to_string(),
      FilesystemFile {
        file_path: "other.rs".to_string(),
        current_hash: "other_hash".to_string(),
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    let renamed: Vec<_> = changes
      .iter()
      .filter_map(|c| match c {
        FileChange::Renamed { old_path, new_path, .. } => Some((old_path.clone(), new_path.clone())),
        _ => None,
      })
      .collect();

    assert_eq!(renamed.len(), 1);
    assert_eq!(renamed[0].0, "old/path/file.rs");
    assert_eq!(renamed[0].1, "new/path/file.rs");

    // other.rs should be unchanged
    let unchanged_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Unchanged { .. }))
      .count();
    assert_eq!(unchanged_count, 1);

    // No deletes or adds (rename was detected)
    let deleted_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Deleted { .. }))
      .count();
    let added_count = changes.iter().filter(|c| matches!(c, FileChange::Added { .. })).count();
    assert_eq!(deleted_count, 0);
    assert_eq!(added_count, 0);
  }

  #[test]
  fn test_classify_renames_ambiguous_hash() {
    // When multiple files have the same hash, don't treat as rename (ambiguous)
    let scanner = StartupScanner::new(StartupScanConfig::default());

    let mut indexed = HashMap::new();
    indexed.insert(
      "a.rs".to_string(),
      IndexedFile {
        file_path: "a.rs".to_string(),
        file_hash: "same_hash".to_string(),
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );
    indexed.insert(
      "b.rs".to_string(),
      IndexedFile {
        file_path: "b.rs".to_string(),
        file_hash: "same_hash".to_string(), // Same hash as a.rs
        indexed_at_ms: 1000,
        chunk_count: 1,
      },
    );

    // Both files deleted, one new file with same hash
    let mut filesystem = HashMap::new();
    filesystem.insert(
      "c.rs".to_string(),
      FilesystemFile {
        file_path: "c.rs".to_string(),
        current_hash: "same_hash".to_string(),
        mtime: 2000,
        size: 100,
      },
    );

    let changes = scanner.classify_changes(&indexed, &filesystem);

    // Should NOT detect rename (ambiguous - two deleted files with same hash)
    let renamed_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Renamed { .. }))
      .count();
    assert_eq!(renamed_count, 0);

    // Should have 2 deletes and 1 add
    let deleted_count = changes
      .iter()
      .filter(|c| matches!(c, FileChange::Deleted { .. }))
      .count();
    let added_count = changes.iter().filter(|c| matches!(c, FileChange::Added { .. })).count();
    assert_eq!(deleted_count, 2);
    assert_eq!(added_count, 1);
  }

  #[test]
  fn test_scan_result_includes_renamed() {
    let result = ScanResult {
      deleted: vec!["a.rs".to_string()],
      added: vec!["b.rs".to_string()],
      modified: vec!["c.rs".to_string()],
      renamed: vec![("old.rs".to_string(), "new.rs".to_string())],
      unchanged_count: 5,
      scan_duration: Duration::from_secs(1),
      errors: vec![],
    };

    assert_eq!(result.total_changes(), 4); // 1 deleted + 1 added + 1 modified + 1 renamed
    assert!(!result.is_empty());
  }
}
