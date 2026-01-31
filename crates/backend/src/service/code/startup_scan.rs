//! Startup scan service for detecting file changes while daemon was down.
//!
//! This service compares the current filesystem state against the `indexed_files` table
//! to detect:
//! - Added files: exist on disk, not in DB
//! - Deleted files: in DB, not on disk
//! - Modified files: mtime changed → verify with content hash
//! - Moved files: same content hash, different path
//!
//! ## Usage
//!
//! Called when a ProjectActor starts watching a previously indexed project.
//! If the project was never manually indexed, the scan is skipped.

use std::{collections::HashMap, path::PathBuf};

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use sha2::{Digest, Sha256};
use tracing::{debug, info, trace, warn};

use crate::{
  context::files::is_document_extension,
  db::{IndexedFile, ProjectDb},
  domain::code::Language,
};

/// Result of a startup scan
#[derive(Debug, Default)]
pub struct StartupScanResult {
  /// Files that were added while daemon was down
  pub added: Vec<PathBuf>,
  /// Files that were modified (content changed)
  pub modified: Vec<PathBuf>,
  /// Files that were deleted
  pub deleted: Vec<String>,
  /// Files that were moved (old_path, new_path)
  pub moved: Vec<(String, PathBuf)>,
  /// Whether the project was previously indexed
  pub was_indexed: bool,
}

impl StartupScanResult {
  /// Total number of changes detected
  pub fn change_count(&self) -> usize {
    self.added.len() + self.modified.len() + self.deleted.len() + self.moved.len()
  }

  /// Returns true if any changes were detected
  pub fn has_changes(&self) -> bool {
    self.change_count() > 0
  }

  /// Get all files that need reindexing (added + modified + moved destinations)
  pub fn files_to_index(&self) -> Vec<PathBuf> {
    let mut files = Vec::new();
    files.extend(self.added.iter().cloned());
    files.extend(self.modified.iter().cloned());
    files.extend(self.moved.iter().map(|(_, new_path)| new_path.clone()));
    files
  }
}

/// Perform a startup scan to detect file changes while the daemon was down.
///
/// Returns `None` if the project was never indexed (no startup scan needed).
/// Returns `Some(result)` with the detected changes if the project was indexed.
pub async fn startup_scan(db: &ProjectDb, project_root: &PathBuf) -> Option<StartupScanResult> {
  let project_id = db.project_id.as_str();

  // Check if project was previously indexed
  let was_indexed = match db.is_manually_indexed(project_id).await {
    Ok(indexed) => indexed,
    Err(e) => {
      warn!(error = %e, "Failed to check if project was indexed");
      return None;
    }
  };

  if !was_indexed {
    debug!(project_id = %project_id, "Project was never indexed, skipping startup scan");
    return Some(StartupScanResult {
      was_indexed: false,
      ..Default::default()
    });
  }

  info!(project_id = %project_id, "Performing startup scan for previously indexed project");

  // Load indexed files from DB
  let indexed_files = match db.list_indexed_files(project_id).await {
    Ok(files) => files,
    Err(e) => {
      warn!(error = %e, "Failed to load indexed files");
      return None;
    }
  };

  // Build lookup maps
  let mut db_files: HashMap<String, IndexedFile> =
    indexed_files.into_iter().map(|f| (f.file_path.clone(), f)).collect();

  // Build hash -> path map for move detection
  let hash_to_path: HashMap<String, String> = db_files
    .iter()
    .map(|(path, f)| (f.content_hash.clone(), path.clone()))
    .collect();

  // Build gitignore matcher
  let gitignore = build_gitignore(project_root);

  // Scan current files on disk
  let mut result = StartupScanResult {
    was_indexed: true,
    ..Default::default()
  };

  let current_files = scan_source_files(project_root, gitignore.as_ref());

  for full_path in current_files {
    let relative = match full_path.strip_prefix(project_root) {
      Ok(rel) => rel.to_string_lossy().to_string(),
      Err(_) => continue,
    };

    if let Some(db_file) = db_files.remove(&relative) {
      // File exists in both DB and disk - check if modified
      let current_mtime = get_mtime(&full_path).await;

      if current_mtime != db_file.mtime {
        // mtime changed - check content hash
        let current_hash = compute_file_hash(&full_path).await;

        if current_hash != db_file.content_hash {
          trace!(path = %relative, "File modified (hash changed)");
          result.modified.push(full_path);
        } else {
          trace!(path = %relative, "File touched but content unchanged");
          // Just mtime changed, content same - no reindex needed, but update DB
        }
      }
    } else {
      // File on disk but not in DB
      // Check if it might be a move (same content hash exists elsewhere)
      let current_hash = compute_file_hash(&full_path).await;

      if let Some(old_path) = hash_to_path.get(&current_hash) {
        // This might be a move - check if old path is now missing
        if !project_root.join(old_path).exists() {
          trace!(from = %old_path, to = %relative, "File moved");
          result.moved.push((old_path.clone(), full_path));
          db_files.remove(old_path); // Don't double-count as deleted
        } else {
          // Both paths exist - this is a copy or new file
          trace!(path = %relative, "New file (possibly copied)");
          result.added.push(full_path);
        }
      } else {
        trace!(path = %relative, "New file added");
        result.added.push(full_path);
      }
    }
  }

  // Any remaining files in db_files are deleted
  for (path, _) in db_files {
    trace!(path = %path, "File deleted");
    result.deleted.push(path);
  }

  info!(
    project_id = %project_id,
    added = result.added.len(),
    modified = result.modified.len(),
    deleted = result.deleted.len(),
    moved = result.moved.len(),
    "Startup scan complete"
  );

  Some(result)
}

/// Get file mtime as Unix timestamp (seconds)
async fn get_mtime(path: &PathBuf) -> i64 {
  tokio::fs::metadata(path)
    .await
    .ok()
    .and_then(|m| m.modified().ok())
    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
    .map(|d| d.as_secs() as i64)
    .unwrap_or(0)
}

/// Compute SHA-256 hash of file content (truncated to 16 hex chars)
async fn compute_file_hash(path: &PathBuf) -> String {
  match tokio::fs::read(path).await {
    Ok(content) => {
      let result = Sha256::digest(&content);
      format!("{:016x}", u64::from_be_bytes(result[0..8].try_into().unwrap()))
    }
    Err(_) => "unknown".to_string(),
  }
}

/// Scan for source files in a directory, respecting gitignore
fn scan_source_files(root: &PathBuf, gitignore: Option<&Gitignore>) -> Vec<PathBuf> {
  let mut files = Vec::new();

  // Use walkdir for recursive traversal
  let walker = walkdir::WalkDir::new(root)
    .follow_links(false)
    .into_iter()
    .filter_entry(|e| {
      // Skip hidden directories
      if e.file_name().to_string_lossy().starts_with('.') && e.depth() > 0 {
        return false;
      }
      // Skip common ignore patterns
      let name = e.file_name().to_string_lossy();
      if matches!(
        name.as_ref(),
        "node_modules" | "target" | "__pycache__" | ".venv" | "venv" | "dist" | "build"
      ) {
        return false;
      }
      true
    });

  for entry in walker.filter_map(|e| e.ok()) {
    let path = entry.path();

    // Skip directories
    if !entry.file_type().is_file() {
      continue;
    }

    // Check gitignore
    if let Some(gi) = gitignore
      && gi.matched(path, false).is_ignore()
    {
      continue;
    }

    // Check if this is a supported file type (code or document)
    if path
      .extension()
      .and_then(|ext| ext.to_str())
      .is_some_and(|ext| Language::from_extension(ext).is_some() || is_document_extension(ext))
    {
      files.push(path.to_path_buf());
    }
  }

  files
}

/// Build a gitignore matcher for the given root directory
fn build_gitignore(root: &PathBuf) -> Option<Gitignore> {
  let gitignore_path = root.join(".gitignore");

  if !gitignore_path.exists() {
    return None;
  }

  let mut builder = GitignoreBuilder::new(root);

  // Add .gitignore rules
  if let Some(err) = builder.add(&gitignore_path) {
    warn!(error = %err, "Error parsing .gitignore");
  }

  // Add .ccengramignore if present
  let ccengramignore_path = root.join(".ccengramignore");
  if ccengramignore_path.exists()
    && let Some(err) = builder.add(&ccengramignore_path)
  {
    warn!(error = %err, "Error parsing .ccengramignore");
  }

  // Add common patterns
  let _ = builder.add_line(None, ".git/");
  let _ = builder.add_line(None, "node_modules/");
  let _ = builder.add_line(None, "target/");
  let _ = builder.add_line(None, "__pycache__/");
  let _ = builder.add_line(None, ".venv/");

  builder.build().ok()
}
