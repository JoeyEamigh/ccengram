//! Caching utilities for the daemon
//!
//! This module provides caches to improve performance of repeated operations.

use moka::sync::Cache;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, trace};

/// Cache entry for file content (used for incremental parsing)
#[derive(Clone)]
pub struct CachedFileContent {
  pub content: Arc<String>,
  pub content_hash: u64,
}

/// LRU cache for file contents to enable incremental tree-sitter parsing.
///
/// When a file is modified, we can compare the new content with the cached
/// old content to compute a minimal diff, enabling tree-sitter's incremental
/// parsing which is much faster for small edits.
///
/// Key: (project_root, relative_file_path)
/// Value: CachedFileContent
pub struct FileContentCache {
  cache: Cache<(PathBuf, String), CachedFileContent>,
}

impl FileContentCache {
  /// Create a new file content cache with default settings.
  ///
  /// Default capacity: 1000 files
  /// Default TTI: 1 hour (files idle for 1 hour are evicted)
  pub fn new() -> Self {
    Self::with_capacity(1000)
  }

  /// Create a cache with custom capacity.
  pub fn with_capacity(capacity: u64) -> Self {
    Self {
      cache: Cache::builder()
        .max_capacity(capacity)
        .time_to_idle(Duration::from_secs(3600)) // 1 hour idle timeout
        .build(),
    }
  }

  /// Get cached content for a file.
  pub fn get(&self, project_root: &Path, file_path: &str) -> Option<CachedFileContent> {
    let key = (project_root.to_path_buf(), file_path.to_string());
    let result = self.cache.get(&key);
    trace!(
      file = %file_path,
      hit = result.is_some(),
      "File content cache lookup"
    );
    result
  }

  /// Store content for a file.
  pub fn insert(&self, project_root: &Path, file_path: &str, content: String) {
    let hash = Self::hash_content(&content);
    let content_len = content.len();
    let key = (project_root.to_path_buf(), file_path.to_string());
    self.cache.insert(
      key,
      CachedFileContent {
        content: Arc::new(content),
        content_hash: hash,
      },
    );
    trace!(file = %file_path, content_len = content_len, "File content cached");
  }

  /// Remove cached content for a file (e.g., on delete).
  pub fn remove(&self, project_root: &Path, file_path: &str) {
    let key = (project_root.to_path_buf(), file_path.to_string());
    self.cache.invalidate(&key);
    debug!(file = %file_path, "File content cache invalidated");
  }

  /// Remove all cached content for a project.
  pub fn remove_project(&self, project_root: &Path) {
    // Note: moka doesn't support prefix invalidation, so we iterate
    // This is O(n) but should be rare (only on project close)
    let root = project_root.to_path_buf();
    let count_before = self.cache.entry_count();
    let _ = self.cache.invalidate_entries_if(move |key, _| key.0 == root);
    let count_after = self.cache.entry_count();
    let invalidated = count_before.saturating_sub(count_after);
    debug!(
      project = %project_root.display(),
      entries_invalidated = invalidated,
      "Project cache invalidated"
    );
  }

  /// Get cache statistics.
  pub fn stats(&self) -> CacheStats {
    CacheStats {
      entry_count: self.cache.entry_count(),
      weighted_size: self.cache.weighted_size(),
    }
  }

  /// Simple hash for content (same as TreeSitterParser::hash_content)
  fn hash_content(content: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
  }
}

impl Default for FileContentCache {
  fn default() -> Self {
    Self::new()
  }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
  pub entry_count: u64,
  pub weighted_size: u64,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_file_content_cache_basic() {
    let cache = FileContentCache::new();
    let root = PathBuf::from("/project");
    let path = "src/main.rs";

    // Initially empty
    assert!(cache.get(&root, path).is_none());

    // Insert content
    let content = "fn main() {}".to_string();
    cache.insert(&root, path, content.clone());

    // Should be retrievable
    let cached = cache.get(&root, path).unwrap();
    assert_eq!(*cached.content, content);

    // Remove
    cache.remove(&root, path);
    assert!(cache.get(&root, path).is_none());
  }

  #[test]
  fn test_file_content_cache_different_projects() {
    let cache = FileContentCache::new();
    let root1 = PathBuf::from("/project1");
    let root2 = PathBuf::from("/project2");
    let path = "src/main.rs";

    cache.insert(&root1, path, "project1 content".to_string());
    cache.insert(&root2, path, "project2 content".to_string());

    // Different projects should have different content
    let cached1 = cache.get(&root1, path).unwrap();
    let cached2 = cache.get(&root2, path).unwrap();
    assert_eq!(*cached1.content, "project1 content");
    assert_eq!(*cached2.content, "project2 content");
  }
}
