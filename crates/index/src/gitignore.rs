//! Gitignore pattern matching with caching for efficient file filtering.
//!
//! This module provides:
//! - A thread-safe cache for compiled gitignore patterns per project
//! - Utilities for tracking gitignore file changes
//! - A deprecated fallback function for simple pattern matching

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, RwLock};
use std::time::SystemTime;

/// Global patterns that apply to all projects regardless of .gitignore
const GLOBAL_PATTERNS: &[&str] = &[
  // Version control (directory and all contents)
  ".git/",
  ".hg/",
  ".svn/",
  // Dependencies (directory and all contents)
  "node_modules/",
  "vendor/",
  ".venv/",
  "venv/",
  ".env/",
  "env/",
  // Build outputs (directory and all contents)
  "target/",
  "dist/",
  "build/",
  ".next/",
  ".nuxt/",
  // Caches (directory and all contents)
  ".cache/",
  "__pycache__/",
  ".pytest_cache/",
  ".mypy_cache/",
  ".ruff_cache/",
  ".tox/",
  // Coverage (directory and all contents)
  "coverage/",
  ".coverage/",
  ".nyc_output/",
  // Minified files
  "*.min.js",
  "*.min.css",
  "*.map",
  // Lock files
  "package-lock.json",
  "yarn.lock",
  "pnpm-lock.yaml",
  "Cargo.lock",
  "poetry.lock",
  "Pipfile.lock",
  "composer.lock",
  "Gemfile.lock",
];

/// Cached compiled gitignore matcher with mtime for invalidation
struct CompiledIgnore {
  matcher: Gitignore,
  gitignore_mtime: Option<SystemTime>,
}

/// Thread-safe cache for compiled gitignore patterns per project.
///
/// This cache compiles gitignore patterns once per project and reuses them,
/// avoiding the overhead of pattern parsing and string allocations on every
/// file check. The cache automatically invalidates when .gitignore files change.
pub struct GitignoreCache {
  cache: RwLock<HashMap<PathBuf, CompiledIgnore>>,
}

impl GitignoreCache {
  /// Create a new empty cache.
  pub fn new() -> Self {
    Self {
      cache: RwLock::new(HashMap::new()),
    }
  }

  /// Check if a path should be ignored for the given project root.
  ///
  /// This is the main entry point - it handles cache lookup, invalidation,
  /// and matcher creation transparently.
  ///
  /// # Arguments
  ///
  /// * `project_root` - The root directory of the project
  /// * `path` - The absolute path to check
  ///
  /// # Returns
  ///
  /// `true` if the path should be ignored, `false` otherwise
  pub fn should_ignore(&self, project_root: &Path, path: &Path) -> bool {
    // Fast path: check cache with read lock
    {
      let cache = self.cache.read().unwrap();
      if let Some(compiled) = cache.get(project_root)
        && self.is_cache_valid(project_root, compiled)
      {
        return self.check_match(&compiled.matcher, project_root, path);
      }
    }

    // Slow path: build and cache matcher
    let matcher = self.build_and_cache_matcher(project_root);
    self.check_match(&matcher, project_root, path)
  }

  /// Check if the path matches the gitignore rules.
  fn check_match(&self, matcher: &Gitignore, project_root: &Path, path: &Path) -> bool {
    // Use relative path for matching
    let relative_path = path.strip_prefix(project_root).unwrap_or(path);

    // Use matched_path_or_any_parents to check if ANY parent directory is ignored.
    // This is crucial for matching files inside ignored directories (e.g., node_modules/foo.js)
    // The is_dir parameter: we assume directories if the path doesn't exist (deleted files)
    // or if it actually is a directory.
    let is_dir = path.is_dir();
    matcher.matched_path_or_any_parents(relative_path, is_dir).is_ignore()
  }

  /// Build matcher and cache it under write lock.
  fn build_and_cache_matcher(&self, project_root: &Path) -> Gitignore {
    let matcher = self.build_matcher(project_root);
    let mtime = self.get_gitignore_mtime(project_root);

    let mut cache = self.cache.write().unwrap();
    cache.insert(
      project_root.to_path_buf(),
      CompiledIgnore {
        matcher: matcher.clone(),
        gitignore_mtime: mtime,
      },
    );

    matcher
  }

  /// Build a new gitignore matcher for the given project root.
  fn build_matcher(&self, project_root: &Path) -> Gitignore {
    let mut builder = GitignoreBuilder::new(project_root);

    // Add global patterns
    for pattern in GLOBAL_PATTERNS {
      let _ = builder.add_line(None, pattern);
    }

    // Add project .gitignore if exists
    let gitignore_path = project_root.join(".gitignore");
    if gitignore_path.exists() {
      let _ = builder.add(&gitignore_path);
    }

    // Add .git/info/exclude if exists
    let exclude_path = project_root.join(".git/info/exclude");
    if exclude_path.exists() {
      let _ = builder.add(&exclude_path);
    }

    // Add .ccengramignore if exists
    let ccengram_ignore = project_root.join(".ccengramignore");
    if ccengram_ignore.exists() {
      let _ = builder.add(&ccengram_ignore);
    }

    builder.build().unwrap_or_else(|_| {
      // Fallback to matcher with just global patterns on error
      let mut fallback = GitignoreBuilder::new(project_root);
      for pattern in GLOBAL_PATTERNS {
        let _ = fallback.add_line(None, pattern);
      }
      fallback.build().unwrap()
    })
  }

  /// Check if the cached matcher is still valid.
  fn is_cache_valid(&self, project_root: &Path, compiled: &CompiledIgnore) -> bool {
    let current_mtime = self.get_gitignore_mtime(project_root);
    compiled.gitignore_mtime == current_mtime
  }

  /// Get the modification time of the .gitignore file.
  fn get_gitignore_mtime(&self, project_root: &Path) -> Option<SystemTime> {
    fs::metadata(project_root.join(".gitignore"))
      .ok()
      .and_then(|m| m.modified().ok())
  }

  /// Invalidate the cache for a specific project.
  ///
  /// Call this when you know the .gitignore has changed.
  pub fn invalidate(&self, project_root: &Path) {
    let mut cache = self.cache.write().unwrap();
    cache.remove(project_root);
  }

  /// Clear all cached matchers.
  pub fn clear(&self) {
    let mut cache = self.cache.write().unwrap();
    cache.clear();
  }
}

impl Default for GitignoreCache {
  fn default() -> Self {
    Self::new()
  }
}

/// Global gitignore cache instance.
///
/// Use this for efficient gitignore checking across the daemon.
pub static GITIGNORE_CACHE: LazyLock<GitignoreCache> = LazyLock::new(GitignoreCache::new);

/// Compute a hash of all gitignore patterns in a project directory.
/// This allows detecting when gitignore rules have changed, triggering a re-index.
pub fn compute_gitignore_hash(project_path: &Path) -> String {
  let mut hasher = Sha256::new();

  // Collect content from common ignore file locations
  let ignore_files = [".gitignore", ".git/info/exclude", ".ccengramignore"];

  for filename in ignore_files {
    let file_path = project_path.join(filename);
    if let Ok(content) = fs::read_to_string(&file_path) {
      hasher.update(filename.as_bytes());
      hasher.update(b":");
      hasher.update(content.as_bytes());
      hasher.update(b"\n");
    }
  }

  // Also check for nested .gitignore files (one level deep for performance)
  if let Ok(entries) = fs::read_dir(project_path) {
    for entry in entries.flatten() {
      if entry.file_type().is_ok_and(|t| t.is_dir()) {
        let nested_gitignore = entry.path().join(".gitignore");
        if let Ok(content) = fs::read_to_string(&nested_gitignore) {
          let relative_path = entry.file_name();
          hasher.update(relative_path.to_string_lossy().as_bytes());
          hasher.update(b"/.gitignore:");
          hasher.update(content.as_bytes());
          hasher.update(b"\n");
        }
      }
    }
  }

  format!("{:x}", hasher.finalize())
}

/// State for tracking gitignore changes
#[derive(Debug, Clone)]
pub struct GitignoreState {
  pub hash: String,
  pub computed_at: chrono::DateTime<chrono::Utc>,
}

impl GitignoreState {
  pub fn new(project_path: &Path) -> Self {
    Self {
      hash: compute_gitignore_hash(project_path),
      computed_at: chrono::Utc::now(),
    }
  }

  /// Load gitignore state from a project path (alias for new)
  pub fn load(project_path: &Path) -> Result<Self, std::io::Error> {
    Ok(Self::new(project_path))
  }

  /// Check if gitignore has changed since last scan
  pub fn has_changed(&self, project_path: &Path) -> bool {
    let current_hash = compute_gitignore_hash(project_path);
    current_hash != self.hash
  }
}

/// Check if a path should be ignored based on common patterns.
///
/// **Deprecated**: Use [`GITIGNORE_CACHE.should_ignore(project_root, path)`](GITIGNORE_CACHE)
/// instead for better performance. This function recreates patterns on every call
/// and doesn't support project-specific .gitignore files.
#[deprecated(
  since = "0.1.0",
  note = "Use GITIGNORE_CACHE.should_ignore(project_root, path) for cached, per-project pattern matching"
)]
pub fn should_ignore(path: &Path) -> bool {
  let path_str = path.to_string_lossy();

  for pattern in GLOBAL_PATTERNS {
    if let Some(dir_name) = pattern.strip_suffix('/') {
      // Directory pattern - check if path contains this directory
      if path_str.contains(&format!("/{dir_name}/")) || path_str.starts_with(&format!("{dir_name}/")) {
        return true;
      }
    } else if let Some(suffix) = pattern.strip_prefix('*') {
      // Glob pattern
      if path_str.ends_with(suffix) {
        return true;
      }
    } else {
      // Exact match
      if path.file_name().is_some_and(|n| n.to_string_lossy() == *pattern) {
        return true;
      }
    }
  }

  false
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::thread;
  use tempfile::TempDir;

  // --- GitignoreCache tests ---

  #[test]
  fn test_cache_should_ignore_global_patterns() {
    let dir = TempDir::new().unwrap();
    let cache = GitignoreCache::new();

    // Test global patterns - files inside ignored directories
    assert!(cache.should_ignore(dir.path(), &dir.path().join("node_modules/foo.js")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join(".git/config")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("target/debug/main")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("dist/bundle.js")));

    // Source files should not be ignored
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("src/main.rs")));
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("lib/index.ts")));
  }

  #[test]
  fn test_cache_should_ignore_lockfiles() {
    let dir = TempDir::new().unwrap();
    let cache = GitignoreCache::new();

    assert!(cache.should_ignore(dir.path(), &dir.path().join("package-lock.json")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("yarn.lock")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("Cargo.lock")));
  }

  #[test]
  fn test_cache_should_ignore_minified() {
    let dir = TempDir::new().unwrap();
    let cache = GitignoreCache::new();

    assert!(cache.should_ignore(dir.path(), &dir.path().join("bundle.min.js")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("styles.min.css")));
  }

  #[test]
  fn test_cache_uses_project_gitignore() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join(".gitignore"), "custom_ignored/\n*.custom").unwrap();

    let cache = GitignoreCache::new();

    // Custom patterns from .gitignore
    assert!(cache.should_ignore(dir.path(), &dir.path().join("custom_ignored/file.rs")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("test.custom")));

    // Regular files still allowed
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("src/main.rs")));
  }

  #[test]
  fn test_cache_invalidation_on_gitignore_change() {
    let dir = TempDir::new().unwrap();
    let cache = GitignoreCache::new();

    // Initially no .gitignore
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("custom/file.rs")));

    // Add .gitignore
    fs::write(dir.path().join(".gitignore"), "custom/").unwrap();

    // Wait a moment for mtime to differ (some filesystems have low resolution)
    thread::sleep(std::time::Duration::from_millis(10));

    // Cache should detect the change and rebuild
    assert!(cache.should_ignore(dir.path(), &dir.path().join("custom/file.rs")));
  }

  #[test]
  fn test_cache_different_projects_independent() {
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();

    fs::write(dir1.path().join(".gitignore"), "ignored_in_1/").unwrap();
    fs::write(dir2.path().join(".gitignore"), "ignored_in_2/").unwrap();

    let cache = GitignoreCache::new();

    // Each project has its own patterns
    assert!(cache.should_ignore(dir1.path(), &dir1.path().join("ignored_in_1/file.rs")));
    assert!(!cache.should_ignore(dir1.path(), &dir1.path().join("ignored_in_2/file.rs")));

    assert!(!cache.should_ignore(dir2.path(), &dir2.path().join("ignored_in_1/file.rs")));
    assert!(cache.should_ignore(dir2.path(), &dir2.path().join("ignored_in_2/file.rs")));
  }

  #[test]
  fn test_cache_reused_on_repeated_calls() {
    let dir = TempDir::new().unwrap();
    let cache = GitignoreCache::new();

    // First call builds cache
    let _ = cache.should_ignore(dir.path(), &dir.path().join("file.rs"));

    // Subsequent calls should use cache (verified by checking cache contains entry)
    for _ in 0..1000 {
      let _ = cache.should_ignore(dir.path(), &dir.path().join("file.rs"));
    }

    // Verify cache contains the entry
    let cache_lock = cache.cache.read().unwrap();
    assert!(cache_lock.contains_key(dir.path()));
  }

  #[test]
  fn test_cache_explicit_invalidate() {
    let dir = TempDir::new().unwrap();
    let cache = GitignoreCache::new();

    // Build cache
    let _ = cache.should_ignore(dir.path(), &dir.path().join("file.rs"));

    // Verify cached
    {
      let cache_lock = cache.cache.read().unwrap();
      assert!(cache_lock.contains_key(dir.path()));
    }

    // Invalidate
    cache.invalidate(dir.path());

    // Verify removed
    {
      let cache_lock = cache.cache.read().unwrap();
      assert!(!cache_lock.contains_key(dir.path()));
    }
  }

  #[test]
  fn test_global_cache_instance() {
    let dir = TempDir::new().unwrap();

    // Use the global instance
    assert!(!GITIGNORE_CACHE.should_ignore(dir.path(), &dir.path().join("src/main.rs")));
    assert!(GITIGNORE_CACHE.should_ignore(dir.path(), &dir.path().join("node_modules/pkg/index.js")));
  }

  // --- Legacy should_ignore tests (deprecated but still need to work) ---

  #[test]
  #[allow(deprecated)]
  fn test_should_ignore_node_modules() {
    assert!(should_ignore(Path::new("project/node_modules/foo.js")));
    assert!(should_ignore(Path::new("node_modules/package/index.js")));
  }

  #[test]
  #[allow(deprecated)]
  fn test_should_ignore_git() {
    assert!(should_ignore(Path::new(".git/config")));
    assert!(should_ignore(Path::new("project/.git/objects/abc")));
  }

  #[test]
  #[allow(deprecated)]
  fn test_should_ignore_lockfiles() {
    assert!(should_ignore(Path::new("package-lock.json")));
    assert!(should_ignore(Path::new("yarn.lock")));
    assert!(should_ignore(Path::new("Cargo.lock")));
  }

  #[test]
  #[allow(deprecated)]
  fn test_should_not_ignore_source() {
    assert!(!should_ignore(Path::new("src/main.rs")));
    assert!(!should_ignore(Path::new("lib/index.ts")));
    assert!(!should_ignore(Path::new("app.py")));
  }

  #[test]
  #[allow(deprecated)]
  fn test_should_ignore_minified() {
    assert!(should_ignore(Path::new("dist/bundle.min.js")));
    assert!(should_ignore(Path::new("styles.min.css")));
  }

  // --- Gitignore compliance tests ---
  // These tests verify that .gitignore file parsing follows standard gitignore behavior

  #[test]
  fn test_gitignore_comments_ignored() {
    let dir = TempDir::new().unwrap();
    fs::write(
      dir.path().join(".gitignore"),
      "# This is a comment\nsecret/\n# Another comment",
    )
    .unwrap();

    let cache = GitignoreCache::new();

    // The comment text itself should not be ignored
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("This is a comment")));
    // But the actual pattern should work
    assert!(cache.should_ignore(dir.path(), &dir.path().join("secret/file.txt")));
  }

  #[test]
  fn test_gitignore_glob_patterns() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join(".gitignore"), "*.log\n*.tmp\n!important.log").unwrap();

    let cache = GitignoreCache::new();

    // Should match glob patterns
    assert!(cache.should_ignore(dir.path(), &dir.path().join("debug.log")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("deep/nested/error.log")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("test.tmp")));

    // Negation pattern should un-ignore
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("important.log")));

    // Non-matching files should not be ignored
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("main.rs")));
  }

  #[test]
  fn test_gitignore_anchored_patterns() {
    let dir = TempDir::new().unwrap();
    // /build only matches at root, build matches anywhere
    fs::write(dir.path().join(".gitignore"), "/root_only\nanywhere").unwrap();

    let cache = GitignoreCache::new();

    // Anchored pattern - only matches at root
    assert!(cache.should_ignore(dir.path(), &dir.path().join("root_only/file.txt")));
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("subdir/root_only/file.txt")));

    // Unanchored pattern - matches anywhere
    assert!(cache.should_ignore(dir.path(), &dir.path().join("anywhere/file.txt")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("deep/nested/anywhere/file.txt")));
  }

  #[test]
  fn test_gitignore_directory_vs_file_patterns() {
    let dir = TempDir::new().unwrap();
    // logs/ matches only directories, logfile matches files named logfile
    fs::write(dir.path().join(".gitignore"), "logs/\nlogfile").unwrap();

    let cache = GitignoreCache::new();

    // Directory pattern matches directory contents
    assert!(cache.should_ignore(dir.path(), &dir.path().join("logs/app.log")));

    // File pattern matches files
    assert!(cache.should_ignore(dir.path(), &dir.path().join("logfile")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("subdir/logfile")));
  }

  #[test]
  fn test_gitignore_trailing_slash_optional_for_dirs() {
    let dir = TempDir::new().unwrap();
    // Both with and without trailing slash should work for directories
    fs::write(dir.path().join(".gitignore"), "with_slash/\nwithout_slash").unwrap();

    let cache = GitignoreCache::new();

    // Pattern with trailing slash - matches directory contents
    assert!(cache.should_ignore(dir.path(), &dir.path().join("with_slash/file.txt")));

    // Pattern without trailing slash - also matches directory contents
    assert!(cache.should_ignore(dir.path(), &dir.path().join("without_slash/file.txt")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("nested/without_slash/file.txt")));
  }

  #[test]
  fn test_gitignore_doublestar_patterns() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join(".gitignore"), "**/cache\nlogs/**").unwrap();

    let cache = GitignoreCache::new();

    // **/cache matches cache directory at any level
    assert!(cache.should_ignore(dir.path(), &dir.path().join("cache/file.txt")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("a/b/c/cache/file.txt")));

    // logs/** matches everything inside logs
    assert!(cache.should_ignore(dir.path(), &dir.path().join("logs/app.log")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("logs/deep/nested/file.txt")));
  }

  #[test]
  fn test_gitignore_complex_real_world() {
    let dir = TempDir::new().unwrap();
    // A realistic .gitignore similar to common projects
    fs::write(
      dir.path().join(".gitignore"),
      r#"# Dependencies
node_modules/
vendor/

# Build outputs
dist/
*.o
*.so

# IDE files
.idea/
*.swp

# Logs
*.log
!important.log

# Temp files
tmp/
*.tmp
"#,
    )
    .unwrap();

    let cache = GitignoreCache::new();

    // Dependencies
    assert!(cache.should_ignore(dir.path(), &dir.path().join("node_modules/lodash/index.js")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("vendor/github.com/pkg/foo.go")));

    // Build outputs
    assert!(cache.should_ignore(dir.path(), &dir.path().join("dist/bundle.js")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("src/main.o")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("lib/libfoo.so")));

    // IDE files
    assert!(cache.should_ignore(dir.path(), &dir.path().join(".idea/workspace.xml")));
    assert!(cache.should_ignore(dir.path(), &dir.path().join("main.rs.swp")));

    // Logs (with negation)
    assert!(cache.should_ignore(dir.path(), &dir.path().join("debug.log")));
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("important.log")));

    // Source files should NOT be ignored
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("src/main.rs")));
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("lib/utils.ts")));
    assert!(!cache.should_ignore(dir.path(), &dir.path().join("README.md")));
  }

  // --- GitignoreState tests ---

  #[test]
  fn test_compute_gitignore_hash_empty() {
    let temp_dir = TempDir::new().unwrap();
    let hash = compute_gitignore_hash(temp_dir.path());
    // Empty project still produces a hash (empty input)
    assert!(!hash.is_empty());
    assert_eq!(hash.len(), 64); // SHA256 hex is 64 chars
  }

  #[test]
  fn test_compute_gitignore_hash_with_gitignore() {
    let temp_dir = TempDir::new().unwrap();

    // Write a .gitignore
    fs::write(temp_dir.path().join(".gitignore"), "node_modules/\n*.log").unwrap();

    let hash1 = compute_gitignore_hash(temp_dir.path());
    assert!(!hash1.is_empty());

    // Same content = same hash
    let hash2 = compute_gitignore_hash(temp_dir.path());
    assert_eq!(hash1, hash2);

    // Different content = different hash
    fs::write(temp_dir.path().join(".gitignore"), "target/\n*.tmp").unwrap();
    let hash3 = compute_gitignore_hash(temp_dir.path());
    assert_ne!(hash1, hash3);
  }

  #[test]
  fn test_gitignore_state_has_changed() {
    let temp_dir = TempDir::new().unwrap();

    // Create initial state
    fs::write(temp_dir.path().join(".gitignore"), "node_modules/").unwrap();
    let state = GitignoreState::new(temp_dir.path());

    // No change yet
    assert!(!state.has_changed(temp_dir.path()));

    // Modify gitignore
    fs::write(temp_dir.path().join(".gitignore"), "target/").unwrap();

    // Now it should detect a change
    assert!(state.has_changed(temp_dir.path()));
  }
}
