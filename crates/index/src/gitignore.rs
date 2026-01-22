// Re-export from ignore crate - gitignore handling is done through the scanner
// which uses the `ignore` crate for proper .gitignore support.
//
// The Scanner in scanner.rs handles:
// - .gitignore files
// - .git/info/exclude
// - Global gitignore (~/.config/git/ignore)
// - Custom .ccengramignore files
//
// This module is kept for potential future extensions like:
// - Custom ignore pattern parsing
// - Programmatic ignore rules
// - Integration with project-specific settings

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

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
      if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
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

/// Check if a path should be ignored based on common patterns
/// This is a simple fallback for when the full ignore crate isn't needed
pub fn should_ignore(path: &Path) -> bool {
  let path_str = path.to_string_lossy();

  // Common patterns to always ignore
  let ignore_patterns = [
    "node_modules/",
    ".git/",
    "target/",
    ".cache/",
    "__pycache__/",
    ".pytest_cache/",
    "dist/",
    "build/",
    ".next/",
    ".nuxt/",
    "vendor/",
    ".venv/",
    "venv/",
    ".env/",
    "env/",
    ".tox/",
    ".mypy_cache/",
    ".ruff_cache/",
    "coverage/",
    ".coverage/",
    ".nyc_output/",
    "*.min.js",
    "*.min.css",
    "*.map",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
  ];

  for pattern in ignore_patterns {
    if let Some(dir_name) = pattern.strip_suffix('/') {
      // Directory pattern
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
      if path.file_name().is_some_and(|n| n.to_string_lossy() == pattern) {
        return true;
      }
    }
  }

  false
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_should_ignore_node_modules() {
    assert!(should_ignore(Path::new("project/node_modules/foo.js")));
    assert!(should_ignore(Path::new("node_modules/package/index.js")));
  }

  #[test]
  fn test_should_ignore_git() {
    assert!(should_ignore(Path::new(".git/config")));
    assert!(should_ignore(Path::new("project/.git/objects/abc")));
  }

  #[test]
  fn test_should_ignore_lockfiles() {
    assert!(should_ignore(Path::new("package-lock.json")));
    assert!(should_ignore(Path::new("yarn.lock")));
    assert!(should_ignore(Path::new("Cargo.lock")));
  }

  #[test]
  fn test_should_not_ignore_source() {
    assert!(!should_ignore(Path::new("src/main.rs")));
    assert!(!should_ignore(Path::new("lib/index.ts")));
    assert!(!should_ignore(Path::new("app.py")));
  }

  #[test]
  fn test_should_ignore_minified() {
    assert!(should_ignore(Path::new("dist/bundle.min.js")));
    assert!(should_ignore(Path::new("styles.min.css")));
  }

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
