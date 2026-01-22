use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Find the git root directory by walking upward from the given path
pub fn find_git_root(path: &Path) -> Option<PathBuf> {
  let mut current = path.to_path_buf();

  loop {
    let git_dir = current.join(".git");
    if git_dir.exists() {
      return Some(current);
    }

    if !current.pop() {
      return None;
    }
  }
}

/// Get the project root path, preferring git root over the given path
pub fn resolve_project_path(path: &Path) -> PathBuf {
  let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
  find_git_root(&canonical).unwrap_or(canonical)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectId(String);

impl ProjectId {
  /// Create a ProjectId from a path, using git root detection if available
  pub fn from_path(path: &Path) -> Self {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Try to find git root first for more stable project identity
    let project_path = find_git_root(&canonical).unwrap_or(canonical);

    let hash = Self::hash_path(&project_path);
    ProjectId(hash)
  }

  /// Create a ProjectId from a path without git root detection
  pub fn from_path_exact(path: &Path) -> Self {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let hash = Self::hash_path(&canonical);
    ProjectId(hash)
  }

  fn hash_path(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
  }

  pub fn as_str(&self) -> &str {
    &self.0
  }

  pub fn data_dir(&self, base: &Path) -> PathBuf {
    base.join("projects").join(&self.0)
  }
}

impl std::fmt::Display for ProjectId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
  pub id: ProjectId,
  pub path: PathBuf,
  pub name: String,
  pub created_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;

  #[test]
  fn test_project_id_stable_across_subdirs() {
    let temp = std::env::temp_dir().join(format!("test_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let root = temp.as_path();

    // Create a git repo
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(root.join("src/components")).unwrap();

    // ProjectId should be the same from any subdirectory
    let id_root = ProjectId::from_path(root);
    let id_src = ProjectId::from_path(&root.join("src"));
    let id_components = ProjectId::from_path(&root.join("src/components"));

    assert_eq!(id_root, id_src);
    assert_eq!(id_root, id_components);

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }

  #[test]
  fn test_project_id_exact_differs() {
    let temp = std::env::temp_dir().join(format!("test_exact_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let root = temp.as_path();

    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(root.join("src")).unwrap();

    // from_path_exact should give different IDs for subdirs
    let id_root = ProjectId::from_path_exact(root);
    let id_src = ProjectId::from_path_exact(&root.join("src"));

    assert_ne!(id_root, id_src);

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }

  #[test]
  fn test_find_git_root() {
    let temp = std::env::temp_dir().join(format!("test_git_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let root = temp.as_path();

    // No .git -> None
    assert!(find_git_root(root).is_none());

    // Create .git
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(root.join("src/deep/nested")).unwrap();

    // Should find root from any subdir
    let canonical_root = root.canonicalize().unwrap();
    assert_eq!(find_git_root(root), Some(canonical_root.clone()));
    assert_eq!(find_git_root(&root.join("src")), Some(canonical_root.clone()));
    assert_eq!(find_git_root(&root.join("src/deep/nested")), Some(canonical_root));

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }

  #[test]
  fn test_resolve_project_path_with_git() {
    let temp = std::env::temp_dir().join(format!("test_resolve_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let root = temp.as_path();

    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(root.join("src")).unwrap();

    let resolved = resolve_project_path(&root.join("src"));
    assert_eq!(resolved, root.canonicalize().unwrap());

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }

  #[test]
  fn test_resolve_project_path_without_git() {
    let temp = std::env::temp_dir().join(format!("test_no_git_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let root = temp.as_path();
    fs::create_dir_all(root.join("src")).unwrap();

    let resolved = resolve_project_path(&root.join("src"));
    assert_eq!(resolved, root.join("src").canonicalize().unwrap());

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }
}
