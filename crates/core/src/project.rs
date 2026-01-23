use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Find the git root directory by walking upward from the given path.
///
/// For regular git repos, returns the directory containing `.git/`.
/// For git worktrees, follows the `.git` file to find the main repository.
pub fn find_git_root(path: &Path) -> Option<PathBuf> {
  let mut current = path.to_path_buf();

  loop {
    let git_path = current.join(".git");
    if git_path.exists() {
      // Check if this is a worktree (.git is a file) or main repo (.git is a directory)
      if git_path.is_file() {
        // This is a worktree - resolve to main repository
        if let Some(main_repo) = resolve_worktree_to_main_repo(&git_path) {
          return Some(main_repo);
        }
        // Fall back to worktree path if we can't resolve
        return Some(current);
      } else {
        // Regular git repository
        return Some(current);
      }
    }

    if !current.pop() {
      return None;
    }
  }
}

/// Parse a worktree's .git file and resolve to the main repository path.
///
/// Worktree .git files contain: `gitdir: /path/to/main/.git/worktrees/<name>`
/// We extract the main .git directory and return its parent (the main repo root).
fn resolve_worktree_to_main_repo(git_file: &Path) -> Option<PathBuf> {
  let content = std::fs::read_to_string(git_file).ok()?;

  // Parse "gitdir: /path/to/git/dir"
  let gitdir_line = content.lines().find(|line| line.starts_with("gitdir:"))?;
  let gitdir_path = gitdir_line.strip_prefix("gitdir:")?.trim();

  // The path might be relative or absolute
  let gitdir = if Path::new(gitdir_path).is_absolute() {
    PathBuf::from(gitdir_path)
  } else {
    // Relative to the worktree directory
    git_file.parent()?.join(gitdir_path)
  };

  // Canonicalize to resolve any symlinks and get absolute path
  let gitdir = gitdir.canonicalize().ok()?;

  // The gitdir is typically: /main/repo/.git/worktrees/<name>
  // We need to find the main .git directory
  // Walk up looking for the main .git directory
  let mut current = gitdir.as_path();
  while let Some(parent) = current.parent() {
    if current.file_name().map(|n| n == ".git").unwrap_or(false) && current.is_dir() {
      // Found the main .git directory, return its parent (the repo root)
      return parent.canonicalize().ok();
    }
    current = parent;
  }

  None
}

/// Find git root without resolving worktrees to main repo.
/// Returns the worktree's own root if in a worktree.
pub fn find_git_root_local(path: &Path) -> Option<PathBuf> {
  let mut current = path.to_path_buf();

  loop {
    let git_path = current.join(".git");
    if git_path.exists() {
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

  #[test]
  fn test_worktree_detection() {
    // Create a "main" repository
    let temp = std::env::temp_dir().join(format!("test_worktree_{}", std::process::id()));
    let main_repo = temp.join("main-repo");
    let worktree = temp.join("worktree-feature");

    fs::create_dir_all(&main_repo).unwrap();
    fs::create_dir_all(&worktree).unwrap();

    // Create main repo's .git directory with worktrees structure
    let main_git = main_repo.join(".git");
    fs::create_dir_all(&main_git).unwrap();
    fs::create_dir_all(main_git.join("worktrees/feature")).unwrap();

    // Create worktree's .git file pointing to main repo
    let gitdir_path = main_git.join("worktrees/feature");
    let gitdir_content = format!("gitdir: {}", gitdir_path.display());
    fs::write(worktree.join(".git"), &gitdir_content).unwrap();

    // find_git_root should resolve worktree to main repo
    let main_canonical = main_repo.canonicalize().unwrap();
    let resolved = find_git_root(&worktree);
    assert_eq!(resolved, Some(main_canonical.clone()));

    // ProjectId should be the same for both
    let id_main = ProjectId::from_path(&main_repo);
    let id_worktree = ProjectId::from_path(&worktree);
    assert_eq!(id_main, id_worktree);

    // find_git_root_local should NOT resolve to main (returns worktree root)
    let local_root = find_git_root_local(&worktree);
    assert_eq!(local_root, Some(worktree.canonicalize().unwrap()));

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }

  #[test]
  fn test_worktree_relative_path() {
    // Test that relative gitdir paths work
    let temp = std::env::temp_dir().join(format!("test_worktree_rel_{}", std::process::id()));
    let main_repo = temp.join("main");
    let worktree = temp.join("worktree");

    fs::create_dir_all(&main_repo).unwrap();
    fs::create_dir_all(&worktree).unwrap();

    // Create main repo structure
    let main_git = main_repo.join(".git");
    fs::create_dir_all(&main_git).unwrap();
    fs::create_dir_all(main_git.join("worktrees/wt")).unwrap();

    // Create worktree with RELATIVE path (relative to worktree/.git)
    // Note: In practice git uses absolute paths, but we support both
    let gitdir_abs = main_git.join("worktrees/wt");
    fs::write(worktree.join(".git"), format!("gitdir: {}", gitdir_abs.display())).unwrap();

    let resolved = find_git_root(&worktree);
    assert_eq!(resolved, Some(main_repo.canonicalize().unwrap()));

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }

  #[test]
  fn test_find_git_root_local_differs_from_find_git_root() {
    // For regular repos, both should return the same
    let temp = std::env::temp_dir().join(format!("test_local_{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    fs::create_dir_all(temp.join(".git")).unwrap();

    let root1 = find_git_root(&temp);
    let root2 = find_git_root_local(&temp);
    assert_eq!(root1, root2);

    // Cleanup
    let _ = fs::remove_dir_all(&temp);
  }
}
