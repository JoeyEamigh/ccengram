//! Test fixture generation utilities for benchmarks.
//!
//! Creates, modifies, and cleans up test files within repositories
//! for measuring incremental indexing and watcher performance.

use std::path::{Path, PathBuf};

use tokio::fs;
use tracing::{debug, info};
use uuid::Uuid;

use crate::Result;

/// Directory name for benchmark fixtures within target repos.
/// Uses "ccengram_bench" instead of "__bench_fixtures__" to avoid
/// potential gitignore patterns that match double-underscore prefixes.
const FIXTURES_DIR: &str = "ccengram_bench";

/// Generator for test fixtures within a repository.
///
/// Creates files in a dedicated subdirectory that can be safely
/// cleaned up after benchmarks complete.
pub struct FixtureGenerator {
  fixtures_dir: PathBuf,
  generated: Vec<PathBuf>,
}

impl FixtureGenerator {
  /// Create a new fixture generator for the given repository path.
  pub async fn new(repo_path: &Path) -> Result<Self> {
    let fixtures_dir = repo_path.join(FIXTURES_DIR);
    fs::create_dir_all(&fixtures_dir).await?;

    Ok(Self {
      fixtures_dir,
      generated: Vec::new(),
    })
  }

  /// Create a file with the given relative path and content.
  ///
  /// The path is relative to the fixtures directory.
  /// Returns the absolute path to the created file.
  pub async fn create_file(&mut self, relative_path: &str, content: &str) -> Result<PathBuf> {
    let path = self.fixtures_dir.join(relative_path);

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent).await?;
    }

    fs::write(&path, content).await?;
    self.generated.push(path.clone());

    debug!("Created fixture: {}", path.display());
    Ok(path)
  }

  /// Create a Rust source file with a unique function.
  ///
  /// Returns the path and the unique identifier used in the content.
  /// Uses UUID in filename to avoid conflicts with stale fixtures from previous runs.
  pub async fn create_rust_file(&mut self, name: &str) -> Result<(PathBuf, String)> {
    let uuid = Uuid::new_v4().to_string();
    let short_id = &uuid[..8];
    let content = format!(
      r#"//! Benchmark fixture file
/// Function with unique identifier for search verification.
pub fn benchmark_fixture_{}_{}() {{
    // Unique marker: {}
    println!("Fixture function");
}}
"#,
      name.replace('-', "_"),
      short_id,
      uuid
    );

    // Include short UUID in filename to avoid conflicts with stale fixtures
    let path = self.create_file(&format!("{}_{}.rs", name, short_id), &content).await?;
    Ok((path, uuid))
  }

  /// Modify an existing file by appending content.
  pub async fn modify_file(&self, path: &Path, append: &str) -> Result<()> {
    let content = fs::read_to_string(path).await?;
    let new_content = format!("{}\n{}", content, append);
    fs::write(path, new_content).await?;

    debug!("Modified fixture: {}", path.display());
    Ok(())
  }

  /// Delete a file.
  pub async fn delete_file(&self, path: &Path) -> Result<()> {
    if path.exists() {
      fs::remove_file(path).await?;
      debug!("Deleted fixture: {}", path.display());
    }
    Ok(())
  }

  /// Rename a file.
  pub async fn rename_file(&mut self, from: &Path, to: &Path) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = to.parent() {
      fs::create_dir_all(parent).await?;
    }

    fs::rename(from, to).await?;

    // Update tracking
    if let Some(idx) = self.generated.iter().position(|p| p == from) {
      self.generated[idx] = to.to_path_buf();
    } else {
      self.generated.push(to.to_path_buf());
    }

    debug!("Renamed fixture: {} -> {}", from.display(), to.display());
    Ok(())
  }

  /// Create a large file of approximately the given size.
  ///
  /// Returns the path and a unique identifier for search verification.
  pub async fn create_large_file(&mut self, size_bytes: u64) -> Result<(PathBuf, String)> {
    let uuid = Uuid::new_v4().to_string();
    let name = format!("large_file_{}", uuid.split('-').next().unwrap_or("x"));

    // Create content that's roughly the target size
    // Use a repeating pattern with the unique marker
    let header = format!(
      r#"//! Large benchmark fixture file
/// Unique marker: {}
pub fn large_file_marker() {{}}

// Padding content follows...
"#,
      uuid
    );

    // Use varied content to avoid triggering the repetitive file heuristic
    // Each line includes its number to ensure uniqueness
    let avg_line_len = 60u64; // approximate length of each numbered line
    let lines_needed = (size_bytes.saturating_sub(header.len() as u64)) / avg_line_len;

    let mut content = header;
    for i in 0..lines_needed {
      content.push_str(&format!(
        "// Padding line {:06} for benchmark fixture target size.\n",
        i
      ));
    }

    let path = self.create_file(&format!("{}.rs", name), &content).await?;
    info!("Created large file: {} ({} bytes)", path.display(), content.len());
    Ok((path, uuid))
  }

  /// Create multiple files in a gitignored directory (e.g., node_modules).
  ///
  /// These files should NOT be indexed by the watcher.
  pub async fn create_ignored_files(&mut self, ignored_dir: &str, count: usize) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(count);
    let ignored_path = self.fixtures_dir.join(ignored_dir);
    fs::create_dir_all(&ignored_path).await?;

    for i in 0..count {
      let uuid = Uuid::new_v4().to_string();
      let short_id = &uuid[..8];
      let content = format!(
        r#"// Ignored fixture file {}
export function ignoredFunction{}_{}() {{
    // Marker: {}
}}
"#,
        i, i, short_id, uuid
      );

      let path = ignored_path.join(format!("ignored_{}_{}.ts", i, short_id));
      fs::write(&path, content).await?;
      self.generated.push(path.clone());
      paths.push(path);
    }

    debug!("Created {} ignored files in {}", count, ignored_dir);
    Ok(paths)
  }

  /// Create files that should be tracked (in src/ directory).
  pub async fn create_tracked_files(&mut self, count: usize) -> Result<Vec<(PathBuf, String)>> {
    let mut results = Vec::with_capacity(count);
    let src_path = self.fixtures_dir.join("src");
    fs::create_dir_all(&src_path).await?;

    for i in 0..count {
      let uuid = Uuid::new_v4().to_string();
      let short_id = &uuid[..8];
      let content = format!(
        r#"//! Tracked fixture file {}
/// Unique marker: {}
pub fn tracked_function_{}_{}() {{
    println!("Tracked");
}}
"#,
        i, uuid, i, short_id
      );

      let path = src_path.join(format!("tracked_{}_{}.rs", i, short_id));
      fs::write(&path, &content).await?;
      self.generated.push(path.clone());
      results.push((path, uuid));
    }

    debug!("Created {} tracked files in src/", count);
    Ok(results)
  }

  /// Create a batch of files rapidly for debounce testing.
  pub async fn create_batch(&mut self, count: usize) -> Result<Vec<(PathBuf, String)>> {
    let mut results = Vec::with_capacity(count);

    for i in 0..count {
      let (path, uuid) = self.create_rust_file(&format!("batch_{}", i)).await?;
      results.push((path, uuid));
    }

    debug!("Created batch of {} files", count);
    Ok(results)
  }

  /// Clean up all generated fixtures.
  pub async fn cleanup(&mut self) -> Result<()> {
    // Remove individual files first
    for path in &self.generated {
      if path.exists()
        && let Err(e) = fs::remove_file(path).await
      {
        debug!("Failed to remove {}: {}", path.display(), e);
      }
    }
    self.generated.clear();

    // Remove the fixtures directory if it exists
    if self.fixtures_dir.exists() {
      fs::remove_dir_all(&self.fixtures_dir).await?;
      info!("Cleaned up fixtures directory: {}", self.fixtures_dir.display());
    }

    Ok(())
  }
}

impl Drop for FixtureGenerator {
  fn drop(&mut self) {
    // Note: async cleanup should be called explicitly before drop
    // This is just a safety fallback using sync operations
    if self.fixtures_dir.exists()
      && let Err(e) = std::fs::remove_dir_all(&self.fixtures_dir)
    {
      debug!("Failed to cleanup fixtures on drop: {}", e);
    }
  }
}
