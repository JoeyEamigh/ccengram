use engram_core::Language;
use ignore::WalkBuilder;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant, UNIX_EPOCH};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScanError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Walk error: {0}")]
  Walk(#[from] ignore::Error),
}

/// Result of scanning a single file
#[derive(Debug, Clone)]
pub struct ScannedFile {
  pub path: PathBuf,
  pub relative_path: String,
  pub language: Language,
  pub size: u64,
  pub mtime: u64,
  pub checksum: String,
}

/// Result of scanning a directory
#[derive(Debug)]
pub struct ScanResult {
  pub files: Vec<ScannedFile>,
  pub skipped_count: u32,
  pub total_bytes: u64,
  pub scan_duration: Duration,
}

/// Progress callback data
#[derive(Debug, Clone)]
pub struct ScanProgress {
  pub scanned: u32,
  pub path: PathBuf,
}

/// File scanner with gitignore support
pub struct Scanner {
  max_file_size: u64,
  follow_links: bool,
}

impl Default for Scanner {
  fn default() -> Self {
    Self::new()
  }
}

impl Scanner {
  pub fn new() -> Self {
    Self {
      max_file_size: 1024 * 1024, // 1MB
      follow_links: false,
    }
  }

  pub fn with_max_file_size(mut self, size: u64) -> Self {
    self.max_file_size = size;
    self
  }

  /// Scan directory in parallel, respecting .gitignore
  pub fn scan<F>(&self, root: &Path, progress: F) -> ScanResult
  where
    F: Fn(ScanProgress) + Send + Sync,
  {
    let start = Instant::now();
    let scanned = AtomicU32::new(0);
    let skipped = AtomicU32::new(0);
    let total_bytes = AtomicU64::new(0);

    let walker = WalkBuilder::new(root)
      .follow_links(self.follow_links)
      .hidden(true) // Include hidden files
      .git_ignore(true)
      .git_global(true)
      .git_exclude(true)
      .add_custom_ignore_filename(".ccengramignore")
      .build();

    // Use par_bridge to parallelize without collecting first - better for 100k+ files
    let files: Vec<ScannedFile> = walker
      .filter_map(|e| e.ok())
      .par_bridge()
      .filter_map(|entry| {
        let path = entry.path();

        // Skip directories
        if entry.file_type().is_none_or(|ft| ft.is_dir()) {
          return None;
        }

        // Progress callback every 100 files
        let count = scanned.fetch_add(1, Ordering::Relaxed);
        if count.is_multiple_of(100) {
          progress(ScanProgress {
            scanned: count,
            path: path.to_path_buf(),
          });
        }

        // Get language from extension
        let ext = path.extension()?.to_str()?;
        let language = Language::from_extension(ext)?;

        // Skip empty or large files
        let metadata = entry.metadata().ok()?;
        if metadata.len() == 0 {
          skipped.fetch_add(1, Ordering::Relaxed);
          return None;
        }
        if metadata.len() > self.max_file_size {
          skipped.fetch_add(1, Ordering::Relaxed);
          return None;
        }

        // Compute quick checksum
        let checksum = quick_checksum(path).ok()?;

        let mtime = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_secs();

        total_bytes.fetch_add(metadata.len(), Ordering::Relaxed);

        Some(ScannedFile {
          path: path.to_path_buf(),
          relative_path: path.strip_prefix(root).ok()?.to_string_lossy().into(),
          language,
          size: metadata.len(),
          mtime,
          checksum,
        })
      })
      .collect();

    ScanResult {
      files,
      skipped_count: skipped.load(Ordering::Relaxed),
      total_bytes: total_bytes.load(Ordering::Relaxed),
      scan_duration: start.elapsed(),
    }
  }

  /// Scan a single file
  pub fn scan_file(&self, path: &Path, root: &Path) -> Option<ScannedFile> {
    let ext = path.extension()?.to_str()?;
    let language = Language::from_extension(ext)?;

    let metadata = path.metadata().ok()?;
    // Skip empty files
    if metadata.len() == 0 {
      return None;
    }
    if metadata.len() > self.max_file_size {
      return None;
    }

    let checksum = quick_checksum(path).ok()?;
    let mtime = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_secs();

    Some(ScannedFile {
      path: path.to_path_buf(),
      relative_path: path.strip_prefix(root).ok()?.to_string_lossy().into(),
      language,
      size: metadata.len(),
      mtime,
      checksum,
    })
  }
}

/// Quick checksum using first 4KB + file size
fn quick_checksum(path: &Path) -> Result<String, std::io::Error> {
  let mut file = File::open(path)?;
  let mut buffer = [0u8; 4096];
  let n = file.read(&mut buffer)?;

  let mut hasher = DefaultHasher::new();
  buffer[..n].hash(&mut hasher);
  file.metadata()?.len().hash(&mut hasher);

  Ok(format!("{:016x}", hasher.finish()))
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_scan_basic() {
    let dir = TempDir::new().unwrap();

    // Create a test file
    std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
    std::fs::write(dir.path().join("test.txt"), "hello").unwrap();

    let scanner = Scanner::new();
    let result = scanner.scan(dir.path(), |_| {});

    // Should find the .rs file but not the .txt file (unsupported)
    assert_eq!(result.files.len(), 1);
    assert_eq!(result.files[0].language, Language::Rust);
  }

  #[test]
  fn test_scan_respects_gitignore() {
    let dir = TempDir::new().unwrap();

    // Create .git directory so the ignore crate recognizes this as a git repo
    std::fs::create_dir(dir.path().join(".git")).unwrap();

    // Create .gitignore
    std::fs::write(dir.path().join(".gitignore"), "ignored/\n*.log").unwrap();

    // Create files
    std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
    std::fs::create_dir(dir.path().join("ignored")).unwrap();
    std::fs::write(dir.path().join("ignored/hidden.rs"), "fn hidden() {}").unwrap();
    std::fs::write(dir.path().join("debug.log"), "log").unwrap();

    let scanner = Scanner::new();
    let result = scanner.scan(dir.path(), |_| {});

    // Should only find main.rs (hidden.rs is in ignored dir, debug.log is not a source file)
    let paths: Vec<_> = result.files.iter().map(|f| &f.relative_path).collect();
    assert_eq!(
      result.files.len(),
      1,
      "Expected 1 file, found {}: {:?}",
      result.files.len(),
      paths
    );
    assert!(result.files[0].relative_path.contains("main.rs"));
  }

  #[test]
  fn test_scan_skips_large_files() {
    let dir = TempDir::new().unwrap();

    // Create a small file
    std::fs::write(dir.path().join("small.rs"), "fn small() {}").unwrap();

    // Create a large file (2MB)
    let large_content = "x".repeat(2 * 1024 * 1024);
    std::fs::write(dir.path().join("large.rs"), large_content).unwrap();

    let scanner = Scanner::new().with_max_file_size(1024 * 1024); // 1MB limit
    let result = scanner.scan(dir.path(), |_| {});

    assert_eq!(result.files.len(), 1);
    assert!(result.files[0].relative_path.contains("small.rs"));
    assert_eq!(result.skipped_count, 1);
  }

  #[test]
  fn test_quick_checksum() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.rs");
    std::fs::write(&path, "fn test() {}").unwrap();

    let checksum1 = quick_checksum(&path).unwrap();
    let checksum2 = quick_checksum(&path).unwrap();

    assert_eq!(checksum1, checksum2);
    assert_eq!(checksum1.len(), 16); // 64-bit hex
  }

  #[test]
  fn test_scan_skips_empty_files() {
    let dir = TempDir::new().unwrap();

    // Create a normal file
    std::fs::write(dir.path().join("normal.rs"), "fn normal() {}").unwrap();

    // Create an empty file
    std::fs::write(dir.path().join("empty.rs"), "").unwrap();

    let scanner = Scanner::new();
    let result = scanner.scan(dir.path(), |_| {});

    // Should only find the non-empty file
    assert_eq!(result.files.len(), 1);
    assert!(result.files[0].relative_path.contains("normal.rs"));
    assert_eq!(result.skipped_count, 1); // Empty file was skipped
  }
}
