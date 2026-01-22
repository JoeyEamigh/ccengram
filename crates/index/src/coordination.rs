// Watcher coordination - lock files and process management
//
// Ensures only one watcher runs per project:
// - Lock files at ~/.local/share/ccengram/watchers/<hash>.lock
// - Stale lock detection via process alive check
// - Activity tracking for health monitoring

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum CoordinationError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("JSON error: {0}")]
  Json(#[from] serde_json::Error),
  #[error("Lock held by process {0}")]
  LockHeld(u32),
  #[error("Lock file corrupted")]
  CorruptedLock,
}

/// Lock file contents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherLock {
  pub project_id: String,
  pub project_path: String,
  pub pid: u32,
  pub started_at: u64,
  pub last_activity: u64,
  pub indexed_files: u32,
}

impl WatcherLock {
  /// Create a new lock for the current process
  pub fn new(project_id: &str, project_path: &str) -> Self {
    let now = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_secs();

    Self {
      project_id: project_id.to_string(),
      project_path: project_path.to_string(),
      pid: std::process::id(),
      started_at: now,
      last_activity: now,
      indexed_files: 0,
    }
  }
}

/// Watcher coordinator for managing lock files
pub struct WatcherCoordinator {
  locks_dir: PathBuf,
}

impl Default for WatcherCoordinator {
  fn default() -> Self {
    Self::new()
  }
}

impl WatcherCoordinator {
  /// Create a new coordinator with default locks directory
  pub fn new() -> Self {
    let locks_dir = db::default_data_dir().join("watchers");
    Self { locks_dir }
  }

  /// Create a coordinator with a custom locks directory (for testing)
  pub fn with_locks_dir(locks_dir: PathBuf) -> Self {
    Self { locks_dir }
  }

  /// Get the lock file path for a project
  pub fn lock_path(&self, project_path: &Path) -> PathBuf {
    let hash = project_hash(project_path);
    self.locks_dir.join(format!("{}.lock", hash))
  }

  /// Try to acquire a lock for a project
  ///
  /// Returns Ok(true) if lock was acquired, Ok(false) if already held by a live process
  pub fn try_acquire(&self, project_id: &str, project_path: &Path) -> Result<bool, CoordinationError> {
    fs::create_dir_all(&self.locks_dir)?;

    let lock_path = self.lock_path(project_path);

    // Check if lock exists and is held by a live process
    if lock_path.exists() {
      match self.read_lock(&lock_path) {
        Ok(existing) => {
          if is_process_running(existing.pid) {
            debug!(
              "Lock held by process {} for project {}",
              existing.pid, existing.project_path
            );
            return Ok(false);
          }
          // Process is dead, clean up stale lock
          info!("Cleaning up stale lock from dead process {}", existing.pid);
          fs::remove_file(&lock_path)?;
        }
        Err(e) => {
          warn!("Corrupted lock file, removing: {}", e);
          fs::remove_file(&lock_path)?;
        }
      }
    }

    // Create new lock
    let lock = WatcherLock::new(project_id, &project_path.to_string_lossy());
    self.write_lock(&lock_path, &lock)?;

    info!("Acquired watcher lock for project: {}", project_path.display());
    Ok(true)
  }

  /// Release a lock for a project
  pub fn release(&self, project_path: &Path) -> Result<(), CoordinationError> {
    let lock_path = self.lock_path(project_path);

    if lock_path.exists() {
      // Verify we own the lock before releasing
      if let Ok(lock) = self.read_lock(&lock_path) {
        if lock.pid == std::process::id() {
          fs::remove_file(&lock_path)?;
          info!("Released watcher lock for project: {}", project_path.display());
        } else {
          warn!(
            "Not releasing lock owned by different process {} (we are {})",
            lock.pid,
            std::process::id()
          );
        }
      }
    }

    Ok(())
  }

  /// Update the activity timestamp for a lock
  pub fn update_activity(&self, project_path: &Path, indexed_files: u32) -> Result<(), CoordinationError> {
    let lock_path = self.lock_path(project_path);

    if !lock_path.exists() {
      return Ok(());
    }

    let mut lock = self.read_lock(&lock_path)?;

    // Only update if we own the lock
    if lock.pid != std::process::id() {
      return Ok(());
    }

    lock.last_activity = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_secs();
    lock.indexed_files = indexed_files;

    self.write_lock(&lock_path, &lock)?;
    Ok(())
  }

  /// Check if a watcher is running for a project
  pub fn is_running(&self, project_path: &Path) -> Result<bool, CoordinationError> {
    let lock_path = self.lock_path(project_path);

    if !lock_path.exists() {
      return Ok(false);
    }

    let lock = self.read_lock(&lock_path)?;
    Ok(is_process_running(lock.pid))
  }

  /// Get the lock info for a project
  pub fn get_lock(&self, project_path: &Path) -> Result<Option<WatcherLock>, CoordinationError> {
    let lock_path = self.lock_path(project_path);

    if !lock_path.exists() {
      return Ok(None);
    }

    let lock = self.read_lock(&lock_path)?;
    if is_process_running(lock.pid) {
      Ok(Some(lock))
    } else {
      Ok(None)
    }
  }

  /// List all active watchers
  pub fn list_active(&self) -> Result<Vec<WatcherLock>, CoordinationError> {
    if !self.locks_dir.exists() {
      return Ok(Vec::new());
    }

    let mut active = Vec::new();

    for entry in fs::read_dir(&self.locks_dir)? {
      let entry = entry?;
      let path = entry.path();

      if path.extension().map(|e| e == "lock").unwrap_or(false) {
        match self.read_lock(&path) {
          Ok(lock) => {
            if is_process_running(lock.pid) {
              active.push(lock);
            } else {
              // Clean up stale lock
              debug!("Removing stale lock: {:?}", path);
              let _ = fs::remove_file(&path);
            }
          }
          Err(e) => {
            warn!("Failed to read lock {:?}: {}", path, e);
            let _ = fs::remove_file(&path);
          }
        }
      }
    }

    Ok(active)
  }

  /// Stop a watcher by sending SIGTERM (Unix) or TerminateProcess (Windows)
  pub fn stop_watcher(&self, project_path: &Path) -> Result<bool, CoordinationError> {
    let lock_path = self.lock_path(project_path);

    if !lock_path.exists() {
      return Ok(false);
    }

    let lock = self.read_lock(&lock_path)?;

    if !is_process_running(lock.pid) {
      fs::remove_file(&lock_path)?;
      return Ok(false);
    }

    // Send termination signal
    if terminate_process(lock.pid) {
      // Wait a bit for the process to exit
      std::thread::sleep(Duration::from_millis(500));

      // Check if it's still running
      if is_process_running(lock.pid) {
        // Force kill
        kill_process(lock.pid);
        std::thread::sleep(Duration::from_millis(100));
      }

      // Clean up lock file
      let _ = fs::remove_file(&lock_path);
      info!("Stopped watcher for project: {}", project_path.display());
      Ok(true)
    } else {
      Ok(false)
    }
  }

  fn read_lock(&self, path: &Path) -> Result<WatcherLock, CoordinationError> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(serde_json::from_str(&contents)?)
  }

  fn write_lock(&self, path: &Path, lock: &WatcherLock) -> Result<(), CoordinationError> {
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;

    let contents = serde_json::to_string_pretty(lock)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;

    Ok(())
  }
}

/// Hash a project path to a short identifier
fn project_hash(path: &Path) -> String {
  let mut hasher = Sha256::new();
  hasher.update(path.to_string_lossy().as_bytes());
  let hash = hasher.finalize();
  hex::encode(&hash[..8]) // First 16 hex chars
}

/// Check if a process is running
#[cfg(unix)]
fn is_process_running(pid: u32) -> bool {
  // kill(pid, 0) returns 0 if the process exists
  unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(windows)]
fn is_process_running(pid: u32) -> bool {
  use std::os::windows::prelude::*;
  use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
  use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_INFORMATION};

  unsafe {
    let handle = OpenProcess(PROCESS_QUERY_INFORMATION, 0, pid);
    if handle.is_null() {
      return false;
    }
    let mut exit_code = 0;
    let result = GetExitCodeProcess(handle, &mut exit_code);
    CloseHandle(handle);
    result != 0 && exit_code == STILL_ACTIVE
  }
}

#[cfg(not(any(unix, windows)))]
fn is_process_running(_pid: u32) -> bool {
  // Fallback: assume running to be safe
  true
}

/// Send termination signal to a process
#[cfg(unix)]
fn terminate_process(pid: u32) -> bool {
  unsafe { libc::kill(pid as i32, libc::SIGTERM) == 0 }
}

#[cfg(windows)]
fn terminate_process(pid: u32) -> bool {
  use windows_sys::Win32::Foundation::CloseHandle;
  use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess as WinTerminate};

  unsafe {
    let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
    if handle.is_null() {
      return false;
    }
    let result = WinTerminate(handle, 1) != 0;
    CloseHandle(handle);
    result
  }
}

#[cfg(not(any(unix, windows)))]
fn terminate_process(_pid: u32) -> bool {
  false
}

/// Force kill a process
#[cfg(unix)]
fn kill_process(pid: u32) -> bool {
  unsafe { libc::kill(pid as i32, libc::SIGKILL) == 0 }
}

#[cfg(windows)]
fn kill_process(pid: u32) -> bool {
  terminate_process(pid) // TerminateProcess is already forceful on Windows
}

#[cfg(not(any(unix, windows)))]
fn kill_process(_pid: u32) -> bool {
  false
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_project_hash() {
    let path1 = Path::new("/home/user/project1");
    let path2 = Path::new("/home/user/project2");

    let hash1 = project_hash(path1);
    let hash2 = project_hash(path2);

    assert_ne!(hash1, hash2);
    assert_eq!(hash1.len(), 16);

    // Same path should produce same hash
    assert_eq!(project_hash(path1), hash1);
  }

  #[test]
  fn test_acquire_and_release_lock() {
    let temp_dir = TempDir::new().unwrap();
    let coordinator = WatcherCoordinator::with_locks_dir(temp_dir.path().join("watchers"));

    let project_path = Path::new("/test/project");

    // Should acquire successfully
    let acquired = coordinator.try_acquire("test-id", project_path).unwrap();
    assert!(acquired);

    // Lock should exist
    assert!(coordinator.is_running(project_path).unwrap());

    // Should not acquire again (same process owns it)
    let acquired2 = coordinator.try_acquire("test-id", project_path).unwrap();
    assert!(!acquired2);

    // Release
    coordinator.release(project_path).unwrap();

    // Should be able to acquire again
    let acquired3 = coordinator.try_acquire("test-id", project_path).unwrap();
    assert!(acquired3);

    coordinator.release(project_path).unwrap();
  }

  #[test]
  fn test_update_activity() {
    let temp_dir = TempDir::new().unwrap();
    let coordinator = WatcherCoordinator::with_locks_dir(temp_dir.path().join("watchers"));

    let project_path = Path::new("/test/project");

    coordinator.try_acquire("test-id", project_path).unwrap();

    // Update activity
    coordinator.update_activity(project_path, 100).unwrap();

    let lock = coordinator.get_lock(project_path).unwrap().unwrap();
    assert_eq!(lock.indexed_files, 100);

    coordinator.release(project_path).unwrap();
  }

  #[test]
  fn test_list_active() {
    let temp_dir = TempDir::new().unwrap();
    let coordinator = WatcherCoordinator::with_locks_dir(temp_dir.path().join("watchers"));

    // Acquire locks for multiple projects
    let p1 = Path::new("/test/project1");
    let p2 = Path::new("/test/project2");

    coordinator.try_acquire("id1", p1).unwrap();
    coordinator.try_acquire("id2", p2).unwrap();

    let active = coordinator.list_active().unwrap();
    assert_eq!(active.len(), 2);

    coordinator.release(p1).unwrap();
    coordinator.release(p2).unwrap();
  }

  #[test]
  fn test_lock_file_contents() {
    let temp_dir = TempDir::new().unwrap();
    let coordinator = WatcherCoordinator::with_locks_dir(temp_dir.path().join("watchers"));

    let project_path = Path::new("/test/project");
    coordinator.try_acquire("test-id", project_path).unwrap();

    let lock = coordinator.get_lock(project_path).unwrap().unwrap();
    assert_eq!(lock.project_id, "test-id");
    assert_eq!(lock.project_path, "/test/project");
    assert_eq!(lock.pid, std::process::id());
    assert!(lock.started_at > 0);

    coordinator.release(project_path).unwrap();
  }

  #[test]
  fn test_is_process_running_current() {
    // Current process should be running
    assert!(is_process_running(std::process::id()));
  }

  #[test]
  fn test_is_process_running_invalid() {
    // Very high PID should not exist
    assert!(!is_process_running(u32::MAX - 1));
  }
}
