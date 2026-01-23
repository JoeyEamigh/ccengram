//! Repository management for benchmark targets.
//!
//! Handles downloading, caching, and managing target repositories
//! (Zed, VSCode) for benchmarking.

mod clone;
mod registry;

pub use clone::RepoCache;
pub use registry::{RepoConfig, RepoRegistry, TargetRepo};

use crate::Result;
use std::path::PathBuf;

/// Get the default cache directory for benchmark repositories.
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("ccengram-bench")
        .join("repos")
}

/// Prepare a repository for benchmarking (download if needed, return path).
pub async fn prepare_repo(repo: TargetRepo, cache_dir: Option<PathBuf>) -> Result<PathBuf> {
    let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
    let cache = RepoCache::new(cache_dir);
    cache.ensure_repo(repo).await
}
