//! Repository downloading and cache management.

use super::registry::{RepoConfig, RepoRegistry, TargetRepo};
use crate::{BenchmarkError, Result};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use tar::Archive;
use tracing::{debug, info, warn};

/// Cache manager for benchmark repositories.
pub struct RepoCache {
    cache_dir: PathBuf,
}

impl RepoCache {
    /// Create a new cache manager.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the path where a repo would be cached.
    pub fn repo_path(&self, repo: TargetRepo) -> PathBuf {
        let config = RepoRegistry::get(repo);
        self.cache_dir.join(config.extracted_dir_name())
    }

    /// Check if a repo is already cached.
    pub fn is_cached(&self, repo: TargetRepo) -> bool {
        let path = self.repo_path(repo);
        path.exists() && path.is_dir()
    }

    /// Ensure a repo is available (download if needed).
    pub async fn ensure_repo(&self, repo: TargetRepo) -> Result<PathBuf> {
        let path = self.repo_path(repo);
        if self.is_cached(repo) {
            info!("Using cached repository: {}", path.display());
            return Ok(path);
        }

        info!("Downloading repository: {}", repo);
        self.download_repo(repo).await?;
        Ok(path)
    }

    /// Download a repository tarball and extract it.
    async fn download_repo(&self, repo: TargetRepo) -> Result<()> {
        let config = RepoRegistry::get(repo);

        // Ensure cache directory exists
        fs::create_dir_all(&self.cache_dir)?;

        let tarball_path = self.cache_dir.join(format!("{}.tar.gz", config.name));

        // Download the tarball
        self.download_tarball(&config, &tarball_path).await?;

        // Extract the tarball
        self.extract_tarball(&tarball_path, &self.cache_dir)?;

        // Clean up tarball
        if let Err(e) = fs::remove_file(&tarball_path) {
            warn!("Failed to remove tarball: {}", e);
        }

        info!("Repository extracted to: {}", self.repo_path(repo).display());
        Ok(())
    }

    /// Download a tarball from GitHub.
    async fn download_tarball(&self, config: &RepoConfig, dest: &Path) -> Result<()> {
        let url = config.tarball_url();
        info!("Downloading from: {}", url);

        let client = reqwest::Client::new();
        let response = client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(BenchmarkError::Repo(format!(
                "Failed to download {}: HTTP {}",
                url,
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(0);
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let file = File::create(dest)?;
        let mut writer = BufWriter::new(file);

        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            io::copy(&mut chunk.as_ref(), &mut writer)?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message("Download complete");
        debug!("Downloaded {} bytes to {}", downloaded, dest.display());
        Ok(())
    }

    /// Extract a tarball to a directory.
    fn extract_tarball(&self, tarball: &Path, dest: &Path) -> Result<()> {
        info!("Extracting {} to {}", tarball.display(), dest.display());

        let file = File::open(tarball)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} Extracting... {msg}")
                .unwrap(),
        );

        let mut count = 0;
        for entry in archive.entries()? {
            let mut entry = entry?;
            entry.unpack_in(dest)?;
            count += 1;
            if count % 100 == 0 {
                pb.set_message(format!("{} files", count));
            }
        }

        pb.finish_with_message(format!("{} files extracted", count));
        Ok(())
    }

    /// Remove a cached repository.
    pub fn remove(&self, repo: TargetRepo) -> Result<()> {
        let path = self.repo_path(repo);
        if path.exists() {
            fs::remove_dir_all(&path)?;
            info!("Removed cached repository: {}", path.display());
        }
        Ok(())
    }

    /// Remove all cached repositories.
    pub fn clean_all(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
            info!("Cleaned all cached repositories");
        }
        Ok(())
    }

    /// List all cached repositories.
    pub fn list_cached(&self) -> Vec<TargetRepo> {
        TargetRepo::all()
            .iter()
            .filter(|r| self.is_cached(**r))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_repo_cache_path() {
        let temp = TempDir::new().unwrap();
        let cache = RepoCache::new(temp.path().to_path_buf());

        let path = cache.repo_path(TargetRepo::Zed);
        assert!(path.to_string_lossy().contains("zed-0.220.3"));
    }

    #[test]
    fn test_is_not_cached() {
        let temp = TempDir::new().unwrap();
        let cache = RepoCache::new(temp.path().to_path_buf());

        assert!(!cache.is_cached(TargetRepo::Zed));
    }

    #[test]
    fn test_list_cached_empty() {
        let temp = TempDir::new().unwrap();
        let cache = RepoCache::new(temp.path().to_path_buf());

        assert!(cache.list_cached().is_empty());
    }
}
