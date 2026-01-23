//! Repository registry with predefined configurations.

use serde::{Deserialize, Serialize};

/// Target repository for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetRepo {
    Zed,
    Vscode,
}

impl TargetRepo {
    /// Get all available target repositories.
    pub fn all() -> &'static [TargetRepo] {
        &[TargetRepo::Zed, TargetRepo::Vscode]
    }

    /// Get the repository name.
    pub fn name(&self) -> &'static str {
        match self {
            TargetRepo::Zed => "zed",
            TargetRepo::Vscode => "vscode",
        }
    }

    /// Parse from string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "zed" => Some(TargetRepo::Zed),
            "vscode" => Some(TargetRepo::Vscode),
            _ => None,
        }
    }
}

impl std::fmt::Display for TargetRepo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for TargetRepo {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        TargetRepo::from_name(s).ok_or_else(|| format!("Unknown repo: {}", s))
    }
}

/// Configuration for a target repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoConfig {
    /// Repository identifier
    pub repo: TargetRepo,
    /// GitHub organization/user
    pub owner: String,
    /// Repository name on GitHub
    pub name: String,
    /// Release tag for tarball download
    pub release_tag: String,
    /// Primary programming language
    pub language: String,
    /// Approximate lines of code
    pub approx_loc: String,
    /// Documentation directory (if any)
    pub docs_dir: Option<String>,
    /// Directories to exclude from indexing
    pub exclude_dirs: Vec<String>,
}

impl RepoConfig {
    /// Get the tarball URL for downloading.
    pub fn tarball_url(&self) -> String {
        format!(
            "https://github.com/{}/{}/archive/refs/tags/{}.tar.gz",
            self.owner, self.name, self.release_tag
        )
    }

    /// Get the expected directory name after extraction.
    pub fn extracted_dir_name(&self) -> String {
        // GitHub tarballs extract to repo-tag format
        let tag = self.release_tag.trim_start_matches('v');
        format!("{}-{}", self.name, tag)
    }
}

/// Registry of all benchmark target repositories.
pub struct RepoRegistry;

impl RepoRegistry {
    /// Get configuration for a specific repository.
    pub fn get(repo: TargetRepo) -> RepoConfig {
        match repo {
            TargetRepo::Zed => Self::zed_config(),
            TargetRepo::Vscode => Self::vscode_config(),
        }
    }

    /// Get all repository configurations.
    pub fn all() -> Vec<RepoConfig> {
        TargetRepo::all().iter().map(|r| Self::get(*r)).collect()
    }

    fn zed_config() -> RepoConfig {
        RepoConfig {
            repo: TargetRepo::Zed,
            owner: "zed-industries".to_string(),
            name: "zed".to_string(),
            release_tag: "v0.220.3".to_string(),
            language: "Rust".to_string(),
            approx_loc: "~1M".to_string(),
            docs_dir: Some("docs".to_string()),
            exclude_dirs: vec![
                "target".to_string(),
                ".git".to_string(),
                "node_modules".to_string(),
                "assets".to_string(),
            ],
        }
    }

    fn vscode_config() -> RepoConfig {
        RepoConfig {
            repo: TargetRepo::Vscode,
            owner: "microsoft".to_string(),
            name: "vscode".to_string(),
            release_tag: "1.108.1".to_string(),
            language: "TypeScript".to_string(),
            approx_loc: "~1M".to_string(),
            docs_dir: Some("docs".to_string()),
            exclude_dirs: vec![
                "node_modules".to_string(),
                ".git".to_string(),
                "out".to_string(),
                "out-build".to_string(),
                ".build".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_repo_name() {
        assert_eq!(TargetRepo::Zed.name(), "zed");
        assert_eq!(TargetRepo::Vscode.name(), "vscode");
    }

    #[test]
    fn test_target_repo_from_name() {
        assert_eq!(TargetRepo::from_name("zed"), Some(TargetRepo::Zed));
        assert_eq!(TargetRepo::from_name("ZED"), Some(TargetRepo::Zed));
        assert_eq!(TargetRepo::from_name("vscode"), Some(TargetRepo::Vscode));
        assert_eq!(TargetRepo::from_name("unknown"), None);
    }

    #[test]
    fn test_repo_config_tarball_url() {
        let config = RepoRegistry::get(TargetRepo::Zed);
        assert!(config.tarball_url().contains("zed-industries/zed"));
        assert!(config.tarball_url().contains("v0.220.3"));
    }

    #[test]
    fn test_repo_config_extracted_dir() {
        let config = RepoRegistry::get(TargetRepo::Zed);
        assert_eq!(config.extracted_dir_name(), "zed-0.220.3");

        let config = RepoRegistry::get(TargetRepo::Vscode);
        assert_eq!(config.extracted_dir_name(), "vscode-1.108.1");
    }

    #[test]
    fn test_all_repos() {
        let configs = RepoRegistry::all();
        assert_eq!(configs.len(), 2);
    }
}
