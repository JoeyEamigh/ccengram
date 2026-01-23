//! Manual annotations for ground truth.
//!
//! JSON files with critical files, symbols, and exploration paths
//! that are manually curated per-scenario.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Exploration path defining expected navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPath {
    /// Starting query
    pub start: String,
    /// Expected intermediate discoveries
    #[serde(default)]
    pub through: Vec<String>,
    /// Final target
    pub target: String,
    /// Maximum acceptable hops
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,
}

fn default_max_hops() -> usize {
    3
}

/// Manual annotations for a scenario.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Annotations {
    /// Scenario ID this applies to
    #[serde(default)]
    pub scenario_id: String,
    /// Critical files that MUST be found
    #[serde(default)]
    pub critical_files: Vec<String>,
    /// Critical symbols that MUST be found
    #[serde(default)]
    pub critical_symbols: Vec<String>,
    /// File:line locations that are key entry points
    #[serde(default)]
    pub key_locations: Vec<String>,
    /// Expected exploration paths
    #[serde(default)]
    pub exploration_paths: Vec<ExplorationPath>,
    /// Notes about why these are critical
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Annotations {
    /// Create empty annotations.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Load annotations from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let annotations: Annotations = serde_json::from_str(&content)?;
        Ok(annotations)
    }

    /// Try to load annotations, returning empty if not found.
    pub fn load_optional(path: &Path) -> Self {
        Self::load(path).unwrap_or_default()
    }

    /// Save annotations to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Check if a file is in critical files.
    pub fn is_critical_file(&self, file: &str) -> bool {
        self.critical_files.iter().any(|f| {
            file == f
                || file.ends_with(f)
                || glob::Pattern::new(f).is_ok_and(|p| p.matches(file))
        })
    }

    /// Check if a symbol is in critical symbols.
    pub fn is_critical_symbol(&self, symbol: &str) -> bool {
        self.critical_symbols.contains(&symbol.to_string())
    }

    /// Get all critical items (files + symbols).
    pub fn all_critical(&self) -> Vec<&str> {
        self.critical_files
            .iter()
            .chain(self.critical_symbols.iter())
            .map(|s| s.as_str())
            .collect()
    }

    /// Check if annotations have any content.
    pub fn is_empty(&self) -> bool {
        self.critical_files.is_empty()
            && self.critical_symbols.is_empty()
            && self.key_locations.is_empty()
            && self.exploration_paths.is_empty()
    }

    /// Merge with another annotations set (other takes precedence).
    pub fn merge(&mut self, other: &Annotations) {
        for file in &other.critical_files {
            if !self.critical_files.contains(file) {
                self.critical_files.push(file.clone());
            }
        }
        for symbol in &other.critical_symbols {
            if !self.critical_symbols.contains(symbol) {
                self.critical_symbols.push(symbol.clone());
            }
        }
        for location in &other.key_locations {
            if !self.key_locations.contains(location) {
                self.key_locations.push(location.clone());
            }
        }
        self.exploration_paths.extend(other.exploration_paths.clone());
        self.notes.extend(other.notes.clone());
    }
}

/// Load annotations for a specific scenario from an annotations directory.
pub fn load_scenario_annotations(annotations_dir: &Path, scenario_id: &str) -> Annotations {
    // Try scenario-specific file first
    let specific_path = annotations_dir.join(format!("{}.json", scenario_id));
    if specific_path.exists()
        && let Ok(ann) = Annotations::load(&specific_path) {
            return ann;
        }

    // Fall back to default.json
    let default_path = annotations_dir.join("default.json");
    Annotations::load_optional(&default_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_empty_annotations() {
        let ann = Annotations::empty();
        assert!(ann.is_empty());
        assert!(ann.critical_files.is_empty());
    }

    #[test]
    fn test_is_critical_file() {
        let ann = Annotations {
            critical_files: vec!["src/commands.rs".to_string(), "**/keymap.rs".to_string()],
            ..Default::default()
        };

        assert!(ann.is_critical_file("src/commands.rs"));
        assert!(ann.is_critical_file("crates/gpui/src/keymap.rs"));
        assert!(!ann.is_critical_file("src/other.rs"));
    }

    #[test]
    fn test_is_critical_symbol() {
        let ann = Annotations {
            critical_symbols: vec!["Command".to_string(), "execute".to_string()],
            ..Default::default()
        };

        assert!(ann.is_critical_symbol("Command"));
        assert!(ann.is_critical_symbol("execute"));
        assert!(!ann.is_critical_symbol("Other"));
    }

    #[test]
    fn test_save_and_load() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test.json");

        let ann = Annotations {
            scenario_id: "test-scenario".to_string(),
            critical_files: vec!["src/main.rs".to_string()],
            critical_symbols: vec!["main".to_string()],
            key_locations: vec!["src/main.rs:1".to_string()],
            exploration_paths: vec![ExplorationPath {
                start: "main".to_string(),
                through: vec!["run".to_string()],
                target: "execute".to_string(),
                max_hops: 3,
            }],
            notes: vec!["Test note".to_string()],
        };

        ann.save(&path).unwrap();
        let loaded = Annotations::load(&path).unwrap();

        assert_eq!(loaded.scenario_id, "test-scenario");
        assert_eq!(loaded.critical_files.len(), 1);
        assert_eq!(loaded.exploration_paths.len(), 1);
    }

    #[test]
    fn test_merge() {
        let mut ann1 = Annotations {
            critical_files: vec!["a.rs".to_string()],
            critical_symbols: vec!["A".to_string()],
            ..Default::default()
        };

        let ann2 = Annotations {
            critical_files: vec!["b.rs".to_string()],
            critical_symbols: vec!["B".to_string()],
            ..Default::default()
        };

        ann1.merge(&ann2);

        assert_eq!(ann1.critical_files.len(), 2);
        assert_eq!(ann1.critical_symbols.len(), 2);
    }

    #[test]
    fn test_load_optional_missing() {
        let ann = Annotations::load_optional(Path::new("/nonexistent/path.json"));
        assert!(ann.is_empty());
    }

    #[test]
    fn test_exploration_path() {
        let path = ExplorationPath {
            start: "main".to_string(),
            through: vec!["run".to_string(), "setup".to_string()],
            target: "execute".to_string(),
            max_hops: 4,
        };

        let json = serde_json::to_string(&path).unwrap();
        let parsed: ExplorationPath = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.start, "main");
        assert_eq!(parsed.through.len(), 2);
        assert_eq!(parsed.target, "execute");
        assert_eq!(parsed.max_hops, 4);
    }
}
