//! Scenario definition and execution.
//!
//! Scenarios are TOML-defined multi-step exploration tasks that test
//! CCEngram's ability to navigate and discover code in large codebases.

mod definition;
mod runner;

pub use definition::{
    Expected, Scenario, ScenarioMetadata, Step, SuccessCriteria, Task,
};
pub use runner::{run_scenarios_parallel, ScenarioResult, ScenarioRunner, StepResult};

use crate::Result;
use std::path::Path;
use tracing::info;

/// Load all scenarios from a directory.
pub fn load_scenarios_from_dir(dir: &Path) -> Result<Vec<Scenario>> {
    let mut scenarios = Vec::new();

    if !dir.exists() {
        return Ok(scenarios);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "toml") {
            info!("Loading scenario: {}", path.display());
            let scenario = Scenario::load(&path)?;
            scenarios.push(scenario);
        }
    }

    // Sort by ID for consistent ordering
    scenarios.sort_by(|a, b| a.metadata.id.cmp(&b.metadata.id));
    Ok(scenarios)
}

/// Load built-in scenarios from the crate's scenarios directory.
pub fn load_builtin_scenarios() -> Vec<Scenario> {
    // In a real implementation, these would be embedded or loaded from a known path
    // For now, return empty - scenarios are loaded from the scenarios/ directory
    Vec::new()
}

/// Filter scenarios by pattern (supports glob-style wildcards).
pub fn filter_scenarios<'a>(scenarios: &'a [Scenario], pattern: &str) -> Vec<&'a Scenario> {
    let pattern = glob::Pattern::new(pattern).ok();

    scenarios
        .iter()
        .filter(|s| {
            pattern
                .as_ref()
                .is_none_or(|p| p.matches(&s.metadata.id) || p.matches(&s.metadata.name))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_scenarios_wildcard() {
        let scenarios = vec![
            Scenario::new_test("zed-commands", "Zed Commands"),
            Scenario::new_test("zed-lsp", "Zed LSP"),
            Scenario::new_test("vscode-extensions", "VSCode Extensions"),
        ];

        let filtered = filter_scenarios(&scenarios, "zed*");
        assert_eq!(filtered.len(), 2);

        let filtered = filter_scenarios(&scenarios, "*extensions*");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_scenarios_all() {
        let scenarios = vec![
            Scenario::new_test("test1", "Test 1"),
            Scenario::new_test("test2", "Test 2"),
        ];

        let filtered = filter_scenarios(&scenarios, "*");
        assert_eq!(filtered.len(), 2);
    }
}
