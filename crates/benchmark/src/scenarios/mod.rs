//! Scenario definition and execution.
//!
//! Scenarios are TOML-defined multi-step exploration tasks that test
//! CCEngram's ability to navigate and discover code in large codebases.

mod definition;
pub mod runner;

use std::path::Path;

pub use definition::{
  ComprehensionQuestion, Expected, LlmJudgeConfig, PreviousStepResults, Scenario, Step, SuccessCriteria, TaskIntent,
  TaskRequirements, TaskRequirementsResult,
};
pub use runner::{ScenarioResult, ScenarioRunner, run_scenarios_parallel};
use tracing::info;

use crate::Result;

/// Load all scenarios from a directory.
pub async fn load_scenarios_from_dir(dir: &Path) -> Result<Vec<Scenario>> {
  let mut scenarios = Vec::new();

  if !dir.exists() {
    return Ok(scenarios);
  }

  let mut entries = tokio::fs::read_dir(dir).await?;
  while let Some(entry) = entries.next_entry().await? {
    let path = entry.path();
    if path.extension().is_some_and(|e| e == "toml") {
      info!("Loading scenario: {}", path.display());
      let scenario = Scenario::load(&path).await?;
      scenarios.push(scenario);
    }
  }

  // Sort by ID for consistent ordering
  scenarios.sort_by(|a, b| a.metadata.id.cmp(&b.metadata.id));
  Ok(scenarios)
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
}
