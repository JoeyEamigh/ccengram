//! Report generation for benchmark results.
//!
//! - JSON: Machine-readable format for CI and comparison
//! - Markdown: Human-readable summary
//! - Comparison: Regression detection between runs

mod comparison;
mod json;
mod markdown;

pub use comparison::{ComparisonReport, Regression};
pub use json::BenchmarkReport;
pub use markdown::MarkdownReport;

use crate::scenarios::ScenarioResult;
use std::path::Path;

/// Generate all report formats for benchmark results.
pub fn generate_reports(
    results: &[ScenarioResult],
    output_dir: &Path,
    run_name: Option<&str>,
) -> crate::Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let run_name = run_name.unwrap_or("benchmark");

    // Generate JSON report
    let json_path = output_dir.join(format!("{}.json", run_name));
    let report = BenchmarkReport::from_results(results);
    report.save(&json_path)?;

    // Generate Markdown report
    let md_path = output_dir.join(format!("{}.md", run_name));
    let md_report = MarkdownReport::from_results(results);
    md_report.save(&md_path)?;

    Ok(())
}
