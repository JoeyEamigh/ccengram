//! Markdown report generation.

use crate::metrics::MetricTargets;
use crate::scenarios::ScenarioResult;
use crate::Result;
use chrono::Utc;
use std::fmt::Write as _;
use std::path::Path;

/// Markdown report generator.
pub struct MarkdownReport {
    content: String,
}

impl MarkdownReport {
    /// Create a markdown report from scenario results.
    pub fn from_results(results: &[ScenarioResult]) -> Self {
        let mut content = String::new();

        Self::write_header(&mut content, results);
        Self::write_summary(&mut content, results);
        Self::write_performance_table(&mut content, results);
        Self::write_accuracy_table(&mut content, results);
        Self::write_scenario_details(&mut content, results);

        Self { content }
    }

    fn write_header(out: &mut String, results: &[ScenarioResult]) {
        let _ = writeln!(out, "# CCEngram Benchmark Report");
        let _ = writeln!(out);
        let _ = writeln!(out, "**Generated:** {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        let _ = writeln!(out, "**Version:** {}", env!("CARGO_PKG_VERSION"));
        let _ = writeln!(out, "**Scenarios:** {}", results.len());
        let _ = writeln!(out);
    }

    fn write_summary(out: &mut String, results: &[ScenarioResult]) {
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.len() - passed;
        let pass_rate = if results.is_empty() {
            0.0
        } else {
            passed as f64 / results.len() as f64 * 100.0
        };

        let total_time: u64 = results.iter().map(|r| r.total_duration_ms).sum();

        let _ = writeln!(out, "## Summary");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Metric | Value |");
        let _ = writeln!(out, "|--------|-------|");
        let _ = writeln!(out, "| **Pass Rate** | {:.1}% ({}/{}) |", pass_rate, passed, results.len());
        let _ = writeln!(out, "| **Passed** | {} |", passed);
        let _ = writeln!(out, "| **Failed** | {} |", failed);
        let _ = writeln!(out, "| **Total Time** | {:.2}s |", total_time as f64 / 1000.0);
        let _ = writeln!(out);

        // Pass/fail emoji summary
        let _ = writeln!(out, "### Quick Status");
        let _ = writeln!(out);
        for result in results {
            let emoji = if result.passed { "✅" } else { "❌" };
            let _ = writeln!(out, "- {} {} ({})", emoji, result.scenario_name, result.scenario_id);
        }
        let _ = writeln!(out);
    }

    fn write_performance_table(out: &mut String, results: &[ScenarioResult]) {
        let _ = writeln!(out, "## Performance");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Scenario | Search p50 | Search p95 | Context p50 | Total |");
        let _ = writeln!(out, "|----------|------------|------------|-------------|-------|");

        for result in results {
            let _ = writeln!(
                out,
                "| {} | {}ms | {}ms | {}ms | {:.2}s |",
                result.scenario_id,
                result.performance.search_latency.p50_ms,
                result.performance.search_latency.p95_ms,
                result.performance.context_latency.p50_ms,
                result.total_duration_ms as f64 / 1000.0
            );
        }
        let _ = writeln!(out);
    }

    fn write_accuracy_table(out: &mut String, results: &[ScenarioResult]) {
        let targets = MetricTargets::default();

        let _ = writeln!(out, "## Accuracy");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| Scenario | File Recall | Symbol Recall | MRR | Noise | Steps |"
        );
        let _ = writeln!(out, "|----------|-------------|---------------|-----|-------|-------|");

        for result in results {
            let file_icon = if result.accuracy.file_recall >= targets.file_recall {
                "✅"
            } else {
                "❌"
            };
            let symbol_icon = if result.accuracy.symbol_recall >= targets.symbol_recall {
                "✅"
            } else {
                "❌"
            };
            let mrr_icon = if result.accuracy.mrr >= targets.mrr {
                "✅"
            } else {
                "❌"
            };
            let noise_icon = if result.accuracy.noise_ratio <= targets.noise_ratio {
                "✅"
            } else {
                "❌"
            };
            let steps_icon = result
                .accuracy
                .steps_to_core
                .map(|s| {
                    if s <= targets.max_steps_to_core {
                        "✅"
                    } else {
                        "❌"
                    }
                })
                .unwrap_or("➖");

            let steps_str = result
                .accuracy
                .steps_to_core
                .map(|s: usize| s.to_string())
                .unwrap_or_else(|| "N/A".to_string());

            let _ = writeln!(
                out,
                "| {} | {} {:.0}% | {} {:.0}% | {} {:.2} | {} {:.0}% | {} {} |",
                result.scenario_id,
                file_icon,
                result.accuracy.file_recall * 100.0,
                symbol_icon,
                result.accuracy.symbol_recall * 100.0,
                mrr_icon,
                result.accuracy.mrr,
                noise_icon,
                result.accuracy.noise_ratio * 100.0,
                steps_icon,
                steps_str
            );
        }
        let _ = writeln!(out);

        // Targets legend
        let _ = writeln!(out, "**Targets:** File Recall ≥{:.0}%, Symbol Recall ≥{:.0}%, MRR ≥{:.1}, Noise ≤{:.0}%, Steps ≤{}",
            targets.file_recall * 100.0,
            targets.symbol_recall * 100.0,
            targets.mrr,
            targets.noise_ratio * 100.0,
            targets.max_steps_to_core
        );
        let _ = writeln!(out);
    }

    fn write_scenario_details(out: &mut String, results: &[ScenarioResult]) {
        let _ = writeln!(out, "## Scenario Details");
        let _ = writeln!(out);

        for result in results {
            let status = if result.passed { "✅ PASSED" } else { "❌ FAILED" };
            let _ = writeln!(out, "### {} - {}", result.scenario_id, status);
            let _ = writeln!(out);

            // Errors if any
            if !result.errors.is_empty() {
                let _ = writeln!(out, "**Errors:**");
                for error in &result.errors {
                    let _ = writeln!(out, "- {}", error);
                }
                let _ = writeln!(out);
            }

            // Files found/missed
            if !result.accuracy.files_found.is_empty() || !result.accuracy.files_missed.is_empty() {
                let _ = writeln!(out, "**Files:**");
                if !result.accuracy.files_found.is_empty() {
                    let _ = writeln!(
                        out,
                        "- Found: {}",
                        result.accuracy.files_found.join(", ")
                    );
                }
                if !result.accuracy.files_missed.is_empty() {
                    let _ = writeln!(
                        out,
                        "- Missed: {}",
                        result.accuracy.files_missed.join(", ")
                    );
                }
                let _ = writeln!(out);
            }

            // Symbols found/missed
            if !result.accuracy.symbols_found.is_empty()
                || !result.accuracy.symbols_missed.is_empty()
            {
                let _ = writeln!(out, "**Symbols:**");
                if !result.accuracy.symbols_found.is_empty() {
                    let _ = writeln!(
                        out,
                        "- Found: `{}`",
                        result.accuracy.symbols_found.join("`, `")
                    );
                }
                if !result.accuracy.symbols_missed.is_empty() {
                    let _ = writeln!(
                        out,
                        "- Missed: `{}`",
                        result.accuracy.symbols_missed.join("`, `")
                    );
                }
                let _ = writeln!(out);
            }

            // Step breakdown
            if !result.steps.is_empty() {
                let _ = writeln!(out, "**Steps:**");
                let _ = writeln!(out);
                let _ = writeln!(out, "| # | Query | Results | Noise | Latency |");
                let _ = writeln!(out, "|---|-------|---------|-------|---------|");

                for step in &result.steps {
                    let query_preview = if step.query.len() > 40 {
                        format!("{}...", &step.query[..40])
                    } else {
                        step.query.clone()
                    };

                    let _ = writeln!(
                        out,
                        "| {} | {} | {} | {:.0}% | {}ms |",
                        step.step_index + 1,
                        query_preview,
                        step.result_count,
                        step.noise_ratio * 100.0,
                        step.latency_ms
                    );
                }
                let _ = writeln!(out);
            }
        }
    }

    /// Save to a markdown file.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, &self.content)?;
        Ok(())
    }

    /// Get the markdown content.
    pub fn content(&self) -> &str {
        &self.content
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{AccuracyMetrics, LatencyStats, PerformanceMetrics};
    use crate::scenarios::StepResult;
    use tempfile::TempDir;

    fn sample_result(id: &str, passed: bool) -> ScenarioResult {
        ScenarioResult {
            scenario_id: id.to_string(),
            scenario_name: format!("Test {}", id),
            passed,
            performance: PerformanceMetrics {
                search_latency: LatencyStats {
                    min_ms: 50,
                    max_ms: 200,
                    mean_ms: 100,
                    p50_ms: 90,
                    p95_ms: 180,
                    p99_ms: 195,
                    count: 5,
                },
                context_latency: LatencyStats::default(),
                total_time_ms: 500,
                steps: vec![],
                peak_memory_bytes: None,
                avg_cpu_percent: None,
            },
            accuracy: AccuracyMetrics {
                file_recall: 0.8,
                symbol_recall: 0.75,
                steps_to_core: Some(2),
                mrr: 0.6,
                noise_ratio: 0.15,
                top3_noise: 0.0,
                hint_utility: 0.7,
                suggestion_quality: 0.5,
                files_found: vec!["found.rs".to_string()],
                files_missed: vec!["missed.rs".to_string()],
                symbols_found: vec!["Found".to_string()],
                symbols_missed: vec!["Missed".to_string()],
            },
            steps: vec![StepResult {
                step_index: 0,
                query: "test query".to_string(),
                result_count: 5,
                noise_ratio: 0.2,
                result_ids: vec![],
                files_found: vec![],
                symbols_found: vec![],
                latency_ms: 100,
                passed: true,
            }],
            errors: vec![],
            total_duration_ms: 500,
        }
    }

    #[test]
    fn test_markdown_generation() {
        let results = vec![sample_result("test-1", true), sample_result("test-2", false)];

        let report = MarkdownReport::from_results(&results);
        let content = report.content();

        assert!(content.contains("# CCEngram Benchmark Report"));
        assert!(content.contains("## Summary"));
        assert!(content.contains("## Performance"));
        assert!(content.contains("## Accuracy"));
        assert!(content.contains("test-1"));
        assert!(content.contains("✅"));
        assert!(content.contains("❌"));
    }

    #[test]
    fn test_save_markdown() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("report.md");

        let results = vec![sample_result("test-1", true)];
        let report = MarkdownReport::from_results(&results);

        report.save(&path).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("# CCEngram Benchmark Report"));
    }

    #[test]
    fn test_empty_results() {
        let report = MarkdownReport::from_results(&[]);
        let content = report.content();

        assert!(content.contains("# CCEngram Benchmark Report"));
        assert!(content.contains("0%"));
    }
}
