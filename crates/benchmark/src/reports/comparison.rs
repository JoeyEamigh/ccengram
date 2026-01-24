//! Comparison and regression detection between benchmark runs.

use super::json::BenchmarkReport;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A detected regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    /// Scenario ID
    pub scenario_id: String,
    /// Metric name
    pub metric: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Change percentage
    pub change_percent: f64,
    /// Whether this is a degradation (negative change)
    pub is_degradation: bool,
}

/// Comparison report between two benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Baseline report timestamp
    pub baseline_timestamp: String,
    /// Current report timestamp
    pub current_timestamp: String,
    /// Detected regressions
    pub regressions: Vec<Regression>,
    /// Detected improvements
    pub improvements: Vec<Regression>,
    /// Summary statistics
    pub summary: ComparisonSummary,
}

/// Summary of comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Total scenarios compared
    pub total_scenarios: usize,
    /// Scenarios with regressions
    pub scenarios_regressed: usize,
    /// Scenarios with improvements
    pub scenarios_improved: usize,
    /// Scenarios unchanged
    pub scenarios_unchanged: usize,
    /// Whether the comparison passes (no significant regressions)
    pub passes: bool,
}

impl ComparisonReport {
    /// Compare two reports with a given threshold.
    pub fn compare(baseline: &BenchmarkReport, current: &BenchmarkReport, threshold_percent: f64) -> Self {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Compare scenarios that exist in both reports
        for current_scenario in &current.scenarios {
            if let Some(baseline_scenario) = baseline
                .scenarios
                .iter()
                .find(|s| s.scenario_id == current_scenario.scenario_id)
            {
                // Compare file recall
                Self::compare_metric(
                    &current_scenario.scenario_id,
                    "file_recall",
                    baseline_scenario.accuracy.file_recall,
                    current_scenario.accuracy.file_recall,
                    threshold_percent,
                    true, // higher is better
                    &mut regressions,
                    &mut improvements,
                );

                // Compare symbol recall
                Self::compare_metric(
                    &current_scenario.scenario_id,
                    "symbol_recall",
                    baseline_scenario.accuracy.symbol_recall,
                    current_scenario.accuracy.symbol_recall,
                    threshold_percent,
                    true,
                    &mut regressions,
                    &mut improvements,
                );

                // Compare MRR
                Self::compare_metric(
                    &current_scenario.scenario_id,
                    "mrr",
                    baseline_scenario.accuracy.mrr,
                    current_scenario.accuracy.mrr,
                    threshold_percent,
                    true,
                    &mut regressions,
                    &mut improvements,
                );

                // Compare noise ratio (lower is better)
                Self::compare_metric(
                    &current_scenario.scenario_id,
                    "noise_ratio",
                    baseline_scenario.accuracy.noise_ratio,
                    current_scenario.accuracy.noise_ratio,
                    threshold_percent,
                    false,
                    &mut regressions,
                    &mut improvements,
                );

                // Compare search latency p50 (lower is better)
                Self::compare_metric(
                    &current_scenario.scenario_id,
                    "search_latency_p50",
                    baseline_scenario.performance.search_latency.p50_ms as f64,
                    current_scenario.performance.search_latency.p50_ms as f64,
                    threshold_percent,
                    false,
                    &mut regressions,
                    &mut improvements,
                );

                // Compare search latency p95 (lower is better)
                Self::compare_metric(
                    &current_scenario.scenario_id,
                    "search_latency_p95",
                    baseline_scenario.performance.search_latency.p95_ms as f64,
                    current_scenario.performance.search_latency.p95_ms as f64,
                    threshold_percent,
                    false,
                    &mut regressions,
                    &mut improvements,
                );
            }
        }

        // Build summary
        let scenario_ids: std::collections::HashSet<_> = regressions
            .iter()
            .chain(improvements.iter())
            .map(|r| r.scenario_id.clone())
            .collect();

        let regressed_ids: std::collections::HashSet<_> =
            regressions.iter().map(|r| r.scenario_id.clone()).collect();
        let improved_ids: std::collections::HashSet<_> =
            improvements.iter().map(|r| r.scenario_id.clone()).collect();

        let total = current.scenarios.len();
        let regressed = regressed_ids.len();
        let improved = improved_ids.len();
        let unchanged = total.saturating_sub(scenario_ids.len());

        Self {
            baseline_timestamp: baseline.metadata.timestamp.to_rfc3339(),
            current_timestamp: current.metadata.timestamp.to_rfc3339(),
            regressions,
            improvements,
            summary: ComparisonSummary {
                total_scenarios: total,
                scenarios_regressed: regressed,
                scenarios_improved: improved,
                scenarios_unchanged: unchanged,
                passes: regressed == 0,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_metric(
        scenario_id: &str,
        metric: &str,
        baseline: f64,
        current: f64,
        threshold: f64,
        higher_is_better: bool,
        regressions: &mut Vec<Regression>,
        improvements: &mut Vec<Regression>,
    ) {
        if baseline == 0.0 && current == 0.0 {
            return;
        }

        let change_percent = if baseline != 0.0 {
            ((current - baseline) / baseline) * 100.0
        } else if current > 0.0 { 100.0 } else { 0.0 };

        let abs_change = change_percent.abs();
        if abs_change < threshold {
            return;
        }

        let is_degradation = if higher_is_better {
            change_percent < 0.0
        } else {
            change_percent > 0.0
        };

        let regression = Regression {
            scenario_id: scenario_id.to_string(),
            metric: metric.to_string(),
            baseline,
            current,
            change_percent,
            is_degradation,
        };

        if is_degradation {
            regressions.push(regression);
        } else {
            improvements.push(regression);
        }
    }

    /// Load comparison between two report files.
    pub fn from_files(baseline_path: &Path, current_path: &Path, threshold: f64) -> Result<Self> {
        let baseline = BenchmarkReport::load(baseline_path)?;
        let current = BenchmarkReport::load(current_path)?;
        Ok(Self::compare(&baseline, &current, threshold))
    }

    /// Save comparison to JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Generate markdown summary.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str("# Benchmark Comparison\n\n");
        out.push_str(&format!("**Baseline:** {}\n", self.baseline_timestamp));
        out.push_str(&format!("**Current:** {}\n\n", self.current_timestamp));

        // Summary
        let status = if self.summary.passes { "✅ PASS" } else { "❌ FAIL" };
        out.push_str(&format!("## Summary: {}\n\n", status));
        out.push_str("| Metric | Value |\n");
        out.push_str("|--------|-------|\n");
        out.push_str(&format!("| Scenarios | {} |\n", self.summary.total_scenarios));
        out.push_str(&format!("| Regressed | {} |\n", self.summary.scenarios_regressed));
        out.push_str(&format!("| Improved | {} |\n", self.summary.scenarios_improved));
        out.push_str(&format!("| Unchanged | {} |\n\n", self.summary.scenarios_unchanged));

        // Regressions
        if !self.regressions.is_empty() {
            out.push_str("## Regressions ❌\n\n");
            out.push_str("| Scenario | Metric | Baseline | Current | Change |\n");
            out.push_str("|----------|--------|----------|---------|--------|\n");
            for r in &self.regressions {
                out.push_str(&format!(
                    "| {} | {} | {:.2} | {:.2} | {:+.1}% |\n",
                    r.scenario_id, r.metric, r.baseline, r.current, r.change_percent
                ));
            }
            out.push('\n');
        }

        // Improvements
        if !self.improvements.is_empty() {
            out.push_str("## Improvements ✅\n\n");
            out.push_str("| Scenario | Metric | Baseline | Current | Change |\n");
            out.push_str("|----------|--------|----------|---------|--------|\n");
            for r in &self.improvements {
                out.push_str(&format!(
                    "| {} | {} | {:.2} | {:.2} | {:+.1}% |\n",
                    r.scenario_id, r.metric, r.baseline, r.current, r.change_percent
                ));
            }
            out.push('\n');
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{AccuracyMetrics, LatencyStats, PerformanceMetrics};
    use crate::scenarios::ScenarioResult;
    use chrono::Utc;

    fn make_report(file_recall: f64, noise_ratio: f64, latency_p50: u64) -> BenchmarkReport {
        use crate::reports::json::{ReportMetadata, ReportSummary, AggregatePerformance, AggregateAccuracy};

        BenchmarkReport {
            metadata: ReportMetadata {
                timestamp: Utc::now(),
                version: "0.1.0".to_string(),
                git_commit: None,
                hostname: None,
                total_scenarios: 1,
            },
            summary: ReportSummary {
                passed: 1,
                failed: 0,
                pass_rate: 1.0,
                performance: AggregatePerformance {
                    avg_search_latency_p50_ms: latency_p50 as f64,
                    avg_search_latency_p95_ms: latency_p50 as f64 * 2.0,
                    avg_context_latency_p50_ms: 20.0,
                    total_queries: 5,
                },
                accuracy: AggregateAccuracy {
                    avg_file_recall: file_recall,
                    avg_symbol_recall: 0.75,
                    avg_mrr: 0.6,
                    avg_noise_ratio: noise_ratio,
                    avg_convergence_rate: 0.85,
                    avg_hint_utility: 0.7,
                    avg_suggestion_quality: 0.6,
                    avg_context_bloat: 0.15,
                    avg_dead_end_ratio: 0.1,
                },
                total_time_ms: 500,
            },
            scenarios: vec![ScenarioResult {
                scenario_id: "test-1".to_string(),
                scenario_name: "Test 1".to_string(),
                passed: true,
                performance: PerformanceMetrics {
                    search_latency: LatencyStats {
                        min_ms: 50,
                        max_ms: 200,
                        mean_ms: 100,
                        p50_ms: latency_p50,
                        p95_ms: latency_p50 * 2,
                        p99_ms: latency_p50 * 3,
                        count: 5,
                    },
                    context_latency: LatencyStats::default(),
                    total_time_ms: 500,
                    steps: vec![],
                    peak_memory_bytes: None,
                    avg_cpu_percent: None,
                },
                accuracy: AccuracyMetrics {
                    file_recall,
                    symbol_recall: 0.75,
                    steps_to_core: Some(2),
                    mrr: 0.6,
                    noise_ratio,
                    top3_noise: 0.0,
                    hint_utility: 0.7,
                    suggestion_quality: 0.5,
                    convergence_rate: 0.85,
                    avg_info_gain: 0.4,
                    context_bloat: 0.1,
                    navigation_efficiency: 0.7,
                    dead_end_ratio: 0.1,
                    files_found: vec![],
                    files_missed: vec![],
                    symbols_found: vec![],
                    symbols_missed: vec![],
                },
                steps: vec![],
                errors: vec![],
                total_duration_ms: 500,
            }],
        }
    }

    #[test]
    fn test_no_regression() {
        let baseline = make_report(0.8, 0.2, 100);
        let current = make_report(0.8, 0.2, 100);

        let comparison = ComparisonReport::compare(&baseline, &current, 10.0);

        assert!(comparison.regressions.is_empty());
        assert!(comparison.improvements.is_empty());
        assert!(comparison.summary.passes);
    }

    #[test]
    fn test_regression_detected() {
        let baseline = make_report(0.8, 0.2, 100);
        let current = make_report(0.6, 0.4, 200); // Worse on all metrics

        let comparison = ComparisonReport::compare(&baseline, &current, 10.0);

        assert!(!comparison.regressions.is_empty());
        assert!(!comparison.summary.passes);

        // Should detect regressions in file_recall, noise_ratio, and latency
        let metrics: Vec<_> = comparison.regressions.iter().map(|r| r.metric.as_str()).collect();
        assert!(metrics.contains(&"file_recall"));
        assert!(metrics.contains(&"noise_ratio"));
        assert!(metrics.contains(&"search_latency_p50"));
    }

    #[test]
    fn test_improvement_detected() {
        let baseline = make_report(0.6, 0.4, 200);
        let current = make_report(0.8, 0.2, 100); // Better on all metrics

        let comparison = ComparisonReport::compare(&baseline, &current, 10.0);

        assert!(!comparison.improvements.is_empty());
        assert!(comparison.summary.passes);
    }

    #[test]
    fn test_threshold_filtering() {
        let baseline = make_report(0.80, 0.20, 100);
        let current = make_report(0.78, 0.22, 105); // 2.5% worse, 10% worse, 5% worse

        // With 15% threshold, nothing should trigger
        let comparison = ComparisonReport::compare(&baseline, &current, 15.0);
        assert!(comparison.regressions.is_empty());

        // With 5% threshold, noise_ratio should trigger
        let comparison = ComparisonReport::compare(&baseline, &current, 5.0);
        assert!(comparison.regressions.iter().any(|r| r.metric == "noise_ratio"));
    }

    #[test]
    fn test_markdown_output() {
        let baseline = make_report(0.8, 0.2, 100);
        let current = make_report(0.6, 0.4, 200);

        let comparison = ComparisonReport::compare(&baseline, &current, 10.0);
        let md = comparison.to_markdown();

        assert!(md.contains("# Benchmark Comparison"));
        assert!(md.contains("FAIL"));
        assert!(md.contains("Regressions"));
    }
}
