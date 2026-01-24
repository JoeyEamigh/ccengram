//! JSON report format for benchmark results.

use crate::scenarios::ScenarioResult;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete benchmark report in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Per-scenario results
    pub scenarios: Vec<ScenarioResult>,
}

/// Report metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report generation timestamp
    pub timestamp: DateTime<Utc>,
    /// CCEngram version
    pub version: String,
    /// Git commit hash (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    /// Hostname
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// Total scenarios run
    pub total_scenarios: usize,
}

/// Summary statistics across all scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Number of scenarios that passed
    pub passed: usize,
    /// Number of scenarios that failed
    pub failed: usize,
    /// Pass rate (0.0-1.0)
    pub pass_rate: f64,
    /// Aggregate performance metrics
    pub performance: AggregatePerformance,
    /// Aggregate accuracy metrics
    pub accuracy: AggregateAccuracy,
    /// Total execution time in milliseconds
    pub total_time_ms: u64,
}

/// Aggregate performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatePerformance {
    /// Average search latency p50 in ms
    pub avg_search_latency_p50_ms: f64,
    /// Average search latency p95 in ms
    pub avg_search_latency_p95_ms: f64,
    /// Average context latency p50 in ms
    pub avg_context_latency_p50_ms: f64,
    /// Total queries executed
    pub total_queries: usize,
}

/// Aggregate accuracy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateAccuracy {
    /// Average file recall
    pub avg_file_recall: f64,
    /// Average symbol recall
    pub avg_symbol_recall: f64,
    /// Average MRR
    pub avg_mrr: f64,
    /// Average noise ratio
    pub avg_noise_ratio: f64,

    // === Exploration metrics ===

    /// Average convergence rate
    pub avg_convergence_rate: f64,
    /// Average hint utility
    pub avg_hint_utility: f64,
    /// Average suggestion quality
    pub avg_suggestion_quality: f64,
    /// Average context bloat
    pub avg_context_bloat: f64,
    /// Average dead end ratio
    pub avg_dead_end_ratio: f64,
}

impl BenchmarkReport {
    /// Create a report from scenario results.
    pub fn from_results(results: &[ScenarioResult]) -> Self {
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.len() - passed;
        let pass_rate = if results.is_empty() {
            0.0
        } else {
            passed as f64 / results.len() as f64
        };

        let total_time_ms: u64 = results.iter().map(|r| r.total_duration_ms).sum();

        let performance = Self::aggregate_performance(results);
        let accuracy = Self::aggregate_accuracy(results);

        Self {
            metadata: ReportMetadata {
                timestamp: Utc::now(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                git_commit: Self::get_git_commit(),
                hostname: hostname::get().ok().and_then(|h| h.into_string().ok()),
                total_scenarios: results.len(),
            },
            summary: ReportSummary {
                passed,
                failed,
                pass_rate,
                performance,
                accuracy,
                total_time_ms,
            },
            scenarios: results.to_vec(),
        }
    }

    fn aggregate_performance(results: &[ScenarioResult]) -> AggregatePerformance {
        if results.is_empty() {
            return AggregatePerformance {
                avg_search_latency_p50_ms: 0.0,
                avg_search_latency_p95_ms: 0.0,
                avg_context_latency_p50_ms: 0.0,
                total_queries: 0,
            };
        }

        let search_p50: f64 = results
            .iter()
            .map(|r| r.performance.search_latency.p50_ms as f64)
            .sum::<f64>()
            / results.len() as f64;

        let search_p95: f64 = results
            .iter()
            .map(|r| r.performance.search_latency.p95_ms as f64)
            .sum::<f64>()
            / results.len() as f64;

        let context_p50: f64 = results
            .iter()
            .map(|r| r.performance.context_latency.p50_ms as f64)
            .sum::<f64>()
            / results.len() as f64;

        let total_queries: usize = results
            .iter()
            .map(|r| r.performance.search_latency.count)
            .sum();

        AggregatePerformance {
            avg_search_latency_p50_ms: search_p50,
            avg_search_latency_p95_ms: search_p95,
            avg_context_latency_p50_ms: context_p50,
            total_queries,
        }
    }

    fn aggregate_accuracy(results: &[ScenarioResult]) -> AggregateAccuracy {
        if results.is_empty() {
            return AggregateAccuracy {
                avg_file_recall: 0.0,
                avg_symbol_recall: 0.0,
                avg_mrr: 0.0,
                avg_noise_ratio: 0.0,
                avg_convergence_rate: 0.0,
                avg_hint_utility: 0.0,
                avg_suggestion_quality: 0.0,
                avg_context_bloat: 0.0,
                avg_dead_end_ratio: 0.0,
            };
        }

        let n = results.len() as f64;

        AggregateAccuracy {
            avg_file_recall: results.iter().map(|r| r.accuracy.file_recall).sum::<f64>() / n,
            avg_symbol_recall: results.iter().map(|r| r.accuracy.symbol_recall).sum::<f64>() / n,
            avg_mrr: results.iter().map(|r| r.accuracy.mrr).sum::<f64>() / n,
            avg_noise_ratio: results.iter().map(|r| r.accuracy.noise_ratio).sum::<f64>() / n,
            avg_convergence_rate: results.iter().map(|r| r.accuracy.convergence_rate).sum::<f64>() / n,
            avg_hint_utility: results.iter().map(|r| r.accuracy.hint_utility).sum::<f64>() / n,
            avg_suggestion_quality: results.iter().map(|r| r.accuracy.suggestion_quality).sum::<f64>() / n,
            avg_context_bloat: results.iter().map(|r| r.accuracy.context_bloat).sum::<f64>() / n,
            avg_dead_end_ratio: results.iter().map(|r| r.accuracy.dead_end_ratio).sum::<f64>() / n,
        }
    }

    fn get_git_commit() -> Option<String> {
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Save report to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load report from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let report = serde_json::from_str(&json)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{AccuracyMetrics, LatencyStats, PerformanceMetrics};
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
                context_latency: LatencyStats {
                    min_ms: 20,
                    max_ms: 50,
                    mean_ms: 30,
                    p50_ms: 28,
                    p95_ms: 45,
                    p99_ms: 48,
                    count: 3,
                },
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
                convergence_rate: 0.85,
                avg_info_gain: 0.4,
                context_bloat: 0.1,
                navigation_efficiency: 0.7,
                dead_end_ratio: 0.1,
                files_found: vec!["found.rs".to_string()],
                files_missed: vec!["missed.rs".to_string()],
                symbols_found: vec!["Found".to_string()],
                symbols_missed: vec!["Missed".to_string()],
            },
            steps: vec![],
            errors: vec![],
            total_duration_ms: 500,
        }
    }

    #[test]
    fn test_from_results() {
        let results = vec![
            sample_result("test-1", true),
            sample_result("test-2", true),
            sample_result("test-3", false),
        ];

        let report = BenchmarkReport::from_results(&results);

        assert_eq!(report.summary.passed, 2);
        assert_eq!(report.summary.failed, 1);
        assert!((report.summary.pass_rate - 2.0 / 3.0).abs() < f64::EPSILON);
        assert_eq!(report.scenarios.len(), 3);
    }

    #[test]
    fn test_aggregate_performance() {
        let results = vec![sample_result("test-1", true), sample_result("test-2", true)];

        let report = BenchmarkReport::from_results(&results);

        assert_eq!(report.summary.performance.avg_search_latency_p50_ms, 90.0);
        assert_eq!(report.summary.performance.total_queries, 10);
    }

    #[test]
    fn test_aggregate_accuracy() {
        let results = vec![sample_result("test-1", true), sample_result("test-2", true)];

        let report = BenchmarkReport::from_results(&results);

        assert!((report.summary.accuracy.avg_file_recall - 0.8).abs() < f64::EPSILON);
        assert!((report.summary.accuracy.avg_noise_ratio - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_save_and_load() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("report.json");

        let results = vec![sample_result("test-1", true)];
        let report = BenchmarkReport::from_results(&results);

        report.save(&path).unwrap();
        let loaded = BenchmarkReport::load(&path).unwrap();

        assert_eq!(loaded.summary.passed, 1);
        assert_eq!(loaded.scenarios.len(), 1);
    }

    #[test]
    fn test_empty_results() {
        let report = BenchmarkReport::from_results(&[]);

        assert_eq!(report.summary.passed, 0);
        assert_eq!(report.summary.failed, 0);
        assert!((report.summary.pass_rate - 0.0).abs() < f64::EPSILON);
    }
}
