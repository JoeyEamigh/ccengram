//! Indexing performance benchmarking.
//!
//! Measures initial indexing performance for repositories, including
//! scan time, chunking throughput, embedding generation, and resource usage.

use crate::metrics::{IndexingMetrics, ResourceMonitor};
use crate::repos::{prepare_repo, TargetRepo};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::{debug, info};

/// Result of a single indexing benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingBenchResult {
    /// Repository name
    pub repo: String,
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Whether this was a cold start (no cache)
    pub cold_start: bool,
    /// Detailed indexing metrics
    pub metrics: IndexingMetrics,
    /// Files scanned
    pub files_scanned: usize,
    /// Files indexed
    pub files_indexed: usize,
    /// Bytes processed
    pub bytes_processed: u64,
}

/// Aggregate statistics across multiple indexing runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingSummary {
    /// Repository name
    pub repo: String,
    /// Number of iterations
    pub iterations: usize,
    /// Average wall time in milliseconds
    pub avg_wall_time_ms: u64,
    /// Median wall time in milliseconds
    pub p50_wall_time_ms: u64,
    /// 95th percentile wall time in milliseconds
    pub p95_wall_time_ms: u64,
    /// Average chunks per second
    pub avg_chunks_per_sec: f64,
    /// Average embeddings per second
    pub avg_embeddings_per_sec: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average files per second
    pub avg_files_per_sec: f64,
}

/// Full indexing benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingReport {
    /// Timestamp of the benchmark run
    pub timestamp: String,
    /// CCEngram version
    pub version: String,
    /// Individual run results
    pub results: Vec<IndexingBenchResult>,
    /// Per-repository summaries
    pub summaries: Vec<IndexingSummary>,
}

impl IndexingReport {
    /// Create a new report from results.
    pub fn from_results(results: Vec<IndexingBenchResult>) -> Self {
        let summaries = Self::compute_summaries(&results);

        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            results,
            summaries,
        }
    }

    fn compute_summaries(results: &[IndexingBenchResult]) -> Vec<IndexingSummary> {
        use std::collections::HashMap;

        // Group by repo
        let mut by_repo: HashMap<String, Vec<&IndexingBenchResult>> = HashMap::new();
        for result in results {
            by_repo.entry(result.repo.clone()).or_default().push(result);
        }

        by_repo
            .into_iter()
            .map(|(repo, runs)| {
                let mut wall_times: Vec<u64> = runs.iter().map(|r| r.metrics.wall_time_ms).collect();
                wall_times.sort_unstable();

                let iterations = runs.len();
                let avg_wall_time_ms = wall_times.iter().sum::<u64>() / iterations as u64;
                let p50_wall_time_ms = wall_times.get(iterations / 2).copied().unwrap_or(0);
                let p95_wall_time_ms = wall_times
                    .get((iterations as f64 * 0.95) as usize)
                    .copied()
                    .unwrap_or(*wall_times.last().unwrap_or(&0));

                let avg_chunks_per_sec =
                    runs.iter().map(|r| r.metrics.chunks_per_sec).sum::<f64>() / iterations as f64;
                let avg_embeddings_per_sec =
                    runs.iter().map(|r| r.metrics.embeddings_per_sec).sum::<f64>() / iterations as f64;
                let peak_memory_bytes = runs.iter().map(|r| r.metrics.peak_memory_bytes).max().unwrap_or(0);

                let avg_files_per_sec = runs
                    .iter()
                    .map(|r| {
                        if r.metrics.wall_time_ms > 0 {
                            r.files_indexed as f64 / (r.metrics.wall_time_ms as f64 / 1000.0)
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / iterations as f64;

                IndexingSummary {
                    repo,
                    iterations,
                    avg_wall_time_ms,
                    p50_wall_time_ms,
                    p95_wall_time_ms,
                    avg_chunks_per_sec,
                    avg_embeddings_per_sec,
                    peak_memory_bytes,
                    avg_files_per_sec,
                }
            })
            .collect()
    }

    /// Generate markdown report.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str("# Indexing Performance Report\n\n");
        out.push_str(&format!("**Timestamp:** {}\n", self.timestamp));
        out.push_str(&format!("**Version:** {}\n\n", self.version));

        out.push_str("## Summary\n\n");
        out.push_str("| Repository | Iterations | Avg Time | p50 | p95 | Chunks/sec | Files/sec | Peak Memory |\n");
        out.push_str("|------------|------------|----------|-----|-----|------------|-----------|-------------|\n");

        for summary in &self.summaries {
            out.push_str(&format!(
                "| {} | {} | {:.1}s | {:.1}s | {:.1}s | {:.0} | {:.0} | {:.1} MB |\n",
                summary.repo,
                summary.iterations,
                summary.avg_wall_time_ms as f64 / 1000.0,
                summary.p50_wall_time_ms as f64 / 1000.0,
                summary.p95_wall_time_ms as f64 / 1000.0,
                summary.avg_chunks_per_sec,
                summary.avg_files_per_sec,
                summary.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            ));
        }

        out.push_str("\n## Detailed Results\n\n");

        for result in &self.results {
            out.push_str(&format!(
                "### {} (iteration {})\n\n",
                result.repo, result.iteration
            ));
            out.push_str(&format!("- **Cold start:** {}\n", result.cold_start));
            out.push_str(&format!("- **Wall time:** {:.2}s\n", result.metrics.wall_time_ms as f64 / 1000.0));
            out.push_str(&format!("- **Files scanned:** {}\n", result.files_scanned));
            out.push_str(&format!("- **Files indexed:** {}\n", result.files_indexed));
            out.push_str(&format!("- **Chunks created:** {}\n", result.metrics.chunks_processed));
            out.push_str(&format!("- **Chunks/sec:** {:.0}\n", result.metrics.chunks_per_sec));
            out.push_str(&format!("- **Embeddings/sec:** {:.0}\n", result.metrics.embeddings_per_sec));
            out.push_str(&format!(
                "- **Peak memory:** {:.1} MB\n",
                result.metrics.peak_memory_bytes as f64 / (1024.0 * 1024.0)
            ));
            out.push_str(&format!(
                "- **Bytes processed:** {:.1} MB\n\n",
                result.bytes_processed as f64 / (1024.0 * 1024.0)
            ));
        }

        out
    }

    /// Save report to files (JSON and Markdown).
    pub fn save(&self, output_dir: &PathBuf) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // Save JSON
        let json_path = output_dir.join("indexing.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&json_path, json)?;
        info!("Saved JSON report: {}", json_path.display());

        // Save Markdown
        let md_path = output_dir.join("indexing.md");
        std::fs::write(&md_path, self.to_markdown())?;
        info!("Saved Markdown report: {}", md_path.display());

        Ok(())
    }
}

/// Indexing benchmark runner.
pub struct IndexingBenchmark {
    socket_path: String,
    cache_dir: Option<PathBuf>,
}

impl IndexingBenchmark {
    /// Create a new indexing benchmark runner.
    pub fn new(socket_path: &str, cache_dir: Option<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            cache_dir,
        }
    }

    /// Get default daemon socket path.
    pub fn default_socket_path() -> String {
        dirs::runtime_dir()
            .or_else(dirs::state_dir)
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("ccengram.sock")
            .to_string_lossy()
            .to_string()
    }

    /// Check if daemon is running.
    pub async fn check_daemon(&self) -> bool {
        UnixStream::connect(&self.socket_path).await.is_ok()
    }

    /// Run indexing benchmark for specified repositories.
    pub async fn run(
        &self,
        repos: &[TargetRepo],
        iterations: usize,
        cold_start: bool,
    ) -> Result<IndexingReport> {
        let mut results = Vec::new();

        for repo in repos {
            info!("Benchmarking indexing for: {}", repo);

            // Prepare repository (download if needed)
            let repo_path = prepare_repo(*repo, self.cache_dir.clone()).await?;
            info!("Repository path: {}", repo_path.display());

            for i in 0..iterations {
                info!("  Iteration {}/{}", i + 1, iterations);

                // If cold start requested and not first iteration, clear the index
                if cold_start && i > 0 {
                    self.clear_index(&repo_path).await?;
                }

                // Run indexing and collect metrics
                let result = self.run_single_index(&repo.to_string(), &repo_path, i, cold_start && i == 0).await?;
                results.push(result);
            }
        }

        Ok(IndexingReport::from_results(results))
    }

    /// Run a single indexing operation and collect metrics.
    async fn run_single_index(
        &self,
        repo_name: &str,
        repo_path: &Path,
        iteration: usize,
        cold_start: bool,
    ) -> Result<IndexingBenchResult> {
        let mut monitor = ResourceMonitor::new();
        monitor.snapshot();

        let start = Instant::now();

        // Send index request to daemon
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "code_index",
            "params": {
                "project_path": repo_path.to_string_lossy(),
                "force": cold_start
            }
        });

        let response = self.send_request(&request).await?;
        let elapsed = start.elapsed();

        monitor.snapshot();

        // Parse response
        let result = response
            .get("result")
            .ok_or_else(|| crate::BenchmarkError::Execution("No result in response".into()))?;

        let files_scanned = result.get("files_scanned").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let files_indexed = result.get("files_indexed").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let chunks_created = result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let _scan_duration_ms = result.get("scan_duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let _index_duration_ms = result.get("index_duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let bytes_processed = result.get("bytes_processed").and_then(|v| v.as_u64()).unwrap_or(0);

        let wall_time_ms = elapsed.as_millis() as u64;
        let chunks_per_sec = if wall_time_ms > 0 {
            chunks_created as f64 / (wall_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        // Estimate embeddings (typically 1 embedding per chunk)
        let embeddings_generated = chunks_created;
        let embeddings_per_sec = chunks_per_sec;

        let metrics = IndexingMetrics {
            wall_time_ms,
            peak_memory_bytes: monitor.peak_memory(),
            avg_cpu_percent: monitor.avg_cpu(),
            chunks_processed: chunks_created,
            chunks_per_sec,
            embeddings_generated,
            embeddings_per_sec,
        };

        debug!(
            "  Indexed {} files, {} chunks in {:.2}s ({:.0} chunks/sec)",
            files_indexed,
            chunks_created,
            wall_time_ms as f64 / 1000.0,
            chunks_per_sec
        );

        Ok(IndexingBenchResult {
            repo: repo_name.to_string(),
            iteration,
            cold_start,
            metrics,
            files_scanned,
            files_indexed,
            bytes_processed,
        })
    }

    /// Clear index for a project (for cold start testing).
    async fn clear_index(&self, repo_path: &Path) -> Result<()> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "clear_code_index",
            "params": {
                "project_path": repo_path.to_string_lossy()
            }
        });

        let _ = self.send_request(&request).await;
        Ok(())
    }

    /// Send a JSON-RPC request to the daemon.
    async fn send_request(&self, request: &serde_json::Value) -> Result<serde_json::Value> {
        let mut stream = UnixStream::connect(&self.socket_path).await.map_err(|e| {
            crate::BenchmarkError::Execution(format!("Failed to connect to daemon: {}", e))
        })?;

        let request_str = serde_json::to_string(request)? + "\n";
        stream.write_all(request_str.as_bytes()).await.map_err(|e| {
            crate::BenchmarkError::Execution(format!("Failed to send request: {}", e))
        })?;

        let reader = BufReader::new(stream);
        let mut lines = reader.lines();

        if let Some(line) = lines.next_line().await.map_err(|e| {
            crate::BenchmarkError::Execution(format!("Failed to read response: {}", e))
        })? {
            let response: serde_json::Value = serde_json::from_str(&line)?;

            if let Some(error) = response.get("error") {
                return Err(crate::BenchmarkError::Execution(format!(
                    "Daemon error: {}",
                    error
                )));
            }

            Ok(response)
        } else {
            Err(crate::BenchmarkError::Execution("Empty response from daemon".into()))
        }
    }
}

/// Compare two indexing reports for regressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingComparison {
    /// Baseline report timestamp
    pub baseline_timestamp: String,
    /// Current report timestamp
    pub current_timestamp: String,
    /// Per-repo comparisons
    pub comparisons: Vec<RepoComparison>,
    /// Overall pass/fail
    pub passes: bool,
}

/// Comparison for a single repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoComparison {
    /// Repository name
    pub repo: String,
    /// Wall time change percentage (positive = slower)
    pub wall_time_change_pct: f64,
    /// Chunks/sec change percentage (positive = faster)
    pub throughput_change_pct: f64,
    /// Memory change percentage (positive = more memory)
    pub memory_change_pct: f64,
    /// Whether this repo passes threshold
    pub passes: bool,
}

impl IndexingComparison {
    /// Compare two reports with a regression threshold.
    pub fn compare(baseline: &IndexingReport, current: &IndexingReport, threshold_pct: f64) -> Self {
        let mut comparisons = Vec::new();
        let mut all_pass = true;

        for current_summary in &current.summaries {
            if let Some(baseline_summary) = baseline
                .summaries
                .iter()
                .find(|s| s.repo == current_summary.repo)
            {
                let wall_time_change_pct = if baseline_summary.avg_wall_time_ms > 0 {
                    ((current_summary.avg_wall_time_ms as f64 - baseline_summary.avg_wall_time_ms as f64)
                        / baseline_summary.avg_wall_time_ms as f64)
                        * 100.0
                } else {
                    0.0
                };

                let throughput_change_pct = if baseline_summary.avg_chunks_per_sec > 0.0 {
                    ((current_summary.avg_chunks_per_sec - baseline_summary.avg_chunks_per_sec)
                        / baseline_summary.avg_chunks_per_sec)
                        * 100.0
                } else {
                    0.0
                };

                let memory_change_pct = if baseline_summary.peak_memory_bytes > 0 {
                    ((current_summary.peak_memory_bytes as f64 - baseline_summary.peak_memory_bytes as f64)
                        / baseline_summary.peak_memory_bytes as f64)
                        * 100.0
                } else {
                    0.0
                };

                // Regression if wall time increased or throughput decreased beyond threshold
                let passes = wall_time_change_pct <= threshold_pct && throughput_change_pct >= -threshold_pct;

                if !passes {
                    all_pass = false;
                }

                comparisons.push(RepoComparison {
                    repo: current_summary.repo.clone(),
                    wall_time_change_pct,
                    throughput_change_pct,
                    memory_change_pct,
                    passes,
                });
            }
        }

        Self {
            baseline_timestamp: baseline.timestamp.clone(),
            current_timestamp: current.timestamp.clone(),
            comparisons,
            passes: all_pass,
        }
    }

    /// Generate markdown comparison report.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str("# Indexing Performance Comparison\n\n");
        out.push_str(&format!("**Baseline:** {}\n", self.baseline_timestamp));
        out.push_str(&format!("**Current:** {}\n\n", self.current_timestamp));

        let status = if self.passes { "PASS" } else { "FAIL" };
        out.push_str(&format!("**Status:** {}\n\n", status));

        out.push_str("| Repository | Wall Time | Throughput | Memory | Status |\n");
        out.push_str("|------------|-----------|------------|--------|--------|\n");

        for comp in &self.comparisons {
            let wall_icon = if comp.wall_time_change_pct > 10.0 { "🔴" } else if comp.wall_time_change_pct < -10.0 { "🟢" } else { "⚪" };
            let throughput_icon = if comp.throughput_change_pct < -10.0 { "🔴" } else if comp.throughput_change_pct > 10.0 { "🟢" } else { "⚪" };
            let status_icon = if comp.passes { "✅" } else { "❌" };

            out.push_str(&format!(
                "| {} | {} {:+.1}% | {} {:+.1}% | {:+.1}% | {} |\n",
                comp.repo,
                wall_icon,
                comp.wall_time_change_pct,
                throughput_icon,
                comp.throughput_change_pct,
                comp.memory_change_pct,
                status_icon,
            ));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_summary_computation() {
        let results = vec![
            IndexingBenchResult {
                repo: "test".to_string(),
                iteration: 0,
                cold_start: true,
                metrics: IndexingMetrics {
                    wall_time_ms: 1000,
                    peak_memory_bytes: 100_000_000,
                    avg_cpu_percent: 50.0,
                    chunks_processed: 1000,
                    chunks_per_sec: 1000.0,
                    embeddings_generated: 1000,
                    embeddings_per_sec: 1000.0,
                },
                files_scanned: 100,
                files_indexed: 100,
                bytes_processed: 1_000_000,
            },
            IndexingBenchResult {
                repo: "test".to_string(),
                iteration: 1,
                cold_start: false,
                metrics: IndexingMetrics {
                    wall_time_ms: 800,
                    peak_memory_bytes: 90_000_000,
                    avg_cpu_percent: 45.0,
                    chunks_processed: 1000,
                    chunks_per_sec: 1250.0,
                    embeddings_generated: 1000,
                    embeddings_per_sec: 1250.0,
                },
                files_scanned: 100,
                files_indexed: 100,
                bytes_processed: 1_000_000,
            },
        ];

        let report = IndexingReport::from_results(results);
        assert_eq!(report.summaries.len(), 1);

        let summary = &report.summaries[0];
        assert_eq!(summary.repo, "test");
        assert_eq!(summary.iterations, 2);
        assert_eq!(summary.avg_wall_time_ms, 900);
    }

    #[test]
    fn test_comparison() {
        let baseline = IndexingReport {
            timestamp: "2024-01-01".to_string(),
            version: "0.1.0".to_string(),
            results: vec![],
            summaries: vec![IndexingSummary {
                repo: "test".to_string(),
                iterations: 1,
                avg_wall_time_ms: 1000,
                p50_wall_time_ms: 1000,
                p95_wall_time_ms: 1000,
                avg_chunks_per_sec: 1000.0,
                avg_embeddings_per_sec: 1000.0,
                peak_memory_bytes: 100_000_000,
                avg_files_per_sec: 100.0,
            }],
        };

        let current = IndexingReport {
            timestamp: "2024-01-02".to_string(),
            version: "0.1.0".to_string(),
            results: vec![],
            summaries: vec![IndexingSummary {
                repo: "test".to_string(),
                iterations: 1,
                avg_wall_time_ms: 1100, // 10% slower
                p50_wall_time_ms: 1100,
                p95_wall_time_ms: 1100,
                avg_chunks_per_sec: 900.0, // 10% slower
                avg_embeddings_per_sec: 900.0,
                peak_memory_bytes: 110_000_000,
                avg_files_per_sec: 90.0,
            }],
        };

        let comparison = IndexingComparison::compare(&baseline, &current, 10.0);
        assert_eq!(comparison.comparisons.len(), 1);
        assert!(comparison.passes); // Within 10% threshold
    }
}
