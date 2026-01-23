//! Scenario execution against the daemon.

use super::{Scenario, Step};
use crate::ground_truth::load_scenario_annotations;
use crate::metrics::{AccuracyMetrics, PerformanceMetrics, ResourceMonitor};
use crate::session::ExplorationSession;
use crate::{BenchmarkError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::{debug, info, warn};

/// Result of running a single scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    /// Scenario ID
    pub scenario_id: String,
    /// Scenario name
    pub scenario_name: String,
    /// Whether the scenario passed all criteria
    pub passed: bool,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Accuracy metrics
    pub accuracy: AccuracyMetrics,
    /// Per-step results
    pub steps: Vec<StepResult>,
    /// Errors encountered (if any)
    pub errors: Vec<String>,
    /// Total execution time
    pub total_duration_ms: u64,
}

/// Result of a single step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Step index (0-based)
    pub step_index: usize,
    /// Query executed
    pub query: String,
    /// Number of results returned
    pub result_count: usize,
    /// Noise ratio for this step
    pub noise_ratio: f64,
    /// IDs of results
    pub result_ids: Vec<String>,
    /// Files found in results
    pub files_found: Vec<String>,
    /// Symbols found in results
    pub symbols_found: Vec<String>,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Whether expected criteria were met
    pub passed: bool,
}

/// Runner for executing scenarios against the daemon.
pub struct ScenarioRunner {
    /// Path to the daemon socket
    socket_path: String,
    /// Project path for the benchmark target
    project_path: String,
    /// Optional annotations directory for ground truth
    annotations_dir: Option<PathBuf>,
}

impl ScenarioRunner {
    /// Create a new scenario runner.
    pub fn new(socket_path: &str, project_path: &str, annotations_dir: Option<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            project_path: project_path.to_string(),
            annotations_dir,
        }
    }

    /// Get the default socket path.
    pub fn default_socket_path() -> String {
        let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
            .unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/ccengram.sock", runtime_dir)
    }

    /// Run a single scenario.
    pub async fn run(&self, scenario: &Scenario) -> Result<ScenarioResult> {
        let start = Instant::now();
        let mut session = ExplorationSession::new(&scenario.metadata.id);
        let mut step_results = Vec::new();
        let mut errors = Vec::new();

        // Initialize resource monitor for memory/CPU tracking
        let mut resource_monitor = ResourceMonitor::new();
        resource_monitor.snapshot(); // Initial snapshot

        info!(
            "Running scenario: {} ({})",
            scenario.metadata.name, scenario.metadata.id
        );

        // Load annotations if available and merge with expected values
        let mut expected = scenario.expected.clone();
        if let Some(annotations_dir) = &self.annotations_dir {
            // Determine repo-specific annotations directory
            let repo_annotations_dir = annotations_dir.join(scenario.metadata.repo.to_string());
            let annotations =
                load_scenario_annotations(&repo_annotations_dir, &scenario.metadata.id);

            if !annotations.is_empty() {
                debug!(
                    "Loaded {} critical files and {} critical symbols from annotations",
                    annotations.critical_files.len(),
                    annotations.critical_symbols.len()
                );

                // Merge annotations into expected values
                for file in &annotations.critical_files {
                    if !expected.must_find_files.contains(file) {
                        expected.must_find_files.push(file.clone());
                    }
                }
                for symbol in &annotations.critical_symbols {
                    if !expected.must_find_symbols.contains(symbol) {
                        expected.must_find_symbols.push(symbol.clone());
                    }
                }
            }
        }

        for (i, step) in scenario.steps.iter().enumerate() {
            debug!("Executing step {}: {}", i + 1, step.query);

            // Take resource snapshot before step
            resource_monitor.snapshot();

            match self.execute_step(step, i, &mut session).await {
                Ok(result) => {
                    step_results.push(result);
                }
                Err(e) => {
                    warn!("Step {} failed: {}", i + 1, e);
                    errors.push(format!("Step {}: {}", i + 1, e));
                    // Create a failed step result
                    step_results.push(StepResult {
                        step_index: i,
                        query: step.query.clone(),
                        result_count: 0,
                        noise_ratio: 1.0,
                        result_ids: vec![],
                        files_found: vec![],
                        symbols_found: vec![],
                        latency_ms: 0,
                        passed: false,
                    });
                }
            }

            // Take resource snapshot after step
            resource_monitor.snapshot();
        }

        let total_duration = start.elapsed();

        // Compute metrics with resource monitoring data
        let mut performance = session.compute_performance_metrics();
        performance.peak_memory_bytes = Some(resource_monitor.peak_memory());
        performance.avg_cpu_percent = Some(resource_monitor.avg_cpu());

        let accuracy = session.compute_accuracy_metrics(&expected, &scenario.success_criteria);

        // Determine if scenario passed
        let passed = errors.is_empty()
            && accuracy.file_recall >= scenario.success_criteria.min_discovery_score
            && accuracy.noise_ratio <= scenario.success_criteria.max_noise_ratio
            && accuracy
                .steps_to_core
                .is_none_or(|s| s <= scenario.success_criteria.max_steps_to_core);

        Ok(ScenarioResult {
            scenario_id: scenario.metadata.id.clone(),
            scenario_name: scenario.metadata.name.clone(),
            passed,
            performance,
            accuracy,
            steps: step_results,
            errors,
            total_duration_ms: total_duration.as_millis() as u64,
        })
    }

    /// Execute a single step.
    async fn execute_step(
        &self,
        step: &Step,
        index: usize,
        session: &mut ExplorationSession,
    ) -> Result<StepResult> {
        let start = Instant::now();

        // Build explore request
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": format!("step-{}", index),
            "method": "explore",
            "params": {
                "query": step.query,
                "scope": step.scope.as_deref().unwrap_or("all"),
                "expand_top": 3,
                "limit": 10,
                "cwd": self.project_path,
            }
        });

        // Send request to daemon
        let response = self.send_request(&request).await?;
        let latency = start.elapsed();

        // Parse response
        let result = response
            .get("result")
            .ok_or_else(|| BenchmarkError::Execution("No result in response".into()))?;

        let results = result
            .get("results")
            .and_then(|r| r.as_array())
            .ok_or_else(|| BenchmarkError::Execution("No results array".into()))?;

        // Extract result info
        let result_ids: Vec<String> = results
            .iter()
            .filter_map(|r| r.get("id").and_then(|id| id.as_str()).map(String::from))
            .collect();

        let files_found: Vec<String> = results
            .iter()
            .filter_map(|r| r.get("file").and_then(|f| f.as_str()).map(String::from))
            .collect();

        let symbols_found: Vec<String> = results
            .iter()
            .filter_map(|r| {
                r.get("symbols").and_then(|s| s.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();

        // Record in session
        session.record_explore_step(
            &step.query,
            &result_ids,
            &files_found,
            &symbols_found,
            latency,
        );

        // Execute context requests if specified
        if !step.context_ids.is_empty() {
            for id in &step.context_ids {
                self.execute_context(id, session).await?;
            }
        }

        // Calculate noise ratio for this step
        let noise_count = session.count_noise_results(&result_ids);
        let noise_ratio = if results.is_empty() {
            0.0
        } else {
            noise_count as f64 / results.len() as f64
        };

        // Determine if step passed its criteria
        let passed = step.expected_results.is_none_or(|expected| results.len() >= expected)
            && step.max_noise_ratio.is_none_or(|max| noise_ratio <= max);

        Ok(StepResult {
            step_index: index,
            query: step.query.clone(),
            result_count: results.len(),
            noise_ratio,
            result_ids,
            files_found,
            symbols_found,
            latency_ms: latency.as_millis() as u64,
            passed,
        })
    }

    /// Execute a context request.
    async fn execute_context(&self, id: &str, session: &mut ExplorationSession) -> Result<()> {
        let start = Instant::now();

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": format!("context-{}", id),
            "method": "context",
            "params": {
                "id": id,
                "depth": 5,
                "cwd": self.project_path,
            }
        });

        let response = self.send_request(&request).await?;
        let latency = start.elapsed();

        // Check for success
        if response.get("error").is_some() {
            return Err(BenchmarkError::Execution(format!(
                "Context request failed for {}",
                id
            )));
        }

        session.record_context_call(id, latency);
        Ok(())
    }

    /// Send a JSON-RPC request to the daemon.
    async fn send_request(&self, request: &serde_json::Value) -> Result<serde_json::Value> {
        let mut stream = UnixStream::connect(&self.socket_path).await.map_err(|e| {
            BenchmarkError::Execution(format!("Failed to connect to daemon: {}", e))
        })?;

        // Write request
        let request_str = serde_json::to_string(request)?;
        stream.write_all(request_str.as_bytes()).await?;
        stream.write_all(b"\n").await?;
        stream.flush().await?;

        // Read response
        let mut reader = BufReader::new(stream);
        let mut response_str = String::new();
        reader.read_line(&mut response_str).await?;

        let response: serde_json::Value = serde_json::from_str(&response_str)?;
        Ok(response)
    }

    /// Check if the daemon is running.
    pub async fn check_daemon(&self) -> bool {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": "health-check",
            "method": "health",
            "params": {}
        });

        match self.send_request(&request).await {
            Ok(response) => response.get("error").is_none(),
            Err(_) => false,
        }
    }
}

/// Run multiple scenarios in parallel.
pub async fn run_scenarios_parallel(
    runner: &ScenarioRunner,
    scenarios: &[Scenario],
) -> Vec<ScenarioResult> {
    use futures::future::join_all;

    let futures: Vec<_> = scenarios
        .iter()
        .map(|s| async { runner.run(s).await })
        .collect();

    let results = join_all(futures).await;

    results
        .into_iter()
        .filter_map(|r| r.ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_socket_path() {
        let path = ScenarioRunner::default_socket_path();
        assert!(path.ends_with("ccengram.sock"));
    }

    #[test]
    fn test_step_result_serialization() {
        let result = StepResult {
            step_index: 0,
            query: "test query".to_string(),
            result_count: 5,
            noise_ratio: 0.2,
            result_ids: vec!["id1".to_string()],
            files_found: vec!["file.rs".to_string()],
            symbols_found: vec!["func".to_string()],
            latency_ms: 100,
            passed: true,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("test query"));
        assert!(json.contains("result_count"));
    }
}
