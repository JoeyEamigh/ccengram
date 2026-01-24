//! Scenario execution against the daemon.

use super::{Expected, PreviousStepResults, Scenario, Step};
use crate::ground_truth::load_scenario_annotations;
use crate::metrics::{AccuracyMetrics, PerformanceMetrics, ResourceMonitor};
use crate::session::{ExplorationSession, HintType};
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
  /// Caller symbols from expanded context (for adaptive templates)
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub callers: Vec<String>,
  /// Callee symbols from expanded context (for adaptive templates)
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub callees: Vec<String>,
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
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
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
    let mut exploration_paths = Vec::new();

    if let Some(annotations_dir) = &self.annotations_dir {
      // Determine repo-specific annotations directory
      let repo_annotations_dir = annotations_dir.join(scenario.metadata.repo.to_string());
      let annotations = load_scenario_annotations(&repo_annotations_dir, &scenario.metadata.id);

      if !annotations.is_empty() {
        debug!(
          "Loaded {} critical files, {} critical symbols, {} exploration paths from annotations",
          annotations.critical_files.len(),
          annotations.critical_symbols.len(),
          annotations.exploration_paths.len()
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

        // Store exploration paths for navigation efficiency calculation
        exploration_paths = annotations.exploration_paths.clone();
      }
    }

    // Track previous step results for adaptive template resolution
    let mut previous_results = PreviousStepResults::default();

    for (i, step) in scenario.steps.iter().enumerate() {
      // Resolve templates in step if it depends on previous results
      let resolved_step = if step.has_templates() {
        let resolved = previous_results.resolve_step(step);
        debug!("Executing step {} (resolved): {}", i + 1, resolved.query);
        resolved
      } else {
        debug!("Executing step {}: {}", i + 1, step.query);
        step.clone()
      };

      // Take resource snapshot before step
      resource_monitor.snapshot();

      match self
        .execute_step_with_context(&resolved_step, i, &mut session, &expected, &mut previous_results)
        .await
      {
        Ok(result) => {
          step_results.push(result);
        }
        Err(e) => {
          warn!("Step {} failed: {}", i + 1, e);
          errors.push(format!("Step {}: {}", i + 1, e));
          // Create a failed step result
          step_results.push(StepResult {
            step_index: i,
            query: resolved_step.query.clone(),
            result_count: 0,
            noise_ratio: 1.0,
            result_ids: vec![],
            files_found: vec![],
            symbols_found: vec![],
            callers: vec![],
            callees: vec![],
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

    let accuracy =
      session.compute_accuracy_metrics_with_paths(&expected, &scenario.success_criteria, &exploration_paths);

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
    expected: &Expected,
  ) -> Result<StepResult> {
    let start = Instant::now();

    // Check if query uses a previous suggestion
    session.check_suggestion_used(&step.query);

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
          arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>()
        })
      })
      .flatten()
      .collect();

    // Record MRR data - check if each result is relevant
    for (rank, r) in results.iter().enumerate() {
      let file = r.get("file").and_then(|f| f.as_str()).unwrap_or("");
      let result_symbols: Vec<&str> = r
        .get("symbols")
        .and_then(|s| s.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

      let is_relevant = self.is_result_relevant(file, &result_symbols, expected);
      session.record_result_rank(is_relevant, rank + 1);
    }

    // Track context budget - calculate bytes returned
    let response_str = serde_json::to_string(&response).unwrap_or_default();
    let total_bytes = response_str.len();

    // Calculate useful bytes - content containing expected files/symbols
    let mut useful_bytes = 0;
    for expected_file in &expected.must_find_files {
      if response_str.contains(expected_file) {
        useful_bytes += expected_file.len();
      }
    }
    for expected_symbol in &expected.must_find_symbols {
      if response_str.contains(expected_symbol) {
        useful_bytes += expected_symbol.len();
      }
    }

    // Record explore bytes for context budget tracking
    session.record_explore_bytes(total_bytes, useful_bytes);

    // Track step relevance for rabbit hole detection
    let found_expected_file = files_found
      .iter()
      .any(|f| self.file_matches_expected(f, &expected.must_find_files));
    let found_expected_symbol = symbols_found.iter().any(|s| expected.must_find_symbols.contains(s));
    let relevant_count = results
      .iter()
      .filter(|r| {
        let file = r.get("file").and_then(|f| f.as_str()).unwrap_or("");
        let result_symbols: Vec<&str> = r
          .get("symbols")
          .and_then(|s| s.as_array())
          .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
          .unwrap_or_default();
        self.is_result_relevant(file, &result_symbols, expected)
      })
      .count();

    session.record_step_relevance(
      found_expected_file,
      found_expected_symbol,
      relevant_count,
      results.len(),
    );

    // Track callers and callees for adaptive templates
    let mut all_callers = Vec::new();
    let mut all_callees = Vec::new();

    // Extract hints from expanded context and build call graph
    for r in results {
      let id = r.get("id").and_then(|id| id.as_str()).unwrap_or("");

      // Get the main symbols of this result for call graph
      let source_symbols: Vec<&str> = r
        .get("symbols")
        .and_then(|s| s.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

      // Record symbol discovery times
      for symbol in &source_symbols {
        session.record_symbol_discovery(symbol);
      }

      if let Some(context) = r.get("context") {
        // Record caller hints and build call graph
        if let Some(callers) = context.get("callers").and_then(|c| c.as_array()) {
          for caller in callers {
            if let Some(target) = caller.get("file").and_then(|f| f.as_str()) {
              session.record_hint(id, HintType::Caller, target);
            }
            // Build call graph: caller_symbol -> source_symbol
            if let Some(caller_symbols) = caller.get("symbols").and_then(|s| s.as_array()) {
              for caller_sym in caller_symbols.iter().filter_map(|v| v.as_str()) {
                all_callers.push(caller_sym.to_string());
                for source_sym in &source_symbols {
                  session.record_call_relation(caller_sym, source_sym);
                }
              }
            }
          }
        }
        // Record callee hints and build call graph
        if let Some(callees) = context.get("callees").and_then(|c| c.as_array()) {
          for callee in callees {
            if let Some(target) = callee.get("file").and_then(|f| f.as_str()) {
              session.record_hint(id, HintType::Callee, target);
            }
            // Build call graph: source_symbol -> callee_symbol
            if let Some(callee_symbols) = callee.get("symbols").and_then(|s| s.as_array()) {
              for callee_sym in callee_symbols.iter().filter_map(|v| v.as_str()) {
                all_callees.push(callee_sym.to_string());
                for source_sym in &source_symbols {
                  session.record_call_relation(source_sym, callee_sym);
                }
              }
            }
          }
        }
        // Record sibling hints
        if let Some(siblings) = context.get("siblings").and_then(|c| c.as_array()) {
          for sibling in siblings {
            if let Some(target) = sibling.get("file").and_then(|f| f.as_str()) {
              session.record_hint(id, HintType::Sibling, target);
            }
          }
        }
      }
    }

    // Extract and record suggestions
    if let Some(suggestions) = result.get("suggestions").and_then(|s| s.as_array()) {
      let suggestion_strs: Vec<String> = suggestions
        .iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
      session.record_suggestions(&suggestion_strs);
    }

    // Record in session
    session.record_explore_step(&step.query, &result_ids, &files_found, &symbols_found, latency);

    // Record step discoveries for convergence tracking
    session.record_step_discoveries(expected);

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
      callers: all_callers,
      callees: all_callees,
      latency_ms: latency.as_millis() as u64,
      passed,
    })
  }

  /// Execute a single step and update previous results for template resolution.
  async fn execute_step_with_context(
    &self,
    step: &Step,
    index: usize,
    session: &mut ExplorationSession,
    expected: &Expected,
    previous_results: &mut PreviousStepResults,
  ) -> Result<StepResult> {
    let result = self.execute_step(step, index, session, expected).await?;

    // Update previous_results for next step's template resolution
    previous_results.ids = result.result_ids.clone();
    previous_results.files = result.files_found.clone();
    previous_results.symbols = result.symbols_found.clone();
    previous_results.callers = result.callers.clone();
    previous_results.callees = result.callees.clone();

    Ok(result)
  }

  /// Check if a result is relevant to the expected values.
  fn is_result_relevant(&self, file: &str, symbols: &[&str], expected: &Expected) -> bool {
    // Check if file matches expected files
    if self.file_matches_expected(file, &expected.must_find_files) {
      return true;
    }

    // Check if any symbol matches expected symbols
    for symbol in symbols {
      if expected.must_find_symbols.contains(&symbol.to_string()) {
        return true;
      }
    }

    false
  }

  /// Check if a file matches expected files (with glob support).
  fn file_matches_expected(&self, file: &str, expected_files: &[String]) -> bool {
    for expected_file in expected_files {
      if let Ok(pattern) = glob::Pattern::new(expected_file)
        && pattern.matches(file)
      {
        return true;
      }
      if file.ends_with(expected_file) || file == expected_file {
        return true;
      }
    }
    false
  }

  /// Execute a context request.
  async fn execute_context(&self, id: &str, session: &mut ExplorationSession) -> Result<()> {
    let start = Instant::now();

    // Track files/symbols before context call
    let files_before = session.discovered_files().len();
    let symbols_before = session.discovered_symbols().len();

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
      return Err(BenchmarkError::Execution(format!("Context request failed for {}", id)));
    }

    // Extract new files/symbols from context response
    let result = response.get("result");
    if let Some(ctx) = result {
      // Extract files from callers/callees
      let mut new_files = Vec::new();
      let mut new_symbols = Vec::new();

      for section in ["callers", "callees", "siblings"] {
        if let Some(items) = ctx.get(section).and_then(|s| s.as_array()) {
          for item in items {
            if let Some(file) = item.get("file").and_then(|f| f.as_str())
              && !session.discovered_files().contains(file)
            {
              new_files.push(file.to_string());
            }
            if let Some(syms) = item.get("symbols").and_then(|s| s.as_array()) {
              for sym in syms {
                if let Some(s) = sym.as_str()
                  && !session.discovered_symbols().contains(s)
                {
                  new_symbols.push(s.to_string());
                }
              }
            }
          }
        }
      }

      // Mark hints as followed (context call = following a hint)
      session.mark_hint_followed(id);

      // Record context call latency
      session.record_context_call(id, latency);

      // Record context value (how many new things it revealed)
      let new_file_count = session.discovered_files().len() - files_before + new_files.len();
      let new_symbol_count = session.discovered_symbols().len() - symbols_before + new_symbols.len();
      session.record_context_value(id, new_file_count, new_symbol_count);
    } else {
      session.record_context_call(id, latency);
      session.record_context_value(id, 0, 0);
    }

    Ok(())
  }

  /// Send a JSON-RPC request to the daemon.
  async fn send_request(&self, request: &serde_json::Value) -> Result<serde_json::Value> {
    let mut stream = UnixStream::connect(&self.socket_path)
      .await
      .map_err(|e| BenchmarkError::Execution(format!("Failed to connect to daemon: {}", e)))?;

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
pub async fn run_scenarios_parallel(runner: &ScenarioRunner, scenarios: &[Scenario]) -> Vec<ScenarioResult> {
  use futures::future::join_all;

  let futures: Vec<_> = scenarios.iter().map(|s| async { runner.run(s).await }).collect();

  let results = join_all(futures).await;

  results.into_iter().filter_map(|r| r.ok()).collect()
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
      callers: vec!["caller_func".to_string()],
      callees: vec!["callee_func".to_string()],
      latency_ms: 100,
      passed: true,
    };

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("test query"));
    assert!(json.contains("result_count"));
  }
}
