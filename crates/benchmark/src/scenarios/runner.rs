//! Scenario execution against the daemon.

use std::{path::PathBuf, time::Instant};

use ccengram::ipc::{
  Client,
  search::{ContextItem, ContextParams, ExploreParams, ExploreResult},
  system::HealthCheckParams,
};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tracing::{debug, info, warn};

use super::{Expected, PreviousStepResults, Scenario, Step, TaskIntent, TaskRequirementsResult};
use crate::{
  BenchmarkError, Result,
  ground_truth::{Annotations, load_scenario_annotations},
  llm_judge::{ComprehensionResult, LlmJudge},
  metrics::{AccuracyMetrics, PerformanceMetrics, ResourceMonitor},
  session::ExplorationSession,
};

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
  /// LLM comprehension evaluation (if enabled)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub comprehension: Option<ComprehensionResult>,
  /// Task requirements evaluation (for task_completion scenarios)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub task_requirements_result: Option<TaskRequirementsResult>,
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
  client: Client,
  /// Optional annotations directory for ground truth
  annotations_dir: Option<PathBuf>,
  /// Optional LLM judge for comprehension evaluation
  llm_judge: Option<LlmJudge>,
  /// Cached daemon PID for resource monitoring (lazily initialized)
  daemon_pid: OnceCell<u32>,
}

impl ScenarioRunner {
  /// Create a new scenario runner.
  pub fn new(client: Client, annotations_dir: Option<PathBuf>) -> Self {
    Self {
      client,
      annotations_dir,
      llm_judge: None,
      daemon_pid: OnceCell::new(),
    }
  }

  /// Get the daemon PID, fetching it if not cached.
  async fn get_daemon_pid(&self) -> Result<u32> {
    use ccengram::ipc::system::StatusParams;

    let pid = self
      .daemon_pid
      .get_or_try_init(|| async {
        let status = self
          .client
          .call(StatusParams)
          .await
          .map_err(|e| BenchmarkError::Execution(format!("Failed to get daemon status: {}", e)))?;

        std::result::Result::<u32, BenchmarkError>::Ok(status.pid)
      })
      .await?;

    Ok(*pid)
  }

  /// Enable LLM-as-judge evaluation for comprehension testing.
  ///
  /// When enabled, the runner will evaluate understanding of the codebase
  /// using Claude to answer comprehension questions defined in scenarios.
  pub fn with_llm_judge(mut self) -> Result<Self> {
    let judge = LlmJudge::new();
    if !judge.is_configured() {
      return Err(BenchmarkError::Execution(
        "LLM judge requires 'claude' CLI in PATH. Install it or run without --llm-judge.".into(),
      ));
    }
    self.llm_judge = Some(judge);
    Ok(self)
  }

  /// Get the default socket path.
  pub fn default_socket_path() -> String {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    format!("{}/ccengram.sock", runtime_dir)
  }

  /// Run a single scenario.
  pub async fn run(&self, scenario: &Scenario) -> Result<ScenarioResult> {
    let start = Instant::now();
    let mut session = ExplorationSession::new();
    let mut step_results = Vec::new();
    let mut errors = Vec::new();

    // Initialize resource monitor for memory/CPU tracking
    let daemon_pid = self.get_daemon_pid().await?;
    let mut resource_monitor = ResourceMonitor::new(daemon_pid);
    resource_monitor.snapshot(); // Initial snapshot

    info!(
      "Running scenario: {} ({})",
      scenario.metadata.name, scenario.metadata.id
    );

    // Load annotations if available and merge with expected values
    let mut expected = scenario.expected.clone();
    let mut exploration_paths = Vec::new();
    // Start with empty annotations as a baseline
    let mut annotations: Option<Annotations> = Some(Annotations::empty());

    if let Some(annotations_dir) = &self.annotations_dir {
      // Determine repo-specific annotations directory
      let repo_annotations_dir = annotations_dir.join(scenario.metadata.repo.to_string());
      let loaded_annotations = load_scenario_annotations(&repo_annotations_dir, &scenario.metadata.id).await;

      if !loaded_annotations.is_empty() {
        // Log all critical items for debugging
        let all_critical = loaded_annotations.all_critical();
        debug!(
          "Loaded {} critical files, {} critical symbols, {} exploration paths from annotations",
          loaded_annotations.critical_files.len(),
          loaded_annotations.critical_symbols.len(),
          loaded_annotations.exploration_paths.len()
        );
        tracing::trace!("All critical items: {:?}", all_critical);

        // Merge annotations into expected values
        for file in &loaded_annotations.critical_files {
          if !expected.must_find_files.contains(file) {
            expected.must_find_files.push(file.clone());
          }
        }
        for symbol in &loaded_annotations.critical_symbols {
          if !expected.must_find_symbols.contains(symbol) {
            expected.must_find_symbols.push(symbol.clone());
          }
        }

        // Store exploration paths for navigation efficiency calculation
        exploration_paths = loaded_annotations.exploration_paths.clone();

        // Keep annotations for critical item checks during step execution
        annotations = Some(loaded_annotations);
      }
    }

    // Also try to load default annotations and merge if scenario-specific ones exist
    if let Some(annotations_dir) = &self.annotations_dir {
      let default_annotations_path = annotations_dir.join("default.json");
      if default_annotations_path.exists() {
        let default_annotations = Annotations::load_optional(&default_annotations_path).await;
        if !default_annotations.is_empty()
          && let Some(ref mut ann) = annotations
        {
          // Merge default annotations into scenario-specific ones
          ann.merge(&default_annotations);
          debug!(
            "Merged default annotations: now {} critical files, {} critical symbols",
            ann.critical_files.len(),
            ann.critical_symbols.len()
          );
        }
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
        .execute_step_with_context(
          &resolved_step,
          i,
          &mut session,
          &expected,
          &mut previous_results,
          annotations.as_ref(),
        )
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

    let mut accuracy =
      session.compute_accuracy_metrics_with_paths(&expected, &scenario.success_criteria, &exploration_paths);

    // Compute diagnostics for actionable insights when metrics are poor
    let convergence_threshold = scenario.success_criteria.min_convergence_rate.unwrap_or(0.7);
    let bloat_threshold = scenario.success_criteria.max_context_bloat.unwrap_or(0.3);
    let recall_threshold = scenario.success_criteria.min_discovery_score;

    let diagnostics = session.compute_diagnostics(
      &expected,
      &accuracy,
      convergence_threshold,
      bloat_threshold,
      recall_threshold,
    );

    if diagnostics.has_issues {
      accuracy.diagnostics = Some(diagnostics);
    }

    // Evaluate task requirements for task_completion scenarios
    let task_requirements_result = if scenario.task.intent == TaskIntent::TaskCompletion {
      let result = session.evaluate_task_requirements(&scenario.task_requirements);
      Some(result)
    } else {
      None
    };

    // Determine if scenario passed (before LLM judge)
    let mut passed = errors.is_empty()
      && accuracy.file_recall >= scenario.success_criteria.min_discovery_score
      && accuracy.noise_ratio <= scenario.success_criteria.max_noise_ratio
      && accuracy
        .steps_to_core
        .is_none_or(|s| s <= scenario.success_criteria.max_steps_to_core);

    // For task_completion scenarios, also check task requirements
    if let Some(ref req_result) = task_requirements_result {
      // Require at least 50% of requirements to be met
      if req_result.success_rate < 0.5 {
        passed = false;
      }
    }

    // Build preliminary result for LLM judge evaluation
    let mut result = ScenarioResult {
      scenario_id: scenario.metadata.id.clone(),
      scenario_name: scenario.metadata.name.clone(),
      passed,
      performance,
      accuracy,
      steps: step_results,
      errors,
      total_duration_ms: total_duration.as_millis() as u64,
      comprehension: None,
      task_requirements_result,
    };

    // Run LLM judge evaluation if enabled and scenario has comprehension questions
    if let Some(judge) = &self.llm_judge {
      if !scenario.llm_judge.comprehension_questions.is_empty() {
        info!("Running LLM comprehension evaluation for {}", scenario.metadata.id);

        match judge.evaluate(&result, &scenario.llm_judge).await {
          Ok(comprehension_result) => {
            // Update passed status based on comprehension
            if !comprehension_result.passed {
              result.passed = false;
            }
            result.comprehension = Some(comprehension_result);
          }
          Err(e) => {
            warn!("LLM judge evaluation failed: {}", e);
            result.errors.push(format!("LLM judge error: {}", e));
          }
        }
      } else {
        debug!(
          "Skipping LLM judge for {} (no comprehension questions defined)",
          scenario.metadata.id
        );
      }
    }

    Ok(result)
  }

  /// Execute a single step.
  async fn execute_step(
    &self,
    step: &Step,
    index: usize,
    session: &mut ExplorationSession,
    expected: &Expected,
    annotations: Option<&Annotations>,
  ) -> Result<StepResult> {
    let start = Instant::now();

    // Check if query uses a previous suggestion
    session.check_suggestion_used(&step.query);

    // Build explore request - use step's expand_top or default to 3
    let expand_top = step.expand_top.unwrap_or(3);
    let result: ExploreResult = self
      .client
      .call(ExploreParams {
        query: step.query.clone(),
        scope: Some(step.scope.as_deref().unwrap_or("all").to_string()),
        expand_top: Some(expand_top),
        limit: Some(10),
        depth: None,
      })
      .await?;
    let latency = start.elapsed();

    // Extract result info from typed response
    let results = &result.results;

    let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();

    let files_found: Vec<String> = results.iter().filter_map(|r| r.file_path.clone()).collect();

    let symbols_found: Vec<String> = results.iter().flat_map(|r| r.symbols.clone()).collect();

    // Record MRR data - check if each result is relevant
    for (rank, r) in results.iter().enumerate() {
      let file = r.file_path.as_deref().unwrap_or("");
      let result_symbols: Vec<&str> = r.symbols.iter().map(|s| s.as_str()).collect();

      let is_relevant = self.is_result_relevant(file, &result_symbols, expected, annotations);
      session.record_result_rank(is_relevant, rank + 1);
    }

    // Track context budget - calculate bytes returned
    let response_str = serde_json::to_string(&result).unwrap_or_default();
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
        let file = r.file_path.as_deref().unwrap_or("");
        let result_symbols: Vec<&str> = r.symbols.iter().map(|s| s.as_str()).collect();
        self.is_result_relevant(file, &result_symbols, expected, annotations)
      })
      .count();

    session.record_step_relevance(found_expected_file, found_expected_symbol, relevant_count);

    // Track callers and callees for adaptive templates
    let mut all_callers = Vec::new();
    let mut all_callees = Vec::new();

    // Extract hints from expanded context and build call graph
    for r in results {
      // Get the main symbols of this result for call graph
      let source_symbols: Vec<&str> = r.symbols.iter().map(|s| s.as_str()).collect();

      // Record symbol discovery times
      for symbol in &source_symbols {
        session.record_symbol_discovery(symbol);
      }

      if let Some(context) = &r.context {
        // Record caller hints and build call graph
        for caller in &context.callers {
          session.record_hint(&caller.file);
          // Build call graph: caller_symbol -> source_symbol
          for caller_sym in &caller.symbols {
            all_callers.push(caller_sym.clone());
            for source_sym in &source_symbols {
              session.record_call_relation(caller_sym, source_sym);
            }
          }
        }
        // Record callee hints and build call graph
        for callee in &context.callees {
          session.record_hint(&callee.file);
          // Build call graph: source_symbol -> callee_symbol
          for callee_sym in &callee.symbols {
            all_callees.push(callee_sym.clone());
            for source_sym in &source_symbols {
              session.record_call_relation(source_sym, callee_sym);
            }
          }
        }
        // Record sibling hints
        for sibling in &context.siblings {
          if let Some(file) = &sibling.file {
            session.record_hint(file);
          }
        }
      }
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

    // Calculate noise ratio for this step using comprehensive noise detection
    // Check each result for noise based on file, symbols, and content
    let noise_count: usize = results
      .iter()
      .filter(|r| {
        let file = r.file_path.as_deref();
        // Get first symbol if available (main symbol for the chunk)
        let symbol = r.symbols.first().map(|s| s.as_str());
        // Use response content to check for noise patterns
        let content_str = serde_json::to_string(r).unwrap_or_default();
        session.is_noise_result(file, symbol, Some(&content_str))
      })
      .count();
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
    annotations: Option<&Annotations>,
  ) -> Result<StepResult> {
    let result = self.execute_step(step, index, session, expected, annotations).await?;

    // Update previous_results for next step's template resolution
    previous_results.ids = result.result_ids.clone();
    previous_results.files = result.files_found.clone();
    previous_results.symbols = result.symbols_found.clone();
    previous_results.callers = result.callers.clone();
    previous_results.callees = result.callees.clone();

    Ok(result)
  }

  /// Check if a result is relevant to the expected values.
  /// Also checks against annotations for critical files/symbols if available.
  fn is_result_relevant(
    &self,
    file: &str,
    symbols: &[&str],
    expected: &Expected,
    annotations: Option<&Annotations>,
  ) -> bool {
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

    // Also check against annotations if available
    if let Some(annotations) = annotations {
      // Check if file is critical according to annotations
      if annotations.is_critical_file(file) {
        return true;
      }

      // Check if any symbol is critical according to annotations
      for symbol in symbols {
        if annotations.is_critical_symbol(symbol) {
          return true;
        }
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

    let items: Vec<ContextItem> = self
      .client
      .call(ContextParams {
        id: Some(id.to_string()),
        ids: None,
        depth: Some(5),
      })
      .await?;
    let latency = start.elapsed();

    // Extract new files/symbols from context response
    let mut new_files = Vec::new();
    let mut new_symbols = Vec::new();

    for ctx in &items {
      // Extract from callers
      if let Some(callers) = &ctx.callers {
        for caller in callers {
          if !session.discovered_files().contains(&caller.file_path) {
            new_files.push(caller.file_path.clone());
          }
          for sym in &caller.symbols {
            if !session.discovered_symbols().contains(sym) {
              new_symbols.push(sym.clone());
            }
          }
        }
      }
      // Extract from callees
      if let Some(callees) = &ctx.callees {
        for callee in callees {
          if !session.discovered_files().contains(&callee.file_path) {
            new_files.push(callee.file_path.clone());
          }
          for sym in &callee.symbols {
            if !session.discovered_symbols().contains(sym) {
              new_symbols.push(sym.clone());
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

    Ok(())
  }

  /// Check if the daemon is running.
  pub async fn check_daemon(&self) -> bool {
    match self.client.call(HealthCheckParams).await {
      Ok(result) => result.healthy,
      Err(_) => false,
    }
  }

  /// Export discovered annotations from a scenario run.
  /// This can be used to generate initial annotations for new scenarios.
  pub async fn export_annotations(
    &self,
    scenario_id: &str,
    discovered_files: &[String],
    discovered_symbols: &[String],
    output_path: &std::path::Path,
  ) -> Result<()> {
    use crate::ground_truth::Annotations;

    let annotations = Annotations {
      scenario_id: scenario_id.to_string(),
      critical_files: discovered_files.to_vec(),
      critical_symbols: discovered_symbols.to_vec(),
      key_locations: Vec::new(),
      exploration_paths: Vec::new(),
      notes: vec![format!("Auto-generated from scenario run at {}", chrono::Utc::now())],
    };

    annotations.save(output_path).await?;
    info!("Exported annotations to: {}", output_path.display());

    Ok(())
  }

  /// Build a call graph from a set of discovered call relationships.
  /// Useful for analyzing navigation patterns from scenario results.
  pub fn build_call_graph_from_results(&self, calls: Vec<(String, String)>) -> crate::ground_truth::CallGraph {
    crate::ground_truth::CallGraph::from_calls(calls)
  }
}

/// Run multiple scenarios in parallel.
pub async fn run_scenarios_parallel(runner: &ScenarioRunner, scenarios: &[Scenario]) -> Vec<ScenarioResult> {
  use futures::future::join_all;

  let futures: Vec<_> = scenarios.iter().map(|s| async { runner.run(s).await }).collect();

  let results = join_all(futures).await;

  results.into_iter().filter_map(|r| r.ok()).collect()
}
