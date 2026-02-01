//! Exploration session state tracking.
//!
//! Tracks the state of a multi-step exploration scenario, accumulating
//! discovered files, symbols, and metrics across steps.
//!
//! ## Exploration Metrics
//!
//! Beyond basic recall, tracks exploration-specific metrics:
//! - **Convergence**: How quickly discoveries plateau
//! - **Information gain**: New discoveries per step
//! - **Context value**: Whether context calls provide new information
//! - **Hint tracking**: Navigation hints shown vs followed
//! - **MRR tracking**: Rank of first relevant result
//! - **Navigation efficiency**: Optimal hops vs actual hops to reach targets
//! - **Path-based failures**: Consecutive steps in wrong direction (rabbit holes)
//! - **Context budget**: Cumulative bytes returned vs useful bytes

use std::{
  collections::{HashMap, HashSet},
  time::{Duration, Instant},
};

use crate::{
  ground_truth::{CallGraph, ExplorationPath, NoisePatterns},
  metrics::{
    AccuracyMetrics, BloatDiagnosis, ConvergenceDiagnosis, DiscoveryPattern, ExplorationDiagnostics, LatencyTracker,
    OverExpandedStep, PerformanceMetrics, RecallCategoryBreakdown, RecallDiagnosis, StepMetrics,
  },
  scenarios::{Expected, SuccessCriteria, TaskRequirements, TaskRequirementsResult},
};

/// Per-step discovery record for convergence analysis.
#[derive(Debug, Clone, Default)]
pub struct StepDiscovery {
  /// Step index
  pub step: usize,
  /// New files discovered in this step
  pub new_files: usize,
  /// New symbols discovered in this step
  pub new_symbols: usize,
  /// Total expected files (for info gain calculation)
  pub total_expected_files: usize,
  /// Total expected symbols (for info gain calculation)
  pub total_expected_symbols: usize,
}

/// Record of a navigation hint shown to the user.
#[derive(Debug, Clone)]
pub struct HintRecord {
  /// Target identifier (symbol or file)
  pub target: String,
}

/// Record of a context call and its value.
#[derive(Debug, Clone)]
pub struct ContextCallRecord {
  /// Chunk ID that was expanded
  pub chunk_id: String,
  /// Step when context was fetched
  pub step: usize,
  /// Number of new files discovered from this context call
  pub new_files: usize,
  /// Number of new symbols discovered from this context call
  pub new_symbols: usize,
  /// Bytes returned by this context call
  pub bytes_returned: usize,
}

/// Per-step relevance tracking for path-based failure detection.
#[derive(Debug, Clone)]
pub struct StepRelevance {
  /// Whether this step found any expected files
  pub found_expected_file: bool,
  /// Whether this step found any expected symbols
  pub found_expected_symbol: bool,
  /// Count of relevant results in this step
  pub relevant_count: usize,
}

/// Context budget tracking for cumulative efficiency.
#[derive(Debug, Clone, Default)]
pub struct ContextBudget {
  /// Total bytes returned across all explore/context calls
  pub total_bytes: usize,
  /// Bytes containing expected symbols or files
  pub useful_bytes: usize,
  /// Per-step breakdown of bytes
  pub step_bytes: Vec<(usize, usize)>, // (step_total_bytes, step_useful_bytes)
}

/// State for a multi-step exploration session.
#[derive(Debug)]
pub struct ExplorationSession {
  /// All discovered files
  discovered_files: HashSet<String>,
  /// All discovered symbols
  discovered_symbols: HashSet<String>,
  /// All result IDs seen
  all_result_ids: HashSet<String>,
  /// Per-step metrics
  step_metrics: Vec<StepMetrics>,
  /// Search latency tracker
  search_latencies: LatencyTracker,
  /// Context fetch latency tracker
  context_latencies: LatencyTracker,
  /// Noise patterns for detection
  noise_patterns: NoisePatterns,
  /// Step when first core result was found
  first_core_step: Option<usize>,
  /// Current step index
  current_step: usize,

  // === Exploration-specific tracking ===
  /// Per-step discovery counts (for convergence analysis)
  step_discoveries: Vec<StepDiscovery>,
  /// Hints shown to user (for hint utility calculation)
  hints_shown: Vec<HintRecord>,
  /// Hints that were followed (subset of hints_shown)
  hints_followed: HashSet<String>,
  /// Context calls and their value
  context_calls: Vec<ContextCallRecord>,
  /// Result ranks for MRR calculation: (is_relevant, rank)
  result_ranks: Vec<(bool, usize)>,
  /// Suggestions shown to user
  suggestions_shown: Vec<String>,
  /// Suggestions that were used in subsequent queries
  suggestions_used: HashSet<String>,
  /// Files discovered before each step (for tracking new discoveries)
  files_before_step: usize,
  /// Symbols discovered before each step
  symbols_before_step: usize,

  // === Navigation efficiency tracking ===
  /// Call graph built from exploration (callers/callees from hints)
  call_graph: CallGraph,
  /// When each symbol was first discovered (symbol -> step index)
  symbol_discovery_step: std::collections::HashMap<String, usize>,

  // === Path-based failure tracking (rabbit holes) ===
  /// Per-step relevance for detecting consecutive failures
  step_relevance: Vec<StepRelevance>,

  // === Context budget tracking ===
  /// Cumulative context budget tracking
  context_budget: ContextBudget,

  // === Time-to-first-relevant tracking ===
  /// When the session started (for timing metrics)
  session_start: Instant,
  /// Time elapsed when first relevant result was found
  first_relevant_result_time: Option<Duration>,

  // === File diversity tracking ===
  /// Files discovered per step (step_index -> files in that step's results)
  step_files: HashMap<usize, Vec<String>>,
}

impl ExplorationSession {
  /// Create a new exploration session.
  pub fn new() -> Self {
    Self {
      discovered_files: HashSet::new(),
      discovered_symbols: HashSet::new(),
      all_result_ids: HashSet::new(),
      step_metrics: Vec::new(),
      search_latencies: LatencyTracker::new(),
      context_latencies: LatencyTracker::new(),
      noise_patterns: NoisePatterns::default(),
      first_core_step: None,
      current_step: 0,
      // Exploration tracking
      step_discoveries: Vec::new(),
      hints_shown: Vec::new(),
      hints_followed: HashSet::new(),
      context_calls: Vec::new(),
      result_ranks: Vec::new(),
      suggestions_shown: Vec::new(),
      suggestions_used: HashSet::new(),
      files_before_step: 0,
      symbols_before_step: 0,
      // Navigation efficiency
      call_graph: CallGraph::new(),
      symbol_discovery_step: std::collections::HashMap::new(),
      // Path-based failure tracking
      step_relevance: Vec::new(),
      // Context budget
      context_budget: ContextBudget::default(),
      // Time-to-first-relevant
      session_start: Instant::now(),
      first_relevant_result_time: None,
      // File diversity
      step_files: HashMap::new(),
    }
  }

  /// Record results from an explore step.
  pub fn record_explore_step(
    &mut self,
    _query: &str,
    result_ids: &[String],
    files: &[String],
    symbols: &[String],
    latency: Duration,
  ) {
    // Track discoveries
    for file in files {
      self.discovered_files.insert(file.clone());
    }
    for symbol in symbols {
      self.discovered_symbols.insert(symbol.clone());
    }
    for id in result_ids {
      self.all_result_ids.insert(id.clone());
    }

    // Track files per step for diversity calculation
    self.step_files.insert(self.current_step, files.to_vec());

    // Track latency using LatencyTracker
    self.search_latencies.record(latency);

    // Record step metrics
    self.step_metrics.push(StepMetrics {
      step_index: self.current_step,
      latency_ms: latency.as_millis() as u64,
      result_count: result_ids.len(),
      context_latencies_ms: vec![],
    });

    self.current_step += 1;
  }

  /// Record a context fetch.
  pub fn record_context_call(&mut self, _id: &str, latency: Duration) {
    self.context_latencies.record(latency);

    // Add to current step's context latencies
    if let Some(step) = self.step_metrics.last_mut() {
      step.context_latencies_ms.push(latency.as_millis() as u64);
    }
  }

  // === Exploration tracking methods ===

  /// Record step discoveries for convergence analysis.
  /// Call this after record_explore_step with the expected values.
  pub fn record_step_discoveries(&mut self, expected: &Expected) {
    let new_files = self.discovered_files.len() - self.files_before_step;
    let new_symbols = self.discovered_symbols.len() - self.symbols_before_step;

    self.step_discoveries.push(StepDiscovery {
      step: self.current_step.saturating_sub(1),
      new_files,
      new_symbols,
      total_expected_files: expected.must_find_files.len(),
      total_expected_symbols: expected.must_find_symbols.len(),
    });

    // Update baseline for next step
    self.files_before_step = self.discovered_files.len();
    self.symbols_before_step = self.discovered_symbols.len();
  }

  /// Record a result's relevance and rank for MRR calculation.
  /// Also records time to first relevant result if this is the first relevant result.
  pub fn record_result_rank(&mut self, is_relevant: bool, rank: usize) {
    self.result_ranks.push((is_relevant, rank));
    self.record_first_relevant_if_needed(is_relevant);
  }

  /// Record the time to first relevant result if this is the first one found.
  fn record_first_relevant_if_needed(&mut self, is_relevant: bool) {
    if is_relevant && self.first_relevant_result_time.is_none() {
      self.first_relevant_result_time = Some(self.session_start.elapsed());
    }
  }

  /// Record a navigation hint shown to the user.
  pub fn record_hint(&mut self, target: &str) {
    self.hints_shown.push(HintRecord {
      target: target.to_string(),
    });
  }

  /// Mark a hint as followed (user expanded or queried for it).
  pub fn mark_hint_followed(&mut self, target: &str) {
    self.hints_followed.insert(target.to_string());
  }

  /// Record a context call with its value (whether it provided new info).
  pub fn record_context_value(&mut self, chunk_id: &str, new_files: usize, new_symbols: usize) {
    self.record_context_value_with_bytes(chunk_id, new_files, new_symbols, 0, 0);
  }

  /// Record a context call with full byte tracking.
  pub fn record_context_value_with_bytes(
    &mut self,
    chunk_id: &str,
    new_files: usize,
    new_symbols: usize,
    bytes_returned: usize,
    useful_bytes: usize,
  ) {
    self.context_calls.push(ContextCallRecord {
      chunk_id: chunk_id.to_string(),
      step: self.current_step.saturating_sub(1),
      new_files,
      new_symbols,
      bytes_returned,
    });

    // Update context budget
    self.context_budget.total_bytes += bytes_returned;
    self.context_budget.useful_bytes += useful_bytes;
  }

  /// Record bytes from an explore step for context budget tracking.
  pub fn record_explore_bytes(&mut self, total_bytes: usize, useful_bytes: usize) {
    self.context_budget.total_bytes += total_bytes;
    self.context_budget.useful_bytes += useful_bytes;
    self.context_budget.step_bytes.push((total_bytes, useful_bytes));
  }

  /// Record a call relationship discovered from explore/context hints.
  pub fn record_call_relation(&mut self, caller: &str, callee: &str) {
    self.call_graph.add_call(caller, callee);
  }

  /// Record when a symbol was first discovered (for navigation efficiency).
  pub fn record_symbol_discovery(&mut self, symbol: &str) {
    let step = self.current_step.saturating_sub(1);
    self.symbol_discovery_step.entry(symbol.to_string()).or_insert(step);
  }

  /// Record step relevance for path-based failure tracking.
  pub fn record_step_relevance(
    &mut self,
    found_expected_file: bool,
    found_expected_symbol: bool,
    relevant_count: usize,
  ) {
    self.step_relevance.push(StepRelevance {
      found_expected_file,
      found_expected_symbol,
      relevant_count,
    });
  }

  /// Check if a query matches a previous suggestion and mark it as used.
  pub fn check_suggestion_used(&mut self, query: &str) {
    let query_lower = query.to_lowercase();
    for suggestion in &self.suggestions_shown {
      if query_lower.contains(&suggestion.to_lowercase()) || suggestion.to_lowercase().contains(&query_lower) {
        self.suggestions_used.insert(suggestion.clone());
      }
    }
  }

  /// Calculate convergence rate (how quickly discoveries plateau).
  /// Returns 1.0 if all discoveries happen early, lower if discoveries are spread out.
  pub fn calculate_convergence_rate(&self) -> f64 {
    if self.step_discoveries.is_empty() {
      return 1.0;
    }

    let total_discoveries: usize = self.step_discoveries.iter().map(|s| s.new_files + s.new_symbols).sum();

    if total_discoveries == 0 {
      return 1.0;
    }

    // Weight earlier discoveries higher
    let mut weighted_sum = 0.0;
    let num_steps = self.step_discoveries.len();

    for (i, discovery) in self.step_discoveries.iter().enumerate() {
      let step_discoveries = discovery.new_files + discovery.new_symbols;
      // Earlier steps get higher weight
      let weight = (num_steps - i) as f64 / num_steps as f64;
      weighted_sum += step_discoveries as f64 * weight;
    }

    // Normalize: perfect convergence (all in step 0) = 1.0
    let max_possible = total_discoveries as f64; // if all in first step
    weighted_sum / max_possible
  }

  /// Calculate average information gain per step.
  /// Higher = more productive steps.
  pub fn calculate_avg_info_gain(&self) -> f64 {
    if self.step_discoveries.is_empty() {
      return 0.0;
    }

    let gains: Vec<f64> = self
      .step_discoveries
      .iter()
      .map(|s| {
        let total_expected = s.total_expected_files + s.total_expected_symbols;
        if total_expected == 0 {
          0.0
        } else {
          (s.new_files + s.new_symbols) as f64 / total_expected as f64
        }
      })
      .collect();

    gains.iter().sum::<f64>() / gains.len() as f64
  }

  /// Calculate context bloat (% of context calls that provided no new info).
  pub fn calculate_context_bloat(&self) -> f64 {
    if self.context_calls.is_empty() {
      return 0.0;
    }

    let empty_calls = self
      .context_calls
      .iter()
      .filter(|c| c.new_files == 0 && c.new_symbols == 0)
      .count();

    empty_calls as f64 / self.context_calls.len() as f64
  }

  /// Calculate dead end ratio (% of steps that found nothing relevant).
  pub fn calculate_dead_end_ratio(&self, _expected: &Expected) -> f64 {
    if self.step_discoveries.is_empty() {
      return 0.0;
    }

    let dead_ends = self
      .step_discoveries
      .iter()
      .filter(|s| s.new_files == 0 && s.new_symbols == 0)
      .count();

    dead_ends as f64 / self.step_discoveries.len() as f64
  }

  /// Calculate file diversity for a specific step.
  /// Returns unique_files / min(top_n, total_results).
  /// 1.0 = perfect diversity (all different files), lower = more files from same location.
  pub fn calculate_step_file_diversity(&self, step: usize, top_n: usize) -> f64 {
    if let Some(files) = self.step_files.get(&step) {
      if files.is_empty() {
        return 1.0; // No results = no diversity problem
      }

      // Take top N files
      let top_files: Vec<_> = files.iter().take(top_n).collect();
      let total = top_files.len();

      if total == 0 {
        return 1.0;
      }

      // Count unique files in top N
      let unique_files: HashSet<_> = top_files.into_iter().collect();
      unique_files.len() as f64 / total as f64
    } else {
      1.0 // Step not found = no penalty
    }
  }

  /// Calculate average file diversity across all steps for top-5 results.
  /// Higher = better (more diverse file coverage in results).
  pub fn calculate_avg_file_diversity(&self) -> f64 {
    if self.step_files.is_empty() {
      return 1.0; // No steps = perfect score
    }

    let mut total_diversity = 0.0;
    let mut step_count = 0;

    for step in 0..self.current_step {
      let diversity = self.calculate_step_file_diversity(step, 5);
      total_diversity += diversity;
      step_count += 1;
    }

    if step_count == 0 {
      return 1.0;
    }

    total_diversity / step_count as f64
  }

  /// Calculate navigation efficiency using exploration paths.
  /// Returns optimal_hops / actual_hops averaged across all paths.
  /// A value of 1.0 means optimal navigation, lower means less efficient.
  pub fn calculate_navigation_efficiency(&self, paths: &[ExplorationPath]) -> f64 {
    if paths.is_empty() {
      return 1.0; // No paths defined = no penalty
    }

    let mut total_efficiency = 0.0;
    let mut valid_paths = 0;

    for path in paths {
      // Check if both start and target were discovered
      let start_step = self.symbol_discovery_step.get(&path.start);
      let target_step = self.symbol_discovery_step.get(&path.target);

      if let (Some(&start), Some(&target)) = (start_step, target_step) {
        // Actual hops = steps between discovering start and target
        let actual_hops = if target >= start { target - start + 1 } else { 1 };

        // Use the call graph to check if there's a shorter path
        let graph_hops = self.call_graph.path_length(&path.start, &path.target);

        // Check if the path is reachable within the max_hops constraint
        let within_max_hops = self.is_reachable_within(&path.start, &path.target, path.max_hops);

        // Optimal = min of annotated max_hops and graph path (if available)
        let optimal_hops = graph_hops.map(|g| g.min(path.max_hops)).unwrap_or(path.max_hops);

        // Efficiency = optimal / actual (capped at 1.0)
        let efficiency = (optimal_hops as f64 / actual_hops as f64).min(1.0);
        total_efficiency += efficiency;
        valid_paths += 1;

        // Log path details for debugging
        tracing::trace!(
          "Path {} -> {}: actual={}, optimal={}, within_max={}, efficiency={:.2}",
          path.start,
          path.target,
          actual_hops,
          optimal_hops,
          within_max_hops,
          efficiency
        );
      }
    }

    if valid_paths == 0 {
      return 0.0; // No paths were completed
    }

    total_efficiency / valid_paths as f64
  }

  /// Calculate cumulative context budget efficiency.
  /// Returns the ratio of useful_bytes / total_bytes.
  pub fn calculate_context_budget_efficiency(&self) -> f64 {
    if self.context_budget.total_bytes == 0 {
      return 1.0; // No bytes = no waste
    }

    self.context_budget.useful_bytes as f64 / self.context_budget.total_bytes as f64
  }

  /// Get context budget summary.
  pub fn get_context_budget(&self) -> &ContextBudget {
    &self.context_budget
  }

  /// Calculate path-based failure metrics (rabbit holes).
  /// Returns (max_consecutive_failures, total_rabbit_hole_steps, rabbit_hole_ratio).
  pub fn calculate_rabbit_holes(&self) -> (usize, usize, f64) {
    if self.step_relevance.is_empty() {
      return (0, 0, 0.0);
    }

    let mut max_consecutive = 0;
    let mut current_consecutive = 0;
    let mut total_rabbit_hole_steps = 0;
    let mut in_rabbit_hole = false;

    for step in &self.step_relevance {
      let is_relevant = step.found_expected_file || step.found_expected_symbol || step.relevant_count > 0;

      if !is_relevant {
        current_consecutive += 1;
        if current_consecutive >= 2 {
          // 2+ consecutive failures = rabbit hole
          if !in_rabbit_hole {
            in_rabbit_hole = true;
            // Count the first step too
            total_rabbit_hole_steps += current_consecutive;
          } else {
            total_rabbit_hole_steps += 1;
          }
        }
        max_consecutive = max_consecutive.max(current_consecutive);
      } else {
        current_consecutive = 0;
        in_rabbit_hole = false;
      }
    }

    let ratio = total_rabbit_hole_steps as f64 / self.step_relevance.len() as f64;
    (max_consecutive, total_rabbit_hole_steps, ratio)
  }

  /// Calculate the overall noise ratio for this session.
  /// Uses the NoisePatterns::noise_ratio method for consistent calculation.
  pub fn calculate_noise_ratio(&self) -> f64 {
    let files: Vec<String> = self.discovered_files.iter().cloned().collect();
    let symbols: Vec<String> = self.discovered_symbols.iter().cloned().collect();
    self.noise_patterns.noise_ratio(&files, &symbols)
  }

  /// Check if a specific result is noise (file, symbol, or content).
  pub fn is_noise_result(&self, file: Option<&str>, symbol: Option<&str>, content: Option<&str>) -> bool {
    self.noise_patterns.is_noise(file, symbol, content)
  }

  /// Validate hints using the call graph.
  /// Returns (valid_hints_count, total_hints_count).
  pub fn validate_hints_with_call_graph(&self, expected_symbols: &[String]) -> (usize, usize) {
    let mut valid_count = 0;
    let total = self.hints_shown.len();

    for hint in &self.hints_shown {
      // A hint is valid if it leads to an expected symbol (reachable in call graph)
      let is_valid = expected_symbols.iter().any(|expected| {
        self.call_graph.is_reachable(&hint.target, expected) || self.call_graph.is_reachable(expected, &hint.target)
      });

      if is_valid {
        valid_count += 1;
      }
    }

    (valid_count, total)
  }

  /// Score the hints shown during exploration against the call graph.
  /// Returns the ratio of hints that exist in the discovered call graph.
  pub fn score_hints(&self) -> (usize, usize) {
    let hints: Vec<String> = self.hints_shown.iter().map(|h| h.target.clone()).collect();
    self.call_graph.score_hints(&hints)
  }

  /// Get the callers of a symbol from the discovered call graph.
  pub fn get_callers(&self, symbol: &str) -> Vec<String> {
    self.call_graph.callers(symbol)
  }

  /// Get the callees of a symbol from the discovered call graph.
  pub fn get_callees(&self, symbol: &str) -> Vec<String> {
    self.call_graph.callees(symbol)
  }

  /// Check if a path exists between two symbols in the discovered call graph.
  pub fn is_reachable(&self, source: &str, target: &str) -> bool {
    self.call_graph.is_reachable(source, target)
  }

  /// Check if a path exists within N hops in the discovered call graph.
  pub fn is_reachable_within(&self, source: &str, target: &str, max_hops: usize) -> bool {
    self.call_graph.is_reachable_within(source, target, max_hops)
  }

  /// Get call graph statistics.
  pub fn call_graph_stats(&self) -> (usize, usize) {
    (self.call_graph.symbol_count(), self.call_graph.edge_count())
  }

  /// Get all discovered files.
  pub fn discovered_files(&self) -> &HashSet<String> {
    &self.discovered_files
  }

  /// Get all discovered symbols.
  pub fn discovered_symbols(&self) -> &HashSet<String> {
    &self.discovered_symbols
  }

  /// Compute performance metrics for this session.
  pub fn compute_performance_metrics(&self) -> PerformanceMetrics {
    let search_latency = self.search_latencies.stats();
    let context_latency = self.context_latencies.stats();

    // Calculate total time from step metrics
    let total_time_ms: u64 = self
      .step_metrics
      .iter()
      .map(|s| s.latency_ms + s.context_latencies_ms.iter().sum::<u64>())
      .sum();

    PerformanceMetrics {
      search_latency,
      context_latency,
      total_time_ms,
      steps: self.step_metrics.clone(),
      peak_memory_bytes: None,
      avg_cpu_percent: None,
    }
  }

  /// Compute accuracy metrics for this session.
  pub fn compute_accuracy_metrics(&self, expected: &Expected, _criteria: &SuccessCriteria) -> AccuracyMetrics {
    let mut builder = AccuracyMetrics::builder()
      .expected_files(expected.must_find_files.iter().cloned())
      .expected_symbols(expected.must_find_symbols.iter().cloned())
      .record_files(self.discovered_files.iter().cloned())
      .record_symbols(self.discovered_symbols.iter().cloned());

    // Record noise for all discovered files and symbols
    for file in &self.discovered_files {
      builder = builder.record_noise(self.noise_patterns.is_noise_file(file));
    }
    for symbol in &self.discovered_symbols {
      builder = builder.record_noise(self.noise_patterns.is_noise_symbol(symbol));
    }

    // Also calculate the overall noise ratio using the more comprehensive method
    let noise_ratio = self.calculate_noise_ratio();
    tracing::debug!("Overall noise ratio: {:.2}%", noise_ratio * 100.0);

    // Record MRR data from result ranks
    for (is_relevant, rank) in &self.result_ranks {
      builder = builder.record_result_rank(*is_relevant, *rank);
    }

    // Record hint utility (hints followed / hints shown)
    for hint in &self.hints_shown {
      let was_followed = self.hints_followed.contains(&hint.target);
      builder = builder.record_hint_relevance(was_followed);
    }

    // Record suggestion quality (suggestions used / suggestions shown)
    for suggestion in &self.suggestions_shown {
      let was_used = self.suggestions_used.contains(suggestion);
      builder = builder.record_suggestion_usefulness(was_used);
    }

    // Set steps to core if found
    if let Some(step) = self.first_core_step {
      builder = builder.set_step_found_core(step);
    } else {
      // Check if any expected file was found
      for (i, step) in self.step_metrics.iter().enumerate() {
        // If this step had results and we found an expected file
        if step.result_count > 0 {
          for expected_file in &expected.must_find_files {
            if self.discovered_files.iter().any(|f| {
              f.ends_with(expected_file) || glob::Pattern::new(expected_file).map(|p| p.matches(f)).unwrap_or(false)
            }) {
              builder = builder.set_step_found_core(i);
              break;
            }
          }
        }
      }
    }

    // Set exploration metrics (computed from session state)
    builder = builder
      .set_convergence_rate(self.calculate_convergence_rate())
      .set_avg_info_gain(self.calculate_avg_info_gain())
      .set_context_bloat(self.calculate_context_bloat())
      .set_dead_end_ratio(self.calculate_dead_end_ratio(expected));

    // Set context budget metrics
    let budget = self.get_context_budget();
    builder = builder.set_context_budget(
      self.calculate_context_budget_efficiency(),
      budget.total_bytes,
      budget.useful_bytes,
    );

    // Set rabbit hole metrics
    let (max_consecutive, total_rabbit, ratio) = self.calculate_rabbit_holes();
    builder = builder.set_rabbit_holes(max_consecutive, total_rabbit, ratio);

    // Set time-to-first-relevant metric
    let time_to_first_ms = self.first_relevant_result_time.map(|d| d.as_millis() as u64);
    builder = builder.set_time_to_first_relevant_ms(time_to_first_ms);

    // Set file diversity metric
    let avg_diversity = self.calculate_avg_file_diversity();
    builder = builder.set_avg_file_diversity_top5(avg_diversity);

    // Validate hints against the discovered call graph
    // This helps understand if our hints point to real code structure
    let (hints_in_graph, total_hints) = self.score_hints();
    if total_hints > 0 {
      let hint_graph_coverage = hints_in_graph as f64 / total_hints as f64;
      // If many hints aren't in our graph, the call graph may be incomplete
      // or hints may be pointing to irrelevant code
      tracing::debug!(
        "Hint graph coverage: {}/{} ({:.1}%)",
        hints_in_graph,
        total_hints,
        hint_graph_coverage * 100.0
      );
    }

    // Validate hints lead to expected symbols using call graph reachability
    let (valid_hints, _total) = self.validate_hints_with_call_graph(&expected.must_find_symbols);
    tracing::debug!(
      "Hints validated against expected symbols: {}/{}",
      valid_hints,
      total_hints
    );

    // Log call graph statistics for debugging
    let (symbol_count, edge_count) = self.call_graph_stats();
    tracing::debug!("Discovered call graph: {} symbols, {} edges", symbol_count, edge_count);

    // Log the actual symbols in the call graph at trace level
    let graph_symbols = self.call_graph.symbols();
    tracing::trace!("Call graph symbols: {:?}", graph_symbols);

    builder.build()
  }

  /// Compute accuracy metrics with navigation efficiency from exploration paths.
  pub fn compute_accuracy_metrics_with_paths(
    &self,
    expected: &Expected,
    criteria: &SuccessCriteria,
    exploration_paths: &[ExplorationPath],
  ) -> AccuracyMetrics {
    let mut metrics = self.compute_accuracy_metrics(expected, criteria);

    // Calculate and set navigation efficiency
    metrics.navigation_efficiency = self.calculate_navigation_efficiency(exploration_paths);

    metrics
  }

  /// Compute exploration diagnostics for actionable insights.
  ///
  /// Returns diagnostics only when metrics indicate problems:
  /// - Convergence diagnosis when convergence_rate < threshold
  /// - Bloat diagnosis when context_bloat > threshold
  /// - Recall diagnosis when file_recall or symbol_recall < threshold
  ///
  /// These diagnostics explain WHY metrics are poor and suggest fixes.
  pub fn compute_diagnostics(
    &self,
    expected: &Expected,
    accuracy: &AccuracyMetrics,
    convergence_threshold: f64,
    bloat_threshold: f64,
    recall_threshold: f64,
  ) -> ExplorationDiagnostics {
    let mut diagnostics = ExplorationDiagnostics::default();

    // Compute convergence diagnosis if needed
    if accuracy.convergence_rate < convergence_threshold {
      diagnostics.convergence = Some(self.diagnose_convergence(expected));
      diagnostics.has_issues = true;
    }

    // Compute bloat diagnosis if needed
    if accuracy.context_bloat > bloat_threshold {
      diagnostics.bloat = Some(self.diagnose_bloat());
      diagnostics.has_issues = true;
    }

    // Compute recall diagnosis if needed
    if accuracy.file_recall < recall_threshold || accuracy.symbol_recall < recall_threshold {
      diagnostics.recall = Some(self.diagnose_recall(expected, accuracy));
      diagnostics.has_issues = true;
    }

    diagnostics
  }

  /// Diagnose why convergence is low.
  fn diagnose_convergence(&self, _expected: &Expected) -> ConvergenceDiagnosis {
    let mut diagnosis = ConvergenceDiagnosis::default();

    // Find empty and productive steps
    for discovery in &self.step_discoveries {
      if discovery.new_files == 0 && discovery.new_symbols == 0 {
        diagnosis.empty_steps.push(discovery.step);
      } else {
        diagnosis.productive_steps.push(discovery.step);
      }
    }

    // Determine discovery pattern
    let total_steps = self.step_discoveries.len();
    if total_steps == 0 || diagnosis.productive_steps.is_empty() {
      diagnosis.discovery_pattern = DiscoveryPattern::NoDiscoveries;
      diagnosis.recommendation =
        "No discoveries were made. Try broader initial queries or check if the index is populated.".to_string();
    } else {
      // Calculate weighted position of discoveries
      let total_discoveries: usize = self.step_discoveries.iter().map(|d| d.new_files + d.new_symbols).sum();

      let mut weighted_position = 0.0;
      for discovery in &self.step_discoveries {
        let step_discoveries = discovery.new_files + discovery.new_symbols;
        if step_discoveries > 0 {
          weighted_position += (discovery.step as f64 / total_steps as f64) * (step_discoveries as f64);
        }
      }
      weighted_position /= total_discoveries as f64;

      diagnosis.discovery_pattern = if weighted_position < 0.35 {
        DiscoveryPattern::FrontLoaded
      } else if weighted_position > 0.65 {
        DiscoveryPattern::BackLoaded
      } else {
        DiscoveryPattern::EvenlySpread
      };

      // Generate recommendation based on pattern
      diagnosis.recommendation = match diagnosis.discovery_pattern {
        DiscoveryPattern::FrontLoaded => "Discoveries are front-loaded (good). Consider fewer steps.".to_string(),
        DiscoveryPattern::EvenlySpread => {
          "Discoveries spread evenly. Earlier queries may be too narrow - try broader initial queries.".to_string()
        }
        DiscoveryPattern::BackLoaded => {
          "Most discoveries happen late. Initial queries are too narrow or missing key terms. \
           Try starting with broader architectural questions."
            .to_string()
        }
        DiscoveryPattern::NoDiscoveries => {
          "No discoveries were made. Check index population and try different query terms.".to_string()
        }
      };
    }

    diagnosis
  }

  /// Diagnose why context bloat is high.
  fn diagnose_bloat(&self) -> BloatDiagnosis {
    let mut diagnosis = BloatDiagnosis::default();

    if self.context_calls.is_empty() {
      diagnosis.recommendation = "No context calls were made. Bloat may be from explore results.".to_string();
      return diagnosis;
    }

    // Count redundant expansions (same chunk expanded multiple times)
    let mut seen_chunks: HashSet<&str> = HashSet::new();
    let mut redundant_count = 0;
    let mut unhelpful_count = 0;

    for call in &self.context_calls {
      if seen_chunks.contains(call.chunk_id.as_str()) {
        redundant_count += 1;
        diagnosis.redundant_chunks.push(call.chunk_id.clone());
      } else {
        seen_chunks.insert(&call.chunk_id);
      }

      if call.new_files == 0 && call.new_symbols == 0 {
        unhelpful_count += 1;
      }
    }

    let total_calls = self.context_calls.len();
    diagnosis.redundant_expansion_pct = redundant_count as f64 / total_calls as f64;
    diagnosis.unhelpful_hints_pct = unhelpful_count as f64 / total_calls as f64;

    // Calculate wasted bytes
    diagnosis.wasted_bytes = self
      .context_calls
      .iter()
      .filter(|c| c.new_files == 0 && c.new_symbols == 0)
      .map(|c| c.bytes_returned)
      .sum();

    // Identify over-expanded steps by comparing expand_top to useful expansions
    // TODO: For now, we track steps where context calls yielded nothing
    for (step, calls_in_step) in self.context_calls.iter().fold(
      std::collections::HashMap::<usize, Vec<&ContextCallRecord>>::new(),
      |mut acc, call| {
        acc.entry(call.step).or_default().push(call);
        acc
      },
    ) {
      let useful = calls_in_step
        .iter()
        .filter(|c| c.new_files > 0 || c.new_symbols > 0)
        .count();
      let total = calls_in_step.len();
      if total > 2 && useful < total / 2 {
        diagnosis.over_expanded_steps.push(OverExpandedStep {
          step,
          expand_top_used: total,
          useful_expansions: useful,
        });
      }
    }

    diagnosis.over_expansion_pct = if !diagnosis.over_expanded_steps.is_empty() {
      diagnosis.over_expanded_steps.len() as f64 / self.step_discoveries.len().max(1) as f64
    } else {
      0.0
    };

    // Generate recommendation
    let mut issues = Vec::new();
    if diagnosis.redundant_expansion_pct > 0.1 {
      issues.push("redundant expansions (same chunk expanded multiple times)");
    }
    if diagnosis.unhelpful_hints_pct > 0.3 {
      issues.push("unhelpful context calls (callers/callees not relevant)");
    }
    if diagnosis.over_expansion_pct > 0.2 {
      issues.push("over-expansion (expand_top too high for query specificity)");
    }

    diagnosis.recommendation = if issues.is_empty() {
      "Context bloat is marginal. Consider reducing expand_top for narrow queries.".to_string()
    } else {
      format!(
        "Bloat caused by: {}. Consider: reducing expand_top, caching expanded chunks, or improving hint relevance scoring.",
        issues.join(", ")
      )
    };

    diagnosis
  }

  /// Diagnose why recall is low.
  fn diagnose_recall(&self, expected: &Expected, accuracy: &AccuracyMetrics) -> RecallDiagnosis {
    let mut diagnosis = RecallDiagnosis {
      // Copy found/missed items
      files_found: accuracy.files_found.clone(),
      symbols_found: accuracy.symbols_found.clone(),
      ..Default::default()
    };

    // Categorize missed items
    // TODO: For now, we can't distinguish between "not in index" and "not retrieved"
    // without access to the index. We'll mark all as "not retrieved" and let
    // the user investigate further.

    for missed_file in &accuracy.files_missed {
      // Check if any discovered file partially matches (could indicate ranking issue)
      let partial_match = self.discovered_files.iter().any(|f| {
        f.contains(missed_file.trim_start_matches("**/"))
          || missed_file.contains(f.split('/').next_back().unwrap_or(""))
      });

      if partial_match {
        // Found something similar - likely a ranking or specificity issue
        diagnosis.in_index_not_retrieved.push(missed_file.clone());
      } else {
        // Completely missed - could be indexing or query issue
        diagnosis.not_in_index.push(missed_file.clone());
      }
    }

    for missed_symbol in &accuracy.symbols_missed {
      // Check if discovered symbols contain partial matches
      let partial_match = self
        .discovered_symbols
        .iter()
        .any(|s| s.contains(missed_symbol) || missed_symbol.contains(s));

      // Also check if we can reach the missed symbol from any discovered symbol
      // This helps determine if it's a navigation issue vs. retrieval issue
      let reachable_from_discovered = accuracy
        .symbols_found
        .iter()
        .any(|found| self.is_reachable(found, missed_symbol) || self.is_reachable(missed_symbol, found));

      if partial_match || reachable_from_discovered {
        diagnosis.in_index_not_retrieved.push(missed_symbol.clone());

        // Log callers/callees that might help navigate to the missed symbol
        let callers = self.get_callers(missed_symbol);
        let callees = self.get_callees(missed_symbol);
        if !callers.is_empty() || !callees.is_empty() {
          tracing::debug!(
            "Missed symbol '{}' has {} callers and {} callees in discovered graph",
            missed_symbol,
            callers.len(),
            callees.len()
          );
        }
      } else {
        diagnosis.not_in_index.push(missed_symbol.clone());
      }
    }

    // Calculate category breakdown
    let total_missed = expected.must_find_files.len() + expected.must_find_symbols.len()
      - accuracy.files_found.len()
      - accuracy.symbols_found.len();

    if total_missed > 0 {
      let not_retrieved = diagnosis.in_index_not_retrieved.len();
      let not_indexed = diagnosis.not_in_index.len();
      let low_ranked = diagnosis.retrieved_low_rank.len();

      diagnosis.category_breakdown = RecallCategoryBreakdown {
        indexing_issues_pct: not_indexed as f64 / total_missed as f64,
        retrieval_issues_pct: not_retrieved as f64 / total_missed as f64,
        ranking_issues_pct: low_ranked as f64 / total_missed.max(1) as f64,
      };
    }

    // Generate recommendation
    let mut recommendations: Vec<String> = Vec::new();

    if diagnosis.category_breakdown.indexing_issues_pct > 0.3 {
      recommendations.push("Check if expected files are being indexed (file patterns, gitignore)".to_string());
    }
    if diagnosis.category_breakdown.retrieval_issues_pct > 0.3 {
      recommendations.push("Queries may not match expected items semantically - try different phrasings".to_string());
    }
    if !diagnosis.not_in_index.is_empty() {
      recommendations.push(format!(
        "These items weren't found at all: {}",
        diagnosis.not_in_index.join(", ")
      ));
    }

    diagnosis.recommendation = if recommendations.is_empty() {
      "Recall issues are minor. Consider adding more exploration steps.".to_string()
    } else {
      recommendations.join(". ")
    };

    diagnosis
  }

  /// Evaluate task requirements against discoveries.
  ///
  /// For task_completion scenarios, this evaluates whether the exploration
  /// discovered enough to complete the specified task, using pattern matching
  /// against discovered files and symbols.
  pub fn evaluate_task_requirements(&self, requirements: &TaskRequirements) -> TaskRequirementsResult {
    let mut result = TaskRequirementsResult::default();
    let mut requirements_met = 0;
    let mut total_requirements = 0;

    // Check modification points
    if requirements.must_identify_modification_points {
      total_requirements += 1;

      for indicator in &requirements.modification_point_indicators {
        // Check if any discovered file or symbol matches the indicator
        let matching_files: Vec<_> = self
          .discovered_files
          .iter()
          .filter(|f| f.to_lowercase().contains(&indicator.to_lowercase()))
          .cloned()
          .collect();

        let matching_symbols: Vec<_> = self
          .discovered_symbols
          .iter()
          .filter(|s| s.to_lowercase().contains(&indicator.to_lowercase()))
          .cloned()
          .collect();

        result.modification_points.extend(matching_files);
        result.modification_points.extend(matching_symbols);
      }

      result.modification_point_found = !result.modification_points.is_empty();
      if result.modification_point_found {
        requirements_met += 1;
      }
    }

    // Check examples
    if requirements.must_find_example {
      total_requirements += 1;

      for indicator in &requirements.example_indicators {
        let matching_files: Vec<_> = self
          .discovered_files
          .iter()
          .filter(|f| f.to_lowercase().contains(&indicator.to_lowercase()))
          .cloned()
          .collect();

        let matching_symbols: Vec<_> = self
          .discovered_symbols
          .iter()
          .filter(|s| s.to_lowercase().contains(&indicator.to_lowercase()))
          .cloned()
          .collect();

        result.examples.extend(matching_files);
        result.examples.extend(matching_symbols);
      }

      result.example_found = !result.examples.is_empty();
      if result.example_found {
        requirements_met += 1;
      }
    }

    // Check related concerns
    for concern in &requirements.must_find_related_concerns {
      total_requirements += 1;

      let indicators = requirements
        .concern_indicators
        .get(concern)
        .cloned()
        .unwrap_or_default();
      let mut evidence = Vec::new();

      for indicator in &indicators {
        let matching: Vec<_> = self
          .discovered_files
          .iter()
          .chain(self.discovered_symbols.iter())
          .filter(|item| item.to_lowercase().contains(&indicator.to_lowercase()))
          .cloned()
          .collect();
        evidence.extend(matching);
      }

      // Also check if the concern itself appears in discovered items
      let concern_lower = concern.to_lowercase();
      let direct_matches: Vec<_> = self
        .discovered_files
        .iter()
        .chain(self.discovered_symbols.iter())
        .filter(|item| item.to_lowercase().contains(&concern_lower))
        .cloned()
        .collect();
      evidence.extend(direct_matches);

      let found = !evidence.is_empty();
      result.concerns_found.insert(concern.clone(), found);
      result.concern_evidence.insert(concern.clone(), evidence);

      if found {
        requirements_met += 1;
      }
    }

    // Calculate success rate
    result.success_rate = if total_requirements > 0 {
      requirements_met as f64 / total_requirements as f64
    } else {
      1.0 // No requirements = automatic success
    };

    result
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_record_context_call() {
    let mut session = ExplorationSession::new();

    // First record an explore step
    session.record_explore_step("q", &["id1".to_string()], &[], &[], Duration::from_millis(50));

    // Then record a context call
    session.record_context_call("id1", Duration::from_millis(30));

    assert_eq!(session.context_latencies.stats().count, 1);
    assert_eq!(session.step_metrics[0].context_latencies_ms.len(), 1);
  }

  #[test]
  fn test_performance_metrics() {
    let mut session = ExplorationSession::new();

    session.record_explore_step("q1", &["id1".to_string()], &[], &[], Duration::from_millis(100));
    session.record_explore_step("q2", &["id2".to_string()], &[], &[], Duration::from_millis(200));

    let metrics = session.compute_performance_metrics();

    assert_eq!(metrics.steps.len(), 2);
    assert_eq!(metrics.search_latency.count, 2);
    assert_eq!(metrics.total_time_ms, 300);
  }

  #[test]
  fn test_accuracy_metrics_with_expectations() {
    let mut session = ExplorationSession::new();

    session.record_explore_step(
      "test",
      &["id1".to_string()],
      &["src/commands.rs".to_string(), "src/tests/test.rs".to_string()],
      &["Command".to_string(), "execute".to_string()],
      Duration::from_millis(100),
    );

    let expected = Expected {
      must_find_files: vec!["src/commands.rs".to_string(), "src/keymap.rs".to_string()],
      must_find_symbols: vec!["Command".to_string(), "Keymap".to_string()],
      noise_patterns: vec!["**/tests/**".to_string()],
      must_find_locations: vec![],
    };

    let criteria = SuccessCriteria::default();
    let metrics = session.compute_accuracy_metrics(&expected, &criteria);

    // Should find 1/2 files and 1/2 symbols
    assert!((metrics.file_recall - 0.5).abs() < f64::EPSILON);
    assert!((metrics.symbol_recall - 0.5).abs() < f64::EPSILON);
  }

  #[test]
  fn test_file_diversity_all_different_files() {
    let mut session = ExplorationSession::new();

    // Record step with 5 different files - perfect diversity
    session.record_explore_step(
      "query",
      &[
        "id1".to_string(),
        "id2".to_string(),
        "id3".to_string(),
        "id4".to_string(),
        "id5".to_string(),
      ],
      &[
        "src/a.rs".to_string(),
        "src/b.rs".to_string(),
        "src/c.rs".to_string(),
        "src/d.rs".to_string(),
        "src/e.rs".to_string(),
      ],
      &[],
      Duration::from_millis(100),
    );

    let diversity = session.calculate_step_file_diversity(0, 5);
    assert!(
      (diversity - 1.0).abs() < f64::EPSILON,
      "Expected 1.0, got {}",
      diversity
    );
  }

  #[test]
  fn test_file_diversity_all_same_file() {
    let mut session = ExplorationSession::new();

    // Record step with 5 results all from the same file - worst diversity
    session.record_explore_step(
      "query",
      &[
        "id1".to_string(),
        "id2".to_string(),
        "id3".to_string(),
        "id4".to_string(),
        "id5".to_string(),
      ],
      &[
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
      ],
      &[],
      Duration::from_millis(100),
    );

    let diversity = session.calculate_step_file_diversity(0, 5);
    assert!(
      (diversity - 0.2).abs() < f64::EPSILON,
      "Expected 0.2, got {}",
      diversity
    );
  }

  #[test]
  fn test_file_diversity_mixed() {
    let mut session = ExplorationSession::new();

    // Record step with 3 unique files out of 5 results
    session.record_explore_step(
      "query",
      &[
        "id1".to_string(),
        "id2".to_string(),
        "id3".to_string(),
        "id4".to_string(),
        "id5".to_string(),
      ],
      &[
        "src/a.rs".to_string(),
        "src/a.rs".to_string(),
        "src/b.rs".to_string(),
        "src/b.rs".to_string(),
        "src/c.rs".to_string(),
      ],
      &[],
      Duration::from_millis(100),
    );

    let diversity = session.calculate_step_file_diversity(0, 5);
    assert!(
      (diversity - 0.6).abs() < f64::EPSILON,
      "Expected 0.6 (3/5), got {}",
      diversity
    );
  }

  #[test]
  fn test_file_diversity_fewer_than_n_results() {
    let mut session = ExplorationSession::new();

    // Record step with only 2 files, ask for top-5
    session.record_explore_step(
      "query",
      &["id1".to_string(), "id2".to_string()],
      &["src/a.rs".to_string(), "src/b.rs".to_string()],
      &[],
      Duration::from_millis(100),
    );

    // Should calculate diversity based on available results (2), not requested (5)
    let diversity = session.calculate_step_file_diversity(0, 5);
    assert!(
      (diversity - 1.0).abs() < f64::EPSILON,
      "Expected 1.0 (2 unique / 2 total), got {}",
      diversity
    );
  }

  #[test]
  fn test_file_diversity_empty_results() {
    let mut session = ExplorationSession::new();

    // Record step with no files
    session.record_explore_step("query", &[], &[], &[], Duration::from_millis(100));

    // Empty results should return 1.0 (no diversity problem)
    let diversity = session.calculate_step_file_diversity(0, 5);
    assert!(
      (diversity - 1.0).abs() < f64::EPSILON,
      "Expected 1.0 for empty results, got {}",
      diversity
    );
  }

  #[test]
  fn test_file_diversity_missing_step() {
    let session = ExplorationSession::new();

    // Query step that doesn't exist - should return 1.0
    let diversity = session.calculate_step_file_diversity(99, 5);
    assert!(
      (diversity - 1.0).abs() < f64::EPSILON,
      "Expected 1.0 for missing step, got {}",
      diversity
    );
  }

  #[test]
  fn test_avg_file_diversity() {
    let mut session = ExplorationSession::new();

    // Step 0: Perfect diversity (5 unique)
    session.record_explore_step(
      "query1",
      &[
        "id1".to_string(),
        "id2".to_string(),
        "id3".to_string(),
        "id4".to_string(),
        "id5".to_string(),
      ],
      &[
        "src/a.rs".to_string(),
        "src/b.rs".to_string(),
        "src/c.rs".to_string(),
        "src/d.rs".to_string(),
        "src/e.rs".to_string(),
      ],
      &[],
      Duration::from_millis(100),
    );

    // Step 1: Worst diversity (all same)
    session.record_explore_step(
      "query2",
      &[
        "id1".to_string(),
        "id2".to_string(),
        "id3".to_string(),
        "id4".to_string(),
        "id5".to_string(),
      ],
      &[
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
        "src/same.rs".to_string(),
      ],
      &[],
      Duration::from_millis(100),
    );

    // Average should be (1.0 + 0.2) / 2 = 0.6
    let avg_diversity = session.calculate_avg_file_diversity();
    assert!(
      (avg_diversity - 0.6).abs() < f64::EPSILON,
      "Expected 0.6 average, got {}",
      avg_diversity
    );
  }

  #[test]
  fn test_file_diversity_in_accuracy_metrics() {
    let mut session = ExplorationSession::new();

    // Record step with 3 unique files out of 5
    session.record_explore_step(
      "query",
      &[
        "id1".to_string(),
        "id2".to_string(),
        "id3".to_string(),
        "id4".to_string(),
        "id5".to_string(),
      ],
      &[
        "src/a.rs".to_string(),
        "src/a.rs".to_string(),
        "src/b.rs".to_string(),
        "src/b.rs".to_string(),
        "src/c.rs".to_string(),
      ],
      &[],
      Duration::from_millis(100),
    );

    let expected = Expected::default();
    let criteria = SuccessCriteria::default();
    let metrics = session.compute_accuracy_metrics(&expected, &criteria);

    // Should include file diversity metric
    assert!(
      (metrics.avg_file_diversity_top5 - 0.6).abs() < f64::EPSILON,
      "Expected 0.6 in metrics, got {}",
      metrics.avg_file_diversity_top5
    );
  }
}
