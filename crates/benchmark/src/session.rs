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

use crate::ground_truth::{CallGraph, ExplorationPath, NoisePatterns};
use crate::metrics::{AccuracyMetrics, LatencyTracker, PerformanceMetrics, StepMetrics};
use crate::scenarios::{Expected, SuccessCriteria};
use std::collections::HashSet;
use std::time::Duration;

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
  /// Step when hint was shown
  pub step: usize,
  /// Source chunk ID
  pub source_id: String,
  /// Hint type
  pub hint_type: HintType,
  /// Target identifier (symbol or file)
  pub target: String,
}

/// Type of navigation hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HintType {
  /// Caller of the current symbol
  Caller,
  /// Callee of the current symbol
  Callee,
  /// Sibling (same file/module)
  Sibling,
  /// Suggestion for follow-up query
  Suggestion,
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
  /// Bytes that contained useful information (expected symbols/files)
  pub useful_bytes: usize,
}

/// Record of a call relationship discovered during exploration.
#[derive(Debug, Clone)]
pub struct CallRelation {
  /// Caller symbol
  pub caller: String,
  /// Callee symbol
  pub callee: String,
  /// Step when this was discovered
  pub step: usize,
}

/// Per-step relevance tracking for path-based failure detection.
#[derive(Debug, Clone)]
pub struct StepRelevance {
  /// Step index
  pub step: usize,
  /// Whether this step found any expected files
  pub found_expected_file: bool,
  /// Whether this step found any expected symbols
  pub found_expected_symbol: bool,
  /// Count of relevant results in this step
  pub relevant_count: usize,
  /// Total results in this step
  pub total_count: usize,
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
  /// Session/scenario ID
  pub id: String,
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
  /// Call relations discovered during exploration
  call_relations: Vec<CallRelation>,
  /// When each symbol was first discovered (symbol -> step index)
  symbol_discovery_step: std::collections::HashMap<String, usize>,

  // === Path-based failure tracking (rabbit holes) ===
  /// Per-step relevance for detecting consecutive failures
  step_relevance: Vec<StepRelevance>,

  // === Context budget tracking ===
  /// Cumulative context budget tracking
  context_budget: ContextBudget,
}

impl ExplorationSession {
  /// Create a new exploration session.
  pub fn new(id: &str) -> Self {
    Self {
      id: id.to_string(),
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
      call_relations: Vec::new(),
      symbol_discovery_step: std::collections::HashMap::new(),
      // Path-based failure tracking
      step_relevance: Vec::new(),
      // Context budget
      context_budget: ContextBudget::default(),
    }
  }

  /// Create with custom noise patterns.
  pub fn with_noise_patterns(id: &str, patterns: NoisePatterns) -> Self {
    let mut session = Self::new(id);
    session.noise_patterns = patterns;
    session
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

  /// Mark that a core result was found at the current step.
  pub fn mark_core_found(&mut self) {
    if self.first_core_step.is_none() {
      self.first_core_step = Some(self.current_step.saturating_sub(1));
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
  pub fn record_result_rank(&mut self, is_relevant: bool, rank: usize) {
    self.result_ranks.push((is_relevant, rank));
  }

  /// Record a navigation hint shown to the user.
  pub fn record_hint(&mut self, source_id: &str, hint_type: HintType, target: &str) {
    self.hints_shown.push(HintRecord {
      step: self.current_step.saturating_sub(1),
      source_id: source_id.to_string(),
      hint_type,
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
      useful_bytes,
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
    self.call_relations.push(CallRelation {
      caller: caller.to_string(),
      callee: callee.to_string(),
      step: self.current_step.saturating_sub(1),
    });
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
    total_count: usize,
  ) {
    self.step_relevance.push(StepRelevance {
      step: self.current_step.saturating_sub(1),
      found_expected_file,
      found_expected_symbol,
      relevant_count,
      total_count,
    });
  }

  /// Record suggestions shown to user.
  pub fn record_suggestions(&mut self, suggestions: &[String]) {
    self.suggestions_shown.extend(suggestions.iter().cloned());
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

  /// Calculate hint utility (% of hints that were relevant/followed).
  pub fn calculate_hint_utility(&self) -> f64 {
    if self.hints_shown.is_empty() {
      return 1.0; // No hints = no problem
    }

    let followed_count = self
      .hints_shown
      .iter()
      .filter(|h| self.hints_followed.contains(&h.target))
      .count();

    followed_count as f64 / self.hints_shown.len() as f64
  }

  /// Calculate suggestion quality (% of suggestions that were used).
  pub fn calculate_suggestion_quality(&self) -> f64 {
    if self.suggestions_shown.is_empty() {
      return 1.0; // No suggestions = no problem
    }

    self.suggestions_used.len() as f64 / self.suggestions_shown.len() as f64
  }

  /// Calculate MRR (Mean Reciprocal Rank) from recorded result ranks.
  pub fn calculate_mrr(&self) -> f64 {
    for (is_relevant, rank) in &self.result_ranks {
      if *is_relevant && *rank > 0 {
        return 1.0 / *rank as f64;
      }
    }
    0.0
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

        // Optimal = min of annotated max_hops and graph path (if available)
        let optimal_hops = graph_hops.map(|g| g.min(path.max_hops)).unwrap_or(path.max_hops);

        // Efficiency = optimal / actual (capped at 1.0)
        let efficiency = (optimal_hops as f64 / actual_hops as f64).min(1.0);
        total_efficiency += efficiency;
        valid_paths += 1;
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

  /// Get the call graph built during exploration.
  pub fn call_graph(&self) -> &CallGraph {
    &self.call_graph
  }

  /// Check if a file matches expected files (with glob support).
  pub fn file_matches_expected(&self, file: &str, expected: &[String]) -> bool {
    for pattern in expected {
      if let Ok(glob) = glob::Pattern::new(pattern)
        && glob.matches(file)
      {
        return true;
      }
      // Also check suffix match
      if file.ends_with(pattern) || file == pattern {
        return true;
      }
    }
    false
  }

  /// Count noise results among given IDs.
  pub fn count_noise_results(&self, _result_ids: &[String]) -> usize {
    // For now, just count based on file patterns in discovered files
    // In a full implementation, we'd track more metadata per result
    self
      .discovered_files
      .iter()
      .filter(|f| self.noise_patterns.is_noise_file(f))
      .count()
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

    // Record noise for all discovered files
    for file in &self.discovered_files {
      builder = builder.record_noise(self.noise_patterns.is_noise_file(file));
    }

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

  /// Get the number of steps executed.
  pub fn step_count(&self) -> usize {
    self.current_step
  }

  /// Get step metrics.
  pub fn step_metrics(&self) -> &[StepMetrics] {
    &self.step_metrics
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_session_creation() {
    let session = ExplorationSession::new("test-scenario");
    assert_eq!(session.id, "test-scenario");
    assert!(session.discovered_files.is_empty());
    assert!(session.discovered_symbols.is_empty());
  }

  #[test]
  fn test_record_explore_step() {
    let mut session = ExplorationSession::new("test");

    session.record_explore_step(
      "test query",
      &["id1".to_string(), "id2".to_string()],
      &["src/main.rs".to_string()],
      &["main".to_string(), "run".to_string()],
      Duration::from_millis(100),
    );

    assert_eq!(session.step_count(), 1);
    assert!(session.discovered_files.contains("src/main.rs"));
    assert!(session.discovered_symbols.contains("main"));
    assert!(session.discovered_symbols.contains("run"));
  }

  #[test]
  fn test_record_context_call() {
    let mut session = ExplorationSession::new("test");

    // First record an explore step
    session.record_explore_step("q", &["id1".to_string()], &[], &[], Duration::from_millis(50));

    // Then record a context call
    session.record_context_call("id1", Duration::from_millis(30));

    assert_eq!(session.context_latencies.count(), 1);
    assert_eq!(session.step_metrics[0].context_latencies_ms.len(), 1);
  }

  #[test]
  fn test_performance_metrics() {
    let mut session = ExplorationSession::new("test");

    session.record_explore_step("q1", &["id1".to_string()], &[], &[], Duration::from_millis(100));
    session.record_explore_step("q2", &["id2".to_string()], &[], &[], Duration::from_millis(200));

    let metrics = session.compute_performance_metrics();

    assert_eq!(metrics.steps.len(), 2);
    assert_eq!(metrics.search_latency.count, 2);
    assert_eq!(metrics.total_time_ms, 300);
  }

  #[test]
  fn test_accuracy_metrics_with_expectations() {
    let mut session = ExplorationSession::new("test");

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
  fn test_file_matches_expected() {
    let session = ExplorationSession::new("test");

    let expected = vec!["src/commands.rs".to_string(), "**/keymap.rs".to_string()];

    assert!(session.file_matches_expected("src/commands.rs", &expected));
    assert!(session.file_matches_expected("crates/gpui/src/keymap.rs", &expected));
    assert!(!session.file_matches_expected("src/other.rs", &expected));
  }
}
