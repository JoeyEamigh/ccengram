//! Exploration session state tracking.
//!
//! Tracks the state of a multi-step exploration scenario, accumulating
//! discovered files, symbols, and metrics across steps.

use crate::ground_truth::NoisePatterns;
use crate::metrics::{AccuracyMetrics, LatencyTracker, PerformanceMetrics, StepMetrics};
use crate::scenarios::{Expected, SuccessCriteria};
use std::collections::HashSet;
use std::time::Duration;

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

    /// Check if a file matches expected files (with glob support).
    pub fn file_matches_expected(&self, file: &str, expected: &[String]) -> bool {
        for pattern in expected {
            if let Ok(glob) = glob::Pattern::new(pattern)
                && glob.matches(file) {
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
        self.discovered_files
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
                            f.ends_with(expected_file)
                                || glob::Pattern::new(expected_file)
                                    .map(|p| p.matches(f))
                                    .unwrap_or(false)
                        }) {
                            builder = builder.set_step_found_core(i);
                            break;
                        }
                    }
                }
            }
        }

        builder.build()
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

        let expected = vec![
            "src/commands.rs".to_string(),
            "**/keymap.rs".to_string(),
        ];

        assert!(session.file_matches_expected("src/commands.rs", &expected));
        assert!(session.file_matches_expected("crates/gpui/src/keymap.rs", &expected));
        assert!(!session.file_matches_expected("src/other.rs", &expected));
    }
}
