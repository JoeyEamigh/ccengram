//! Accuracy metrics for exploration quality.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Accuracy metrics for a benchmark scenario.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// File recall: % of must-find files discovered (target ≥70%)
    pub file_recall: f64,
    /// Symbol recall: % of must-find symbols discovered (target ≥70%)
    pub symbol_recall: f64,
    /// Steps to core: queries needed to find first core result (target ≤3)
    pub steps_to_core: Option<usize>,
    /// MRR: Mean reciprocal rank of first correct result (target ≥0.5)
    pub mrr: f64,
    /// Noise ratio: % of results matching noise patterns (target ≤25%)
    pub noise_ratio: f64,
    /// Top-3 noise: noise in top 3 results (target ≤10%)
    pub top3_noise: f64,
    /// Hint utility: % of callers/callees that are relevant (target ≥60%)
    pub hint_utility: f64,
    /// Suggestion quality: % of suggestions leading to useful results (target ≥50%)
    pub suggestion_quality: f64,

    // === Exploration-specific metrics ===

    /// Convergence rate: how quickly discoveries plateau (1.0 = all early, target ≥0.7)
    #[serde(default)]
    pub convergence_rate: f64,
    /// Average information gain per step (new discoveries / total expected, target ≥0.3)
    #[serde(default)]
    pub avg_info_gain: f64,
    /// Context bloat: % of context calls with no new info (target ≤0.3)
    #[serde(default)]
    pub context_bloat: f64,
    /// Navigation efficiency: optimal_hops / actual_hops (target ≥0.5)
    #[serde(default)]
    pub navigation_efficiency: f64,
    /// Dead end ratio: % of steps with no useful discoveries (target ≤0.2)
    #[serde(default)]
    pub dead_end_ratio: f64,

    // === Debug fields ===

    /// Files found (for debugging)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files_found: Vec<String>,
    /// Files missed (for debugging)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files_missed: Vec<String>,
    /// Symbols found (for debugging)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub symbols_found: Vec<String>,
    /// Symbols missed (for debugging)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub symbols_missed: Vec<String>,
}

impl AccuracyMetrics {
    /// Create a builder for computing accuracy metrics.
    pub fn builder() -> AccuracyMetricsBuilder {
        AccuracyMetricsBuilder::new()
    }

    /// Check if metrics meet minimum thresholds.
    pub fn passes_thresholds(
        &self,
        min_file_recall: f64,
        min_symbol_recall: f64,
        max_noise_ratio: f64,
        max_steps_to_core: usize,
    ) -> bool {
        self.file_recall >= min_file_recall
            && self.symbol_recall >= min_symbol_recall
            && self.noise_ratio <= max_noise_ratio
            && self.steps_to_core.is_none_or(|s| s <= max_steps_to_core)
    }
}

/// Builder for computing accuracy metrics.
#[derive(Debug, Default)]
pub struct AccuracyMetricsBuilder {
    expected_files: HashSet<String>,
    expected_symbols: HashSet<String>,
    found_files: HashSet<String>,
    found_symbols: HashSet<String>,
    result_ranks: Vec<(bool, usize)>, // (is_relevant, rank)
    noise_results: Vec<bool>,          // true if result is noise
    hint_relevance: Vec<bool>,         // true if hint was relevant
    suggestion_usefulness: Vec<bool>,  // true if suggestion was useful
    step_found_core: Option<usize>,    // step index when first core result found

    // Exploration metrics (set directly, computed externally by session)
    convergence_rate: Option<f64>,
    avg_info_gain: Option<f64>,
    context_bloat: Option<f64>,
    navigation_efficiency: Option<f64>,
    dead_end_ratio: Option<f64>,
}

impl AccuracyMetricsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set expected files (glob patterns supported in comparison).
    pub fn expected_files(mut self, files: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.expected_files = files.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Set expected symbols.
    pub fn expected_symbols(mut self, symbols: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.expected_symbols = symbols.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Record files found across all results.
    pub fn record_files(mut self, files: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for file in files {
            self.found_files.insert(file.into());
        }
        self
    }

    /// Record symbols found across all results.
    pub fn record_symbols(mut self, symbols: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for symbol in symbols {
            self.found_symbols.insert(symbol.into());
        }
        self
    }

    /// Record a result's relevance and rank for MRR calculation.
    pub fn record_result_rank(mut self, is_relevant: bool, rank: usize) -> Self {
        self.result_ranks.push((is_relevant, rank));
        self
    }

    /// Record whether a result is noise.
    pub fn record_noise(mut self, is_noise: bool) -> Self {
        self.noise_results.push(is_noise);
        self
    }

    /// Record multiple noise statuses.
    pub fn record_noise_batch(mut self, noise_flags: impl IntoIterator<Item = bool>) -> Self {
        for flag in noise_flags {
            self.noise_results.push(flag);
        }
        self
    }

    /// Record whether a hint (caller/callee) was relevant.
    pub fn record_hint_relevance(mut self, is_relevant: bool) -> Self {
        self.hint_relevance.push(is_relevant);
        self
    }

    /// Record whether a suggestion was useful.
    pub fn record_suggestion_usefulness(mut self, is_useful: bool) -> Self {
        self.suggestion_usefulness.push(is_useful);
        self
    }

    /// Record when first core result was found.
    pub fn set_step_found_core(mut self, step: usize) -> Self {
        if self.step_found_core.is_none() {
            self.step_found_core = Some(step);
        }
        self
    }

    // === Exploration metric setters ===

    /// Set convergence rate (computed externally by session).
    pub fn set_convergence_rate(mut self, rate: f64) -> Self {
        self.convergence_rate = Some(rate);
        self
    }

    /// Set average information gain (computed externally by session).
    pub fn set_avg_info_gain(mut self, gain: f64) -> Self {
        self.avg_info_gain = Some(gain);
        self
    }

    /// Set context bloat (computed externally by session).
    pub fn set_context_bloat(mut self, bloat: f64) -> Self {
        self.context_bloat = Some(bloat);
        self
    }

    /// Set navigation efficiency (computed externally by session).
    pub fn set_navigation_efficiency(mut self, efficiency: f64) -> Self {
        self.navigation_efficiency = Some(efficiency);
        self
    }

    /// Set dead end ratio (computed externally by session).
    pub fn set_dead_end_ratio(mut self, ratio: f64) -> Self {
        self.dead_end_ratio = Some(ratio);
        self
    }

    /// Build the final metrics.
    pub fn build(self) -> AccuracyMetrics {
        // Calculate file recall
        let (file_recall, files_found, files_missed) = self.calculate_file_recall();

        // Calculate symbol recall
        let (symbol_recall, symbols_found, symbols_missed) = self.calculate_symbol_recall();

        // Calculate MRR
        let mrr = self.calculate_mrr();

        // Calculate noise ratios
        let (noise_ratio, top3_noise) = self.calculate_noise_ratios();

        // Calculate hint utility
        let hint_utility = self.calculate_hint_utility();

        // Calculate suggestion quality
        let suggestion_quality = self.calculate_suggestion_quality();

        AccuracyMetrics {
            file_recall,
            symbol_recall,
            steps_to_core: self.step_found_core,
            mrr,
            noise_ratio,
            top3_noise,
            hint_utility,
            suggestion_quality,
            // Exploration metrics (default to neutral values if not set)
            convergence_rate: self.convergence_rate.unwrap_or(1.0),
            avg_info_gain: self.avg_info_gain.unwrap_or(0.0),
            context_bloat: self.context_bloat.unwrap_or(0.0),
            navigation_efficiency: self.navigation_efficiency.unwrap_or(1.0),
            dead_end_ratio: self.dead_end_ratio.unwrap_or(0.0),
            // Debug fields
            files_found,
            files_missed,
            symbols_found,
            symbols_missed,
        }
    }

    fn calculate_file_recall(&self) -> (f64, Vec<String>, Vec<String>) {
        if self.expected_files.is_empty() {
            return (1.0, vec![], vec![]);
        }

        let mut found = Vec::new();
        let mut missed = Vec::new();

        for expected in &self.expected_files {
            // Support glob patterns
            if let Ok(pattern) = glob::Pattern::new(expected) {
                if self.found_files.iter().any(|f| pattern.matches(f) || f.ends_with(expected)) {
                    found.push(expected.clone());
                } else {
                    missed.push(expected.clone());
                }
            } else {
                // Exact match or suffix match
                if self.found_files.iter().any(|f| f == expected || f.ends_with(expected)) {
                    found.push(expected.clone());
                } else {
                    missed.push(expected.clone());
                }
            }
        }

        let recall = found.len() as f64 / self.expected_files.len() as f64;
        (recall, found, missed)
    }

    fn calculate_symbol_recall(&self) -> (f64, Vec<String>, Vec<String>) {
        if self.expected_symbols.is_empty() {
            return (1.0, vec![], vec![]);
        }

        let mut found = Vec::new();
        let mut missed = Vec::new();

        for expected in &self.expected_symbols {
            if self.found_symbols.contains(expected) {
                found.push(expected.clone());
            } else {
                missed.push(expected.clone());
            }
        }

        let recall = found.len() as f64 / self.expected_symbols.len() as f64;
        (recall, found, missed)
    }

    fn calculate_mrr(&self) -> f64 {
        // Find the rank of the first relevant result
        for (is_relevant, rank) in &self.result_ranks {
            if *is_relevant && *rank > 0 {
                return 1.0 / *rank as f64;
            }
        }
        0.0
    }

    fn calculate_noise_ratios(&self) -> (f64, f64) {
        if self.noise_results.is_empty() {
            return (0.0, 0.0);
        }

        let noise_count = self.noise_results.iter().filter(|&&n| n).count();
        let noise_ratio = noise_count as f64 / self.noise_results.len() as f64;

        // Top-3 noise
        let top3_noise_count = self.noise_results.iter().take(3).filter(|&&n| n).count();
        let top3_total = self.noise_results.len().min(3);
        let top3_noise = if top3_total > 0 {
            top3_noise_count as f64 / top3_total as f64
        } else {
            0.0
        };

        (noise_ratio, top3_noise)
    }

    fn calculate_hint_utility(&self) -> f64 {
        if self.hint_relevance.is_empty() {
            return 1.0; // No hints to evaluate
        }

        let relevant_count = self.hint_relevance.iter().filter(|&&r| r).count();
        relevant_count as f64 / self.hint_relevance.len() as f64
    }

    fn calculate_suggestion_quality(&self) -> f64 {
        if self.suggestion_usefulness.is_empty() {
            return 1.0; // No suggestions to evaluate
        }

        let useful_count = self.suggestion_usefulness.iter().filter(|&&u| u).count();
        useful_count as f64 / self.suggestion_usefulness.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_recall() {
        let metrics = AccuracyMetrics::builder()
            .expected_files(["src/main.rs", "src/lib.rs"])
            .record_files(["src/main.rs", "src/utils.rs"])
            .build();

        assert!((metrics.file_recall - 0.5).abs() < f64::EPSILON);
        assert_eq!(metrics.files_found.len(), 1);
        assert_eq!(metrics.files_missed.len(), 1);
    }

    #[test]
    fn test_symbol_recall() {
        let metrics = AccuracyMetrics::builder()
            .expected_symbols(["main", "run", "setup"])
            .record_symbols(["main", "run"])
            .build();

        assert!((metrics.symbol_recall - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mrr() {
        let metrics = AccuracyMetrics::builder()
            .record_result_rank(false, 1)
            .record_result_rank(true, 2)  // First relevant at rank 2
            .record_result_rank(true, 3)
            .build();

        assert!((metrics.mrr - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_ratio() {
        let metrics = AccuracyMetrics::builder()
            .record_noise_batch([false, true, false, false, true])
            .build();

        assert!((metrics.noise_ratio - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_top3_noise() {
        let metrics = AccuracyMetrics::builder()
            .record_noise_batch([true, false, false, true, true])
            .build();

        assert!((metrics.top3_noise - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hint_utility() {
        let metrics = AccuracyMetrics::builder()
            .record_hint_relevance(true)
            .record_hint_relevance(true)
            .record_hint_relevance(false)
            .record_hint_relevance(true)
            .build();

        assert!((metrics.hint_utility - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_passes_thresholds() {
        let metrics = AccuracyMetrics {
            file_recall: 0.8,
            symbol_recall: 0.75,
            noise_ratio: 0.2,
            steps_to_core: Some(2),
            ..Default::default()
        };

        assert!(metrics.passes_thresholds(0.7, 0.7, 0.25, 3));
        assert!(!metrics.passes_thresholds(0.9, 0.7, 0.25, 3)); // file_recall too low
    }

    #[test]
    fn test_empty_expectations() {
        let metrics = AccuracyMetrics::builder()
            .record_files(["src/main.rs"])
            .record_symbols(["main"])
            .build();

        // With no expectations, recall should be 1.0
        assert!((metrics.file_recall - 1.0).abs() < f64::EPSILON);
        assert!((metrics.symbol_recall - 1.0).abs() < f64::EPSILON);
    }
}
