//! Metrics collection for benchmarking.
//!
//! Two categories:
//! - Performance: latency, throughput, resource usage
//! - Accuracy: discovery rate, noise ratio, navigation quality

mod accuracy;
mod performance;

pub use accuracy::{AccuracyMetrics, AccuracyMetricsBuilder};
pub use performance::{
    IndexingMetrics, LatencyStats, LatencyTracker, PerformanceMetrics, ResourceMonitor,
    ResourceSnapshot, SearchMetrics, StepMetrics,
};

/// All metrics targets from the plan.
pub struct MetricTargets {
    /// File recall target (≥70%)
    pub file_recall: f64,
    /// Symbol recall target (≥70%)
    pub symbol_recall: f64,
    /// Max steps to core (≤3)
    pub max_steps_to_core: usize,
    /// MRR target (≥0.5)
    pub mrr: f64,
    /// Max noise ratio (≤25%)
    pub noise_ratio: f64,
    /// Max top-3 noise (≤10%)
    pub top3_noise: f64,
    /// Hint utility target (≥60%)
    pub hint_utility: f64,
    /// Suggestion quality target (≥50%)
    pub suggestion_quality: f64,
}

impl Default for MetricTargets {
    fn default() -> Self {
        Self {
            file_recall: 0.70,
            symbol_recall: 0.70,
            max_steps_to_core: 3,
            mrr: 0.50,
            noise_ratio: 0.25,
            top3_noise: 0.10,
            hint_utility: 0.60,
            suggestion_quality: 0.50,
        }
    }
}

impl MetricTargets {
    /// Check if accuracy metrics meet all targets.
    pub fn check_accuracy(&self, metrics: &AccuracyMetrics) -> bool {
        metrics.file_recall >= self.file_recall
            && metrics.symbol_recall >= self.symbol_recall
            && metrics.steps_to_core.is_none_or(|s| s <= self.max_steps_to_core)
            && metrics.mrr >= self.mrr
            && metrics.noise_ratio <= self.noise_ratio
            && metrics.top3_noise <= self.top3_noise
    }
}
