//! Performance metrics collection.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use sysinfo::{Pid, System};

/// Latency statistics with percentiles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Minimum latency in milliseconds
    pub min_ms: u64,
    /// Maximum latency in milliseconds
    pub max_ms: u64,
    /// Mean latency in milliseconds
    pub mean_ms: u64,
    /// Median (p50) latency in milliseconds
    pub p50_ms: u64,
    /// 95th percentile latency in milliseconds
    pub p95_ms: u64,
    /// 99th percentile latency in milliseconds
    pub p99_ms: u64,
    /// Sample count
    pub count: usize,
}

impl LatencyStats {
    /// Create from a list of durations.
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self::default();
        }

        let mut ms_values: Vec<u64> = durations.iter().map(|d| d.as_millis() as u64).collect();
        ms_values.sort_unstable();

        let count = ms_values.len();
        let sum: u64 = ms_values.iter().sum();

        Self {
            min_ms: *ms_values.first().unwrap_or(&0),
            max_ms: *ms_values.last().unwrap_or(&0),
            mean_ms: sum / count as u64,
            p50_ms: percentile(&ms_values, 50),
            p95_ms: percentile(&ms_values, 95),
            p99_ms: percentile(&ms_values, 99),
            count,
        }
    }
}

/// Calculate percentile from sorted values.
fn percentile(sorted_values: &[u64], pct: usize) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }
    let idx = (pct * sorted_values.len() / 100).saturating_sub(1).min(sorted_values.len() - 1);
    sorted_values[idx]
}

/// Resource usage snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// Timestamp (Unix millis)
    pub timestamp_ms: u64,
}

/// Metrics for indexing operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexingMetrics {
    /// Total wall clock time in milliseconds
    pub wall_time_ms: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average CPU usage percentage
    pub avg_cpu_percent: f32,
    /// Total chunks processed
    pub chunks_processed: usize,
    /// Chunks per second throughput
    pub chunks_per_sec: f64,
    /// Total embeddings generated
    pub embeddings_generated: usize,
    /// Embeddings per second throughput
    pub embeddings_per_sec: f64,
}

/// Metrics for search operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Latency statistics
    pub latency: LatencyStats,
    /// Queries per second throughput
    pub qps: f64,
    /// Average result count per query
    pub avg_result_count: f64,
    /// Total queries executed
    pub total_queries: usize,
}

/// Metrics for a single exploration step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Step index
    pub step_index: usize,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Number of results returned
    pub result_count: usize,
    /// Context fetch latencies (if any)
    pub context_latencies_ms: Vec<u64>,
}

/// Aggregate performance metrics for a scenario.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Search latency stats
    pub search_latency: LatencyStats,
    /// Context fetch latency stats
    pub context_latency: LatencyStats,
    /// Total execution time in milliseconds
    pub total_time_ms: u64,
    /// Per-step metrics
    pub steps: Vec<StepMetrics>,
    /// Peak memory during execution (if measured)
    pub peak_memory_bytes: Option<u64>,
    /// Average CPU usage during execution (if measured)
    pub avg_cpu_percent: Option<f32>,
}

/// Resource monitor using sysinfo.
pub struct ResourceMonitor {
    system: System,
    pid: Pid,
    snapshots: Vec<ResourceSnapshot>,
}

impl ResourceMonitor {
    /// Create a new resource monitor for the current process.
    pub fn new() -> Self {
        let system = System::new_all();
        let pid = Pid::from_u32(std::process::id());
        Self {
            system,
            pid,
            snapshots: Vec::new(),
        }
    }

    /// Take a resource snapshot.
    pub fn snapshot(&mut self) -> ResourceSnapshot {
        self.system.refresh_all();

        let snapshot = if let Some(process) = self.system.process(self.pid) {
            ResourceSnapshot {
                memory_bytes: process.memory(),
                cpu_percent: process.cpu_usage(),
                timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
            }
        } else {
            ResourceSnapshot {
                timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
                ..Default::default()
            }
        };

        self.snapshots.push(snapshot.clone());
        snapshot
    }

    /// Get peak memory usage from all snapshots.
    pub fn peak_memory(&self) -> u64 {
        self.snapshots.iter().map(|s| s.memory_bytes).max().unwrap_or(0)
    }

    /// Get average CPU usage from all snapshots.
    pub fn avg_cpu(&self) -> f32 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.snapshots.iter().map(|s| s.cpu_percent).sum();
        sum / self.snapshots.len() as f32
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency tracker for collecting timing measurements.
#[derive(Debug, Default)]
pub struct LatencyTracker {
    durations: Vec<Duration>,
}

impl LatencyTracker {
    /// Create a new latency tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a duration.
    pub fn record(&mut self, duration: Duration) {
        self.durations.push(duration);
    }

    /// Get statistics from recorded durations.
    pub fn stats(&self) -> LatencyStats {
        LatencyStats::from_durations(&self.durations)
    }

    /// Get the number of recorded samples.
    pub fn count(&self) -> usize {
        self.durations.len()
    }

    /// Get total time in milliseconds.
    pub fn total_ms(&self) -> u64 {
        self.durations.iter().map(|d| d.as_millis() as u64).sum()
    }

    /// Clear all recorded durations.
    pub fn clear(&mut self) {
        self.durations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats_from_durations() {
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
            Duration::from_millis(500),
        ];

        let stats = LatencyStats::from_durations(&durations);
        assert_eq!(stats.min_ms, 100);
        assert_eq!(stats.max_ms, 500);
        assert_eq!(stats.mean_ms, 300);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_latency_stats_empty() {
        let stats = LatencyStats::from_durations(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.min_ms, 0);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new();
        tracker.record(Duration::from_millis(100));
        tracker.record(Duration::from_millis(200));

        assert_eq!(tracker.count(), 2);
        let stats = tracker.stats();
        assert_eq!(stats.count, 2);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&values, 50), 5);
        assert_eq!(percentile(&values, 90), 9);
    }

    #[test]
    fn test_resource_monitor() {
        let mut monitor = ResourceMonitor::new();
        let snapshot = monitor.snapshot();
        assert!(snapshot.timestamp_ms > 0);
    }
}
