//! Performance metrics collection.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, System};

// ============================================================================
// Incremental Indexing Metrics
// ============================================================================

/// Result of an incremental indexing benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalBenchResult {
  /// Repository name
  pub repo: String,
  /// Number of files modified
  pub files_modified: usize,
  /// Time per file in milliseconds
  pub time_per_file_ms: f64,
  /// Files correctly detected as changed
  pub true_positives: usize,
  /// Files incorrectly detected as changed
  pub false_positives: usize,
  /// Changed files missed
  pub false_negatives: usize,
  /// Total reindexing time in milliseconds
  pub total_time_ms: u64,
  /// Peak memory usage during reindex
  pub peak_memory_bytes: u64,
}

/// Result of large file handling benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeFileBenchResult {
  /// File size in bytes
  pub file_size_bytes: u64,
  /// Whether the file was indexed
  pub indexed: bool,
  /// Number of chunks created (if indexed)
  pub chunks_created: Option<usize>,
  /// Processing time in milliseconds
  pub processing_time_ms: u64,
  /// Peak memory usage
  pub peak_memory_bytes: u64,
  /// Reason for skipping (if not indexed)
  pub skip_reason: Option<String>,
}

/// Full incremental indexing benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalReport {
  /// Timestamp of the benchmark run
  pub timestamp: String,
  /// CCEngram version
  pub version: String,
  /// Incremental benchmark results
  pub results: Vec<IncrementalBenchResult>,
  /// Large file benchmark results
  pub large_file_results: Vec<LargeFileBenchResult>,
  /// Summary statistics
  pub summary: IncrementalSummary,
}

/// Summary statistics for incremental benchmarks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncrementalSummary {
  /// Average time per file across all tests
  pub avg_time_per_file_ms: f64,
  /// Maximum time per file
  pub max_time_per_file_ms: f64,
  /// Detection accuracy (true positives / (TP + FN))
  pub detection_accuracy: f64,
  /// False positive rate
  pub false_positive_rate: f64,
  /// Largest file successfully indexed
  pub max_indexed_file_bytes: u64,
  /// Whether all tests passed thresholds
  pub passes: bool,
}

// ============================================================================
// File Watcher Metrics
// ============================================================================

/// Result of watcher lifecycle benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherLifecycleResult {
  /// Startup latency in milliseconds
  pub startup_latency_ms: u64,
  /// Shutdown latency in milliseconds
  pub shutdown_latency_ms: u64,
  /// File descriptors before test (if available)
  pub fd_before: Option<usize>,
  /// File descriptors after test (if available)
  pub fd_after: Option<usize>,
  /// Whether a resource leak was detected
  pub leak_detected: bool,
}

/// Result of single file change benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleChangeResult {
  /// Time to detect the change
  pub detection_latency_ms: u64,
  /// Time to complete indexing
  pub indexing_latency_ms: u64,
  /// Total end-to-end latency (save to searchable)
  pub end_to_end_latency_ms: u64,
  /// Whether the file became searchable
  pub searchable: bool,
  /// Type of operation (create, modify, delete, rename)
  pub operation: String,
}

/// Result of batch file change benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchChangeResult {
  /// Number of files modified
  pub files_modified: usize,
  /// Number of reindex triggers observed
  pub reindex_triggers: usize,
  /// Whether debouncing worked correctly (1 trigger for batch)
  pub debounce_correct: bool,
  /// Total processing time for the batch
  pub total_processing_time_ms: u64,
}

/// Result of a single file operation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationResult {
  /// Operation type
  pub operation: String,
  /// Detection latency
  pub detection_latency_ms: u64,
  /// Whether it was handled correctly
  pub success: bool,
  /// Optional error message
  pub error: Option<String>,
}

/// Result of file operations benchmark (create/modify/delete/rename).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperationsResult {
  /// Create operation result
  pub create: OperationResult,
  /// Modify operation result
  pub modify: OperationResult,
  /// Delete operation result
  pub delete: OperationResult,
  /// Rename operation result
  pub rename: OperationResult,
}

/// Result of gitignore respect benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitignoreResult {
  /// Number of ignored files modified
  pub ignored_files_modified: usize,
  /// Triggers observed for ignored files (should be 0)
  pub false_positive_triggers: usize,
  /// Number of tracked files modified
  pub tracked_files_modified: usize,
  /// Tracked files correctly detected
  pub tracked_files_detected: usize,
  /// Gitignore respect rate (1.0 = perfect)
  pub respect_rate: f64,
}

/// Full watcher benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherReport {
  /// Timestamp of the benchmark run
  pub timestamp: String,
  /// CCEngram version
  pub version: String,
  /// Repository tested
  pub repo: String,
  /// Lifecycle test results
  pub lifecycle: Vec<WatcherLifecycleResult>,
  /// Single change test results
  pub single_change: Vec<SingleChangeResult>,
  /// Batch change test results
  pub batch_change: Vec<BatchChangeResult>,
  /// File operations test results
  pub file_operations: Vec<FileOperationsResult>,
  /// Gitignore test results
  pub gitignore: Vec<GitignoreResult>,
  /// Summary statistics
  pub summary: WatcherSummary,
}

/// Summary statistics for watcher benchmarks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatcherSummary {
  /// Average end-to-end latency
  pub avg_e2e_latency_ms: f64,
  /// p95 end-to-end latency
  pub p95_e2e_latency_ms: u64,
  /// Maximum end-to-end latency
  pub max_e2e_latency_ms: u64,
  /// Debounce accuracy (correct batches / total batches)
  pub debounce_accuracy: f64,
  /// Gitignore respect rate
  pub gitignore_respect_rate: f64,
  /// Resource leak count
  pub resource_leaks: usize,
  /// Whether all tests passed thresholds
  pub passes: bool,
}

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
  let idx = (pct * sorted_values.len() / 100)
    .saturating_sub(1)
    .min(sorted_values.len() - 1);
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
  /// Create a new resource monitor for a specific process.
  pub fn new(pid: u32) -> Self {
    let system = System::new_all();
    Self {
      system,
      pid: Pid::from_u32(pid),
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
  fn test_percentile() {
    let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(percentile(&values, 50), 5);
    assert_eq!(percentile(&values, 90), 9);
  }
}
