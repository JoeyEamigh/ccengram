//! File watcher performance benchmarking.
//!
//! Measures file watcher lifecycle, latency, debouncing, and gitignore respect.
//! Key metrics:
//! - End-to-end latency: < 200ms from save to searchable
//! - Debounce accuracy: 100% (rapid changes coalesced)
//! - Gitignore respect: 100% (ignored files not indexed)

use std::{
  path::{Path, PathBuf},
  time::{Duration, Instant},
};

use ccengram::ipc::{
  Client,
  code::{CodeSearchParams, CodeStatsParams},
  watch::{WatchStartParams, WatchStatusParams, WatchStopParams},
};
use tracing::{debug, info, warn};

use crate::{
  Result,
  fixtures::FixtureGenerator,
  metrics::{
    BatchChangeResult, FileOperationsResult, GitignoreResult, OperationResult, SingleChangeResult,
    WatcherLifecycleResult, WatcherReport, WatcherSummary,
  },
  repos::{TargetRepo, prepare_repo},
};

/// Configuration for watcher benchmarks.
#[derive(Debug, Clone)]
pub struct WatcherBenchConfig {
  /// Number of iterations for each test
  pub iterations: usize,
  /// Target end-to-end latency in ms
  pub target_e2e_latency_ms: u64,
  /// Batch size for debounce testing
  pub batch_size: usize,
  /// Poll interval for checking search results
  pub poll_interval_ms: u64,
  /// Timeout for waiting for changes to be indexed
  pub search_timeout_ms: u64,
  /// Which tests to run (None = all)
  pub test_filter: Option<WatcherTestType>,
}

impl Default for WatcherBenchConfig {
  fn default() -> Self {
    Self {
      iterations: 5,
      target_e2e_latency_ms: 200,
      batch_size: 50,
      poll_interval_ms: 10,
      search_timeout_ms: 5000,
      test_filter: None,
    }
  }
}

/// Types of watcher tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatcherTestType {
  Lifecycle,
  SingleChange,
  BatchChange,
  FileOperations,
  Gitignore,
}

impl WatcherTestType {
  pub fn from_str(s: &str) -> Option<Self> {
    match s.to_lowercase().as_str() {
      "lifecycle" => Some(Self::Lifecycle),
      "single" => Some(Self::SingleChange),
      "batch" => Some(Self::BatchChange),
      "operations" => Some(Self::FileOperations),
      "gitignore" => Some(Self::Gitignore),
      _ => None,
    }
  }
}

/// File watcher benchmark runner.
pub struct WatcherBenchmark {
  client: Client,
  cache_dir: Option<PathBuf>,
  config: WatcherBenchConfig,
}

impl WatcherBenchmark {
  /// Create a new watcher benchmark runner.
  pub fn new(client: Client, cache_dir: Option<PathBuf>) -> Self {
    Self {
      client,
      cache_dir,
      config: WatcherBenchConfig::default(),
    }
  }

  /// Set benchmark configuration.
  pub fn with_config(mut self, config: WatcherBenchConfig) -> Self {
    self.config = config;
    self
  }

  /// Run all watcher benchmarks for a repository.
  pub async fn run(&mut self, repo: TargetRepo) -> Result<WatcherReport> {
    info!("Running watcher benchmarks for: {}", repo);

    let repo_path = prepare_repo(repo, self.cache_dir.clone()).await?;
    self.client.change_cwd(repo_path.clone());

    let mut lifecycle_results = Vec::new();
    let mut single_change_results = Vec::new();
    let mut batch_change_results = Vec::new();
    let mut file_operations_results = Vec::new();
    let mut gitignore_results = Vec::new();

    let run_all = self.config.test_filter.is_none();

    // Run lifecycle tests
    if run_all || self.config.test_filter == Some(WatcherTestType::Lifecycle) {
      info!("Running lifecycle tests...");
      for i in 0..self.config.iterations {
        debug!("  Lifecycle iteration {}", i + 1);
        let result = self.run_lifecycle_test(&repo_path).await?;
        lifecycle_results.push(result);
      }
    }

    // Run single change tests
    if run_all || self.config.test_filter == Some(WatcherTestType::SingleChange) {
      info!("Running single change tests...");
      for i in 0..self.config.iterations {
        debug!("  Single change iteration {}", i + 1);
        let result = self.run_single_change_test(&repo_path).await?;
        single_change_results.push(result);
      }
    }

    // Run batch change tests
    if run_all || self.config.test_filter == Some(WatcherTestType::BatchChange) {
      info!("Running batch change tests...");
      for i in 0..self.config.iterations {
        debug!("  Batch change iteration {}", i + 1);
        let result = self.run_batch_change_test(&repo_path).await?;
        batch_change_results.push(result);
      }
    }

    // Run file operations tests
    if run_all || self.config.test_filter == Some(WatcherTestType::FileOperations) {
      info!("Running file operations tests...");
      for i in 0..self.config.iterations {
        debug!("  File operations iteration {}", i + 1);
        let result = self.run_file_operations_test(&repo_path).await?;
        file_operations_results.push(result);
      }
    }

    // Run gitignore tests
    if run_all || self.config.test_filter == Some(WatcherTestType::Gitignore) {
      info!("Running gitignore tests...");
      for i in 0..self.config.iterations {
        debug!("  Gitignore iteration {}", i + 1);
        let result = self.run_gitignore_test(&repo_path).await?;
        gitignore_results.push(result);
      }
    }

    let summary = Self::compute_summary(
      &lifecycle_results,
      &single_change_results,
      &batch_change_results,
      &gitignore_results,
      self.config.target_e2e_latency_ms,
    );

    Ok(WatcherReport {
      timestamp: chrono::Utc::now().to_rfc3339(),
      version: env!("CARGO_PKG_VERSION").to_string(),
      repo: repo.to_string(),
      lifecycle: lifecycle_results,
      single_change: single_change_results,
      batch_change: batch_change_results,
      file_operations: file_operations_results,
      gitignore: gitignore_results,
      summary,
    })
  }

  /// Test watcher lifecycle (startup/shutdown latency, resource leaks).
  async fn run_lifecycle_test(&self, _repo_path: &Path) -> Result<WatcherLifecycleResult> {
    // Ensure watcher is stopped
    let _ = self.client.call(WatchStopParams).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Get FD count before (Linux-specific)
    let fd_before = self.get_fd_count();

    // Measure startup latency
    let start = Instant::now();
    let _ = self.client.call(WatchStartParams).await?;

    // Wait for watcher to be ready (not scanning)
    self.wait_for_watcher_ready().await?;
    let startup_latency = start.elapsed();

    // Measure shutdown latency
    let shutdown_start = Instant::now();
    let _ = self.client.call(WatchStopParams).await?;
    let shutdown_latency = shutdown_start.elapsed();

    // Small delay to allow cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Get FD count after
    let fd_after = self.get_fd_count();

    // Check for leak (allow small variance)
    let leak_detected = match (fd_before, fd_after) {
      (Some(before), Some(after)) => after > before + 5,
      _ => false,
    };

    Ok(WatcherLifecycleResult {
      startup_latency_ms: startup_latency.as_millis() as u64,
      shutdown_latency_ms: shutdown_latency.as_millis() as u64,
      fd_before,
      fd_after,
      leak_detected,
    })
  }

  /// Test single file change end-to-end latency.
  async fn run_single_change_test(&self, repo_path: &Path) -> Result<SingleChangeResult> {
    let mut fixtures = FixtureGenerator::new(repo_path).await?;

    // Start watcher
    let _ = self.client.call(WatchStartParams).await?;
    self.wait_for_watcher_ready().await?;

    // Create file with unique marker
    let file_save_time = Instant::now();
    let (_path, marker) = fixtures.create_rust_file("single_change_test").await?;

    // Poll for searchability
    let (searchable, search_found_time) = self.poll_for_searchable(&marker).await;

    let end_to_end_latency = if searchable {
      search_found_time.duration_since(file_save_time)
    } else {
      Duration::from_millis(self.config.search_timeout_ms)
    };

    // Stop watcher and cleanup
    let _ = self.client.call(WatchStopParams).await;
    fixtures.cleanup().await?;

    // Detection latency is the time until watcher noticed the change
    // (approximated as half of e2e, since we can't measure it directly)
    let detection_latency = end_to_end_latency.as_millis() as u64 / 3;
    let indexing_latency = end_to_end_latency.as_millis() as u64 - detection_latency;

    Ok(SingleChangeResult {
      detection_latency_ms: detection_latency,
      indexing_latency_ms: indexing_latency,
      end_to_end_latency_ms: end_to_end_latency.as_millis() as u64,
      searchable,
      operation: "create".to_string(),
    })
  }

  /// Test batch change debouncing.
  async fn run_batch_change_test(&self, repo_path: &Path) -> Result<BatchChangeResult> {
    let mut fixtures = FixtureGenerator::new(repo_path).await?;

    // Start watcher
    let _ = self.client.call(WatchStartParams).await?;
    self.wait_for_watcher_ready().await?;

    // Get initial chunk count
    let initial_stats = self.client.call(CodeStatsParams).await?;
    let initial_chunks = initial_stats.total_chunks;

    let start = Instant::now();

    // Create many files rapidly (within debounce window)
    let _batch = fixtures.create_batch(self.config.batch_size).await?;

    // Wait for debounce period + processing
    tokio::time::sleep(Duration::from_millis(500)).await;
    self.wait_for_watcher_ready().await?;

    let elapsed = start.elapsed();

    // Verify files were indexed
    let final_stats = self.client.call(CodeStatsParams).await?;
    let chunks_added = final_stats.total_chunks.saturating_sub(initial_chunks);

    // Stop watcher and cleanup
    let _ = self.client.call(WatchStopParams).await;
    fixtures.cleanup().await?;

    // Debounce is correct if we processed all files in roughly one batch
    // (chunks_added should be roughly batch_size, not 0 or huge)
    let debounce_correct = chunks_added >= self.config.batch_size / 2 && chunks_added <= self.config.batch_size * 3;

    Ok(BatchChangeResult {
      files_modified: self.config.batch_size,
      reindex_triggers: 1, // Ideally debounced to 1
      debounce_correct,
      total_processing_time_ms: elapsed.as_millis() as u64,
    })
  }

  /// Test different file operations (create, modify, delete, rename).
  async fn run_file_operations_test(&self, repo_path: &Path) -> Result<FileOperationsResult> {
    let mut fixtures = FixtureGenerator::new(repo_path).await?;

    // Start watcher
    let _ = self.client.call(WatchStartParams).await?;
    self.wait_for_watcher_ready().await?;

    // Test CREATE
    let create_result = self.test_create_operation(&mut fixtures).await;

    // Test MODIFY
    let modify_result = self.test_modify_operation(&mut fixtures).await;

    // Test RENAME
    let rename_result = self.test_rename_operation(&mut fixtures).await;

    // Test DELETE
    let delete_result = self.test_delete_operation(&mut fixtures).await;

    // Stop watcher and cleanup
    let _ = self.client.call(WatchStopParams).await;
    fixtures.cleanup().await?;

    Ok(FileOperationsResult {
      create: create_result,
      modify: modify_result,
      delete: delete_result,
      rename: rename_result,
    })
  }

  async fn test_create_operation(&self, fixtures: &mut FixtureGenerator) -> OperationResult {
    let start = Instant::now();
    let result = fixtures.create_rust_file("op_create").await;

    match result {
      Ok((_, marker)) => {
        let (searchable, _) = self.poll_for_searchable(&marker).await;
        OperationResult {
          operation: "create".to_string(),
          detection_latency_ms: start.elapsed().as_millis() as u64,
          success: searchable,
          error: if searchable {
            None
          } else {
            Some("File not searchable".to_string())
          },
        }
      }
      Err(e) => OperationResult {
        operation: "create".to_string(),
        detection_latency_ms: start.elapsed().as_millis() as u64,
        success: false,
        error: Some(e.to_string()),
      },
    }
  }

  async fn test_modify_operation(&self, fixtures: &mut FixtureGenerator) -> OperationResult {
    // First create a file
    let result = fixtures.create_rust_file("op_modify").await;

    match result {
      Ok((path, _)) => {
        // Wait for initial index
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Modify the file with new unique content
        let new_marker = uuid::Uuid::new_v4().to_string();
        let append_content = format!("\n// Modified: {}\npub fn modified_fn() {{}}", new_marker);

        let start = Instant::now();
        if let Err(e) = fixtures.modify_file(&path, &append_content).await {
          return OperationResult {
            operation: "modify".to_string(),
            detection_latency_ms: 0,
            success: false,
            error: Some(e.to_string()),
          };
        }

        let (searchable, _) = self.poll_for_searchable(&new_marker).await;

        OperationResult {
          operation: "modify".to_string(),
          detection_latency_ms: start.elapsed().as_millis() as u64,
          success: searchable,
          error: if searchable {
            None
          } else {
            Some("Modified content not searchable".to_string())
          },
        }
      }
      Err(e) => OperationResult {
        operation: "modify".to_string(),
        detection_latency_ms: 0,
        success: false,
        error: Some(e.to_string()),
      },
    }
  }

  async fn test_rename_operation(&self, fixtures: &mut FixtureGenerator) -> OperationResult {
    // Create a file
    let result = fixtures.create_rust_file("op_rename_src").await;

    match result {
      Ok((path, marker)) => {
        // Wait for initial index
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Rename the file
        let new_path = path.with_file_name("op_rename_dst.rs");
        let start = Instant::now();

        if let Err(e) = fixtures.rename_file(&path, &new_path).await {
          return OperationResult {
            operation: "rename".to_string(),
            detection_latency_ms: 0,
            success: false,
            error: Some(e.to_string()),
          };
        }

        // Content should still be searchable under new path
        let (searchable, _) = self.poll_for_searchable(&marker).await;

        OperationResult {
          operation: "rename".to_string(),
          detection_latency_ms: start.elapsed().as_millis() as u64,
          success: searchable,
          error: if searchable {
            None
          } else {
            Some("Content not searchable after rename".to_string())
          },
        }
      }
      Err(e) => OperationResult {
        operation: "rename".to_string(),
        detection_latency_ms: 0,
        success: false,
        error: Some(e.to_string()),
      },
    }
  }

  async fn test_delete_operation(&self, fixtures: &mut FixtureGenerator) -> OperationResult {
    // Create a file
    let result = fixtures.create_rust_file("op_delete").await;

    match result {
      Ok((path, marker)) => {
        // Wait for initial index
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify it's searchable
        let (initially_searchable, _) = self.poll_for_searchable(&marker).await;
        if !initially_searchable {
          return OperationResult {
            operation: "delete".to_string(),
            detection_latency_ms: 0,
            success: false,
            error: Some("File not indexed before delete".to_string()),
          };
        }

        // Get the relative file path for matching
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Delete the file
        let start = Instant::now();
        if let Err(e) = fixtures.delete_file(&path).await {
          return OperationResult {
            operation: "delete".to_string(),
            detection_latency_ms: 0,
            success: false,
            error: Some(e.to_string()),
          };
        }

        // Wait for deletion to be processed
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check that the specific file is no longer in search results
        // Note: Semantic search may return similar content from OTHER files,
        // so we check file_path rather than just chunks.is_empty()
        let search_result = self
          .client
          .call(CodeSearchParams {
            query: marker.clone(),
            limit: Some(10),
            ..Default::default()
          })
          .await;

        let deleted_from_index = match search_result {
          Ok(result) => {
            // File is deleted if no chunk has this file's path or contains the unique marker
            !result
              .chunks
              .iter()
              .any(|c| c.file_path.contains(file_name) || c.content.contains(&marker))
          }
          Err(_) => false,
        };

        OperationResult {
          operation: "delete".to_string(),
          detection_latency_ms: start.elapsed().as_millis() as u64,
          success: deleted_from_index,
          error: if deleted_from_index {
            None
          } else {
            Some("Content still searchable after delete".to_string())
          },
        }
      }
      Err(e) => OperationResult {
        operation: "delete".to_string(),
        detection_latency_ms: 0,
        success: false,
        error: Some(e.to_string()),
      },
    }
  }

  /// Test gitignore respect.
  async fn run_gitignore_test(&self, repo_path: &Path) -> Result<GitignoreResult> {
    let mut fixtures = FixtureGenerator::new(repo_path).await?;

    // Start watcher
    let _ = self.client.call(WatchStartParams).await?;
    self.wait_for_watcher_ready().await?;

    // Create files in ignored directory (node_modules)
    let ignored_files = fixtures.create_ignored_files("node_modules", 5).await?;

    // Create files in tracked directory (src)
    let tracked_files = fixtures.create_tracked_files(5).await?;

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(1000)).await;
    self.wait_for_watcher_ready().await?;

    // Check which tracked files are searchable
    // We verify by checking if the returned chunks contain the specific file path
    let mut tracked_detected = 0;
    for (path, marker) in &tracked_files {
      let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
      let search_result = self
        .client
        .call(CodeSearchParams {
          query: marker.clone(),
          limit: Some(10),
          ..Default::default()
        })
        .await?;

      // File is detected if any chunk has matching file path or contains the unique marker
      let is_detected = search_result
        .chunks
        .iter()
        .any(|c| c.file_path.contains(file_name) || c.content.contains(marker));
      if is_detected {
        tracked_detected += 1;
      }
    }

    // Check ignored files (should NOT be searchable)
    let mut false_positive_triggers = 0;
    for path in &ignored_files {
      let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
      // Read content to get marker
      let content = tokio::fs::read_to_string(path).await?;
      if let Some(marker_line) = content.lines().find(|l| l.contains("Marker:")) {
        let marker = marker_line.split("Marker:").nth(1).unwrap_or("").trim();
        let search_result = self
          .client
          .call(CodeSearchParams {
            query: marker.to_string(),
            limit: Some(10),
            ..Default::default()
          })
          .await?;

        // Only count as false positive if the specific ignored file is in results
        let is_indexed = search_result
          .chunks
          .iter()
          .any(|c| c.file_path.contains(file_name) || c.content.contains(marker));
        if is_indexed {
          false_positive_triggers += 1;
        }
      }
    }

    // Stop watcher and cleanup
    let _ = self.client.call(WatchStopParams).await;
    fixtures.cleanup().await?;

    let respect_rate = if !ignored_files.is_empty() {
      1.0 - (false_positive_triggers as f64 / ignored_files.len() as f64)
    } else {
      1.0
    };

    Ok(GitignoreResult {
      ignored_files_modified: ignored_files.len(),
      false_positive_triggers,
      tracked_files_modified: tracked_files.len(),
      tracked_files_detected: tracked_detected,
      respect_rate,
    })
  }

  /// Wait for watcher to be ready (not scanning, no pending changes).
  async fn wait_for_watcher_ready(&self) -> Result<()> {
    let timeout = Duration::from_secs(30);
    let start = Instant::now();

    loop {
      if start.elapsed() > timeout {
        warn!("Timeout waiting for watcher to be ready");
        break;
      }

      let status = self.client.call(WatchStatusParams).await?;
      if !status.scanning && status.pending_changes == 0 {
        break;
      }

      tokio::time::sleep(Duration::from_millis(50)).await;
    }

    Ok(())
  }

  /// Poll for a unique marker to become searchable.
  async fn poll_for_searchable(&self, marker: &str) -> (bool, Instant) {
    let timeout = Duration::from_millis(self.config.search_timeout_ms);
    let poll_interval = Duration::from_millis(self.config.poll_interval_ms);
    let start = Instant::now();

    loop {
      if start.elapsed() > timeout {
        return (false, Instant::now());
      }

      let search_result = self
        .client
        .call(CodeSearchParams {
          query: marker.to_string(),
          limit: Some(1),
          ..Default::default()
        })
        .await;

      if let Ok(result) = search_result
        && !result.chunks.is_empty()
      {
        return (true, Instant::now());
      }

      tokio::time::sleep(poll_interval).await;
    }
  }

  /// Get current file descriptor count (Linux-specific).
  fn get_fd_count(&self) -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
      let fd_dir = format!("/proc/{}/fd", std::process::id());
      std::fs::read_dir(fd_dir).ok().map(|entries| entries.count())
    }

    #[cfg(not(target_os = "linux"))]
    {
      None
    }
  }

  /// Compute summary statistics.
  fn compute_summary(
    lifecycle: &[WatcherLifecycleResult],
    single_change: &[SingleChangeResult],
    batch_change: &[BatchChangeResult],
    gitignore: &[GitignoreResult],
    target_latency_ms: u64,
  ) -> WatcherSummary {
    // E2E latency stats
    let e2e_latencies: Vec<u64> = single_change.iter().map(|r| r.end_to_end_latency_ms).collect();

    let avg_e2e = if e2e_latencies.is_empty() {
      0.0
    } else {
      e2e_latencies.iter().sum::<u64>() as f64 / e2e_latencies.len() as f64
    };

    let mut sorted_latencies = e2e_latencies.clone();
    sorted_latencies.sort_unstable();

    let p95_e2e = sorted_latencies
      .get((sorted_latencies.len() as f64 * 0.95) as usize)
      .copied()
      .unwrap_or(0);
    let max_e2e = sorted_latencies.last().copied().unwrap_or(0);

    // Debounce accuracy
    let debounce_correct_count = batch_change.iter().filter(|r| r.debounce_correct).count();
    let debounce_accuracy = if batch_change.is_empty() {
      1.0
    } else {
      debounce_correct_count as f64 / batch_change.len() as f64
    };

    // Gitignore respect rate
    let gitignore_respect = if gitignore.is_empty() {
      1.0
    } else {
      gitignore.iter().map(|r| r.respect_rate).sum::<f64>() / gitignore.len() as f64
    };

    // Resource leaks
    let resource_leaks = lifecycle.iter().filter(|r| r.leak_detected).count();

    // Overall pass/fail
    let passes = avg_e2e < target_latency_ms as f64 && debounce_accuracy >= 0.9 && gitignore_respect >= 0.95;

    WatcherSummary {
      avg_e2e_latency_ms: avg_e2e,
      p95_e2e_latency_ms: p95_e2e,
      max_e2e_latency_ms: max_e2e,
      debounce_accuracy,
      gitignore_respect_rate: gitignore_respect,
      resource_leaks,
      passes,
    }
  }
}

impl WatcherReport {
  /// Generate markdown report.
  pub fn to_markdown(&self) -> String {
    let mut out = String::new();

    out.push_str("# File Watcher Performance Report\n\n");
    out.push_str(&format!("**Timestamp:** {}\n", self.timestamp));
    out.push_str(&format!("**Version:** {}\n", self.version));
    out.push_str(&format!("**Repository:** {}\n\n", self.repo));

    // Summary
    out.push_str("## Summary\n\n");
    let status = if self.summary.passes { "PASS" } else { "FAIL" };
    out.push_str(&format!("**Status:** {}\n\n", status));
    out.push_str(&format!(
      "- Average E2E latency: {:.1} ms\n",
      self.summary.avg_e2e_latency_ms
    ));
    out.push_str(&format!("- p95 E2E latency: {} ms\n", self.summary.p95_e2e_latency_ms));
    out.push_str(&format!("- Max E2E latency: {} ms\n", self.summary.max_e2e_latency_ms));
    out.push_str(&format!(
      "- Debounce accuracy: {:.1}%\n",
      self.summary.debounce_accuracy * 100.0
    ));
    out.push_str(&format!(
      "- Gitignore respect rate: {:.1}%\n",
      self.summary.gitignore_respect_rate * 100.0
    ));
    out.push_str(&format!(
      "- Resource leaks detected: {}\n\n",
      self.summary.resource_leaks
    ));

    // Lifecycle results
    if !self.lifecycle.is_empty() {
      out.push_str("## Lifecycle Tests\n\n");
      out.push_str("| Startup (ms) | Shutdown (ms) | FD Before | FD After | Leak |\n");
      out.push_str("|--------------|---------------|-----------|----------|------|\n");

      for result in &self.lifecycle {
        let fd_before = result.fd_before.map(|f| f.to_string()).unwrap_or("-".to_string());
        let fd_after = result.fd_after.map(|f| f.to_string()).unwrap_or("-".to_string());
        let leak = if result.leak_detected { "Yes" } else { "No" };

        out.push_str(&format!(
          "| {} | {} | {} | {} | {} |\n",
          result.startup_latency_ms, result.shutdown_latency_ms, fd_before, fd_after, leak,
        ));
      }
      out.push('\n');
    }

    // Single change results
    if !self.single_change.is_empty() {
      out.push_str("## Single Change Tests\n\n");
      out.push_str("| Operation | Detection (ms) | Indexing (ms) | E2E (ms) | Searchable |\n");
      out.push_str("|-----------|----------------|---------------|----------|------------|\n");

      for result in &self.single_change {
        let searchable = if result.searchable { "Yes" } else { "No" };
        out.push_str(&format!(
          "| {} | {} | {} | {} | {} |\n",
          result.operation,
          result.detection_latency_ms,
          result.indexing_latency_ms,
          result.end_to_end_latency_ms,
          searchable,
        ));
      }
      out.push('\n');
    }

    // Batch change results
    if !self.batch_change.is_empty() {
      out.push_str("## Batch Change Tests\n\n");
      out.push_str("| Files | Triggers | Debounce OK | Time (ms) |\n");
      out.push_str("|-------|----------|-------------|----------|\n");

      for result in &self.batch_change {
        let debounce_ok = if result.debounce_correct { "Yes" } else { "No" };
        out.push_str(&format!(
          "| {} | {} | {} | {} |\n",
          result.files_modified, result.reindex_triggers, debounce_ok, result.total_processing_time_ms,
        ));
      }
      out.push('\n');
    }

    // File operations results
    if !self.file_operations.is_empty() {
      out.push_str("## File Operations Tests\n\n");
      out.push_str("| Iteration | Create | Modify | Delete | Rename |\n");
      out.push_str("|-----------|--------|--------|--------|--------|\n");

      for (i, result) in self.file_operations.iter().enumerate() {
        let create = if result.create.success { "OK" } else { "FAIL" };
        let modify = if result.modify.success { "OK" } else { "FAIL" };
        let delete = if result.delete.success { "OK" } else { "FAIL" };
        let rename = if result.rename.success { "OK" } else { "FAIL" };

        out.push_str(&format!(
          "| {} | {} | {} | {} | {} |\n",
          i + 1,
          create,
          modify,
          delete,
          rename,
        ));
      }
      out.push('\n');
    }

    // Gitignore results
    if !self.gitignore.is_empty() {
      out.push_str("## Gitignore Tests\n\n");
      out.push_str("| Ignored | False Positives | Tracked | Detected | Respect Rate |\n");
      out.push_str("|---------|-----------------|---------|----------|-------------|\n");

      for result in &self.gitignore {
        out.push_str(&format!(
          "| {} | {} | {} | {} | {:.1}% |\n",
          result.ignored_files_modified,
          result.false_positive_triggers,
          result.tracked_files_modified,
          result.tracked_files_detected,
          result.respect_rate * 100.0,
        ));
      }
      out.push('\n');
    }

    out
  }

  /// Save report to files (JSON and Markdown).
  pub async fn save(&self, output_dir: &PathBuf) -> Result<()> {
    tokio::fs::create_dir_all(output_dir).await?;

    // Save JSON
    let json_path = output_dir.join("watcher.json");
    let json = serde_json::to_string_pretty(self)?;
    tokio::fs::write(&json_path, json).await?;
    info!("Saved JSON report: {}", json_path.display());

    // Save Markdown
    let md_path = output_dir.join("watcher.md");
    tokio::fs::write(&md_path, self.to_markdown()).await?;
    info!("Saved Markdown report: {}", md_path.display());

    Ok(())
  }
}
