//! Code indexing and search integration tests for the CCEngram daemon
//!
//! Tests: code index dry run, code index with files, language filter,
//! code list and stats, code import chunk, checkpoint resume, index checkpoints.

mod common;

use daemon::Request;
use tempfile::TempDir;

/// Test that the router handles code operations correctly
#[tokio::test]
async fn test_router_code_index_dry_run() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Test code_index with dry_run
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({
        "dry_run": true,
        "cwd": cwd
    }),
  };

  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "code_index dry_run should succeed");
  let result = index_response.result.expect("Should have result");
  assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("dry_run"));
}

/// Test full code indexing with actual files
#[tokio::test]
async fn test_router_code_index_with_files() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create test source files
  let rust_file = project_dir.path().join("test.rs");
  std::fs::write(
    &rust_file,
    r#"
fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Calculator {
    value: i32,
}

impl Calculator {
    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn add(&mut self, n: i32) {
        self.value += n;
    }
}
"#,
  )
  .expect("Failed to write test.rs");

  let py_file = project_dir.path().join("helper.py");
  std::fs::write(
    &py_file,
    r#"
def process_data(items):
    """Process a list of items and return results."""
    return [item * 2 for item in items]

class DataProcessor:
    def __init__(self):
        self.cache = {}

    def process(self, key, value):
        self.cache[key] = value
        return self.cache
"#,
  )
  .expect("Failed to write helper.py");

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "force": true
    }),
  };

  let index_response = router.handle(index_request).await;
  assert!(
    index_response.error.is_none(),
    "code_index should succeed: {:?}",
    index_response.error
  );
  let result = index_response.result.expect("Should have result");

  let status = result.get("status").and_then(|v| v.as_str());
  assert_eq!(status, Some("complete"), "Index should complete");

  let files_indexed = result.get("files_indexed").and_then(|v| v.as_u64()).unwrap_or(0);
  assert!(
    files_indexed >= 2,
    "Should index at least 2 files, got {}",
    files_indexed
  );

  // Search for code
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_search".to_string(),
    params: serde_json::json!({
        "query": "calculate sum",
        "cwd": cwd,
        "limit": 5
    }),
  };

  let search_response = router.handle(search_request).await;
  assert!(
    search_response.error.is_none(),
    "code_search should succeed: {:?}",
    search_response.error
  );
  let response = search_response.result.expect("Should have results");
  let results_arr = response
    .get("results")
    .expect("Should have results field")
    .as_array()
    .expect("Results should be array");

  // Verify results are returned as a valid array structure
  // Note: without embeddings, we can't guarantee matches, but we can verify structure
  assert!(
    results_arr.iter().all(|r| r.is_object()),
    "All results should be objects with code chunk structure"
  );
}

/// Test code search filtering by language
#[tokio::test]
async fn test_router_code_search_language_filter() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create test files
  std::fs::write(
    project_dir.path().join("app.ts"),
    "export function greet(name: string): string { return `Hello, ${name}!`; }",
  )
  .expect("Failed to write app.ts");

  std::fs::write(
    project_dir.path().join("lib.rs"),
    "pub fn greet(name: &str) -> String { format!(\"Hello, {}!\", name) }",
  )
  .expect("Failed to write lib.rs");

  // Index both files
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Index should succeed");
  let index_result = index_response.result.expect("Should have index result");
  let files_indexed = index_result.get("files_indexed").and_then(|v| v.as_u64()).unwrap_or(0);
  assert!(
    files_indexed >= 2,
    "Should index at least 2 files, got {}",
    files_indexed
  );

  // Search with language filter for TypeScript only
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_search".to_string(),
    params: serde_json::json!({
        "query": "greet function",
        "language": "typescript",
        "cwd": cwd
    }),
  };

  let response = router.handle(search_request).await;
  assert!(response.error.is_none(), "Filtered search should succeed");

  // Verify that results only contain TypeScript files (if any results returned)
  if let Some(results) = response.result
    && let Some(results_arr) = results.as_array()
  {
    for result in results_arr {
      let lang = result.get("language").and_then(|v| v.as_str()).unwrap_or("");
      assert!(
        lang == "typescript" || lang == "ts",
        "Language filter should only return TypeScript, got: {}",
        lang
      );
    }
  }
}

/// Test code_list and code_stats tools
#[tokio::test]
async fn test_router_code_list_and_stats() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create test files
  let src_dir = project_dir.path().join("src");
  std::fs::create_dir(&src_dir).unwrap();
  std::fs::write(
    src_dir.join("main.rs"),
    r#"
fn main() {
    println!("Hello, world!");
}

pub fn helper(x: i32) -> i32 {
    x * 2
}
"#,
  )
  .unwrap();

  std::fs::write(
    src_dir.join("lib.rs"),
    r#"
pub mod utils;

pub fn calculate(a: i32, b: i32) -> i32 {
    a + b
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "force": true
    }),
  };
  let index_response = router.handle(index_request).await;
  assert!(
    index_response.error.is_none(),
    "code_index should succeed: {:?}",
    index_response.error
  );

  // Test code_list
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({
        "cwd": cwd
    }),
  };
  let list_response = router.handle(list_request).await;
  assert!(
    list_response.error.is_none(),
    "code_list should succeed: {:?}",
    list_response.error
  );
  let result = list_response.result.unwrap();
  let chunks = result.as_array().unwrap();
  assert!(!chunks.is_empty(), "Should have code chunks");

  // Verify chunk structure
  let first_chunk = &chunks[0];
  assert!(first_chunk.get("file_path").is_some(), "Chunk should have file_path");
  assert!(first_chunk.get("chunk_type").is_some(), "Chunk should have chunk_type");
  assert!(first_chunk.get("content").is_some(), "Chunk should have content");
  assert!(first_chunk.get("start_line").is_some(), "Chunk should have start_line");
  assert!(first_chunk.get("end_line").is_some(), "Chunk should have end_line");

  // Test code_list with limit
  let list_limited = Request {
    id: Some(serde_json::json!(3)),
    method: "code_list".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "limit": 2
    }),
  };
  let limited_response = router.handle(list_limited).await;
  assert!(limited_response.error.is_none(), "code_list with limit should succeed");
  let limited = limited_response.result.unwrap();
  let limited_chunks = limited.as_array().unwrap();
  assert!(limited_chunks.len() <= 2, "Should respect limit");

  // Test code_stats
  let stats_request = Request {
    id: Some(serde_json::json!(4)),
    method: "code_stats".to_string(),
    params: serde_json::json!({
        "cwd": cwd
    }),
  };
  let stats_response = router.handle(stats_request).await;
  assert!(
    stats_response.error.is_none(),
    "code_stats should succeed: {:?}",
    stats_response.error
  );
  let stats = stats_response.result.unwrap();

  // Verify stats structure
  assert!(stats.get("total_chunks").is_some(), "Should have total_chunks");
  assert!(stats.get("total_files").is_some(), "Should have total_files");
  assert!(
    stats.get("language_breakdown").is_some(),
    "Should have language_breakdown"
  );
  assert!(
    stats.get("chunk_type_breakdown").is_some(),
    "Should have chunk_type_breakdown"
  );

  let total_chunks = stats.get("total_chunks").and_then(|v| v.as_u64()).unwrap();
  assert!(total_chunks > 0, "Should have indexed some chunks");

  let languages = stats.get("language_breakdown").and_then(|v| v.as_object()).unwrap();
  assert!(languages.contains_key("rust"), "Should have rust language");
}

/// Test code_import_chunk tool
#[tokio::test]
async fn test_router_code_import_chunk() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Import a code chunk directly (nested chunk structure required)
  let import_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_import_chunk".to_string(),
    params: serde_json::json!({
        "chunk": {
            "file_path": "external/snippet.rs",
            "content": "pub fn external_func() -> String { \"imported\".to_string() }",
            "language": "rs",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 1,
            "symbols": ["external_func"],
            "file_hash": "abc123"
        },
        "cwd": cwd
    }),
  };
  let import_response = router.handle(import_request).await;
  assert!(
    import_response.error.is_none(),
    "code_import_chunk should succeed: {:?}",
    import_response.error
  );
  let result = import_response.result.unwrap();
  assert!(result.get("id").is_some(), "Should return id");
  assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("imported"));

  // Search for the imported chunk
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_search".to_string(),
    params: serde_json::json!({
        "query": "external function imported string",
        "cwd": cwd
    }),
  };
  let search_response = router.handle(search_request).await;
  assert!(search_response.error.is_none(), "code_search should succeed");
  let response = search_response.result.unwrap();
  let results = response
    .get("results")
    .expect("Should have results field")
    .as_array()
    .expect("Results should be array")
    .clone();
  assert!(!results.is_empty(), "Should find imported chunk");

  // Verify the chunk was found
  let found = results.iter().any(|r| {
    r.get("file_path")
      .and_then(|v| v.as_str())
      .map(|p| p.contains("external/snippet.rs"))
      .unwrap_or(false)
  });
  assert!(found, "Should find the imported chunk by file path");
}

/// Test indexing checkpoint resume functionality
#[tokio::test]
async fn test_code_index_checkpoint_resume() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create test files
  std::fs::write(project_dir.path().join("main.rs"), "fn main() { println!(\"Hello\"); }").unwrap();
  std::fs::write(
    project_dir.path().join("lib.rs"),
    "pub fn add(a: i32, b: i32) -> i32 { a + b }",
  )
  .unwrap();
  std::fs::write(
    project_dir.path().join("utils.rs"),
    "pub fn helper() -> String { String::new() }",
  )
  .unwrap();

  // First index - should create checkpoint and index all files
  let index_request1 = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };

  let response1 = router.handle(index_request1).await;
  assert!(
    response1.error.is_none(),
    "First index should succeed: {:?}",
    response1.error
  );
  let result1 = response1.result.expect("Should have result");
  let files_indexed1 = result1.get("files_indexed").and_then(|v| v.as_u64()).unwrap_or(0);
  assert!(files_indexed1 > 0, "Should have indexed files");

  // Second index with resume=true - should find completed checkpoint and re-index fresh
  let index_request2 = Request {
    id: Some(serde_json::json!(2)),
    method: "code_index".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "resume": true
    }),
  };

  let response2 = router.handle(index_request2).await;
  assert!(
    response2.error.is_none(),
    "Second index should succeed: {:?}",
    response2.error
  );
  let result2 = response2.result.expect("Should have result");
  assert_eq!(result2.get("status").and_then(|v| v.as_str()), Some("complete"));

  // Force re-index should clear and re-index
  let index_request3 = Request {
    id: Some(serde_json::json!(3)),
    method: "code_index".to_string(),
    params: serde_json::json!({
        "cwd": cwd,
        "force": true
    }),
  };

  let response3 = router.handle(index_request3).await;
  assert!(
    response3.error.is_none(),
    "Force index should succeed: {:?}",
    response3.error
  );
  let result3 = response3.result.expect("Should have result");
  let files_indexed3 = result3.get("files_indexed").and_then(|v| v.as_u64()).unwrap_or(0);
  assert!(files_indexed3 > 0, "Force should re-index all files");
}

/// Test index checkpoint system for resumable indexing
#[tokio::test]
async fn test_index_checkpoints() {
  use db::{CheckpointType, IndexCheckpoint, ProjectDb};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 4096).await.unwrap();

  let proj_id = "test_project_checkpoints";
  let files = vec![
    "src/main.rs".to_string(),
    "src/lib.rs".to_string(),
    "Cargo.toml".to_string(),
  ];

  // Create a new checkpoint
  let mut checkpoint = IndexCheckpoint::new(proj_id, CheckpointType::Code, files);
  checkpoint.gitignore_hash = Some("abc123".to_string());

  // Save checkpoint
  db.save_checkpoint(&checkpoint).await.unwrap();

  // Retrieve checkpoint
  let retrieved = db.get_checkpoint(proj_id, CheckpointType::Code).await.unwrap();
  assert!(retrieved.is_some(), "Should retrieve checkpoint");
  let retrieved = retrieved.unwrap();
  assert_eq!(retrieved.total_files, 3);
  assert_eq!(retrieved.pending_files.len(), 3);
  assert_eq!(retrieved.processed_count, 0);
  assert!(!retrieved.is_complete);

  // Mark files as processed
  let mut updated = retrieved;
  updated.mark_processed("src/main.rs");
  updated.mark_processed("src/lib.rs");
  updated.mark_error("Cargo.toml");
  db.save_checkpoint(&updated).await.unwrap();

  // Retrieve and verify
  let after_progress = db.get_checkpoint(proj_id, CheckpointType::Code).await.unwrap().unwrap();
  assert_eq!(after_progress.processed_count, 2);
  assert_eq!(after_progress.error_count, 1);
  assert!(after_progress.pending_files.is_empty());
  assert!((after_progress.progress_percent() - 100.0).abs() < 0.01);

  // Mark complete
  let mut completed = after_progress;
  completed.mark_complete();
  db.save_checkpoint(&completed).await.unwrap();

  let final_check = db.get_checkpoint(proj_id, CheckpointType::Code).await.unwrap().unwrap();
  assert!(final_check.is_complete);

  // Clear checkpoint
  db.clear_checkpoint(proj_id, CheckpointType::Code).await.unwrap();
  let cleared = db.get_checkpoint(proj_id, CheckpointType::Code).await.unwrap();
  assert!(cleared.is_none(), "Checkpoint should be cleared");
}
