//! Integration tests for the CCEngram daemon
//!
//! These tests verify end-to-end functionality of the daemon, tools, and database.
//!
//! Note: These tests expect Ollama to be running locally with the nomic-embed-text model.
//! Run: ollama pull nomic-embed-text

use daemon::{ProjectRegistry, Request, Router};
use embedding::{EmbeddingProvider, OllamaProvider};
use std::sync::Arc;
use tempfile::TempDir;

/// Create a router with Ollama embedding and isolated temp directories
fn create_test_router() -> (TempDir, TempDir, Router) {
  let data_dir = TempDir::new().expect("Failed to create data temp dir");
  let project_dir = TempDir::new().expect("Failed to create project temp dir");

  let registry = Arc::new(ProjectRegistry::with_data_dir(data_dir.path().to_path_buf()));
  let embedding: Arc<dyn EmbeddingProvider> = Arc::new(OllamaProvider::new());
  let router = Router::with_embedding(registry, embedding);

  (data_dir, project_dir, router)
}

/// Test that the router handles memory operations correctly
#[tokio::test]
async fn test_router_memory_lifecycle() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Test memory_add
  let add_request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "This is a test memory for integration testing",
        "sector": "semantic",
        "tags": ["test", "integration"],
        "cwd": cwd
    }),
  };

  let add_response = router.handle(add_request).await;
  assert!(
    add_response.error.is_none(),
    "memory_add should succeed: {:?}",
    add_response.error
  );
  let result = add_response.result.expect("Should have result");
  let memory_id = result
    .get("id")
    .expect("Should have id")
    .as_str()
    .expect("id should be string");

  // Test memory_search - use a substring that exists in the content
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "integration testing",
        "cwd": cwd
    }),
  };

  let search_response = router.handle(search_request).await;
  assert!(
    search_response.error.is_none(),
    "memory_search should succeed: {:?}",
    search_response.error
  );
  let results = search_response.result.expect("Should have results");
  let results_arr = results.as_array().expect("Results should be array");
  assert!(
    !results_arr.is_empty(),
    "Should find the memory we added (searched for 'integration testing')"
  );

  // Test memory_reinforce
  let reinforce_request = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_reinforce".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "amount": 0.2,
        "cwd": cwd
    }),
  };

  let reinforce_response = router.handle(reinforce_request).await;
  assert!(
    reinforce_response.error.is_none(),
    "memory_reinforce should succeed: {:?}",
    reinforce_response.error
  );
  let result = reinforce_response.result.expect("Should have result");
  let new_salience = result
    .get("new_salience")
    .expect("Should have new_salience")
    .as_f64()
    .expect("salience should be number");
  assert!(new_salience > 0.9, "Salience should be high after reinforce");

  // Test memory_deemphasize
  let deemphasize_request = Request {
    id: Some(serde_json::json!(4)),
    method: "memory_deemphasize".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "amount": 0.1,
        "cwd": cwd
    }),
  };

  let deemphasize_response = router.handle(deemphasize_request).await;
  assert!(
    deemphasize_response.error.is_none(),
    "memory_deemphasize should succeed"
  );

  // Test memory_delete (soft)
  let delete_request = Request {
    id: Some(serde_json::json!(5)),
    method: "memory_delete".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "hard": false,
        "cwd": cwd
    }),
  };

  let delete_response = router.handle(delete_request).await;
  assert!(delete_response.error.is_none(), "memory_delete should succeed");
  let result = delete_response.result.expect("Should have result");
  assert_eq!(result.get("hard_delete").and_then(|v| v.as_bool()), Some(false));
}

/// Test that the router handles code operations correctly
#[tokio::test]
async fn test_router_code_index_dry_run() {
  let (_data_dir, project_dir, router) = create_test_router();
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

/// Test document ingestion and search
#[tokio::test]
async fn test_router_docs_lifecycle() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Test docs_ingest with content
  let ingest_request = Request {
    id: Some(serde_json::json!(1)),
    method: "docs_ingest".to_string(),
    params: serde_json::json!({
        "content": "This is a test document about Rust programming and memory management. It covers topics like ownership, borrowing, and lifetimes.",
        "title": "Rust Memory Guide",
        "cwd": cwd
    }),
  };

  let ingest_response = router.handle(ingest_request).await;
  assert!(
    ingest_response.error.is_none(),
    "docs_ingest should succeed: {:?}",
    ingest_response.error
  );
  let result = ingest_response.result.expect("Should have result");
  let doc_id = result.get("document_id").expect("Should have document_id");
  assert!(doc_id.is_string(), "document_id should be string");
  let chunks = result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0);
  assert!(chunks >= 1, "Should create at least one chunk");

  // Test docs_search
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "docs_search".to_string(),
    params: serde_json::json!({
        "query": "Rust ownership borrowing",
        "cwd": cwd
    }),
  };

  let search_response = router.handle(search_request).await;
  assert!(search_response.error.is_none(), "docs_search should succeed");
  // Text search fallback should find results since vector search won't have embeddings
}

/// Test validation errors
#[tokio::test]
async fn test_router_validation_errors() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Content too short
  let request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "hi",
        "cwd": cwd
    }),
  };

  let response = router.handle(request).await;
  assert!(response.error.is_some(), "Should reject short content");
  let err = response.error.unwrap();
  assert!(
    err.message.contains("too short"),
    "Error should mention content too short"
  );

  // Missing required field for docs_ingest
  let request = Request {
    id: Some(serde_json::json!(2)),
    method: "docs_ingest".to_string(),
    params: serde_json::json!({
        "cwd": cwd
    }),
  };

  let response = router.handle(request).await;
  assert!(response.error.is_some(), "Should reject missing content/path/url");
}

/// Test memory timeline
#[tokio::test]
async fn test_router_memory_timeline() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // First add a memory
  let add_request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Timeline test memory content for this test",
        "cwd": cwd
    }),
  };

  let add_response = router.handle(add_request).await;
  assert!(
    add_response.error.is_none(),
    "memory_add should succeed: {:?}",
    add_response.error
  );
  let result = add_response.result.expect("Should have result");
  let memory_id = result
    .get("id")
    .expect("Should have id")
    .as_str()
    .expect("id should be string");

  // Test timeline
  let timeline_request = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_timeline".to_string(),
    params: serde_json::json!({
        "anchor_id": memory_id,
        "depth_before": 3,
        "depth_after": 3,
        "cwd": cwd
    }),
  };

  let timeline_response = router.handle(timeline_request).await;
  assert!(timeline_response.error.is_none(), "memory_timeline should succeed");
  let result = timeline_response.result.expect("Should have result");

  // Verify anchor memory is present and matches the requested ID
  let anchor = result.get("anchor").expect("Should have anchor memory");
  assert!(anchor.is_object(), "Anchor should be an object");
  let anchor_id = anchor
    .get("id")
    .and_then(|v| v.as_str())
    .expect("Anchor should have id");
  assert_eq!(anchor_id, memory_id, "Anchor should match requested memory");

  // Verify before/after are arrays (even if empty)
  let before = result.get("before").expect("Should have before array");
  assert!(before.is_array(), "before should be an array");

  let after = result.get("after").expect("Should have after array");
  assert!(after.is_array(), "after should be an array");
}

/// Test memory supersede
#[tokio::test]
async fn test_router_memory_supersede() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create two memories
  let add1 = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Original memory content that will be superseded",
        "cwd": cwd
    }),
  };
  let response1 = router.handle(add1).await;
  assert!(
    response1.error.is_none(),
    "memory_add should succeed: {:?}",
    response1.error
  );
  let old_id = response1
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  let add2 = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "New memory content that supersedes the original",
        "cwd": cwd
    }),
  };
  let response2 = router.handle(add2).await;
  assert!(
    response2.error.is_none(),
    "memory_add should succeed: {:?}",
    response2.error
  );
  let new_id = response2
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  // Supersede
  let supersede_request = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_supersede".to_string(),
    params: serde_json::json!({
        "old_memory_id": old_id,
        "new_memory_id": new_id,
        "cwd": cwd
    }),
  };

  let supersede_response = router.handle(supersede_request).await;
  assert!(supersede_response.error.is_none(), "memory_supersede should succeed");
}

/// Test ping/status commands
#[tokio::test]
async fn test_router_meta_commands() {
  let (_data_dir, _project_dir, router) = create_test_router();

  // Test ping
  let ping_request = Request {
    id: Some(serde_json::json!(1)),
    method: "ping".to_string(),
    params: serde_json::json!({}),
  };

  let ping_response = router.handle(ping_request).await;
  assert!(ping_response.error.is_none());
  assert_eq!(ping_response.result.unwrap(), serde_json::json!("pong"));

  // Test status
  let status_request = Request {
    id: Some(serde_json::json!(2)),
    method: "status".to_string(),
    params: serde_json::json!({}),
  };

  let status_response = router.handle(status_request).await;
  assert!(status_response.error.is_none());
  let result = status_response.result.expect("Should have result");
  assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("running"));
}

/// Test full code indexing with actual files
#[tokio::test]
async fn test_router_code_index_with_files() {
  let (_data_dir, project_dir, router) = create_test_router();
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
  let results = search_response.result.expect("Should have results");
  let results_arr = results.as_array().expect("Results should be array");

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
  let (_data_dir, project_dir, router) = create_test_router();
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

/// Test document ingestion from file path
#[tokio::test]
async fn test_router_docs_ingest_from_file() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create a markdown document
  let doc_content = r#"# Getting Started Guide

This guide explains how to set up the project.

## Installation

Run the following command:
```
npm install
```

## Configuration

Create a `.env` file with your settings.
"#;

  let doc_path = project_dir.path().join("guide.md");
  std::fs::write(&doc_path, doc_content).expect("Failed to write guide.md");

  // Ingest from file
  let ingest_request = Request {
    id: Some(serde_json::json!(1)),
    method: "docs_ingest".to_string(),
    params: serde_json::json!({
        "path": "guide.md",
        "title": "Getting Started",
        "cwd": cwd
    }),
  };

  let response = router.handle(ingest_request).await;
  assert!(
    response.error.is_none(),
    "docs_ingest from file should succeed: {:?}",
    response.error
  );

  let result = response.result.expect("Should have result");
  assert!(result.get("document_id").is_some(), "Should have document_id");
  assert!(result.get("chunks_created").and_then(|v| v.as_u64()).unwrap_or(0) >= 1);

  // Search for the document
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "docs_search".to_string(),
    params: serde_json::json!({
        "query": "npm install configuration",
        "cwd": cwd
    }),
  };

  let search_response = router.handle(search_request).await;
  assert!(search_response.error.is_none(), "docs_search should succeed");
}

/// Test memory search with extended filter options
#[tokio::test]
async fn test_router_memory_search_filters() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Add memories with different attributes
  let add_preference = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "User prefers tabs over spaces for indentation",
        "sector": "emotional",
        "type": "preference",
        "scope_path": "src/formatter",
        "scope_module": "formatter",
        "categories": ["coding-style", "preferences"],
        "importance": 0.9,
        "cwd": cwd
    }),
  };
  let pref_response = router.handle(add_preference).await;
  assert!(pref_response.error.is_none(), "Should add preference memory");

  let add_codebase = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "The Router struct handles all incoming MCP requests",
        "sector": "semantic",
        "type": "codebase",
        "scope_path": "src/router",
        "scope_module": "router",
        "categories": ["architecture"],
        "importance": 0.7,
        "cwd": cwd
    }),
  };
  let code_response = router.handle(add_codebase).await;
  assert!(code_response.error.is_none(), "Should add codebase memory");

  // Search with sector filter - should only return emotional sector
  let search_sector = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "indentation",
        "sector": "emotional",
        "cwd": cwd
    }),
  };
  let sector_response = router.handle(search_sector).await;
  assert!(sector_response.error.is_none(), "Sector-filtered search should succeed");

  // Verify sector filter is applied
  if let Some(results) = sector_response.result.as_ref().and_then(|r| r.as_array()) {
    for result in results {
      let sector = result.get("sector").and_then(|v| v.as_str()).unwrap_or("");
      assert_eq!(
        sector, "emotional",
        "Sector filter should only return emotional memories, got: {}",
        sector
      );
    }
  }

  // Search with memory type filter - should only return codebase type
  let search_type = Request {
    id: Some(serde_json::json!(4)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "code",
        "type": "codebase",
        "cwd": cwd
    }),
  };
  let type_response = router.handle(search_type).await;
  assert!(type_response.error.is_none(), "Type-filtered search should succeed");

  // Verify type filter is applied
  if let Some(results) = type_response.result.as_ref().and_then(|r| r.as_array()) {
    for result in results {
      let mem_type = result.get("memory_type").and_then(|v| v.as_str()).unwrap_or("");
      assert_eq!(
        mem_type, "codebase",
        "Type filter should only return codebase memories, got: {}",
        mem_type
      );
    }
  }

  // Search with min_salience filter
  let search_salience = Request {
    id: Some(serde_json::json!(5)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "preference",
        "min_salience": 0.5,
        "cwd": cwd
    }),
  };
  let salience_response = router.handle(search_salience).await;
  assert!(
    salience_response.error.is_none(),
    "Salience-filtered search should succeed"
  );

  // Search with scope_path filter (prefix match)
  let search_scope = Request {
    id: Some(serde_json::json!(6)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "code",
        "scope_path": "src/",
        "cwd": cwd
    }),
  };
  let scope_response = router.handle(search_scope).await;
  assert!(scope_response.error.is_none(), "Scope-filtered search should succeed");
}

/// Test memory search result metadata includes new fields
#[tokio::test]
async fn test_router_memory_search_result_metadata() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Add a memory with all fields populated - use unique keyword for reliable text search
  let add_request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "The project uses XYZMETADATA framework for implementation testing",
        "sector": "semantic",
        "type": "codebase",
        "scope_path": "src/frontend",
        "scope_module": "ui",
        "categories": ["tech-stack", "frontend"],
        "tags": ["framework", "frontend"],
        "importance": 0.8,
        "cwd": cwd
    }),
  };
  let add_response = router.handle(add_request).await;
  assert!(add_response.error.is_none(), "Should add memory");

  // Search for it using the unique keyword
  let search_request = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "XYZMETADATA",
        "cwd": cwd
    }),
  };
  let search_response = router.handle(search_request).await;
  assert!(search_response.error.is_none(), "Search should succeed");

  let results = search_response.result.expect("Should have results");
  let results_arr = results.as_array().expect("Results should be array");
  assert!(!results_arr.is_empty(), "Should find the memory with unique keyword");

  // Verify result includes extended metadata
  let first_result = &results_arr[0];
  assert!(first_result.get("sector").is_some(), "Result should include sector");
  assert!(first_result.get("tier").is_some(), "Result should include tier");
  assert!(
    first_result.get("memory_type").is_some(),
    "Result should include memory_type"
  );
  assert!(first_result.get("salience").is_some(), "Result should include salience");
  assert!(
    first_result.get("importance").is_some(),
    "Result should include importance"
  );
  assert!(
    first_result.get("is_superseded").is_some(),
    "Result should include is_superseded"
  );
  assert!(first_result.get("tags").is_some(), "Result should include tags");
  assert!(
    first_result.get("categories").is_some(),
    "Result should include categories"
  );
  assert!(
    first_result.get("scope_path").is_some(),
    "Result should include scope_path"
  );
  assert!(
    first_result.get("scope_module").is_some(),
    "Result should include scope_module"
  );
  assert!(
    first_result.get("created_at").is_some(),
    "Result should include created_at"
  );
  assert!(
    first_result.get("last_accessed").is_some(),
    "Result should include last_accessed"
  );
}

/// Test memory hard delete vs soft delete
#[tokio::test]
async fn test_router_memory_delete_modes() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create two memories
  let add1 = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Memory for soft delete test content here",
        "cwd": cwd
    }),
  };
  let response1 = router.handle(add1).await;
  let soft_delete_id = response1
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  let add2 = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Memory for hard delete test content here",
        "cwd": cwd
    }),
  };
  let response2 = router.handle(add2).await;
  let hard_delete_id = response2
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  // Soft delete first memory
  let soft_delete = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_delete".to_string(),
    params: serde_json::json!({
        "memory_id": soft_delete_id,
        "hard": false,
        "cwd": cwd
    }),
  };
  let soft_response = router.handle(soft_delete).await;
  assert!(soft_response.error.is_none(), "Soft delete should succeed");
  let soft_result = soft_response.result.unwrap();
  assert_eq!(soft_result.get("hard_delete").and_then(|v| v.as_bool()), Some(false));

  // Hard delete second memory
  let hard_delete = Request {
    id: Some(serde_json::json!(4)),
    method: "memory_delete".to_string(),
    params: serde_json::json!({
        "memory_id": hard_delete_id,
        "hard": true,
        "cwd": cwd
    }),
  };
  let hard_response = router.handle(hard_delete).await;
  assert!(hard_response.error.is_none(), "Hard delete should succeed");
  let hard_result = hard_response.result.unwrap();
  assert_eq!(hard_result.get("hard_delete").and_then(|v| v.as_bool()), Some(true));
}

/// Test search with include_superseded flag
#[tokio::test]
async fn test_router_memory_search_include_superseded() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create and supersede a memory
  let add_old = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Project uses PostgreSQL 14 for database needs",
        "cwd": cwd
    }),
  };
  let old_response = router.handle(add_old).await;
  let old_id = old_response
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  let add_new = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Project now uses PostgreSQL 16 (upgraded)",
        "cwd": cwd
    }),
  };
  let new_response = router.handle(add_new).await;
  let new_id = new_response
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  // Supersede the old memory
  let supersede = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_supersede".to_string(),
    params: serde_json::json!({
        "old_memory_id": old_id,
        "new_memory_id": new_id,
        "cwd": cwd
    }),
  };
  let supersede_response = router.handle(supersede).await;
  assert!(
    supersede_response.error.is_none(),
    "Supersede should succeed: {:?}",
    supersede_response.error
  );

  // Search without include_superseded (default) - should only find new
  let search_default = Request {
    id: Some(serde_json::json!(4)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "PostgreSQL database",
        "cwd": cwd
    }),
  };
  let default_response = router.handle(search_default).await;
  let default_results = default_response.result.unwrap();
  let default_arr = default_results.as_array().unwrap();
  // Non-superseded results should not show the old memory
  for result in default_arr {
    let is_superseded = result.get("is_superseded").and_then(|v| v.as_bool()).unwrap_or(false);
    // The old memory (with "14") should be filtered out or marked superseded
    if result
      .get("content")
      .and_then(|v| v.as_str())
      .unwrap_or("")
      .contains("14")
    {
      assert!(is_superseded, "Old memory should be superseded");
    }
  }

  // Search with include_superseded = true - should find both
  let search_all = Request {
    id: Some(serde_json::json!(5)),
    method: "memory_search".to_string(),
    params: serde_json::json!({
        "query": "PostgreSQL database",
        "include_superseded": true,
        "cwd": cwd
    }),
  };
  let all_response = router.handle(search_all).await;
  assert!(
    all_response.error.is_none(),
    "Search with include_superseded should succeed"
  );
}

/// Test memory relationships database operations
#[tokio::test]
async fn test_memory_relationships() {
  use db::ProjectDb;
  use engram_core::{MemoryId, RelationshipType};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

  // Create memory IDs
  let from_id = MemoryId::new();
  let to_id1 = MemoryId::new();
  let to_id2 = MemoryId::new();

  // Create relationships
  let rel1 = db
    .create_relationship(&from_id, &to_id1, RelationshipType::BuildsOn, 0.9, "test")
    .await
    .unwrap();
  assert_eq!(rel1.relationship_type, RelationshipType::BuildsOn);
  assert_eq!(rel1.confidence, 0.9);

  let rel2 = db
    .create_relationship(&from_id, &to_id2, RelationshipType::Contradicts, 0.7, "test")
    .await
    .unwrap();
  assert_eq!(rel2.relationship_type, RelationshipType::Contradicts);

  // Get relationships from
  let rels_from = db.get_relationships_from(&from_id).await.unwrap();
  assert_eq!(rels_from.len(), 2);

  // Get relationships of specific type
  let contradicts = db
    .get_relationships_of_type(&from_id, RelationshipType::Contradicts)
    .await
    .unwrap();
  assert_eq!(contradicts.len(), 1);
  assert_eq!(contradicts[0].to_memory_id, to_id2);

  // Count relationships
  let count = db.count_relationships(&from_id).await.unwrap();
  assert_eq!(count, 2);

  // Delete relationships for a memory
  db.delete_relationships_for_memory(&from_id).await.unwrap();
  let count_after = db.count_relationships(&from_id).await.unwrap();
  assert_eq!(count_after, 0);
}

/// Test document metadata and update detection
#[tokio::test]
async fn test_document_metadata_and_updates() {
  use db::{ProjectDb, compute_content_hash};
  use engram_core::{Document, DocumentSource};
  use std::io::Write;
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create data temp dir");
  let project_dir = TempDir::new().expect("Failed to create project temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

  let doc_project_id = uuid::Uuid::new_v4();

  // Create a test file
  let file_path = project_dir.path().join("test_doc.md");
  let initial_content = "# Initial Content\n\nThis is the initial document content.";
  std::fs::write(&file_path, initial_content).unwrap();

  // Create document metadata
  let content_hash = compute_content_hash(initial_content);
  let doc = Document::new(
    doc_project_id,
    "Test Document".to_string(),
    file_path.to_string_lossy().to_string(),
    DocumentSource::File,
    content_hash.clone(),
    initial_content.len(),
    1,
  );

  // Store metadata
  db.upsert_document_metadata(&doc).await.unwrap();

  // Verify it can be retrieved by ID
  let retrieved = db.get_document_metadata(&doc.id).await.unwrap();
  assert!(retrieved.is_some());
  assert_eq!(retrieved.unwrap().content_hash, content_hash);

  // Verify it can be retrieved by source path
  let by_source = db.get_document_by_source(&file_path.to_string_lossy()).await.unwrap();
  assert!(by_source.is_some());
  assert_eq!(by_source.unwrap().id, doc.id);

  // Check for updates - should be unchanged
  let updates = db.check_document_updates(&doc_project_id.to_string()).await.unwrap();
  assert!(updates.modified.is_empty(), "Should not detect modifications yet");
  assert!(updates.missing.is_empty(), "File should exist");

  // Modify the file
  let mut file = std::fs::File::create(&file_path).unwrap();
  writeln!(file, "# Modified Content\n\nThis content has been changed.").unwrap();
  drop(file);

  // Check for updates again - should detect modification
  let updates_after = db.check_document_updates(&doc_project_id.to_string()).await.unwrap();
  assert_eq!(updates_after.modified.len(), 1, "Should detect one modified document");
  assert_eq!(updates_after.modified[0], doc.id);

  // Delete the file
  std::fs::remove_file(&file_path).unwrap();

  // Check for updates - should detect missing
  let updates_missing = db.check_document_updates(&doc_project_id.to_string()).await.unwrap();
  assert_eq!(updates_missing.missing.len(), 1, "Should detect one missing document");
  assert_eq!(updates_missing.missing[0], doc.id);
}

/// Test session-memory linkage
#[tokio::test]
async fn test_session_memory_links() {
  use db::{ProjectDb, Session, UsageType};
  use engram_core::ProjectId;
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id.clone(), db_path, 768).await.unwrap();

  // Create and add a session
  let mut session = Session::new(uuid::Uuid::new_v4());
  session.user_prompt = Some("Test prompt".to_string());
  db.add_session(&session).await.unwrap();

  // Create memory IDs (as strings)
  let mem1 = uuid::Uuid::new_v4().to_string();
  let mem2 = uuid::Uuid::new_v4().to_string();

  // Link memories to session
  db.link_memory(session.id, &mem1, UsageType::Created).await.unwrap();
  db.link_memory(session.id, &mem2, UsageType::Recalled).await.unwrap();
  db.link_memory(session.id, &mem1, UsageType::Updated).await.unwrap(); // Same memory, different usage

  // Get session links
  let links = db.get_session_memory_links(&session.id).await.unwrap();
  assert_eq!(links.len(), 3, "Should have 3 links");

  // Get session stats
  let stats = db.get_session_stats(&session.id).await.unwrap();
  assert_eq!(stats.total_memories, 3);
  assert_eq!(stats.created, 1);
  assert_eq!(stats.recalled, 1);
  assert_eq!(stats.updated, 1);

  // Get memory usage count
  let mem1_count = db.get_memory_usage_count(&mem1).await.unwrap();
  assert_eq!(mem1_count, 2, "mem1 should have 2 usages");

  // Get memories for session
  let session_memories = db.get_memory_session_links(&mem1).await.unwrap();
  assert!(!session_memories.is_empty(), "Should find links from memory to session");
}

/// Test RelationshipType parsing
#[test]
fn test_relationship_type_from_str() {
  use engram_core::RelationshipType;

  // Test all types
  assert_eq!(
    "supersedes".parse::<RelationshipType>().unwrap(),
    RelationshipType::Supersedes
  );
  assert_eq!(
    "contradicts".parse::<RelationshipType>().unwrap(),
    RelationshipType::Contradicts
  );
  assert_eq!(
    "related_to".parse::<RelationshipType>().unwrap(),
    RelationshipType::RelatedTo
  );
  assert_eq!(
    "builds_on".parse::<RelationshipType>().unwrap(),
    RelationshipType::BuildsOn
  );
  assert_eq!(
    "confirms".parse::<RelationshipType>().unwrap(),
    RelationshipType::Confirms
  );
  assert_eq!(
    "applies_to".parse::<RelationshipType>().unwrap(),
    RelationshipType::AppliesTo
  );
  assert_eq!(
    "depends_on".parse::<RelationshipType>().unwrap(),
    RelationshipType::DependsOn
  );
  assert_eq!(
    "alternative_to".parse::<RelationshipType>().unwrap(),
    RelationshipType::AlternativeTo
  );

  // Test case insensitivity
  assert_eq!(
    "SUPERSEDES".parse::<RelationshipType>().unwrap(),
    RelationshipType::Supersedes
  );

  // Test alternate forms
  assert_eq!(
    "relatedto".parse::<RelationshipType>().unwrap(),
    RelationshipType::RelatedTo
  );

  // Test invalid
  assert!("invalid".parse::<RelationshipType>().is_err());
}

/// Test entity extraction subsystem
#[tokio::test]
async fn test_entity_operations() {
  use db::ProjectDb;
  use engram_core::{Entity, EntityType};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

  // Create an entity
  let entity = Entity {
    id: uuid::Uuid::new_v4(),
    name: "John Doe".to_string(),
    entity_type: EntityType::Person,
    summary: Some("A test user".to_string()),
    aliases: vec!["JD".to_string(), "Johnnie".to_string()],
    first_seen_at: chrono::Utc::now(),
    last_seen_at: chrono::Utc::now(),
    mention_count: 1,
  };

  // Add entity
  db.add_entity(&entity).await.unwrap();

  // Get entity by ID
  let retrieved = db.get_entity(&entity.id).await.unwrap();
  assert!(retrieved.is_some(), "Should retrieve entity by ID");
  let retrieved = retrieved.unwrap();
  assert_eq!(retrieved.name, "John Doe");
  assert_eq!(retrieved.entity_type, EntityType::Person);

  // Find entity by name
  let found = db.find_entity_by_name("John Doe").await.unwrap();
  assert!(found.is_some(), "Should find entity by name");
  assert_eq!(found.unwrap().id, entity.id);

  // Update entity
  let mut updated = retrieved.clone();
  updated.mention_count = 5;
  updated.summary = Some("An updated test user".to_string());
  db.update_entity(&updated).await.unwrap();

  let after_update = db.get_entity(&entity.id).await.unwrap().unwrap();
  assert_eq!(after_update.mention_count, 5);
  assert_eq!(after_update.summary, Some("An updated test user".to_string()));

  // Add another entity of different type
  let entity2 = Entity {
    id: uuid::Uuid::new_v4(),
    name: "Rust".to_string(),
    entity_type: EntityType::Technology,
    summary: Some("A programming language".to_string()),
    aliases: vec!["rust-lang".to_string()],
    first_seen_at: chrono::Utc::now(),
    last_seen_at: chrono::Utc::now(),
    mention_count: 1,
  };
  db.add_entity(&entity2).await.unwrap();

  // List entities by type
  let people = db.list_entities_by_type(EntityType::Person).await.unwrap();
  assert_eq!(people.len(), 1);
  assert_eq!(people[0].name, "John Doe");

  let techs = db.list_entities_by_type(EntityType::Technology).await.unwrap();
  assert_eq!(techs.len(), 1);
  assert_eq!(techs[0].name, "Rust");

  // Get top entities
  let top = db.get_top_entities(10).await.unwrap();
  assert_eq!(top.len(), 2);
  // After update, John Doe has 5 mentions so should be first
  assert_eq!(top[0].name, "John Doe");

  // Record additional mention
  db.record_entity_mention(&entity2.id).await.unwrap();
  let after_mention = db.get_entity(&entity2.id).await.unwrap().unwrap();
  assert_eq!(after_mention.mention_count, 2);

  // Find or create (existing)
  let found_or_created = db.find_or_create_entity("John Doe", EntityType::Person).await.unwrap();
  assert_eq!(found_or_created.id, entity.id, "Should find existing entity");

  // Find or create (new)
  let new_entity = db.find_or_create_entity("React", EntityType::Technology).await.unwrap();
  assert_ne!(new_entity.id, entity.id);
  assert_eq!(new_entity.name, "React");

  // Delete entity
  db.delete_entity(&entity.id).await.unwrap();
  let after_delete = db.get_entity(&entity.id).await.unwrap();
  assert!(after_delete.is_none(), "Entity should be deleted");
}

/// Test index checkpoint system for resumable indexing
#[tokio::test]
async fn test_index_checkpoints() {
  use db::{CheckpointType, IndexCheckpoint, ProjectDb};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

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

/// Test extended session statistics with sector breakdown
#[tokio::test]
async fn test_session_stats_extended() {
  use db::{ProjectDb, Session, UsageType};
  use engram_core::{Memory, ProjectId, Sector};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id.clone(), db_path, 768).await.unwrap();

  // Create a session
  let session = Session::new(uuid::Uuid::new_v4());
  db.add_session(&session).await.unwrap();

  let proj_uuid = uuid::Uuid::new_v4();

  // Create memories with different sectors and saliences
  let mut m1 = Memory::new(proj_uuid, "Semantic memory content".to_string(), Sector::Semantic);
  m1.content_hash = "hash1".to_string();
  m1.salience = 0.8;

  let mut m2 = Memory::new(proj_uuid, "Emotional memory content".to_string(), Sector::Emotional);
  m2.content_hash = "hash2".to_string();
  m2.salience = 0.6;

  let mut m3 = Memory::new(proj_uuid, "Procedural memory content".to_string(), Sector::Procedural);
  m3.content_hash = "hash3".to_string();
  m3.salience = 0.4;

  // Add memories
  db.add_memory(&m1, None).await.unwrap();
  db.add_memory(&m2, None).await.unwrap();
  db.add_memory(&m3, None).await.unwrap();

  // Link to session
  db.link_memory(session.id, &m1.id.to_string(), UsageType::Created)
    .await
    .unwrap();
  db.link_memory(session.id, &m2.id.to_string(), UsageType::Created)
    .await
    .unwrap();
  db.link_memory(session.id, &m3.id.to_string(), UsageType::Recalled)
    .await
    .unwrap();

  // Get extended session stats
  let stats = db.get_session_stats(&session.id).await.unwrap();
  assert_eq!(stats.total_memories, 3);
  assert_eq!(stats.created, 2);
  assert_eq!(stats.recalled, 1);

  // Verify sector breakdown
  assert_eq!(*stats.by_sector.get("semantic").unwrap_or(&0), 1);
  assert_eq!(*stats.by_sector.get("emotional").unwrap_or(&0), 1);
  assert_eq!(*stats.by_sector.get("procedural").unwrap_or(&0), 1);

  // Verify average salience (0.8 + 0.6 + 0.4) / 3 = 0.6
  assert!(
    (stats.average_salience - 0.6).abs() < 0.01,
    "Average salience should be 0.6, got {}",
    stats.average_salience
  );
}

/// Test memory promotion from session tier to project tier
#[tokio::test]
async fn test_memory_promotion() {
  use db::{ProjectDb, Session, UsageType};
  use engram_core::{Memory, ProjectId, Sector, Tier};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id.clone(), db_path, 768).await.unwrap();

  // Create two sessions
  let session1 = Session::new(uuid::Uuid::new_v4());
  let session2 = Session::new(uuid::Uuid::new_v4());
  db.add_session(&session1).await.unwrap();
  db.add_session(&session2).await.unwrap();

  let proj_uuid = uuid::Uuid::new_v4();

  // Create a session-tier memory
  let mut memory = Memory::new(proj_uuid, "Session tier memory".to_string(), Sector::Semantic);
  memory.content_hash = "hash1".to_string();
  memory.tier = Tier::Session;
  db.add_memory(&memory, None).await.unwrap();

  // Link to session1 as created
  db.link_memory(session1.id, &memory.id.to_string(), UsageType::Created)
    .await
    .unwrap();

  // Try promotion with threshold 2 - should not promote (only 1 usage)
  let promoted = db.promote_session_memories(&session1.id, 2).await.unwrap();
  assert_eq!(promoted, 0, "Should not promote with only 1 usage");

  // Verify still session tier
  let still_session = db.get_memory(&memory.id).await.unwrap().unwrap();
  assert_eq!(still_session.tier, Tier::Session);

  // Link to session2 (second usage)
  db.link_memory(session2.id, &memory.id.to_string(), UsageType::Recalled)
    .await
    .unwrap();

  // Now promotion should work (2 usages >= threshold)
  let promoted = db.promote_session_memories(&session1.id, 2).await.unwrap();
  assert_eq!(promoted, 1, "Should promote 1 memory");

  // Verify promoted to project tier
  let promoted_memory = db.get_memory(&memory.id).await.unwrap().unwrap();
  assert_eq!(
    promoted_memory.tier,
    Tier::Project,
    "Memory should be promoted to project tier"
  );
}

/// Test document full content storage
#[tokio::test]
async fn test_document_full_content_storage() {
  use db::ProjectDb;
  use engram_core::{Document, DocumentSource};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

  let project_uuid = uuid::Uuid::new_v4();
  let full_content = "This is the full document content.\n\nWith multiple paragraphs.\n\nAnd more text.";

  // Create document with full content
  let doc = Document::with_content(
    project_uuid,
    "Full Content Doc".to_string(),
    "test://doc".to_string(),
    DocumentSource::Content,
    full_content.to_string(),
    3,
  );

  // Verify hash was computed
  assert!(!doc.content_hash.is_empty());
  assert_eq!(doc.char_count, full_content.len());
  assert!(doc.full_content.is_some());

  // Store and retrieve
  db.upsert_document_metadata(&doc).await.unwrap();

  let retrieved = db.get_document_metadata(&doc.id).await.unwrap().unwrap();
  assert_eq!(retrieved.full_content, Some(full_content.to_string()));
  assert_eq!(retrieved.char_count, full_content.len());
}

/// Test memory-entity linkage
#[tokio::test]
async fn test_memory_entity_links() {
  use db::ProjectDb;
  use engram_core::{Entity, EntityRole, EntityType, MemoryEntityLink};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

  // Create entity
  let entity = Entity {
    id: uuid::Uuid::new_v4(),
    name: "Test Entity".to_string(),
    entity_type: EntityType::Concept,
    summary: None,
    aliases: vec![],
    first_seen_at: chrono::Utc::now(),
    last_seen_at: chrono::Utc::now(),
    mention_count: 1,
  };
  db.add_entity(&entity).await.unwrap();

  // Create memory IDs (as strings)
  let memory_id1 = uuid::Uuid::new_v4().to_string();
  let memory_id2 = uuid::Uuid::new_v4().to_string();

  // Create and link entity to memories
  let link1 = MemoryEntityLink::new(memory_id1.clone(), entity.id, EntityRole::Subject, 0.9);
  db.link_entity_to_memory(&link1).await.unwrap();
  assert_eq!(link1.role, EntityRole::Subject);
  assert_eq!(link1.confidence, 0.9);

  let link2 = MemoryEntityLink::new(memory_id2.clone(), entity.id, EntityRole::Reference, 0.7);
  db.link_entity_to_memory(&link2).await.unwrap();
  assert_eq!(link2.role, EntityRole::Reference);

  // Get entity links for a memory
  let mem1_links = db.get_memory_entity_links(&memory_id1).await.unwrap();
  assert_eq!(mem1_links.len(), 1);
  assert_eq!(mem1_links[0].entity_id, entity.id);

  // Get memory links for an entity
  let entity_links = db.get_entity_memory_links(&entity.id).await.unwrap();
  assert_eq!(entity_links.len(), 2);

  // Check if link exists
  let exists = db.entity_memory_link_exists(&memory_id1, &entity.id).await.unwrap();
  assert!(exists, "Link should exist");

  let not_exists = db.entity_memory_link_exists("fake-id", &entity.id).await.unwrap();
  assert!(!not_exists, "Link should not exist for fake memory");

  // Count entity memories
  let count = db.count_entity_memories(&entity.id).await.unwrap();
  assert_eq!(count, 2);

  // Delete all links for the memory
  db.delete_memory_entity_links(&memory_id1).await.unwrap();
  let after_delete = db.get_memory_entity_links(&memory_id1).await.unwrap();
  assert!(after_delete.is_empty(), "Link should be deleted");

  // Count again
  let count_after = db.count_entity_memories(&entity.id).await.unwrap();
  assert_eq!(count_after, 1);
}

/// Test entity tools via router
#[tokio::test]
async fn test_router_entity_tools() {
  use engram_core::{Entity, EntityType};

  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // First, create an entity directly in the database
  let project_path = project_dir.path();
  let project_id = engram_core::ProjectId::from_path(project_path);
  let data_dir = _data_dir
    .path()
    .join("projects")
    .join(project_id.as_str())
    .join("lancedb");
  std::fs::create_dir_all(&data_dir).unwrap();
  let db = db::ProjectDb::open_at_path(project_id, data_dir, 768).await.unwrap();

  let entity = Entity {
    id: uuid::Uuid::new_v4(),
    name: "Rust".to_string(),
    entity_type: EntityType::Technology,
    summary: Some("A systems programming language".to_string()),
    aliases: vec!["rust-lang".to_string()],
    first_seen_at: chrono::Utc::now(),
    last_seen_at: chrono::Utc::now(),
    mention_count: 5,
  };
  db.add_entity(&entity).await.unwrap();

  // Test entity_list
  let list_request = Request {
    id: Some(serde_json::json!(1)),
    method: "entity_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  assert!(
    list_response.error.is_none(),
    "entity_list should succeed: {:?}",
    list_response.error
  );

  // Test entity_get by name
  let get_request = Request {
    id: Some(serde_json::json!(2)),
    method: "entity_get".to_string(),
    params: serde_json::json!({ "cwd": cwd, "name": "Rust" }),
  };
  let get_response = router.handle(get_request).await;
  assert!(
    get_response.error.is_none(),
    "entity_get should succeed: {:?}",
    get_response.error
  );
  let result = get_response.result.expect("Should have result");
  assert_eq!(result.get("name").and_then(|v| v.as_str()), Some("Rust"));

  // Test entity_top
  let top_request = Request {
    id: Some(serde_json::json!(3)),
    method: "entity_top".to_string(),
    params: serde_json::json!({ "cwd": cwd, "limit": 5 }),
  };
  let top_response = router.handle(top_request).await;
  assert!(
    top_response.error.is_none(),
    "entity_top should succeed: {:?}",
    top_response.error
  );
}

/// Test watcher tools via router
#[tokio::test]
async fn test_router_watcher_tools() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Test watch_start
  let start_request = Request {
    id: Some(serde_json::json!(1)),
    method: "watch_start".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let start_response = router.handle(start_request).await;
  assert!(
    start_response.error.is_none(),
    "watch_start should succeed: {:?}",
    start_response.error
  );
  let result = start_response.result.expect("Should have result");
  assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("started"));

  // Test watch_status
  let status_request = Request {
    id: Some(serde_json::json!(2)),
    method: "watch_status".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let status_response = router.handle(status_request).await;
  assert!(
    status_response.error.is_none(),
    "watch_status should succeed: {:?}",
    status_response.error
  );
  let result = status_response.result.expect("Should have result");
  assert_eq!(result.get("running").and_then(|v| v.as_bool()), Some(true));

  // Test watch_stop
  let stop_request = Request {
    id: Some(serde_json::json!(3)),
    method: "watch_stop".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let stop_response = router.handle(stop_request).await;
  assert!(
    stop_response.error.is_none(),
    "watch_stop should succeed: {:?}",
    stop_response.error
  );
  let result = stop_response.result.expect("Should have result");
  assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("stopped"));

  // Verify status shows stopped
  let status_request2 = Request {
    id: Some(serde_json::json!(4)),
    method: "watch_status".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let status_response2 = router.handle(status_request2).await;
  let result = status_response2.result.expect("Should have result");
  assert_eq!(result.get("running").and_then(|v| v.as_bool()), Some(false));
}

/// Test memory deduplication - adding same content twice should return existing memory
#[tokio::test]
async fn test_memory_deduplication() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  let content = "This is a unique memory about integration testing that should be deduplicated when added twice";

  // Add the memory first time
  let add_request1 = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": content,
        "sector": "semantic",
        "cwd": cwd
    }),
  };

  let response1 = router.handle(add_request1).await;
  assert!(
    response1.error.is_none(),
    "First add should succeed: {:?}",
    response1.error
  );
  let result1 = response1.result.expect("Should have result");
  let id1 = result1
    .get("id")
    .expect("Should have id")
    .as_str()
    .expect("id should be string");

  // Add the exact same content again - should return same ID (deduplication)
  let add_request2 = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": content,
        "sector": "semantic",
        "cwd": cwd
    }),
  };

  let response2 = router.handle(add_request2).await;
  assert!(
    response2.error.is_none(),
    "Second add should succeed: {:?}",
    response2.error
  );
  let result2 = response2.result.expect("Should have result");
  let id2 = result2
    .get("id")
    .expect("Should have id")
    .as_str()
    .expect("id should be string");

  // Should return the same ID due to deduplication
  assert_eq!(id1, id2, "Duplicate content should return same memory ID");

  // Check for is_duplicate flag
  let is_dup = result2.get("is_duplicate").and_then(|v| v.as_bool()).unwrap_or(false);
  assert!(is_dup, "Second add should be marked as duplicate");

  // Completely different content should create new memory
  let different_content = "This is completely different content about debugging database issues";
  let add_request3 = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": different_content,
        "sector": "semantic",
        "cwd": cwd
    }),
  };

  let response3 = router.handle(add_request3).await;
  assert!(
    response3.error.is_none(),
    "Third add should succeed: {:?}",
    response3.error
  );
  let result3 = response3.result.expect("Should have result");
  let id3 = result3
    .get("id")
    .expect("Should have id")
    .as_str()
    .expect("id should be string");

  // Different content should create new memory
  assert_ne!(id1, id3, "Different content should create new memory");
}

/// Test indexing checkpoint resume functionality
#[tokio::test]
async fn test_code_index_checkpoint_resume() {
  let (_data_dir, project_dir, router) = create_test_router();
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

/// Test decay calculation and stats
#[tokio::test]
async fn test_decay_functionality() {
  use chrono::{Duration, Utc};
  use engram_core::{Memory, Sector};
  use extract::{DecayConfig, apply_decay};

  // Create a memory with high salience
  let project_id = uuid::Uuid::new_v4();
  let mut memory = Memory::new(project_id, "Test memory for decay".into(), Sector::Episodic);
  memory.salience = 1.0;
  memory.importance = 0.5;

  let config = DecayConfig::default();

  // Apply decay for 30 days in the future
  let future = Utc::now() + Duration::days(30);
  let result = apply_decay(&mut memory, future, &config);

  // Verify decay happened
  assert!(
    result.new_salience < result.previous_salience,
    "Salience should decrease"
  );
  assert_eq!(result.previous_salience, 1.0);
  assert!(result.days_since_access >= 29.0); // ~30 days

  // Verify memory is not archived yet (needs to be below threshold)
  // For 30 days, episodic memories should still be above archive threshold
  assert!(!result.should_archive || result.new_salience < config.archive_threshold);
}

/// Test deduplication detection with similar content (SimHash)
#[tokio::test]
async fn test_deduplication_simhash_similarity() {
  use extract::{DuplicateChecker, DuplicateMatch, content_hash, simhash};

  let checker = DuplicateChecker::new();

  let content1 = "The user wants to implement a feature for handling authentication in the application";
  let hash1 = content_hash(content1);

  // Create first memory (needs content_hash and simhash set on the memory)
  let mut memory1 = engram_core::Memory::new(uuid::Uuid::new_v4(), content1.into(), engram_core::Sector::Semantic);
  memory1.content_hash = hash1.clone();
  memory1.simhash = simhash(content1);

  // Check exact duplicate - comparing against the same memory
  let exact_match = checker.is_duplicate(content1, &hash1, memory1.simhash, &memory1);
  assert!(
    matches!(exact_match, DuplicateMatch::Exact),
    "Should detect exact duplicate"
  );

  // Check similar content (slight modification)
  let content2 = "The user wants to implement a feature for handling authentication in their application";
  let hash2 = content_hash(content2);
  let simhash2 = simhash(content2);

  // Check against memory1 (different content, potentially similar simhash)
  let similar_match = checker.is_duplicate(content2, &hash2, simhash2, &memory1);

  // May or may not match depending on similarity threshold
  match similar_match {
    DuplicateMatch::Simhash { distance, jaccard } => {
      assert!((0.0..=1.0).contains(&jaccard), "Jaccard should be valid");
      assert!(distance <= 10, "Distance should be reasonable for similar content");
    }
    DuplicateMatch::Exact | DuplicateMatch::None => {
      // Either too different or exact match is fine
    }
  }

  // Check completely different content
  let content3 = "Debugging the database connection timeout issue in production environment";
  let hash3 = content_hash(content3);
  let simhash3 = simhash(content3);

  let different_match = checker.is_duplicate(content3, &hash3, simhash3, &memory1);
  assert!(
    matches!(different_match, DuplicateMatch::None),
    "Completely different content should not match"
  );
}

/// Test resilient provider retry configuration
#[tokio::test]
async fn test_retry_config_presets() {
  use embedding::RetryConfig;
  use std::time::Duration;

  // Test default config
  let default = RetryConfig::default();
  assert_eq!(default.max_retries, 3);
  assert_eq!(default.initial_backoff, Duration::from_secs(1));
  assert!(default.add_jitter);

  // Test local config (faster)
  let local = RetryConfig::for_local();
  assert_eq!(local.max_retries, 2);
  assert!(local.initial_backoff < Duration::from_secs(1));
  assert!(local.request_timeout < Duration::from_secs(60));

  // Test cloud config (more resilient)
  let cloud = RetryConfig::for_cloud();
  assert_eq!(cloud.max_retries, 5);
  assert!(cloud.max_backoff > Duration::from_secs(30));
  assert!(cloud.request_timeout > Duration::from_secs(60));

  // Test backoff calculation
  let backoff_config = RetryConfig {
    max_retries: 5,
    initial_backoff: Duration::from_secs(1),
    max_backoff: Duration::from_secs(60),
    backoff_multiplier: 2.0,
    add_jitter: false,
    request_timeout: Duration::from_secs(30),
  };

  assert_eq!(backoff_config.backoff_for_attempt(0), Duration::from_secs(1));
  assert_eq!(backoff_config.backoff_for_attempt(1), Duration::from_secs(2));
  assert_eq!(backoff_config.backoff_for_attempt(2), Duration::from_secs(4));
}

/// Test scheduler configuration
#[tokio::test]
async fn test_scheduler_configuration() {
  use daemon::SchedulerConfig;

  // Test default config
  let config = SchedulerConfig::default();
  assert_eq!(config.decay_interval_hours, 60);
  assert_eq!(config.session_cleanup_hours, 6);
  assert_eq!(config.max_session_age_hours, 6);
  assert_eq!(config.checkpoint_interval_secs, 30);

  // Test custom config
  let custom = SchedulerConfig {
    decay_interval_hours: 24,
    session_cleanup_hours: 1,
    max_session_age_hours: 2,
    checkpoint_interval_secs: 10,
    decay_batch_size: 1000,
  };
  assert_eq!(custom.decay_interval_hours, 24);
  assert_eq!(custom.session_cleanup_hours, 1);
  assert_eq!(custom.decay_batch_size, 1000);
}

/// Test accumulator extraction triggers
#[tokio::test]
async fn test_accumulator_extraction_triggers() {
  use db::SegmentAccumulator;
  use uuid::Uuid;

  let session_id = Uuid::new_v4();
  let project_id = Uuid::new_v4();

  let mut acc = SegmentAccumulator::new(session_id, project_id);

  // Initially no meaningful work
  assert!(!acc.has_meaningful_work());

  // Add user prompts (with full signature)
  acc.add_user_prompt("first prompt", None, true);
  acc.add_user_prompt("second prompt", Some("question".to_string()), true);

  // Still no meaningful work (just prompts)
  assert!(!acc.has_meaningful_work());

  // Test meaningful work detection via tool calls
  let mut acc2 = SegmentAccumulator::new(session_id, project_id);
  acc2.add_file_modified("src/main.rs");
  assert!(acc2.has_meaningful_work()); // Modified files count

  let mut acc3 = SegmentAccumulator::new(session_id, project_id);
  acc3.increment_tool_calls();
  acc3.increment_tool_calls();
  assert!(!acc3.has_meaningful_work()); // Only 2 tool calls

  acc3.increment_tool_calls();
  assert!(acc3.has_meaningful_work()); // Now 3

  // Test todo completion trigger
  let mut acc4 = SegmentAccumulator::new(session_id, project_id);
  assert!(!acc4.should_trigger_todo_extraction());

  acc4.add_completed_task("Task 1");
  acc4.add_completed_task("Task 2");
  acc4.add_completed_task("Task 3");
  assert!(!acc4.should_trigger_todo_extraction()); // Need 5+ tool calls too

  // Add tool calls
  for _ in 0..5 {
    acc4.increment_tool_calls();
  }
  assert!(acc4.should_trigger_todo_extraction()); // Now has 3 tasks and 5+ tool calls

  // Test reset
  acc4.reset();
  assert!(!acc4.has_meaningful_work());
  assert!(!acc4.should_trigger_todo_extraction());
}

/// Test health check endpoint
#[tokio::test]
async fn test_router_health_check() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  let health_request = Request {
    id: Some(serde_json::json!(1)),
    method: "health_check".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let health_response = router.handle(health_request).await;
  assert!(health_response.error.is_none(), "Health check should succeed");

  let health = health_response.result.unwrap();

  // Verify database section exists and reports status
  let database = health.get("database").expect("Should have database section");
  assert!(database.get("status").is_some(), "Should have database status");

  // Verify ollama section exists
  let ollama = health.get("ollama").expect("Should have ollama section");
  assert!(ollama.get("available").is_some(), "Should report ollama availability");
  assert!(
    ollama.get("configured_model").is_some(),
    "Should report configured model"
  );

  // Verify embedding section exists
  let embedding = health.get("embedding").expect("Should have embedding section");
  assert!(
    embedding.get("configured").is_some(),
    "Should report embedding configuration"
  );
}

/// Test project statistics
#[tokio::test]
async fn test_router_project_stats() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Add some memories with diverse content to avoid deduplication
  let memory_contents = [
    "The authentication system uses JWT tokens with RS256 algorithm for secure session management",
    "Database connections are pooled using pg-pool with a maximum of 10 concurrent connections",
    "The frontend uses React 18 with concurrent rendering features for better performance",
    "API rate limiting is implemented using Redis with a sliding window algorithm",
    "Logging is handled by Winston with structured JSON output for ELK stack integration",
  ];

  for (i, content) in memory_contents.iter().enumerate() {
    let add = Request {
      id: Some(serde_json::json!(i)),
      method: "memory_add".to_string(),
      params: serde_json::json!({
          "content": content,
          "cwd": cwd
      }),
    };
    router.handle(add).await;
  }

  // Get project stats
  let stats_request = Request {
    id: Some(serde_json::json!(100)),
    method: "project_stats".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let stats_response = router.handle(stats_request).await;
  assert!(stats_response.error.is_none(), "Stats request should succeed");

  let stats = stats_response.result.unwrap();

  // Verify memory stats
  let memories = stats.get("memories").expect("Should have memories section");
  let total = memories.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
  assert!(total >= 1, "Should have at least 1 memory, got {}", total);

  // Verify by_sector exists and has values
  let by_sector = memories
    .get("by_sector")
    .and_then(|v| v.as_object())
    .expect("Should have by_sector");
  assert!(by_sector.contains_key("semantic") || by_sector.contains_key("episodic"));

  // Verify by_tier exists
  let by_tier = memories
    .get("by_tier")
    .and_then(|v| v.as_object())
    .expect("Should have by_tier");
  assert!(by_tier.contains_key("session") || by_tier.contains_key("project"));

  // Verify salience distribution exists
  let by_salience = memories.get("by_salience").expect("Should have by_salience");
  assert!(by_salience.get("high").is_some());
  assert!(by_salience.get("medium").is_some());
  assert!(by_salience.get("low").is_some());
  assert!(by_salience.get("very_low").is_some());

  // Verify code stats section exists
  let code = stats.get("code").expect("Should have code section");
  assert!(code.get("total_chunks").is_some());
  assert!(code.get("total_files").is_some());
  assert!(code.get("by_language").is_some());

  // Verify documents stats section exists
  let documents = stats.get("documents").expect("Should have documents section");
  assert!(documents.get("total").is_some());
  assert!(documents.get("total_chunks").is_some());

  // Verify entities stats section exists
  let entities = stats.get("entities").expect("Should have entities section");
  assert!(entities.get("total").is_some());
  assert!(entities.get("by_type").is_some());
}

/// Test database migrations
#[tokio::test]
async fn test_database_migrations() {
  use db::{CURRENT_SCHEMA_VERSION, ProjectDb};
  use std::path::Path;

  let data_dir = TempDir::new().expect("Failed to create temp dir");
  let db_path = data_dir.path().join("test.lancedb");
  let project_id = engram_core::ProjectId::from_path(Path::new("/test/migrations"));

  let db = ProjectDb::open_at_path(project_id, db_path, 768).await.unwrap();

  // New database should need migration
  let needs = db.needs_migration().await.unwrap();
  assert!(needs, "New database should need migration");

  // Run migrations
  let applied = db.run_migrations().await.unwrap();
  assert!(!applied.is_empty(), "Should apply at least one migration");

  // Check version matches target
  let version = db.get_current_version().await.unwrap();
  assert_eq!(version, CURRENT_SCHEMA_VERSION);

  // Should not need migration now
  let needs_after = db.needs_migration().await.unwrap();
  assert!(!needs_after, "Should not need migration after running");

  // Migration history should be populated
  let history = db.get_migration_history().await.unwrap();
  assert!(!history.is_empty(), "Should have migration history");
  assert_eq!(history[0].version, 1);
  assert_eq!(history[0].name, "initial_schema");

  // Running again should be idempotent
  let applied_again = db.run_migrations().await.unwrap();
  assert!(applied_again.is_empty(), "Second run should apply nothing");
}

/// Test router relationship management tools
#[tokio::test]
async fn test_router_relationship_tools() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create two memories to relate
  let add_first = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "The authentication system uses JWT tokens for session management",
        "cwd": cwd
    }),
  };
  let first_response = router.handle(add_first).await;
  assert!(first_response.error.is_none(), "First memory creation should succeed");
  let first_id = first_response
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  let add_second = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "JWT tokens are stored in httpOnly cookies for security",
        "cwd": cwd
    }),
  };
  let second_response = router.handle(add_second).await;
  assert!(second_response.error.is_none(), "Second memory creation should succeed");
  let second_id = second_response
    .result
    .unwrap()
    .get("id")
    .unwrap()
    .as_str()
    .unwrap()
    .to_string();

  // Create a relationship between them
  let add_rel = Request {
    id: Some(serde_json::json!(3)),
    method: "relationship_add".to_string(),
    params: serde_json::json!({
        "from_memory_id": first_id,
        "to_memory_id": second_id,
        "relationship_type": "builds_on",
        "confidence": 0.85,
        "cwd": cwd
    }),
  };
  let rel_response = router.handle(add_rel).await;
  assert!(rel_response.error.is_none(), "Relationship creation should succeed");
  let rel_result = rel_response.result.unwrap();
  let rel_id = rel_result
    .get("id")
    .and_then(|v| v.as_str())
    .expect("Should return relationship ID");
  assert_eq!(
    rel_result.get("relationship_type").and_then(|v| v.as_str()),
    Some("builds_on")
  );
  let confidence = rel_result.get("confidence").and_then(|v| v.as_f64()).unwrap();
  assert!(
    (confidence - 0.85).abs() < 0.001,
    "Confidence should be approximately 0.85"
  );

  // List relationships for the first memory
  let list_rel = Request {
    id: Some(serde_json::json!(4)),
    method: "relationship_list".to_string(),
    params: serde_json::json!({
        "memory_id": first_id,
        "cwd": cwd
    }),
  };
  let list_response = router.handle(list_rel).await;
  assert!(list_response.error.is_none(), "Relationship listing should succeed");
  let list_result = list_response.result.unwrap();
  let relationships = list_result.as_array().unwrap();
  assert_eq!(relationships.len(), 1);
  assert_eq!(
    relationships[0].get("to_memory_id").and_then(|v| v.as_str()),
    Some(second_id.as_str())
  );

  // Get related memories with full context
  let get_related = Request {
    id: Some(serde_json::json!(5)),
    method: "relationship_related".to_string(),
    params: serde_json::json!({
        "memory_id": first_id,
        "cwd": cwd
    }),
  };
  let related_response = router.handle(get_related).await;
  assert!(
    related_response.error.is_none(),
    "Getting related memories should succeed"
  );
  let related_result = related_response.result.unwrap();
  let related_arr = related_result.as_array().unwrap();
  assert_eq!(related_arr.len(), 1);
  // Should have both relationship info and memory content
  assert!(related_arr[0].get("memory").is_some(), "Should include memory content");
  assert!(
    related_arr[0].get("relationship").is_some(),
    "Should include relationship info"
  );

  // Test filtering by relationship type
  let list_filtered = Request {
    id: Some(serde_json::json!(6)),
    method: "relationship_list".to_string(),
    params: serde_json::json!({
        "memory_id": first_id,
        "relationship_type": "contradicts",
        "cwd": cwd
    }),
  };
  let filtered_response = router.handle(list_filtered).await;
  assert!(filtered_response.error.is_none(), "Filtered listing should succeed");
  let filtered_result = filtered_response.result.unwrap();
  let filtered_rels = filtered_result.as_array().unwrap();
  assert_eq!(filtered_rels.len(), 0, "Should find no contradicts relationships");

  // Delete the relationship
  let delete_rel = Request {
    id: Some(serde_json::json!(7)),
    method: "relationship_delete".to_string(),
    params: serde_json::json!({
        "relationship_id": rel_id,
        "cwd": cwd
    }),
  };
  let delete_response = router.handle(delete_rel).await;
  assert!(delete_response.error.is_none(), "Relationship deletion should succeed");
  let delete_result = delete_response.result.unwrap();
  assert_eq!(delete_result.get("deleted").and_then(|v| v.as_bool()), Some(true));

  // Verify relationship was deleted
  let list_after_delete = Request {
    id: Some(serde_json::json!(8)),
    method: "relationship_list".to_string(),
    params: serde_json::json!({
        "memory_id": first_id,
        "cwd": cwd
    }),
  };
  let after_delete_response = router.handle(list_after_delete).await;
  let after_delete_result = after_delete_response.result.unwrap();
  let after_delete_rels = after_delete_result.as_array().unwrap();
  assert_eq!(after_delete_rels.len(), 0, "Relationship should be deleted");
}

/// Test memory_get and memory_list tools
#[tokio::test]
async fn test_router_memory_get_and_list() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Add multiple memories
  let mut memory_ids = Vec::new();
  for i in 0..3 {
    let add_request = Request {
      id: Some(serde_json::json!(i)),
      method: "memory_add".to_string(),
      params: serde_json::json!({
          "content": format!("Test memory #{} for get/list testing with unique content", i),
          "sector": if i % 2 == 0 { "semantic" } else { "procedural" },
          "cwd": cwd
      }),
    };
    let response = router.handle(add_request).await;
    assert!(
      response.error.is_none(),
      "memory_add should succeed: {:?}",
      response.error
    );
    let id = response
      .result
      .unwrap()
      .get("id")
      .unwrap()
      .as_str()
      .unwrap()
      .to_string();
    memory_ids.push(id);
  }

  // Test memory_get
  let get_request = Request {
    id: Some(serde_json::json!(10)),
    method: "memory_get".to_string(),
    params: serde_json::json!({
        "memory_id": memory_ids[0],
        "cwd": cwd
    }),
  };
  let get_response = router.handle(get_request).await;
  assert!(
    get_response.error.is_none(),
    "memory_get should succeed: {:?}",
    get_response.error
  );
  let memory = get_response.result.unwrap();
  assert!(memory.get("content").is_some(), "Should have content");
  assert!(memory.get("sector").is_some(), "Should have sector");
  assert!(memory.get("salience").is_some(), "Should have salience");
  assert!(memory.get("created_at").is_some(), "Should have created_at");

  // Test memory_list without filter
  let list_request = Request {
    id: Some(serde_json::json!(11)),
    method: "memory_list".to_string(),
    params: serde_json::json!({
        "cwd": cwd
    }),
  };
  let list_response = router.handle(list_request).await;
  assert!(
    list_response.error.is_none(),
    "memory_list should succeed: {:?}",
    list_response.error
  );
  let memories = list_response.result.unwrap();
  let memories_arr = memories.as_array().unwrap();
  // Note: memory_list returns memories - count may vary due to dedup or other factors
  assert!(!memories_arr.is_empty(), "Should have memories");

  // Verify memory structure
  let first_mem = &memories_arr[0];
  assert!(first_mem.get("id").is_some(), "Memory should have id");
  assert!(first_mem.get("content").is_some(), "Memory should have content");
  assert!(first_mem.get("sector").is_some(), "Memory should have sector");

  // Test memory_list with limit
  let list_limited = Request {
    id: Some(serde_json::json!(13)),
    method: "memory_list".to_string(),
    params: serde_json::json!({
        "limit": 2,
        "cwd": cwd
    }),
  };
  let limited_response = router.handle(list_limited).await;
  assert!(
    limited_response.error.is_none(),
    "memory_list with limit should succeed"
  );
  let limited = limited_response.result.unwrap();
  let limited_arr = limited.as_array().unwrap();
  assert!(limited_arr.len() <= 2, "Should respect limit");

  // Test memory_get with non-existent ID
  let get_missing = Request {
    id: Some(serde_json::json!(14)),
    method: "memory_get".to_string(),
    params: serde_json::json!({
        "memory_id": "00000000-0000-0000-0000-000000000000",
        "cwd": cwd
    }),
  };
  let missing_response = router.handle(get_missing).await;
  assert!(
    missing_response.error.is_some(),
    "memory_get should fail for non-existent ID"
  );
}

/// Test code_list and code_stats tools
#[tokio::test]
async fn test_router_code_list_and_stats() {
  let (_data_dir, project_dir, router) = create_test_router();
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
  let (_data_dir, project_dir, router) = create_test_router();
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
  let results = search_response.result.unwrap().as_array().unwrap().clone();
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

/// Test hook handler via router
/// Valid hooks: SessionStart, SessionEnd, UserPromptSubmit, PostToolUse, PreCompact, Stop, SubagentStop, Notification
#[tokio::test]
async fn test_router_hook_handler() {
  let (_data_dir, project_dir, router) = create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Test SessionStart hook (uses "event" field and nested "params")
  let session_start = Request {
    id: Some(serde_json::json!(1)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "SessionStart",
        "params": {
            "session_id": "test-session-123",
            "cwd": cwd
        }
    }),
  };
  let start_response = router.handle(session_start).await;
  assert!(
    start_response.error.is_none(),
    "SessionStart hook should succeed: {:?}",
    start_response.error
  );
  let start_result = start_response.result.unwrap();
  assert!(start_result.get("status").is_some(), "Should have status");

  // Test PostToolUse hook (for tool observation capture)
  let post_tool = Request {
    id: Some(serde_json::json!(2)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "PostToolUse",
        "params": {
            "session_id": "test-session-123",
            "tool_name": "Read",
            "tool_input": {"file_path": "/some/file.rs"},
            "tool_output": "fn main() { println!(\"Hello\"); }",
            "cwd": cwd
        }
    }),
  };
  let post_response = router.handle(post_tool).await;
  assert!(
    post_response.error.is_none(),
    "PostToolUse hook should succeed: {:?}",
    post_response.error
  );

  // Test UserPromptSubmit hook
  let user_prompt = Request {
    id: Some(serde_json::json!(3)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "UserPromptSubmit",
        "params": {
            "session_id": "test-session-123",
            "content": "Help me debug the authentication module",
            "cwd": cwd
        }
    }),
  };
  let prompt_response = router.handle(user_prompt).await;
  assert!(
    prompt_response.error.is_none(),
    "UserPromptSubmit hook should succeed: {:?}",
    prompt_response.error
  );

  // Test Stop hook (should trigger extraction)
  let stop_hook = Request {
    id: Some(serde_json::json!(4)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "Stop",
        "params": {
            "session_id": "test-session-123",
            "stop_reason": "end_turn",
            "cwd": cwd
        }
    }),
  };
  let stop_response = router.handle(stop_hook).await;
  assert!(
    stop_response.error.is_none(),
    "Stop hook should succeed: {:?}",
    stop_response.error
  );

  // Test PreCompact hook
  let pre_compact = Request {
    id: Some(serde_json::json!(5)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "PreCompact",
        "params": {
            "session_id": "test-session-123",
            "cwd": cwd
        }
    }),
  };
  let compact_response = router.handle(pre_compact).await;
  assert!(
    compact_response.error.is_none(),
    "PreCompact hook should succeed: {:?}",
    compact_response.error
  );

  // Test SessionEnd hook (should trigger promotion)
  let session_end = Request {
    id: Some(serde_json::json!(6)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "SessionEnd",
        "params": {
            "session_id": "test-session-123",
            "cwd": cwd
        }
    }),
  };
  let end_response = router.handle(session_end).await;
  assert!(
    end_response.error.is_none(),
    "SessionEnd hook should succeed: {:?}",
    end_response.error
  );

  // Test Notification hook
  let notification = Request {
    id: Some(serde_json::json!(7)),
    method: "hook".to_string(),
    params: serde_json::json!({
        "event": "Notification",
        "params": {
            "session_id": "test-session-123",
            "message": "Build completed successfully",
            "cwd": cwd
        }
    }),
  };
  let notif_response = router.handle(notification).await;
  assert!(
    notif_response.error.is_none(),
    "Notification hook should succeed: {:?}",
    notif_response.error
  );

  // Clean up - stop all watchers
  router.registry().stop_all_watchers().await;
}
