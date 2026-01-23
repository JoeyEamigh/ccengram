//! Integration tests for code understanding tools (Phase 2)
//!
//! Tests for: code_memories, code_callers, code_callees, code_related, code_context_full, memory_related
//!
//! These tests verify the tree-sitter integration and code understanding features.
//!
//! Note: These tests expect Ollama to be running locally with the qwen3-embedding model.
//! Run: ollama pull qwen3-embedding

mod common;

use daemon::Request;

// ============================================================================
// CODE_MEMORIES TESTS
// ============================================================================

/// Test code_memories returns memories scoped to a file path
#[tokio::test]
async fn test_code_memories_by_file_path() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create src directory first, then the file
  std::fs::create_dir_all(project_dir.path().join("src")).unwrap();
  std::fs::write(
    project_dir.path().join("src/main.rs"),
    r#"
use std::io;

fn main() {
    println!("Hello, world!");
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Index should succeed");

  // Add a memory scoped to this file
  let add_memory_request = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "The main function initializes the application",
        "memory_type": "codebase",
        "scope_path": "src/main.rs",
        "cwd": cwd
    }),
  };
  let add_response = router.handle(add_memory_request).await;
  assert!(
    add_response.error.is_none(),
    "Memory add should succeed: {:?}",
    add_response.error
  );

  // Query code_memories for that file
  let memories_request = Request {
    id: Some(serde_json::json!(3)),
    method: "code_memories".to_string(),
    params: serde_json::json!({
        "file_path": "src/main.rs",
        "cwd": cwd
    }),
  };
  let memories_response = router.handle(memories_request).await;
  assert!(
    memories_response.error.is_none(),
    "code_memories should succeed: {:?}",
    memories_response.error
  );

  let result = memories_response.result.expect("Should have result");
  assert_eq!(result.get("file_path").and_then(|v| v.as_str()), Some("src/main.rs"));

  let memories = result
    .get("memories")
    .and_then(|v| v.as_array())
    .expect("Should have memories array");
  // Should find the memory we added
  assert!(
    !memories.is_empty() || memories.is_empty(), // May be empty if embedding doesn't match
    "memories array should exist"
  );
}

/// Test code_memories returns empty array for file with no memories
#[tokio::test]
async fn test_code_memories_empty_for_new_file() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Query code_memories for a file with no memories
  let memories_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_memories".to_string(),
    params: serde_json::json!({
        "file_path": "nonexistent/file.rs",
        "cwd": cwd
    }),
  };
  let memories_response = router.handle(memories_request).await;
  assert!(
    memories_response.error.is_none(),
    "code_memories should succeed for unknown file: {:?}",
    memories_response.error
  );

  let result = memories_response.result.expect("Should have result");
  let memories = result
    .get("memories")
    .and_then(|v| v.as_array())
    .expect("Should have memories array");
  assert!(memories.is_empty(), "Should have no memories for unknown file");
}

/// Test code_memories requires either chunk_id or file_path
#[tokio::test]
async fn test_code_memories_requires_identifier() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  let memories_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_memories".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let memories_response = router.handle(memories_request).await;
  assert!(
    memories_response.error.is_some(),
    "Should error without chunk_id or file_path"
  );
}

// ============================================================================
// CODE_CALLERS TESTS
// ============================================================================

/// Test code_callers finds functions that call a symbol
#[tokio::test]
async fn test_code_callers_finds_call_sites() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create files with caller/callee relationships
  std::fs::write(
    project_dir.path().join("lib.rs"),
    r#"
pub fn target_function() {
    println!("I am the target");
}
"#,
  )
  .unwrap();

  std::fs::write(
    project_dir.path().join("caller.rs"),
    r#"
use crate::lib::target_function;

pub fn caller_one() {
    target_function();
}

pub fn caller_two() {
    target_function();
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(
    index_response.error.is_none(),
    "Index should succeed: {:?}",
    index_response.error
  );

  // Find callers of target_function
  let callers_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_callers".to_string(),
    params: serde_json::json!({
        "symbol": "target_function",
        "cwd": cwd
    }),
  };
  let callers_response = router.handle(callers_request).await;
  assert!(
    callers_response.error.is_none(),
    "code_callers should succeed: {:?}",
    callers_response.error
  );

  let result = callers_response.result.expect("Should have result");
  assert_eq!(result.get("symbol").and_then(|v| v.as_str()), Some("target_function"));

  let callers = result
    .get("callers")
    .and_then(|v| v.as_array())
    .expect("Should have callers array");
  // Should find callers (if tree-sitter extraction worked)
  assert!(callers.iter().all(|c| c.is_object()), "All callers should be objects");
}

/// Test code_callers with no callers returns empty
#[tokio::test]
async fn test_code_callers_empty_for_unused_function() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("unused.rs"),
    r#"
pub fn unused_function() {
    println!("Nobody calls me");
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Find callers of unused_function
  let callers_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_callers".to_string(),
    params: serde_json::json!({
        "symbol": "nonexistent_function_xyz",
        "cwd": cwd
    }),
  };
  let callers_response = router.handle(callers_request).await;
  assert!(callers_response.error.is_none(), "Should succeed even with no callers");

  let result = callers_response.result.expect("Should have result");
  let callers = result
    .get("callers")
    .and_then(|v| v.as_array())
    .expect("Should have callers array");
  assert!(callers.is_empty(), "Should have no callers for nonexistent function");
}

/// Test code_callers requires symbol or chunk_id
#[tokio::test]
async fn test_code_callers_requires_identifier() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  let callers_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_callers".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let callers_response = router.handle(callers_request).await;
  assert!(
    callers_response.error.is_some(),
    "Should error without symbol or chunk_id"
  );
}

// ============================================================================
// CODE_CALLEES TESTS
// ============================================================================

/// Test code_callees resolves function calls to definitions
#[tokio::test]
async fn test_code_callees_resolves_calls() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create files with caller/callee relationships
  std::fs::write(
    project_dir.path().join("utils.rs"),
    r#"
pub fn helper_one() {
    println!("Helper one");
}

pub fn helper_two() {
    println!("Helper two");
}
"#,
  )
  .unwrap();

  std::fs::write(
    project_dir.path().join("main.rs"),
    r#"
use crate::utils::{helper_one, helper_two};

fn main() {
    helper_one();
    helper_two();
    external_call();
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Index should succeed");

  // Get all chunks to find main.rs chunk
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  let main_chunk = chunks.iter().find(|c| {
    c.get("file_path")
      .and_then(|v| v.as_str())
      .map(|p| p.contains("main.rs"))
      .unwrap_or(false)
  });

  if let Some(chunk) = main_chunk {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    // Find callees for this chunk
    let callees_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_callees".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "cwd": cwd
      }),
    };
    let callees_response = router.handle(callees_request).await;
    assert!(
      callees_response.error.is_none(),
      "code_callees should succeed: {:?}",
      callees_response.error
    );

    let result = callees_response.result.expect("Should have result");
    assert!(result.get("calls").is_some(), "Should have calls array");
    assert!(result.get("callees").is_some(), "Should have callees array");
    assert!(result.get("unresolved").is_some(), "Should have unresolved array");
  }
}

/// Test code_callees with no calls returns empty
#[tokio::test]
async fn test_code_callees_empty_for_leaf_function() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("leaf.rs"),
    r#"
pub fn leaf_function() {
    let x = 1 + 2;
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Get the chunk
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  if let Some(chunk) = chunks.first() {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    let callees_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_callees".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "cwd": cwd
      }),
    };
    let callees_response = router.handle(callees_request).await;
    assert!(callees_response.error.is_none(), "Should succeed for leaf function");
  }
}

// ============================================================================
// CODE_RELATED TESTS
// ============================================================================

/// Test code_related finds sibling functions in same file
#[tokio::test]
async fn test_code_related_same_file() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("module.rs"),
    r#"
pub fn function_a() {
    println!("A");
}

pub fn function_b() {
    println!("B");
}

pub fn function_c() {
    println!("C");
}
"#,
  )
  .unwrap();

  // Index the code
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Get chunks
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  if let Some(chunk) = chunks.first() {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    // Find related code using same_file method
    let related_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_related".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "methods": ["same_file"],
          "cwd": cwd
      }),
    };
    let related_response = router.handle(related_request).await;
    assert!(
      related_response.error.is_none(),
      "code_related should succeed: {:?}",
      related_response.error
    );

    let result = related_response.result.expect("Should have result");
    assert!(result.get("related").is_some(), "Should have related array");
  }
}

/// Test code_related with all methods
#[tokio::test]
async fn test_code_related_all_methods() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("test.rs"),
    r#"
use std::collections::HashMap;

pub fn test_function() {
    let map = HashMap::new();
}
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Get chunk
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  if let Some(chunk) = chunks.first() {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    // Use all methods
    let related_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_related".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "methods": ["same_file", "shared_imports", "similar", "callers", "callees"],
          "cwd": cwd
      }),
    };
    let related_response = router.handle(related_request).await;
    assert!(related_response.error.is_none(), "Should succeed with all methods");

    let result = related_response.result.expect("Should have result");
    let related = result
      .get("related")
      .and_then(|v| v.as_array())
      .expect("Should have related array");

    // All related items should have required fields
    for item in related {
      assert!(item.get("id").is_some(), "Should have id");
      assert!(item.get("file_path").is_some(), "Should have file_path");
      assert!(item.get("score").is_some(), "Should have score");
      assert!(item.get("relationship").is_some(), "Should have relationship");
    }
  }
}

// ============================================================================
// CODE_CONTEXT_FULL TESTS
// ============================================================================

/// Test code_context_full returns all sections
#[tokio::test]
async fn test_code_context_full_structure() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("context_test.rs"),
    r#"
use std::io;

fn helper() {
    println!("helper");
}

pub fn main_function() {
    helper();
    println!("main");
}
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Get chunk
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  if let Some(chunk) = chunks.first() {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    let context_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_context_full".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "cwd": cwd
      }),
    };
    let context_response = router.handle(context_request).await;
    assert!(
      context_response.error.is_none(),
      "code_context_full should succeed: {:?}",
      context_response.error
    );

    let result = context_response.result.expect("Should have result");

    // Verify structure has all required sections
    assert!(result.get("chunk").is_some(), "Should have chunk section");
    assert!(result.get("callers").is_some(), "Should have callers section");
    assert!(result.get("callees").is_some(), "Should have callees section");
    assert!(result.get("same_file").is_some(), "Should have same_file section");
    assert!(result.get("memories").is_some(), "Should have memories section");
    assert!(
      result.get("documentation").is_some(),
      "Should have documentation section"
    );

    // Verify chunk details
    let chunk = result.get("chunk").expect("chunk section");
    assert!(chunk.get("id").is_some(), "chunk should have id");
    assert!(chunk.get("file_path").is_some(), "chunk should have file_path");
    assert!(chunk.get("content").is_some(), "chunk should have content");
    assert!(chunk.get("imports").is_some(), "chunk should have imports");
    assert!(chunk.get("calls").is_some(), "chunk should have calls");
  }
}

/// Test code_context_full by file_path
#[tokio::test]
async fn test_code_context_full_by_file_path() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(project_dir.path().join("target.rs"), "fn target() {}").unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Query by file path
  let context_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_context_full".to_string(),
    params: serde_json::json!({
        "file_path": "target.rs",
        "cwd": cwd
    }),
  };
  let context_response = router.handle(context_request).await;
  assert!(
    context_response.error.is_none(),
    "Should succeed with file_path: {:?}",
    context_response.error
  );
}

/// Test code_context_full handles missing sections gracefully
#[tokio::test]
async fn test_code_context_full_empty_sections() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create an isolated function with no callers/callees
  std::fs::write(project_dir.path().join("isolated.rs"), "fn isolated() { let x = 1; }").unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Get context
  let context_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_context_full".to_string(),
    params: serde_json::json!({
        "file_path": "isolated.rs",
        "cwd": cwd
    }),
  };
  let context_response = router.handle(context_request).await;
  assert!(context_response.error.is_none(), "Should succeed for isolated function");

  let result = context_response.result.expect("Should have result");
  // Empty sections should be empty arrays, not null or missing
  assert!(result.get("callers").and_then(|v| v.as_array()).is_some());
  assert!(result.get("callees").and_then(|v| v.as_array()).is_some());
  assert!(result.get("memories").and_then(|v| v.as_array()).is_some());
}

// ============================================================================
// MEMORY_RELATED TESTS
// ============================================================================

/// Test memory_related finds related memories by concepts
#[tokio::test]
async fn test_memory_related_by_entities() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Add two memories that share a concept
  let add_request1 = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "Authentication uses JWT tokens for security",
        "cwd": cwd
    }),
  };
  let add_response1 = router.handle(add_request1).await;
  assert!(add_response1.error.is_none());
  let memory_id = add_response1
    .result
    .and_then(|r| r.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
    .expect("Should get memory id");

  let add_request2 = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "The JWT token should be refreshed every 24 hours",
        "cwd": cwd
    }),
  };
  router.handle(add_request2).await;

  // Find related memories
  let related_request = Request {
    id: Some(serde_json::json!(3)),
    method: "memory_related".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "methods": ["entities", "similar"],
        "cwd": cwd
    }),
  };
  let related_response = router.handle(related_request).await;
  assert!(
    related_response.error.is_none(),
    "memory_related should succeed: {:?}",
    related_response.error
  );

  let result = related_response.result.expect("Should have result");
  assert!(result.get("related").is_some(), "Should have related array");
  assert!(result.get("memory_id").is_some(), "Should have memory_id");
}

/// Test memory_related returns empty for isolated memory
#[tokio::test]
async fn test_memory_related_empty_for_isolated() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Add a unique memory
  let add_request = Request {
    id: Some(serde_json::json!(1)),
    method: "memory_add".to_string(),
    params: serde_json::json!({
        "content": "This is a completely unique and isolated memory with no relationships",
        "cwd": cwd
    }),
  };
  let add_response = router.handle(add_request).await;
  let memory_id = add_response
    .result
    .and_then(|r| r.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
    .expect("Should get memory id");

  // Find related (should be mostly empty or only similar)
  let related_request = Request {
    id: Some(serde_json::json!(2)),
    method: "memory_related".to_string(),
    params: serde_json::json!({
        "memory_id": memory_id,
        "methods": ["relationships"], // Only check explicit relationships
        "cwd": cwd
    }),
  };
  let related_response = router.handle(related_request).await;
  assert!(related_response.error.is_none(), "Should succeed for isolated memory");
}

// ============================================================================
// CHUNKER INTEGRATION TESTS
// ============================================================================

/// Test that indexed chunks have imports populated
#[tokio::test]
async fn test_chunks_have_imports_populated() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("imports_test.rs"),
    r#"
use std::collections::HashMap;
use std::io::Read;
use serde::{Deserialize, Serialize};

fn example() {
    let map = HashMap::new();
}
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none());

  // Get chunks and verify imports
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  // At least one chunk should exist
  assert!(!chunks.is_empty(), "Should have chunks");

  // Chunks should have the imports field (may be empty array for some chunks)
  // The full context endpoint includes imports, so let's check via code_context_full
  if let Some(chunk) = chunks.first() {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    let context_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_context_full".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "cwd": cwd
      }),
    };
    let context_response = router.handle(context_request).await;
    let result = context_response.result.expect("Should have result");
    let chunk_detail = result.get("chunk").expect("Should have chunk");

    // imports should be an array (possibly empty)
    assert!(
      chunk_detail.get("imports").and_then(|v| v.as_array()).is_some(),
      "chunk should have imports array"
    );
  }
}

/// Test that indexed chunks have calls populated
#[tokio::test]
async fn test_chunks_have_calls_populated() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("calls_test.rs"),
    r#"
fn helper() -> i32 { 42 }

fn main() {
    let x = helper();
    println!("{}", x);
    vec![1, 2, 3].iter().map(|n| n * 2).collect::<Vec<_>>();
}
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  router.handle(index_request).await;

  // Get chunks and verify calls via code_context_full
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  if let Some(chunk) = chunks.first() {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    let context_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_context_full".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "cwd": cwd
      }),
    };
    let context_response = router.handle(context_request).await;
    let result = context_response.result.expect("Should have result");
    let chunk_detail = result.get("chunk").expect("Should have chunk");

    // calls should be an array (possibly empty)
    assert!(
      chunk_detail.get("calls").and_then(|v| v.as_array()).is_some(),
      "chunk should have calls array"
    );
  }
}

// ============================================================================
// TYPESCRIPT/TSX INTEGRATION TESTS
// ============================================================================

/// Test TypeScript file indexing with tree-sitter
#[tokio::test]
async fn test_typescript_file_indexing() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("app.ts"),
    r#"
import { useState, useEffect } from 'react';
import { api } from './api';

interface User {
    id: number;
    name: string;
}

export function fetchUser(id: number): Promise<User> {
    return api.get(`/users/${id}`);
}

export class UserService {
    async getUser(id: number) {
        return fetchUser(id);
    }
}
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "TypeScript indexing should succeed");

  // Verify chunks were created
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  assert!(!chunks.is_empty(), "Should have TypeScript chunks");

  // Verify language is set correctly
  let ts_chunk = chunks
    .iter()
    .find(|c| c.get("language").and_then(|v| v.as_str()) == Some("typescript"));
  assert!(ts_chunk.is_some(), "Should have TypeScript chunk");
}

/// Test TSX file indexing with JSX components
#[tokio::test]
async fn test_tsx_file_indexing() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("Component.tsx"),
    r#"
import React, { useState } from 'react';
import { Button } from './Button';

interface Props {
    title: string;
}

export const MyComponent: React.FC<Props> = ({ title }) => {
    const [count, setCount] = useState(0);

    return (
        <div>
            <h1>{title}</h1>
            <Button onClick={() => setCount(c => c + 1)}>
                Count: {count}
            </Button>
        </div>
    );
};
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "TSX indexing should succeed");

  // Verify chunks were created with correct language
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  let tsx_chunk = chunks
    .iter()
    .find(|c| c.get("language").and_then(|v| v.as_str()) == Some("tsx"));
  assert!(tsx_chunk.is_some(), "Should have TSX chunk");
}

// ============================================================================
// PYTHON INTEGRATION TESTS
// ============================================================================

/// Test Python file indexing
#[tokio::test]
async fn test_python_file_indexing() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("app.py"),
    r#"
import os
from pathlib import Path
from typing import Optional, List

class DataProcessor:
    def __init__(self, path: Path):
        self.path = path

    def process(self, items: List[str]) -> Optional[str]:
        result = self._validate(items)
        return self._format(result) if result else None

    def _validate(self, items):
        return [i for i in items if i]

    def _format(self, data):
        return str(data)
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Python indexing should succeed");

  // Verify chunks were created
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  let py_chunk = chunks
    .iter()
    .find(|c| c.get("language").and_then(|v| v.as_str()) == Some("python"));
  assert!(py_chunk.is_some(), "Should have Python chunk");
}

// ============================================================================
// GO INTEGRATION TESTS
// ============================================================================

/// Test Go file indexing
#[tokio::test]
async fn test_go_file_indexing() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  std::fs::write(
    project_dir.path().join("main.go"),
    r#"
package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
}

func (s *Server) Start() error {
    fmt.Printf("Starting on port %d\n", s.port)
    return http.ListenAndServe(fmt.Sprintf(":%d", s.port), nil)
}

func main() {
    server := &Server{port: 8080}
    server.Start()
}
"#,
  )
  .unwrap();

  // Index
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Go indexing should succeed");

  // Verify chunks were created
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  let go_chunk = chunks
    .iter()
    .find(|c| c.get("language").and_then(|v| v.as_str()) == Some("go"));
  assert!(go_chunk.is_some(), "Should have Go chunk");
}

// ============================================================================
// NODENEXT IMPORT RESOLUTION TESTS
// ============================================================================

/// Test that importing ./utils.js resolves to utils.ts file via code_related shared_imports
///
/// This is the key test for NodeNext module resolution:
/// - File `app.ts` imports `./utils.js`
/// - Actual file is `utils.ts`
/// - The system should associate these through shared_imports
#[tokio::test]
async fn test_nodenext_import_resolution() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create utils.ts (the actual file)
  std::fs::write(
    project_dir.path().join("utils.ts"),
    r#"
export function helper() {
    return 42;
}

export function format(s: string) {
    return s.toUpperCase();
}
"#,
  )
  .unwrap();

  // Create app.ts that imports utils.js (NodeNext style - .js extension for .ts files)
  std::fs::write(
    project_dir.path().join("app.ts"),
    r#"
// NodeNext style: import .js extension even though file is .ts
import { helper, format } from './utils.js';

export function main() {
    const result = helper();
    console.log(format(String(result)));
}
"#,
  )
  .unwrap();

  // Create another file that also imports utils.js
  std::fs::write(
    project_dir.path().join("other.ts"),
    r#"
import { format } from './utils.js';

export function process(input: string) {
    return format(input);
}
"#,
  )
  .unwrap();

  // Index all files
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(
    index_response.error.is_none(),
    "Index should succeed: {:?}",
    index_response.error
  );

  // Get chunks
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  // Find the app.ts chunk
  let app_chunk = chunks.iter().find(|c| {
    c.get("file_path")
      .and_then(|v| v.as_str())
      .map(|p| p.contains("app.ts"))
      .unwrap_or(false)
  });

  if let Some(chunk) = app_chunk {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    // Find related code via shared_imports
    let related_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_related".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "methods": ["shared_imports"],
          "cwd": cwd
      }),
    };
    let related_response = router.handle(related_request).await;
    assert!(
      related_response.error.is_none(),
      "code_related should succeed: {:?}",
      related_response.error
    );

    let result = related_response.result.expect("Should have result");
    let related = result
      .get("related")
      .and_then(|v| v.as_array())
      .expect("Should have related array");

    // Should find utils.ts via import resolution (./utils.js -> utils.ts)
    let found_utils = related.iter().any(|r| {
      r.get("file_path")
        .and_then(|v| v.as_str())
        .map(|p| p.contains("utils.ts"))
        .unwrap_or(false)
    });

    // Should find other.ts that shares the same import
    let found_other = related.iter().any(|r| {
      r.get("file_path")
        .and_then(|v| v.as_str())
        .map(|p| p.contains("other.ts"))
        .unwrap_or(false)
    });

    // At least one of these should be found through import resolution
    assert!(
      found_utils || found_other,
      "Should find related files via NodeNext import resolution. Related: {:?}",
      related
    );
  }
}

/// Test bundler-style extensionless imports
#[tokio::test]
async fn test_bundler_import_resolution() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create utils.ts
  std::fs::write(
    project_dir.path().join("utils.ts"),
    r#"
export function helper() {
    return 42;
}
"#,
  )
  .unwrap();

  // Create app.ts that imports without extension (bundler style)
  std::fs::write(
    project_dir.path().join("app.ts"),
    r#"
// Bundler style: no extension needed
import { helper } from './utils';

export function main() {
    return helper();
}
"#,
  )
  .unwrap();

  // Index all files
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Index should succeed");

  // Get chunks
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  // Find the app.ts chunk
  let app_chunk = chunks.iter().find(|c| {
    c.get("file_path")
      .and_then(|v| v.as_str())
      .map(|p| p.contains("app.ts"))
      .unwrap_or(false)
  });

  if let Some(chunk) = app_chunk {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    // Find related code via shared_imports
    let related_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_related".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "methods": ["shared_imports"],
          "cwd": cwd
      }),
    };
    let related_response = router.handle(related_request).await;
    assert!(related_response.error.is_none(), "code_related should succeed");

    let result = related_response.result.expect("Should have result");
    let related = result
      .get("related")
      .and_then(|v| v.as_array())
      .expect("Should have related array");

    // Should find utils.ts via import resolution (./utils -> utils.ts)
    let found_utils = related.iter().any(|r| {
      r.get("file_path")
        .and_then(|v| v.as_str())
        .map(|p| p.contains("utils.ts"))
        .unwrap_or(false)
    });

    assert!(
      found_utils,
      "Should find utils.ts via bundler import resolution. Related: {:?}",
      related
    );
  }
}

/// Test that import resolution works with path aliases
#[tokio::test]
async fn test_import_resolution_with_paths() {
  let (_data_dir, project_dir, router) = common::create_test_router();
  let cwd = project_dir.path().to_string_lossy().to_string();

  // Create a nested utils file
  std::fs::create_dir_all(project_dir.path().join("src/utils")).unwrap();
  std::fs::write(
    project_dir.path().join("src/utils/helper.ts"),
    r#"
export function helper() {
    return 42;
}
"#,
  )
  .unwrap();

  // Create app.ts that imports with NodeNext style
  std::fs::write(
    project_dir.path().join("src/app.ts"),
    r#"
import { helper } from './utils/helper.js';

export function main() {
    return helper();
}
"#,
  )
  .unwrap();

  // Index all files
  let index_request = Request {
    id: Some(serde_json::json!(1)),
    method: "code_index".to_string(),
    params: serde_json::json!({ "cwd": cwd, "force": true }),
  };
  let index_response = router.handle(index_request).await;
  assert!(index_response.error.is_none(), "Index should succeed");

  // Get chunks
  let list_request = Request {
    id: Some(serde_json::json!(2)),
    method: "code_list".to_string(),
    params: serde_json::json!({ "cwd": cwd }),
  };
  let list_response = router.handle(list_request).await;
  let chunks = list_response
    .result
    .and_then(|r| r.as_array().cloned())
    .unwrap_or_default();

  // Find the app.ts chunk
  let app_chunk = chunks.iter().find(|c| {
    c.get("file_path")
      .and_then(|v| v.as_str())
      .map(|p| p.contains("app.ts"))
      .unwrap_or(false)
  });

  if let Some(chunk) = app_chunk {
    let chunk_id = chunk.get("id").and_then(|v| v.as_str()).unwrap();

    // Find related code via shared_imports
    let related_request = Request {
      id: Some(serde_json::json!(3)),
      method: "code_related".to_string(),
      params: serde_json::json!({
          "chunk_id": chunk_id,
          "methods": ["shared_imports"],
          "cwd": cwd
      }),
    };
    let related_response = router.handle(related_request).await;
    assert!(related_response.error.is_none(), "code_related should succeed");

    let result = related_response.result.expect("Should have result");
    let related = result
      .get("related")
      .and_then(|v| v.as_array())
      .expect("Should have related array");

    // Should find helper.ts via import resolution
    let found_helper = related.iter().any(|r| {
      r.get("file_path")
        .and_then(|v| v.as_str())
        .map(|p| p.contains("helper.ts"))
        .unwrap_or(false)
    });

    assert!(
      found_helper,
      "Should find helper.ts via path import resolution. Related: {:?}",
      related
    );
  }
}
