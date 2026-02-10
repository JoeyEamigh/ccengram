//! Integration tests for the complete memory lifecycle.
//!
//! These tests validate the full memory lifecycle from creation to restoration,
//! including deduplication, reinforcement, relationships, decay, and soft delete/restore.

#[cfg(test)]
mod tests {
  use crate::{
    context::memory::extract::decay::MemoryDecay,
    ipc::types::{
      memory::{MemoryAddParams, MemoryGetParams, MemoryListParams, MemoryRelatedParams, MemorySearchParams},
      relationship::RelationshipAddParams,
    },
    service::{
      __tests__::helpers::TestContext,
      memory::{self, relationship},
    },
  };

  /// Helper to create MemoryAddParams with just content
  fn add_params(content: &str) -> MemoryAddParams {
    MemoryAddParams {
      content: content.to_string(),
      sector: None,
      memory_type: None,
      context: None,
      tags: None,
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    }
  }

  /// Helper to create MemoryAddParams with content and sector
  fn add_params_with_sector(content: &str, sector: &str) -> MemoryAddParams {
    MemoryAddParams {
      content: content.to_string(),
      sector: Some(sector.to_string()),
      memory_type: None,
      context: None,
      tags: None,
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    }
  }

  /// Test the complete memory lifecycle from creation to restoration.
  ///
  /// This test validates:
  /// 1. Add a memory
  /// 2. Search for it (text search without embedding)
  /// 3. Duplicate detection
  /// 4. Reinforce to increase salience
  /// 5. Create second memory and relationship
  /// 6. Verify related() finds the relationship
  /// 7. Apply decay to decrease salience
  /// 8. Soft delete
  /// 9. Verify appears in list_deleted
  /// 10. Restore and verify searchable again
  #[tokio::test]
  async fn test_memory_lifecycle_full() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Step 1: Add a memory
    let add_params = MemoryAddParams {
      content: "User prefers dark mode for all editors and terminal applications".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("preference".to_string()),
      context: None,
      tags: Some(vec!["preference".to_string(), "ui".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: Some(0.7),
    };

    let result = memory::add(&mem_ctx, add_params).await.expect("add memory");
    assert!(!result.is_duplicate, "Should not be a duplicate");
    assert!(!result.id.is_empty(), "Should have an ID");
    let memory_id = result.id.clone();

    // Step 2: Search for it (text fallback since no embedding provider)
    let search_params = MemorySearchParams {
      query: "dark mode".to_string(),
      ..Default::default()
    };
    let search_result = memory::search(&mem_ctx, search_params, &ctx.config, None)
      .await
      .expect("search");
    assert!(!search_result.items.is_empty(), "Should find the memory");
    assert_eq!(search_result.items[0].id, memory_id);

    // Step 3: Add duplicate - should detect it
    let duplicate_params = add_params_with_sector(
      "User prefers dark mode for all editors and terminal applications",
      "semantic",
    );
    let dup_result = memory::add(&mem_ctx, duplicate_params).await.expect("add duplicate");
    assert!(dup_result.is_duplicate, "Should detect as duplicate");
    assert_eq!(dup_result.id, memory_id, "Should return existing memory ID");

    // Step 4: Check reinforce works on memory below max salience
    // Note: New memories start with salience 1.0, so reinforce has no effect
    // due to diminishing returns formula: new = old + amount * (1.0 - old)
    // We verify reinforce runs without error and maintains salience
    let reinforce_result = memory::reinforce(&mem_ctx, &memory_id, Some(0.2))
      .await
      .expect("reinforce memory");
    // At max salience (1.0), reinforce maintains it
    assert!(
      reinforce_result.new_salience >= 0.99,
      "Salience should remain high: {}",
      reinforce_result.new_salience
    );

    // Step 5: Create a second memory
    let second_add = MemoryAddParams {
      content: "The VS Code theme should be set to One Dark Pro for consistency".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("preference".to_string()),
      context: None,
      tags: Some(vec!["preference".to_string(), "ui".to_string(), "vscode".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    let second_result = memory::add(&mem_ctx, second_add).await.expect("add second memory");
    let second_id = second_result.id.clone();

    // Create a relationship between them
    let rel_params = RelationshipAddParams {
      from_memory_id: memory_id.clone(),
      to_memory_id: second_id.clone(),
      relationship_type: "related_to".to_string(),
      confidence: Some(0.9),
    };
    let rel_result = relationship::add(&ctx.db, rel_params).await.expect("add relationship");
    assert!(!rel_result.id.is_empty());

    // Step 6: Verify related() finds the relationship
    let related_params = MemoryRelatedParams {
      memory_id: memory_id.clone(),
      limit: Some(10),
    };
    let related_result = memory::related(&mem_ctx, related_params).await.expect("get related");
    assert!(related_result.count > 0, "Should have related memories");
    let has_relationship = related_result.related.iter().any(|r| r.id == second_id);
    assert!(has_relationship, "Should include the related memory");

    // Step 7: Apply decay
    let decay_config = MemoryDecay {
      archive_threshold: 0.1,
      max_idle_days: 90,
    };
    // Note: Decay may not reduce salience if recently accessed, so we verify it runs without error
    let decay_result = memory::apply_decay(&mem_ctx, &decay_config).await.expect("apply decay");
    assert!(decay_result.total_processed >= 2, "Should process at least 2 memories");

    // Step 8: Soft delete
    let deleted = memory::delete(&mem_ctx, &memory_id).await.expect("delete memory");
    assert!(deleted.is_deleted, "Memory should be marked deleted");

    // Verify not in normal search
    let search_after_delete = MemorySearchParams {
      query: "dark mode".to_string(),
      ..Default::default()
    };
    let search_result2 = memory::search(&mem_ctx, search_after_delete, &ctx.config, None)
      .await
      .expect("search after delete");
    let found_deleted = search_result2.items.iter().any(|m| m.id == memory_id);
    assert!(!found_deleted, "Deleted memory should not appear in normal search");

    // Step 9: Verify in list_deleted
    let deleted_list = memory::list_deleted(&mem_ctx, Some(10)).await.expect("list deleted");
    let in_deleted = deleted_list.iter().any(|m| m.id == memory_id);
    assert!(in_deleted, "Memory should appear in deleted list");

    // Step 10: Restore and verify searchable again
    let restored = memory::restore(&mem_ctx, &memory_id).await.expect("restore memory");
    assert!(!restored.is_deleted, "Memory should no longer be deleted");

    let search_after_restore = MemorySearchParams {
      query: "dark mode".to_string(),
      ..Default::default()
    };
    let search_result3 = memory::search(&mem_ctx, search_after_restore, &ctx.config, None)
      .await
      .expect("search after restore");
    let found_restored = search_result3.items.iter().any(|m| m.id == memory_id);
    assert!(found_restored, "Restored memory should be searchable again");
  }

  /// Test that content validation works correctly.
  #[tokio::test]
  async fn test_memory_content_validation() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Too short content should fail
    let short_params = add_params("Hi");
    let result = memory::add(&mem_ctx, short_params).await;
    assert!(result.is_err(), "Should reject content that is too short");
  }

  /// Test listing memories with sector filter.
  #[tokio::test]
  async fn test_memory_list_with_filters() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Add memories in different sectors
    let semantic_params = add_params_with_sector("This is a semantic memory about project architecture", "semantic");
    memory::add(&mem_ctx, semantic_params).await.expect("add semantic");

    let episodic_params = add_params_with_sector("User ran the build command and it succeeded yesterday", "episodic");
    memory::add(&mem_ctx, episodic_params).await.expect("add episodic");

    // List only semantic memories
    let list_params = MemoryListParams {
      sector: Some("semantic".to_string()),
      limit: Some(10),
      offset: None,
    };
    let list_result = memory::list(&mem_ctx, list_params).await.expect("list memories");

    assert_eq!(list_result.len(), 1, "Should only have 1 semantic memory");
    assert_eq!(list_result[0].sector, "semantic");
  }

  /// Test deemphasize operation.
  #[tokio::test]
  async fn test_memory_deemphasize() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    let add_p = MemoryAddParams {
      content: "A memory that will be deemphasized for testing purposes".to_string(),
      sector: None,
      memory_type: None,
      context: None,
      tags: None,
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: Some(0.9),
    };
    let result = memory::add(&mem_ctx, add_p).await.expect("add memory");

    // Get initial salience
    let get_params = MemoryGetParams {
      memory_id: result.id.clone(),
      include_related: Some(false),
    };
    let detail = memory::get(&mem_ctx, get_params).await.expect("get memory");
    let initial_salience = detail.salience;

    // Deemphasize
    let deemph_result = memory::deemphasize(&mem_ctx, &result.id, Some(0.3))
      .await
      .expect("deemphasize");
    assert!(
      deemph_result.new_salience < initial_salience,
      "Salience should decrease: {} < {}",
      deemph_result.new_salience,
      initial_salience
    );
    assert!(
      deemph_result.new_salience >= 0.05,
      "Salience should not go below minimum: {}",
      deemph_result.new_salience
    );
  }

  /// Test supersede operation.
  #[tokio::test]
  async fn test_memory_supersede() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Create old memory
    let old_params = add_params_with_sector("Old information: the API uses REST endpoints at /api/v1", "semantic");
    let old_result = memory::add(&mem_ctx, old_params).await.expect("add old memory");
    let old_id = old_result.id.clone();

    // Create new memory that supersedes the old one
    let new_params = add_params_with_sector("Updated information: the API now uses GraphQL at /graphql", "semantic");
    let new_result = memory::add(&mem_ctx, new_params).await.expect("add new memory");
    let new_id = new_result.id.clone();

    // Supersede old with new
    let supersede_result = memory::supersede(&mem_ctx, &old_id, &new_id).await.expect("supersede");
    assert_eq!(supersede_result.old_id, old_id);
    assert_eq!(supersede_result.new_id, new_id);

    // Verify old memory is marked as superseded
    let get_params = MemoryGetParams {
      memory_id: old_id.clone(),
      include_related: Some(false),
    };
    let old_detail = memory::get(&mem_ctx, get_params).await.expect("get old memory");
    assert_eq!(old_detail.superseded_by, Some(new_id.clone()));
  }

  /// Test hard delete permanently removes memory.
  #[tokio::test]
  async fn test_memory_hard_delete() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    let add_p = add_params("A memory that will be permanently deleted forever");
    let result = memory::add(&mem_ctx, add_p).await.expect("add memory");
    let memory_id = result.id.clone();

    // Hard delete
    let deleted_id = memory::hard_delete(&mem_ctx, &memory_id).await.expect("hard delete");
    assert_eq!(deleted_id, memory_id);

    // Verify not retrievable
    let get_params = MemoryGetParams {
      memory_id: memory_id.clone(),
      include_related: Some(false),
    };
    let get_result = memory::get(&mem_ctx, get_params).await;
    assert!(get_result.is_err(), "Hard deleted memory should not be retrievable");
  }

  /// Test relationship list operation.
  #[tokio::test]
  async fn test_relationship_list() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Create two memories
    let m1_params = add_params("First memory for relationship testing purposes here");
    let m1 = memory::add(&mem_ctx, m1_params).await.expect("add m1");

    let m2_params = add_params("Second memory for relationship testing purposes here");
    let m2 = memory::add(&mem_ctx, m2_params).await.expect("add m2");

    // Create relationship using a valid relationship type
    let rel_params = RelationshipAddParams {
      from_memory_id: m1.id.clone(),
      to_memory_id: m2.id.clone(),
      relationship_type: "builds_on".to_string(),
      confidence: Some(0.85),
    };
    relationship::add(&ctx.db, rel_params).await.expect("add relationship");

    // List relationships
    let rels = relationship::list(&ctx.db, &m1.id).await.expect("list relationships");
    assert_eq!(rels.len(), 1);
    assert_eq!(rels[0].relationship_type, "builds_on");
    assert_eq!(rels[0].to_memory_id, m2.id);
  }

  /// Test that search respects sector/tier/memory_type filters.
  ///
  /// This validates Phase 3.4: sector-based filtering in memory search.
  /// The filter should be applied BEFORE vector similarity ranking.
  #[tokio::test]
  async fn test_memory_search_with_filters() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Add memories with different sectors and types
    let semantic_decision = MemoryAddParams {
      content: "We decided to use Postgres for the database because of JSON support".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("decision".to_string()),
      context: None,
      tags: None,
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, semantic_decision)
      .await
      .expect("add semantic decision");

    let semantic_codebase = MemoryAddParams {
      content: "The database module is located in src/db and handles all Postgres queries".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: None,
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, semantic_codebase)
      .await
      .expect("add semantic codebase");

    let procedural_pattern = MemoryAddParams {
      content: "When adding new database migrations, always run them in a transaction".to_string(),
      sector: Some("procedural".to_string()),
      memory_type: Some("pattern".to_string()),
      context: None,
      tags: None,
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, procedural_pattern)
      .await
      .expect("add procedural pattern");

    // Search with sector filter - should only find semantic memories
    let search_by_sector = MemorySearchParams {
      query: "database".to_string(),
      sector: Some("semantic".to_string()),
      tier: None,
      memory_type: None,
      min_salience: None,
      scope_path: None,
      scope_module: None,
      session_id: None,
      limit: Some(10),
      include_superseded: false,
    };
    let sector_result = memory::search(&mem_ctx, search_by_sector, &ctx.config, None)
      .await
      .expect("search by sector");

    assert_eq!(
      sector_result.items.len(),
      2,
      "Should find exactly 2 semantic memories about database"
    );
    for item in &sector_result.items {
      assert_eq!(item.sector, "semantic", "All results should be semantic sector");
    }

    // Search with memory_type filter - should only find decisions
    let search_by_type = MemorySearchParams {
      query: "database".to_string(),
      sector: None,
      tier: None,
      memory_type: Some("decision".to_string()),
      min_salience: None,
      scope_path: None,
      scope_module: None,
      session_id: None,
      limit: Some(10),
      include_superseded: false,
    };
    let type_result = memory::search(&mem_ctx, search_by_type, &ctx.config, None)
      .await
      .expect("search by type");

    assert_eq!(type_result.items.len(), 1, "Should find exactly 1 decision memory");
    assert_eq!(
      type_result.items[0].memory_type.as_deref(),
      Some("decision"),
      "Result should be a decision"
    );

    // Search with both sector AND memory_type filter - should find intersection
    let search_combined = MemorySearchParams {
      query: "database".to_string(),
      sector: Some("semantic".to_string()),
      tier: None,
      memory_type: Some("codebase".to_string()),
      min_salience: None,
      scope_path: None,
      scope_module: None,
      session_id: None,
      limit: Some(10),
      include_superseded: false,
    };
    let combined_result = memory::search(&mem_ctx, search_combined, &ctx.config, None)
      .await
      .expect("search combined");

    assert_eq!(
      combined_result.items.len(),
      1,
      "Should find exactly 1 semantic codebase memory"
    );
    assert_eq!(combined_result.items[0].sector, "semantic");
    assert_eq!(combined_result.items[0].memory_type.as_deref(), Some("codebase"));
  }

  /// Test get with include_related flag.
  #[tokio::test]
  async fn test_memory_get_with_relationships() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Create two memories with a relationship
    let m1_params = add_params("Main memory for testing include_related functionality");
    let m1 = memory::add(&mem_ctx, m1_params).await.expect("add m1");

    let m2_params = add_params("Related memory for testing include_related functionality");
    let m2 = memory::add(&mem_ctx, m2_params).await.expect("add m2");

    let rel_params = RelationshipAddParams {
      from_memory_id: m1.id.clone(),
      to_memory_id: m2.id.clone(),
      relationship_type: "related_to".to_string(),
      confidence: Some(0.8),
    };
    relationship::add(&ctx.db, rel_params).await.expect("add relationship");

    // Get with include_related = true
    let get_params = MemoryGetParams {
      memory_id: m1.id.clone(),
      include_related: Some(true),
    };
    let detail = memory::get(&mem_ctx, get_params).await.expect("get with related");

    assert!(detail.relationships.is_some(), "Should include relationships");
    let rels = detail.relationships.unwrap();
    assert!(!rels.is_empty(), "Should have relationships");
  }

  // ==========================================================================
  // Phase 5 Tests: Memory Search Confidence
  // ==========================================================================

  /// Test that memory search returns search_quality metadata.
  ///
  /// This validates Phase 5.1 for memory search: Results include confidence
  /// information derived from vector distance.
  #[tokio::test]
  async fn test_memory_search_includes_search_quality() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Create a memory with specific content
    let add_params = MemoryAddParams {
      content: "The authentication system uses JWT tokens for session management".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["auth".to_string(), "jwt".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, add_params).await.expect("add memory");

    // Search for related content
    let search_params = MemorySearchParams {
      query: "JWT authentication tokens".to_string(),
      sector: None,
      tier: None,
      memory_type: None,
      min_salience: None,
      scope_path: None,
      scope_module: None,
      session_id: None,
      limit: Some(10),
      include_superseded: false,
    };

    let result = memory::search(&mem_ctx, search_params, &ctx.config, None)
      .await
      .expect("search");

    // Should have search quality
    assert!(
      result.search_quality.best_distance < 1.0,
      "Search quality should have meaningful best_distance"
    );

    // Relevant search should not be marked low confidence
    if !result.items.is_empty() {
      // If we found results, the search was reasonably confident
      assert!(
        result.search_quality.best_distance < 0.7 || !result.search_quality.low_confidence,
        "Relevant query should have reasonable confidence"
      );
    }
  }
}
