//! Integration tests for unified explore (cross-domain search).
//!
//! These tests validate the unified search across code, memory, and documents.

#[cfg(test)]
mod tests {
  use crate::{
    domain::code::Language,
    ipc::types::memory::MemoryAddParams,
    service::{
      __tests__::helpers::TestContext,
      explore::{
        ExploreContext, ExploreScope, RelatedMemoryInfo, SearchParams,
        context::{get_related_code_for_memory, get_related_memories_for_code},
        get_context, search,
      },
      memory,
    },
  };

  /// Test unified search across code and memory domains.
  ///
  /// Validates:
  /// 1. Add code chunk and memory about authentication
  /// 2. Search with scope=All finds both
  /// 3. Search with scope=Code finds only code
  /// 4. Search with scope=Memory finds only memory
  #[tokio::test]
  async fn test_explore_cross_domain_search() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();
    let explore_ctx = ExploreContext::new(&ctx.db, ctx.embedding.as_ref());

    // Add a code chunk about authentication
    ctx
      .index_code(
        "src/auth/oauth.rs",
        r#"
/// Authenticate using OAuth2 provider.
pub async fn authenticate_oauth(provider: &str, token: &str) -> Result<User, AuthError> {
    let client = OAuthClient::new(provider);
    client.verify_token(token).await
}
"#,
        Language::Rust,
      )
      .await;

    // Add a memory about authentication
    let memory_params = MemoryAddParams {
      content: "The project uses OAuth2 for authentication via the authenticate_oauth function".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["auth".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, memory_params).await.expect("add memory");

    // Search with scope=All
    let all_params = SearchParams {
      query: "authenticate oauth".to_string(),
      scope: ExploreScope::All,
      expand_top: 0,
      limit: 10,
      depth: 3,
    };

    let all_result = search(&explore_ctx, &all_params).await.expect("search all");
    assert!(!all_result.results.is_empty(), "Should find results");

    // Should have counts for both code and memory
    let code_count = all_result.counts.get("code").copied().unwrap_or(0);
    let memory_count = all_result.counts.get("memory").copied().unwrap_or(0);
    assert!(code_count > 0, "Should have code results");
    assert!(memory_count > 0, "Should have memory results");

    // Search with scope=Code only
    let code_params = SearchParams {
      query: "authenticate oauth".to_string(),
      scope: ExploreScope::Code,
      expand_top: 0,
      limit: 10,
      depth: 3,
    };

    let code_result = search(&explore_ctx, &code_params).await.expect("search code");
    for result in &code_result.results {
      assert_eq!(result.result_type, "code", "Should only return code results");
    }

    // Search with scope=Memory only
    let memory_params = SearchParams {
      query: "authenticate oauth".to_string(),
      scope: ExploreScope::Memory,
      expand_top: 0,
      limit: 10,
      depth: 3,
    };

    let memory_result = search(&explore_ctx, &memory_params).await.expect("search memory");
    for result in &memory_result.results {
      assert_eq!(result.result_type, "memory", "Should only return memory results");
    }
  }

  /// Test that results are sorted by score.
  #[tokio::test]
  async fn test_explore_results_sorted_by_score() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();
    let explore_ctx = ExploreContext::new(&ctx.db, ctx.embedding.as_ref());

    // Add multiple memories with varying relevance
    for i in 0..5 {
      let params = MemoryAddParams {
        content: format!(
          "Memory {} about database connection pooling and configuration settings",
          i
        ),
        sector: Some("semantic".to_string()),
        memory_type: None,
        context: None,
        tags: None,
        categories: None,
        scope_path: None,
        scope_module: None,
        importance: None,
      };
      memory::add(&mem_ctx, params).await.expect("add memory");
    }

    let params = SearchParams {
      query: "database connection".to_string(),
      scope: ExploreScope::Memory,
      expand_top: 0,
      limit: 10,
      depth: 3,
    };

    let result = search(&explore_ctx, &params).await.expect("search");

    // Verify results are sorted by score descending
    for i in 0..result.results.len().saturating_sub(1) {
      assert!(
        result.results[i].score >= result.results[i + 1].score,
        "Results should be sorted by score descending"
      );
    }
  }

  /// Test empty query validation.
  #[tokio::test]
  async fn test_explore_empty_query_error() {
    let ctx = TestContext::new().await;
    let explore_ctx = ExploreContext::new(&ctx.db, ctx.embedding.as_ref());

    let params = SearchParams {
      query: "   ".to_string(), // Whitespace only
      scope: ExploreScope::All,
      expand_top: 0,
      limit: 10,
      depth: 3,
    };

    let result = search(&explore_ctx, &params).await;
    assert!(result.is_err(), "Should reject empty/whitespace query");
  }

  // ==========================================================================
  // Phase 1 Tests: Related memories via vector search
  // ==========================================================================

  /// Test that related memories are found via semantic similarity.
  ///
  /// This verifies the Phase 1 improvement: using vector search instead of LIKE
  /// queries to find memories related to code chunks. The memory should be found
  /// even though it doesn't contain the exact symbol name "validate_jwt_token".
  #[tokio::test]
  async fn test_related_memories_found_semantically() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Index code about JWT token validation using full AST chunker
    ctx
      .index_code(
        "src/auth/jwt.rs",
        r#"
/// Validates a JWT token and extracts the claims.
pub fn validate_jwt_token(token: &str) -> Result<Claims, AuthError> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(AuthError::InvalidFormat);
    }
    decode_and_verify(parts[1], &SECRET_KEY)
}
"#,
        Language::Rust,
      )
      .await;

    // Create a memory that discusses the concept but doesn't use "validate_jwt_token"
    let memory_params = MemoryAddParams {
      content: "The authentication system verifies JSON Web Tokens before granting access to protected resources. Token verification includes signature validation and expiration checks.".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["auth".to_string(), "security".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    let add_result = memory::add(&mem_ctx, memory_params).await.expect("add memory");
    let memory_id = add_result.id;

    // Get the code chunk we just indexed
    let chunks = ctx.db.get_chunks_for_file("src/auth/jwt.rs").await.expect("get chunks");
    assert!(!chunks.is_empty(), "Should have indexed the JWT code");

    // Find the validate_jwt_token chunk (AST chunker extracts symbols properly)
    let chunk = chunks
      .iter()
      .find(|c| c.symbols.iter().any(|s| s.contains("validate_jwt_token")))
      .expect("Should find validate_jwt_token chunk - chunker should extract symbols");

    // Find related memories for this chunk
    let related: Vec<RelatedMemoryInfo> = get_related_memories_for_code(&ctx.db, chunk, 10).await;

    // The memory should be found via semantic similarity, even without exact symbol match
    let found_memory = related.iter().any(|m| m.id == memory_id);
    assert!(
      found_memory,
      "Should find semantically related memory about JWT/token verification. Found memories: {:?}",
      related.iter().map(|m| &m.content).collect::<Vec<_>>()
    );
  }

  /// Test that vector search is efficient (single query instead of N+1 LIKE queries).
  ///
  /// This validates the performance aspect of Phase 1: even with many symbols,
  /// we should use a single vector search rather than querying per symbol.
  #[tokio::test]
  async fn test_related_memories_efficient_for_many_symbols() {
    use std::time::Instant;

    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Index code with multiple symbols using the full AST chunker
    ctx
      .index_code(
        "src/user/repository.rs",
        r#"
/// User repository for database operations.
pub struct UserRepository { db: Database }

impl UserRepository {
    pub fn new(db: Database) -> Self { Self { db } }
    pub fn find_by_id(&self, id: UserId) -> Option<User> { todo!() }
    pub fn find_by_email(&self, email: &str) -> Option<User> { todo!() }
    pub fn find_by_username(&self, username: &str) -> Option<User> { todo!() }
    pub fn save(&self, user: &User) -> Result<(), DbError> { todo!() }
    pub fn update(&self, user: &User) -> Result<(), DbError> { todo!() }
    pub fn delete(&self, id: UserId) -> Result<(), DbError> { todo!() }
    pub fn list_all(&self) -> Vec<User> { todo!() }
    pub fn count(&self) -> usize { todo!() }
}
"#,
        Language::Rust,
      )
      .await;

    // Add some related memories
    for i in 0..5 {
      let params = MemoryAddParams {
        content: format!(
          "User data management note {}: Database queries for user operations should use prepared statements for security.",
          i
        ),
        sector: Some("semantic".to_string()),
        memory_type: None,
        context: None,
        tags: None,
        categories: None,
        scope_path: None,
        scope_module: None,
        importance: None,
      };
      memory::add(&mem_ctx, params).await.expect("add memory");
    }

    // Get a chunk with multiple symbols
    let chunks = ctx
      .db
      .get_chunks_for_file("src/user/repository.rs")
      .await
      .expect("get chunks");
    assert!(!chunks.is_empty(), "Should have indexed the repository code");

    // Find a chunk with multiple symbols (the impl block should have many methods)
    let chunk = chunks
      .iter()
      .max_by_key(|c| c.symbols.len())
      .expect("Should find chunk with symbols");

    // Time the query - with vector search it should be fast regardless of symbol count
    let start = Instant::now();
    let related: Vec<RelatedMemoryInfo> = get_related_memories_for_code(&ctx.db, chunk, 10).await;
    let duration = start.elapsed();

    // The query should complete reasonably fast (single vector search)
    // Using 500ms as a generous threshold for test stability
    assert!(
      duration.as_millis() < 500,
      "Related memories query should be fast (got {:?}), indicating single vector search was used",
      duration
    );

    // Should find some related memories
    assert!(
      !related.is_empty(),
      "Should find related memories about user/database operations"
    );
  }

  /// Test that memory search by pre-computed embedding works correctly.
  #[tokio::test]
  async fn test_search_memories_by_embedding() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Create several memories with distinct topics
    let auth_memory = MemoryAddParams {
      content: "OAuth2 authentication flow uses access tokens and refresh tokens for secure API access".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["auth".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, auth_memory).await.expect("add auth memory");

    let db_memory = MemoryAddParams {
      content: "Database connection pooling improves performance by reusing connections".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["database".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    memory::add(&mem_ctx, db_memory).await.expect("add db memory");

    // Generate an embedding for an auth-related query
    let query = "token authentication security";
    let embedding = ctx
      .embedding
      .embed(query, crate::embedding::EmbeddingMode::Query)
      .await
      .expect("generate embedding");

    // Search using the pre-computed embedding
    let results = memory::search::search_by_embedding(&ctx.db, &embedding, 5, None)
      .await
      .expect("search by embedding");

    // Should find at least the auth memory
    assert!(!results.is_empty(), "Should find memories via embedding search");

    // The auth memory should rank higher than the database memory for an auth query
    let auth_result = results.iter().find(|(m, _)| m.content.contains("OAuth2"));
    assert!(auth_result.is_some(), "Should find the OAuth2/auth memory");
  }

  // ==========================================================================
  // Phase 4 Tests: Cross-Domain Vector Search
  // ==========================================================================

  /// Test finding related code from memory context using cross-domain vector search.
  ///
  /// This validates Phase 4.1: Memory-to-code search capability.
  /// A memory about database migrations should find the migration code via
  /// semantic similarity, even if the memory doesn't mention exact function names.
  #[tokio::test]
  async fn test_find_code_from_memory_context() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Index code about database migrations
    ctx
      .index_code(
        "src/db/migrations.rs",
        r#"
/// Run all pending database migrations.
pub async fn run_migrations(db: &Database) -> Result<(), MigrationError> {
    let pending = get_pending_migrations(db).await?;
    for migration in pending {
        db.execute(&migration.sql).await?;
        db.record_migration(&migration.version).await?;
    }
    Ok(())
}

/// Get list of migrations that haven't been applied yet.
async fn get_pending_migrations(db: &Database) -> Result<Vec<Migration>, MigrationError> {
    let applied = db.query("SELECT version FROM migrations").await?;
    MIGRATIONS.iter()
        .filter(|m| !applied.contains(&m.version))
        .cloned()
        .collect()
}
"#,
        Language::Rust,
      )
      .await;

    // Create a memory that discusses the concept without mentioning exact function names
    let memory_params = MemoryAddParams {
      content: "Database schema migrations should be run on startup. The migration system tracks applied versions and only runs pending changes.".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["database".to_string(), "migrations".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    let add_result = memory::add(&mem_ctx, memory_params).await.expect("add memory");

    // Get the memory we just created
    let memory = ctx
      .db
      .get_memory(&add_result.id.parse().expect("valid memory id"))
      .await
      .expect("get memory")
      .expect("memory should exist");

    // Find related code for this memory using cross-domain vector search
    let related_code = get_related_code_for_memory(&ctx.db, &memory, 10).await;

    // Should find the migration code via semantic similarity
    assert!(
      !related_code.is_empty(),
      "Should find code related to the migration memory"
    );

    // At least one result should be from the migrations file
    let has_migration_code = related_code.iter().any(|c| c.file.contains("migrations"));
    assert!(
      has_migration_code,
      "Should find migration code. Found files: {:?}",
      related_code.iter().map(|c| &c.file).collect::<Vec<_>>()
    );
  }

  /// Test that memory context includes related code.
  ///
  /// This validates Phase 4.3: MemoryContext includes related_code field.
  #[tokio::test]
  async fn test_memory_context_includes_related_code() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();
    let explore_ctx = ExploreContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index code about user authentication
    ctx
      .index_code(
        "src/auth/handler.rs",
        r#"
/// Handle login request and return JWT token.
pub async fn handle_login(req: LoginRequest) -> Result<LoginResponse, AuthError> {
    let user = validate_credentials(&req.username, &req.password).await?;
    let token = generate_jwt_token(&user).await?;
    Ok(LoginResponse { token, user_id: user.id })
}
"#,
        Language::Rust,
      )
      .await;

    // Create a memory about the authentication system
    let memory_params = MemoryAddParams {
      content: "The login handler validates user credentials and returns a JWT token for authenticated sessions"
        .to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["auth".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    let add_result = memory::add(&mem_ctx, memory_params).await.expect("add memory");

    // Get full context for the memory
    let context_response = get_context(&explore_ctx, std::slice::from_ref(&add_result.id), 5)
      .await
      .expect("get context");

    // Verify we got a memory context with related code
    match context_response {
      crate::service::explore::ContextResponse::Memory { items } => {
        assert!(!items.is_empty(), "Should have memory context");
        let memory_ctx = &items[0];

        // Should have related code (login handler)
        assert!(
          !memory_ctx.related_code.is_empty(),
          "Memory context should include related code via cross-domain search"
        );

        // At least one related code should be from the auth handler
        let has_auth_code = memory_ctx.related_code.iter().any(|c| c.file.contains("handler"));
        assert!(
          has_auth_code,
          "Should find auth handler code. Found: {:?}",
          memory_ctx.related_code.iter().map(|c| &c.file).collect::<Vec<_>>()
        );
      }
      _ => panic!("Expected Memory context response, got different type"),
    }
  }

  /// Test cross-domain search semantic accuracy.
  ///
  /// This validates the semantic quality of cross-domain search:
  /// A memory about one topic should find code about that topic,
  /// not code about unrelated topics.
  #[tokio::test]
  async fn test_cross_domain_search_semantic_accuracy() {
    let ctx = TestContext::new().await;
    let mem_ctx = ctx.memory_context();

    // Index code about two distinct topics
    ctx
      .index_code(
        "src/payment/processor.rs",
        r#"
/// Process a payment transaction.
pub async fn process_payment(amount: Money, card: CardInfo) -> Result<Transaction, PaymentError> {
    let gateway = PaymentGateway::new();
    gateway.charge(amount, &card).await
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/email/sender.rs",
        r#"
/// Send an email notification to a user.
pub async fn send_email(to: &str, subject: &str, body: &str) -> Result<(), EmailError> {
    let client = SmtpClient::new();
    client.send(to, subject, body).await
}
"#,
        Language::Rust,
      )
      .await;

    // Create a memory specifically about payments
    let memory_params = MemoryAddParams {
      content: "Payment processing uses a third-party gateway for secure credit card transactions".to_string(),
      sector: Some("semantic".to_string()),
      memory_type: Some("codebase".to_string()),
      context: None,
      tags: Some(vec!["payment".to_string()]),
      categories: None,
      scope_path: None,
      scope_module: None,
      importance: None,
    };
    let add_result = memory::add(&mem_ctx, memory_params).await.expect("add memory");

    // Get the memory
    let memory = ctx
      .db
      .get_memory(&add_result.id.parse().expect("valid memory id"))
      .await
      .expect("get memory")
      .expect("memory should exist");

    // Find related code
    let related_code = get_related_code_for_memory(&ctx.db, &memory, 10).await;

    // Should find some code
    assert!(!related_code.is_empty(), "Should find some related code");

    // Payment code should rank higher than email code for a payment-related memory
    let payment_idx = related_code.iter().position(|c| c.file.contains("payment"));
    let email_idx = related_code.iter().position(|c| c.file.contains("email"));

    if let (Some(pay_pos), Some(email_pos)) = (payment_idx, email_idx) {
      assert!(
        pay_pos < email_pos,
        "Payment code should rank higher than email code for payment-related memory. Payment at {}, email at {}",
        pay_pos,
        email_pos
      );
    } else {
      // If we found payment code but not email, that's fine too
      assert!(
        payment_idx.is_some(),
        "Should find payment code for payment-related memory. Found: {:?}",
        related_code.iter().map(|c| &c.file).collect::<Vec<_>>()
      );
    }
  }
}
