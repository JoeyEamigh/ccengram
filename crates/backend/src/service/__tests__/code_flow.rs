//! Integration tests for code indexing and search flow.
//!
//! These tests validate the code indexing, search, and call graph navigation.

#[cfg(test)]
mod tests {
  use crate::{
    domain::code::Language,
    service::{
      __tests__::helpers::TestContext,
      code::{CodeContext, RankingConfig, SearchParams, search},
    },
  };

  /// Test basic code indexing and search flow.
  ///
  /// Validates:
  /// 1. Index code chunks with AST parsing
  /// 2. Search finds indexed code
  /// 3. Symbol matching works correctly
  #[tokio::test]
  async fn test_code_import_and_search() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index a Rust function
    ctx
      .index_code(
        "src/auth/login.rs",
        r#"
/// Authenticate a user with username and password.
/// Returns a session token on success.
pub fn authenticate(username: &str, password: &str) -> Result<Token, AuthError> {
    let user = find_user_by_username(username)?;
    verify_password(&user, password)?;
    generate_session_token(&user)
}
"#,
        Language::Rust,
      )
      .await;

    // Index a second function that calls authenticate
    ctx
      .index_code(
        "src/handlers/auth_handler.rs",
        r#"
use crate::auth::login::authenticate;

/// HTTP handler for login requests.
pub async fn handle_login(request: LoginRequest) -> Response {
    match authenticate(&request.username, &request.password) {
        Ok(token) => Response::ok(token),
        Err(e) => Response::unauthorized(e.message()),
    }
}
"#,
        Language::Rust,
      )
      .await;

    // Search for "authenticate" - language filter uses stored format (lowercase enum name)
    let search_params = SearchParams {
      query: "authenticate".to_string(),
      language: Some("rust".to_string()), // This matches the stored format
      limit: Some(10),
      include_context: true,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let search_result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search");

    assert!(!search_result.results.is_empty(), "Should find code chunks");

    // Verify at least one result contains authenticate
    let has_auth = search_result.results.iter().any(|r| r.content.contains("authenticate"));
    assert!(has_auth, "Results should include authenticate function");
  }

  /// Test search with language filter.
  #[tokio::test]
  async fn test_code_search_language_filter() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index Rust code
    ctx
      .index_code(
        "src/lib.rs",
        "pub fn rust_function() { println!(\"Hello from Rust\"); }",
        Language::Rust,
      )
      .await;

    // Index Python code
    ctx
      .index_code(
        "main.py",
        "def python_function():\n    print(\"Hello from Python\")",
        Language::Python,
      )
      .await;

    // Search with Rust filter - use stored format
    let search_params = SearchParams {
      query: "function".to_string(),
      language: Some("rust".to_string()),
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search");

    // Should only find Rust code
    for item in &result.results {
      assert_eq!(item.language.as_deref(), Some("rust"), "Should only return Rust code");
    }
  }

  /// Test semantic search finds related code without hardcoded query expansion.
  ///
  /// This validates Phase 2 of embedding improvements: the embedding model
  /// naturally understands semantic relationships like "auth" → authentication,
  /// jwt, oauth, etc. without needing hardcoded synonym mappings.
  #[tokio::test]
  async fn test_semantic_search_without_expansion() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index auth-related functions with different naming conventions
    ctx
      .index_code(
        "src/auth/user.rs",
        r#"
/// Authenticate a user with credentials.
/// Validates username and password against the database.
pub fn authenticate_user(credentials: &Credentials) -> Result<User, AuthError> {
    let user = find_by_username(&credentials.username)?;
    verify_password(&user, &credentials.password)?;
    Ok(user)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/auth/jwt.rs",
        r#"
/// Validate a JSON Web Token and extract claims.
/// Returns the decoded claims if the token is valid.
pub fn validate_jwt_token(token: &str) -> Result<Claims, TokenError> {
    let decoded = decode_token(token)?;
    verify_signature(&decoded)?;
    verify_expiration(&decoded)?;
    Ok(decoded.claims)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/auth/oauth.rs",
        r#"
/// Handle OAuth2 callback after user authorization.
/// Exchanges the authorization code for access token.
pub fn oauth_callback(code: &str) -> Result<Session, OAuthError> {
    let tokens = exchange_code(code)?;
    let user_info = fetch_user_info(&tokens.access_token)?;
    create_session(&user_info)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/utils/math.rs",
        r#"
/// Calculate the sum of two numbers.
pub fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
"#,
        Language::Rust,
      )
      .await;

    // Search for "auth" with exact=true to ensure we're NOT using hardcoded expansion
    // The embedding model should still find semantically related code
    let search_params = SearchParams {
      query: "auth".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // The embedding model should find auth-related functions via semantic similarity
    let symbols: Vec<String> = result.results.iter().filter_map(|r| r.symbol_name.clone()).collect();

    assert!(
      symbols.iter().any(|s| s.contains("authenticate")),
      "Should find authenticate_user via semantic similarity, found: {:?}",
      symbols
    );

    assert!(
      symbols.iter().any(|s| s.contains("jwt") || s.contains("token")),
      "Should find JWT validation via semantic similarity, found: {:?}",
      symbols
    );

    // Unrelated code should be ranked BELOW auth-related code
    // Vector search returns all results, but relevant ones should rank higher
    let auth_positions: Vec<usize> = symbols
      .iter()
      .enumerate()
      .filter(|(_, s)| s.contains("authenticate") || s.contains("jwt") || s.contains("oauth"))
      .map(|(i, _)| i)
      .collect();
    let unrelated_position = symbols.iter().position(|s| s.contains("add_numbers"));

    if let Some(unrelated_pos) = unrelated_position {
      let max_auth_pos = auth_positions.iter().max().copied().unwrap_or(0);
      assert!(
        unrelated_pos > max_auth_pos,
        "Unrelated function should rank lower than auth functions. Auth positions: {:?}, unrelated: {}",
        auth_positions,
        unrelated_pos
      );
    }
  }

  // ==========================================================================
  // Phase 3 Tests: Metadata Filters and Caller Count Ranking
  // ==========================================================================

  /// Test that visibility filter is applied before vector search.
  ///
  /// This validates Phase 3.2: Metadata filters work correctly in code search.
  /// The visibility filter should restrict results BEFORE ranking.
  #[tokio::test]
  async fn test_visibility_filter_applied_to_vector_search() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index public and private functions about the same topic (auth)
    ctx
      .index_code(
        "src/auth/public_api.rs",
        r#"
/// Public authentication entry point.
pub fn public_authenticate(username: &str, password: &str) -> Result<Token, AuthError> {
    validate_credentials(username, password)?;
    generate_token(username)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/auth/internal.rs",
        r#"
/// Private helper for authentication (internal only).
fn private_auth_helper(username: &str) -> Option<User> {
    database_lookup(username)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/auth/crate_api.rs",
        r#"
/// Crate-visible authentication utility.
pub(crate) fn crate_auth_utility(token: &str) -> bool {
    verify_token_signature(token)
}
"#,
        Language::Rust,
      )
      .await;

    // Search with visibility filter for only public functions
    let search_params = SearchParams {
      query: "authenticate".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec!["pub".to_string()],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Should only find the public function
    assert!(!result.results.is_empty(), "Should find at least one result");

    for item in &result.results {
      // All results should be from the public_api.rs file
      assert!(
        item.file_path.contains("public_api"),
        "Should only return public functions, got: {} (from {})",
        item.symbol_name.as_deref().unwrap_or("unknown"),
        item.file_path
      );
    }
  }

  /// Test that chunk_type filter works correctly.
  ///
  /// This validates Phase 3.2: chunk_type filtering restricts to specific types.
  #[tokio::test]
  async fn test_chunk_type_filter_applied_to_vector_search() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index a struct/class and a function about the same topic
    ctx
      .index_code(
        "src/user/model.rs",
        r#"
/// User data model for the application.
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/user/service.rs",
        r#"
/// Get a user by their ID from the database.
pub fn get_user_by_id(id: u64) -> Option<User> {
    database.find_user(id)
}
"#,
        Language::Rust,
      )
      .await;

    // Search with chunk_type filter for only functions
    let search_params = SearchParams {
      query: "user".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec!["function".to_string()],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Should only find the function, not the struct
    for item in &result.results {
      assert_eq!(
        item.chunk_type.as_deref(),
        Some("function"),
        "Should only return functions, got: {:?}",
        item.chunk_type
      );
    }
  }

  /// Test that caller_count affects ranking (higher callers = higher rank).
  ///
  /// This validates Phase 3.3: Functions with more callers rank higher.
  #[tokio::test]
  async fn test_caller_count_affects_ranking() {
    use chrono::Utc;
    use uuid::Uuid;

    use crate::domain::code::{ChunkType, CodeChunk};

    let ctx = TestContext::new().await;

    // Create two functions with similar content but different caller counts
    let central_content = "pub fn central_utility() { process_data() }";
    let isolated_content = "pub fn isolated_helper() { process_data() }";

    // Create chunks with caller counts set
    let central_chunk = CodeChunk {
      id: Uuid::new_v4(),
      file_path: "src/utils/central.rs".to_string(),
      content: central_content.to_string(),
      language: Language::Rust,
      chunk_type: ChunkType::Function,
      symbols: vec!["central_utility".to_string()],
      imports: vec![],
      calls: vec!["process_data".to_string()],
      start_line: 1,
      end_line: 1,
      file_hash: "hash1".to_string(),
      indexed_at: Utc::now(),
      tokens_estimate: 10,
      definition_kind: Some("function".to_string()),
      definition_name: Some("central_utility".to_string()),
      visibility: Some("pub".to_string()),
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: Some("public function central_utility that calls process_data".to_string()),
      content_hash: Some("central_hash_001".to_string()),
      caller_count: 50, // Called by many other functions
      callee_count: 1,
    };

    let isolated_chunk = CodeChunk {
      id: Uuid::new_v4(),
      file_path: "src/utils/isolated.rs".to_string(),
      content: isolated_content.to_string(),
      language: Language::Rust,
      chunk_type: ChunkType::Function,
      symbols: vec!["isolated_helper".to_string()],
      imports: vec![],
      calls: vec!["process_data".to_string()],
      start_line: 1,
      end_line: 1,
      file_hash: "hash2".to_string(),
      indexed_at: Utc::now(),
      tokens_estimate: 10,
      definition_kind: Some("function".to_string()),
      definition_name: Some("isolated_helper".to_string()),
      visibility: Some("pub".to_string()),
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: Some("public function isolated_helper that calls process_data".to_string()),
      content_hash: Some("isolated_hash_001".to_string()),
      caller_count: 0, // Never called
      callee_count: 1,
    };

    // Generate embeddings and add chunks directly
    let central_embedding = ctx
      .embedding
      .embed(
        central_chunk
          .embedding_text
          .as_deref()
          .unwrap_or(&central_chunk.content),
        crate::embedding::EmbeddingMode::Document,
      )
      .await
      .expect("embed central");

    let isolated_embedding = ctx
      .embedding
      .embed(
        isolated_chunk
          .embedding_text
          .as_deref()
          .unwrap_or(&isolated_chunk.content),
        crate::embedding::EmbeddingMode::Document,
      )
      .await
      .expect("embed isolated");

    ctx
      .db
      .upsert_code_chunks("src/utils/central.rs", &[(central_chunk, central_embedding)])
      .await
      .expect("upsert central chunk");
    ctx
      .db
      .upsert_code_chunks("src/utils/isolated.rs", &[(isolated_chunk, isolated_embedding)])
      .await
      .expect("upsert isolated chunk");

    // Search for "utility" - should find both but central should rank higher
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());
    let search_params = SearchParams {
      query: "utility function that processes data".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Find positions of both chunks in results
    let central_pos = result
      .results
      .iter()
      .position(|r| r.symbol_name.as_deref() == Some("central_utility"));
    let isolated_pos = result
      .results
      .iter()
      .position(|r| r.symbol_name.as_deref() == Some("isolated_helper"));

    assert!(central_pos.is_some(), "Should find central_utility in results");
    assert!(isolated_pos.is_some(), "Should find isolated_helper in results");

    assert!(
      central_pos.unwrap() < isolated_pos.unwrap(),
      "Central function (50 callers) should rank higher than isolated (0 callers). Central at {}, isolated at {}",
      central_pos.unwrap(),
      isolated_pos.unwrap()
    );
  }

  /// Test min_caller_count filter.
  ///
  /// This validates Phase 3.2: min_caller_count filters out code with few callers.
  #[tokio::test]
  async fn test_min_caller_count_filter() {
    use chrono::Utc;
    use uuid::Uuid;

    use crate::domain::code::{ChunkType, CodeChunk};

    let ctx = TestContext::new().await;

    // Create chunks with different caller counts
    let popular_chunk = CodeChunk {
      id: Uuid::new_v4(),
      file_path: "src/utils/popular.rs".to_string(),
      content: "pub fn popular_function() { }".to_string(),
      language: Language::Rust,
      chunk_type: ChunkType::Function,
      symbols: vec!["popular_function".to_string()],
      imports: vec![],
      calls: vec![],
      start_line: 1,
      end_line: 1,
      file_hash: "hash1".to_string(),
      indexed_at: Utc::now(),
      tokens_estimate: 10,
      definition_kind: Some("function".to_string()),
      definition_name: Some("popular_function".to_string()),
      visibility: Some("pub".to_string()),
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: Some("public function popular_function utility".to_string()),
      content_hash: Some("popular_hash_001".to_string()),
      caller_count: 15,
      callee_count: 0,
    };

    let unpopular_chunk = CodeChunk {
      id: Uuid::new_v4(),
      file_path: "src/utils/unpopular.rs".to_string(),
      content: "pub fn unpopular_function() { }".to_string(),
      language: Language::Rust,
      chunk_type: ChunkType::Function,
      symbols: vec!["unpopular_function".to_string()],
      imports: vec![],
      calls: vec![],
      start_line: 1,
      end_line: 1,
      file_hash: "hash2".to_string(),
      indexed_at: Utc::now(),
      tokens_estimate: 10,
      definition_kind: Some("function".to_string()),
      definition_name: Some("unpopular_function".to_string()),
      visibility: Some("pub".to_string()),
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: Some("public function unpopular_function utility".to_string()),
      content_hash: Some("unpopular_hash_001".to_string()),
      caller_count: 2,
      callee_count: 0,
    };

    // Generate embeddings and add chunks
    let popular_embedding = ctx
      .embedding
      .embed(
        popular_chunk
          .embedding_text
          .as_deref()
          .unwrap_or(&popular_chunk.content),
        crate::embedding::EmbeddingMode::Document,
      )
      .await
      .expect("embed popular");

    let unpopular_embedding = ctx
      .embedding
      .embed(
        unpopular_chunk
          .embedding_text
          .as_deref()
          .unwrap_or(&unpopular_chunk.content),
        crate::embedding::EmbeddingMode::Document,
      )
      .await
      .expect("embed unpopular");

    ctx
      .db
      .upsert_code_chunks("src/utils/popular.rs", &[(popular_chunk, popular_embedding)])
      .await
      .expect("upsert popular chunk");
    ctx
      .db
      .upsert_code_chunks("src/utils/unpopular.rs", &[(unpopular_chunk, unpopular_embedding)])
      .await
      .expect("upsert unpopular chunk");

    // Search with min_caller_count filter
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());
    let search_params = SearchParams {
      query: "function utility".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: Some(10), // Only functions with 10+ callers
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Should only find the popular function
    assert!(!result.results.is_empty(), "Should find at least one result");

    let found_popular = result
      .results
      .iter()
      .any(|r| r.symbol_name.as_deref() == Some("popular_function"));
    let found_unpopular = result
      .results
      .iter()
      .any(|r| r.symbol_name.as_deref() == Some("unpopular_function"));

    assert!(found_popular, "Should find popular_function (15 callers >= 10)");
    assert!(!found_unpopular, "Should NOT find unpopular_function (2 callers < 10)");
  }

  /// Test that embedding model understands domain-specific abbreviations.
  ///
  /// This validates that semantic search works for domain terms that would
  /// NOT be in any hardcoded synonym map. The embedding model should
  /// understand that "LTV" relates to "lifetime value" in business contexts.
  #[tokio::test]
  async fn test_domain_specific_semantic_search() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index business domain code
    ctx
      .index_code(
        "src/analytics/ltv.rs",
        r#"
/// Calculate customer LTV (Lifetime Value).
/// Uses historical purchase data to estimate total revenue.
pub fn calculate_ltv(customer: &Customer) -> Money {
    let total_orders = customer.orders.len() as f64;
    let avg_order_value = customer.total_spent / total_orders;
    let retention_rate = calculate_retention_rate(customer);
    Money::from(avg_order_value * retention_rate * 12.0)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/analytics/clv.rs",
        r#"
/// Compute customer lifetime value from order history.
/// Projects future revenue based on purchase patterns.
pub fn compute_customer_lifetime_value(orders: &[Order]) -> Money {
    let purchase_frequency = calculate_frequency(orders);
    let average_value = calculate_average_order_value(orders);
    let lifespan = estimate_customer_lifespan(orders);
    Money::from(purchase_frequency * average_value * lifespan)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/utils/string.rs",
        r#"
/// Convert string to uppercase.
pub fn to_uppercase(s: &str) -> String {
    s.to_uppercase()
}
"#,
        Language::Rust,
      )
      .await;

    // Search for "LTV" - a domain abbreviation NOT in any hardcoded expansion map
    let search_params = SearchParams {
      query: "LTV".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    let symbols: Vec<String> = result.results.iter().filter_map(|r| r.symbol_name.clone()).collect();

    // Should find LTV-related functions via semantic understanding
    // The embedding model should know LTV = Lifetime Value
    let found_ltv = symbols
      .iter()
      .any(|s| s.contains("ltv") || s.contains("lifetime_value"));
    assert!(
      found_ltv,
      "Embedding model should understand LTV = lifetime value and find related functions, found: {:?}",
      symbols
    );

    // Unrelated code should be ranked BELOW LTV-related code
    // Vector search returns all results, but relevant ones should rank higher
    let ltv_positions: Vec<usize> = symbols
      .iter()
      .enumerate()
      .filter(|(_, s)| s.contains("ltv") || s.contains("lifetime_value"))
      .map(|(i, _)| i)
      .collect();
    let unrelated_position = symbols.iter().position(|s| s.contains("uppercase"));

    if let Some(unrelated_pos) = unrelated_position {
      let max_ltv_pos = ltv_positions.iter().max().copied().unwrap_or(0);
      assert!(
        unrelated_pos > max_ltv_pos,
        "Unrelated function should rank lower than LTV functions. LTV positions: {:?}, unrelated: {}",
        ltv_positions,
        unrelated_pos
      );
    }
  }

  // ==========================================================================
  // Phase 5 Tests: Distance-Based Confidence Scoring
  // ==========================================================================

  /// Test that confidence score is returned and reflects vector distance.
  ///
  /// This validates Phase 5.1: Search results include confidence scores derived
  /// from the raw vector distance (1.0 - distance).
  #[tokio::test]
  async fn test_confidence_score_in_search_results() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index a function with a very specific name
    ctx
      .index_code(
        "src/auth/exact_match.rs",
        r#"
/// This function has a very specific unique name for testing exact matches.
pub fn unique_exact_match_function_xyz123() {
    println!("Hello");
}
"#,
        Language::Rust,
      )
      .await;

    // Search for the exact function name
    let search_params = SearchParams {
      query: "unique_exact_match_function_xyz123".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(&code_ctx, search_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    assert!(!result.results.is_empty(), "Should find at least one result");

    // First result should have confidence score
    let first = &result.results[0];
    assert!(
      first.confidence.is_some(),
      "Search results should include confidence score"
    );

    let confidence = first.confidence.unwrap();
    // Exact symbol match should have reasonable confidence
    // (threshold varies by model: larger models score higher, 0.6B LlamaCpp scores ~0.26)
    assert!(
      confidence > 0.2,
      "Exact symbol match should have confidence > 0.2, got {}",
      confidence
    );
  }

  /// Test that search quality metadata is returned and reflects result quality.
  ///
  /// This validates Phase 5.3: SearchQuality indicates when results may not be relevant.
  #[tokio::test]
  async fn test_search_quality_metadata() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index some code
    ctx
      .index_code(
        "src/math/add.rs",
        r#"
/// Add two numbers together.
pub fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
"#,
        Language::Rust,
      )
      .await;

    // Search with a relevant query
    let relevant_params = SearchParams {
      query: "add numbers".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let relevant_result = search::search(&code_ctx, relevant_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Search quality should be reasonable for relevant query
    let quality = &relevant_result.search_quality;
    assert!(
      quality.best_distance < 0.8,
      "Relevant query should have best_distance < 0.8, got {}",
      quality.best_distance
    );

    // Search with completely unrelated query
    let unrelated_params = SearchParams {
      query: "quantum entanglement photon decay".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let unrelated_result = search::search(&code_ctx, unrelated_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Unrelated query should have higher distance (lower confidence)
    let unrelated_quality = &unrelated_result.search_quality;
    // The best distance for unrelated query should be higher
    // If it's low_confidence, that's also acceptable
    assert!(
      unrelated_quality.best_distance > relevant_result.search_quality.best_distance
        || unrelated_quality.low_confidence,
      "Unrelated query should have worse search quality than relevant query"
    );
  }

  /// Test that adaptive limit reduces results when top results are confident.
  ///
  /// This validates Phase 5.2: When adaptive_limit is enabled, confident searches
  /// return fewer results to reduce noise.
  #[tokio::test]
  async fn test_adaptive_limit_reduces_noise() {
    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index multiple functions, one with exact match
    ctx
      .index_code(
        "src/primary.rs",
        r#"
/// Primary authentication handler.
pub fn authenticate_primary() {
    verify_credentials();
}
"#,
        Language::Rust,
      )
      .await;

    for i in 0..10 {
      ctx
        .index_code(
          &format!("src/other_{}.rs", i),
          &format!(
            r#"
/// Some other utility function number {}.
pub fn utility_function_{}() {{
    do_something();
}}
"#,
            i, i
          ),
          Language::Rust,
        )
        .await;
    }

    // Search WITH adaptive_limit
    let adaptive_params = SearchParams {
      query: "authenticate_primary".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: true,
    };

    let adaptive_result = search::search(&code_ctx, adaptive_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Search WITHOUT adaptive_limit
    let normal_params = SearchParams {
      query: "authenticate_primary".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let normal_result = search::search(&code_ctx, normal_params, &RankingConfig::default(), None, None)
      .await
      .expect("search should succeed");

    // Both should find the primary function
    assert!(
      adaptive_result.results.iter().any(|r| {
        r.symbol_name
          .as_ref()
          .map(|s| s.contains("authenticate_primary"))
          .unwrap_or(false)
      }),
      "Adaptive search should find authenticate_primary"
    );
    assert!(
      normal_result.results.iter().any(|r| {
        r.symbol_name
          .as_ref()
          .map(|s| s.contains("authenticate_primary"))
          .unwrap_or(false)
      }),
      "Normal search should find authenticate_primary"
    );

    // If the search was confident, adaptive should return fewer or equal results
    // (Note: This is a soft test - the actual behavior depends on the confidence scores)
    if adaptive_result.search_quality.high_confidence_count >= 3 {
      assert!(
        adaptive_result.results.len() <= 5,
        "With high confidence, adaptive should limit to 5 results, got {}",
        adaptive_result.results.len()
      );
    }
  }

  // ==========================================================================
  // Phase 6 Tests: Hybrid Search Pipeline
  // ==========================================================================

  /// Test that hybrid search (FTS + vector) with fts_enabled=true works end-to-end.
  ///
  /// Validates:
  /// 1. Index code with identifiers that should be FTS-matchable
  /// 2. Search with fts_enabled=true finds results
  /// 3. Search still works (no panics) even if FTS returns empty
  #[tokio::test]
  async fn test_hybrid_search_fts_enabled() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index code with very specific function names
    ctx
      .index_code(
        "src/payment/checkout.rs",
        r#"
/// Process a checkout with payment gateway.
pub fn process_checkout_payment(cart: &Cart, method: &PaymentMethod) -> Result<Receipt, PaymentError> {
    validate_cart(cart)?;
    charge_payment(method, cart.total())?;
    generate_receipt(cart)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/inventory/stock.rs",
        r#"
/// Check if all items in the cart are in stock.
pub fn verify_stock_availability(items: &[Item]) -> Result<(), StockError> {
    for item in items {
        let available = check_warehouse_stock(item.sku)?;
        if available < item.quantity {
            return Err(StockError::OutOfStock(item.sku.clone()));
        }
    }
    Ok(())
}
"#,
        Language::Rust,
      )
      .await;

    let fts_config = SearchConfig {
      fts_enabled: true,
      rrf_k: 60,
      rerank_candidates: 30,
      ..Default::default()
    };

    // Search with FTS enabled - should find results even if FTS indexes aren't built
    // (the search gracefully falls back to vector-only when FTS fails)
    let search_params = SearchParams {
      query: "checkout payment".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(
      &code_ctx,
      search_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("hybrid search should not fail");

    assert!(!result.results.is_empty(), "hybrid search should return results");

    // Payment-related code should be found
    let has_payment = result
      .results
      .iter()
      .any(|r| r.content.contains("checkout") || r.content.contains("payment"));
    assert!(
      has_payment,
      "should find checkout/payment code, found: {:?}",
      result
        .results
        .iter()
        .map(|r| r.symbol_name.as_deref().unwrap_or("?"))
        .collect::<Vec<_>>()
    );
  }

  /// Test that search with fts_enabled=false (default) still works correctly.
  ///
  /// This validates graceful degradation: the pipeline falls back to vector-only
  /// when FTS is disabled.
  #[tokio::test]
  async fn test_search_fts_disabled_fallback() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    ctx
      .index_code(
        "src/router.rs",
        r#"
/// Route HTTP requests to the appropriate handler.
pub fn route_request(method: &str, path: &str) -> Handler {
    match (method, path) {
        ("GET", "/health") => health_check,
        ("POST", "/login") => handle_login,
        _ => not_found,
    }
}
"#,
        Language::Rust,
      )
      .await;

    let fts_off_config = SearchConfig {
      fts_enabled: false,
      ..Default::default()
    };

    let search_params = SearchParams {
      query: "route request".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(
      &code_ctx,
      search_params,
      &RankingConfig::default(),
      Some(&fts_off_config),
      None,
    )
    .await
    .expect("vector-only search should work");

    assert!(!result.results.is_empty(), "vector-only search should return results");
  }

  /// Test full hybrid search pipeline: index code with FTS enabled, search for
  /// exact identifiers, and verify keyword matches appear in results.
  ///
  /// This validates the end-to-end hybrid pipeline:
  /// 1. Index code files with unique identifiers
  /// 2. Enable FTS in SearchConfig
  /// 3. Search for an exact identifier that FTS should match
  /// 4. Verify results contain the expected keyword matches
  #[tokio::test]
  async fn test_hybrid_fts_keyword_match() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index code with very unique, specific function names that FTS should match exactly
    ctx
      .index_code(
        "src/billing/invoice_generator.rs",
        r#"
/// Generate a detailed invoice for the customer order.
/// Calculates line items, taxes, and total amount due.
pub fn generate_invoice_for_customer_order(order: &Order, customer: &Customer) -> Result<Invoice, BillingError> {
    let line_items = calculate_line_items(order)?;
    let subtotal = line_items.iter().map(|li| li.amount).sum::<Money>();
    let tax = calculate_tax(subtotal, customer.tax_region())?;
    Invoice::build(customer, line_items, subtotal, tax)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/billing/refund_processor.rs",
        r#"
/// Process a refund for a previously completed transaction.
/// Validates the original transaction and creates a reversal entry.
pub fn process_refund_for_transaction(transaction_id: &str, reason: &str) -> Result<Refund, BillingError> {
    let original = find_transaction(transaction_id)?;
    validate_refund_eligibility(&original)?;
    create_reversal_entry(&original, reason)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/notifications/email_sender.rs",
        r#"
/// Send a notification email to the specified recipient.
pub fn send_notification_email(recipient: &str, subject: &str, body: &str) -> Result<(), EmailError> {
    let message = build_email_message(recipient, subject, body)?;
    smtp_client().send(message)
}
"#,
        Language::Rust,
      )
      .await;

    let fts_config = SearchConfig {
      fts_enabled: true,
      rrf_k: 60,
      rerank_candidates: 30,
      ..Default::default()
    };

    // Search for a keyword that should match via FTS (exact identifier token)
    let search_params = SearchParams {
      query: "generate_invoice_for_customer_order".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(
      &code_ctx,
      search_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("hybrid search should succeed");

    assert!(
      !result.results.is_empty(),
      "hybrid search with FTS should return results"
    );

    // The invoice generator should be in results - it's an exact identifier match
    let has_invoice = result.results.iter().any(|r| {
      r.content.contains("generate_invoice_for_customer_order")
        || r.symbol_name.as_deref().is_some_and(|s| s.contains("generate_invoice"))
    });
    assert!(
      has_invoice,
      "FTS should find exact identifier 'generate_invoice_for_customer_order', found symbols: {:?}",
      result
        .results
        .iter()
        .map(|r| r.symbol_name.as_deref().unwrap_or("?"))
        .collect::<Vec<_>>()
    );

    // Also search for a more natural query - vector similarity should still work
    let natural_params = SearchParams {
      query: "billing refund".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let natural_result = search::search(
      &code_ctx,
      natural_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("natural language hybrid search should succeed");

    assert!(
      !natural_result.results.is_empty(),
      "natural language hybrid search should return results"
    );

    let has_refund = natural_result
      .results
      .iter()
      .any(|r| r.content.contains("refund") || r.content.contains("billing"));
    assert!(
      has_refund,
      "hybrid search for 'billing refund' should find refund-related code, found: {:?}",
      natural_result
        .results
        .iter()
        .map(|r| r.symbol_name.as_deref().unwrap_or("?"))
        .collect::<Vec<_>>()
    );
  }

  /// Test that FTS boosts exact identifier matches in hybrid search results.
  ///
  /// When searching for a very specific, unique function name, the hybrid
  /// pipeline (vector + FTS with RRF fusion) should rank it highly because
  /// FTS provides an exact keyword match signal on top of vector similarity.
  #[tokio::test]
  async fn test_hybrid_fts_boosts_exact_identifier() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index many functions, but one has a very unique name
    ctx
      .index_code(
        "src/protocol/zxq_handshake.rs",
        r#"
/// Perform the ZXQ protocol handshake with the remote peer.
/// This is a custom binary protocol for low-latency communication.
pub fn perform_zxq_protocol_handshake(peer: &Peer, timeout: Duration) -> Result<Connection, ProtocolError> {
    let syn = build_zxq_syn_packet(peer)?;
    let ack = send_and_await_ack(syn, timeout)?;
    establish_zxq_connection(peer, ack)
}
"#,
        Language::Rust,
      )
      .await;

    // Index semantically related but differently named functions
    ctx
      .index_code(
        "src/network/tcp_connect.rs",
        r#"
/// Establish a TCP connection to the remote server.
pub fn establish_tcp_connection(host: &str, port: u16) -> Result<TcpStream, IoError> {
    let addr = resolve_dns(host, port)?;
    TcpStream::connect(addr)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/network/http_client.rs",
        r#"
/// Send an HTTP request and return the response.
pub fn send_http_request(url: &str, method: &str, body: Option<&str>) -> Result<Response, HttpError> {
    let client = HttpClient::new();
    client.request(method, url).body(body).send()
}
"#,
        Language::Rust,
      )
      .await;

    let fts_config = SearchConfig {
      fts_enabled: true,
      rrf_k: 60,
      rerank_candidates: 30,
      ..Default::default()
    };

    // Search for the exact unique identifier
    let search_params = SearchParams {
      query: "zxq_protocol_handshake".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(
      &code_ctx,
      search_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("search should succeed");

    assert!(!result.results.is_empty(), "should find results for unique identifier");

    // The ZXQ handshake function should be the top result (or near the top)
    let zxq_position = result.results.iter().position(|r| {
      r.content.contains("zxq_protocol_handshake") || r.symbol_name.as_deref().is_some_and(|s| s.contains("zxq"))
    });

    assert!(
      zxq_position.is_some(),
      "should find the ZXQ handshake function in results, found: {:?}",
      result
        .results
        .iter()
        .map(|r| r.symbol_name.as_deref().unwrap_or("?"))
        .collect::<Vec<_>>()
    );

    assert!(
      zxq_position.unwrap() <= 2,
      "ZXQ handshake should be in top 3 results (exact identifier match), found at position {}",
      zxq_position.unwrap()
    );
  }

  /// Test hybrid vs vector-only comparison.
  ///
  /// Run the same query with fts_enabled=true and fts_enabled=false.
  /// Both should return results. The hybrid results should at minimum include
  /// keyword-matchable results, and both pipelines should work without errors.
  #[tokio::test]
  async fn test_hybrid_vs_vector_only_comparison() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index diverse code so both vector and FTS have work to do
    ctx
      .index_code(
        "src/database/connection_pool.rs",
        r#"
/// Manage a pool of database connections for concurrent access.
/// Connections are reused to avoid the overhead of establishing new ones.
pub fn create_connection_pool(config: &DbConfig) -> Result<ConnectionPool, PoolError> {
    let max_connections = config.max_pool_size.unwrap_or(10);
    let pool = ConnectionPool::builder()
        .max_size(max_connections)
        .connection_timeout(config.timeout)
        .build(config.connection_string())?;
    Ok(pool)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/database/query_executor.rs",
        r#"
/// Execute a parameterized SQL query against the database.
pub fn execute_parameterized_query(pool: &ConnectionPool, sql: &str, params: &[Value]) -> Result<QueryResult, DbError> {
    let conn = pool.get_connection()?;
    conn.execute(sql, params)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/cache/redis_adapter.rs",
        r#"
/// Store a value in Redis with expiration.
pub fn redis_cache_set(key: &str, value: &str, ttl_secs: u64) -> Result<(), CacheError> {
    let client = get_redis_client()?;
    client.set_ex(key, value, ttl_secs)
}
"#,
        Language::Rust,
      )
      .await;

    let query = "connection_pool database".to_string();

    // Run with FTS enabled (hybrid)
    let fts_config = SearchConfig {
      fts_enabled: true,
      rrf_k: 60,
      rerank_candidates: 30,
      ..Default::default()
    };

    let hybrid_params = SearchParams {
      query: query.clone(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let hybrid_result = search::search(
      &code_ctx,
      hybrid_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("hybrid search should succeed");

    // Run with FTS disabled (vector-only)
    let no_fts_config = SearchConfig {
      fts_enabled: false,
      ..Default::default()
    };

    let vector_params = SearchParams {
      query: query.clone(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let vector_result = search::search(
      &code_ctx,
      vector_params,
      &RankingConfig::default(),
      Some(&no_fts_config),
      None,
    )
    .await
    .expect("vector-only search should succeed");

    // Both should return results
    assert!(!hybrid_result.results.is_empty(), "hybrid search should return results");
    assert!(
      !vector_result.results.is_empty(),
      "vector-only search should return results"
    );

    // Both should find the connection pool code
    let hybrid_has_pool = hybrid_result
      .results
      .iter()
      .any(|r| r.content.contains("connection_pool") || r.content.contains("ConnectionPool"));
    let vector_has_pool = vector_result
      .results
      .iter()
      .any(|r| r.content.contains("connection_pool") || r.content.contains("ConnectionPool"));

    assert!(hybrid_has_pool, "hybrid search should find connection pool code");
    assert!(vector_has_pool, "vector-only search should find connection pool code");

    // Hybrid results should have the pool code; verify it appears
    let hybrid_pool_pos = hybrid_result
      .results
      .iter()
      .position(|r| r.content.contains("create_connection_pool"));
    assert!(
      hybrid_pool_pos.is_some(),
      "hybrid should include the connection pool function"
    );
  }

  /// Test RRF fusion score sanity in hybrid search results.
  ///
  /// With FTS enabled, verify that:
  /// 1. All results have positive confidence scores
  /// 2. Results are ordered by score descending
  /// 3. No nonsensical scores (negative, NaN, etc.)
  #[tokio::test]
  async fn test_rrf_fusion_sanity() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    // Index several code files to generate enough results for meaningful scoring
    ctx
      .index_code(
        "src/auth/login.rs",
        r#"
/// Handle user login with username and password.
pub fn handle_user_login(username: &str, password: &str) -> Result<Session, AuthError> {
    let user = lookup_user(username)?;
    verify_password_hash(&user, password)?;
    create_new_session(&user)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/auth/logout.rs",
        r#"
/// Handle user logout by invalidating the session.
pub fn handle_user_logout(session_id: &str) -> Result<(), AuthError> {
    let session = find_session(session_id)?;
    invalidate_session(&session)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/auth/register.rs",
        r#"
/// Register a new user account with validation.
pub fn register_new_user(email: &str, username: &str, password: &str) -> Result<User, AuthError> {
    validate_email_format(email)?;
    validate_password_strength(password)?;
    check_username_available(username)?;
    create_user_record(email, username, password)
}
"#,
        Language::Rust,
      )
      .await;

    ctx
      .index_code(
        "src/analytics/tracking.rs",
        r#"
/// Track a user event for analytics purposes.
pub fn track_user_event(user_id: &str, event: &str, metadata: &HashMap<String, String>) -> Result<(), TrackingError> {
    let event = AnalyticsEvent::new(user_id, event, metadata);
    event_queue().push(event)
}
"#,
        Language::Rust,
      )
      .await;

    let fts_config = SearchConfig {
      fts_enabled: true,
      rrf_k: 60,
      rerank_candidates: 30,
      ..Default::default()
    };

    let search_params = SearchParams {
      query: "user login authentication".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    let result = search::search(
      &code_ctx,
      search_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("hybrid search should succeed");

    assert!(
      !result.results.is_empty(),
      "should return results for 'user login authentication'"
    );

    // Verify all confidence scores are positive and not NaN
    for (i, item) in result.results.iter().enumerate() {
      if let Some(confidence) = item.confidence {
        assert!(
          confidence > 0.0,
          "result[{i}] confidence should be positive, got {confidence} (symbol: {:?})",
          item.symbol_name
        );
        assert!(!confidence.is_nan(), "result[{i}] confidence should not be NaN");
        assert!(
          !confidence.is_infinite(),
          "result[{i}] confidence should not be infinite"
        );
      }
    }

    // Verify results are ordered by score descending (via confidence as a proxy)
    // Note: In the hybrid path, the rank_score is stored as the score field,
    // so we check that the ordering is monotonically non-increasing.
    let confidences: Vec<f32> = result.results.iter().filter_map(|r| r.confidence).collect();
    if confidences.len() >= 2 {
      for i in 0..confidences.len() - 1 {
        assert!(
          confidences[i] >= confidences[i + 1] - 0.001, // small epsilon for floating point
          "results should be ordered by score descending: result[{i}] ({}) should be >= result[{}] ({})",
          confidences[i],
          i + 1,
          confidences[i + 1]
        );
      }
    }

    // Verify search quality metadata is populated
    let quality = &result.search_quality;
    assert!(
      quality.best_distance >= 0.0,
      "best_distance should be non-negative, got {}",
      quality.best_distance
    );
  }

  /// Test that passing reranker=None gracefully degrades to RRF-only results.
  ///
  /// Validates spec point 13: When reranker is None, search still works
  /// and returns RRF-fused results.
  #[tokio::test]
  async fn test_reranker_none_graceful_degradation() {
    use crate::config::SearchConfig;

    let ctx = TestContext::new().await;
    let code_ctx = CodeContext::new(&ctx.db, ctx.embedding.as_ref());

    ctx
      .index_code(
        "src/cache.rs",
        r#"
/// Get a value from the cache by key.
pub fn cache_get(key: &str) -> Option<Value> {
    CACHE.lock().get(key).cloned()
}

/// Set a value in the cache with TTL.
pub fn cache_set(key: &str, value: Value, ttl: Duration) {
    CACHE.lock().insert(key.to_string(), value, ttl);
}
"#,
        Language::Rust,
      )
      .await;

    let fts_config = SearchConfig {
      fts_enabled: true,
      rrf_k: 60,
      rerank_candidates: 30,
      ..Default::default()
    };

    let search_params = SearchParams {
      query: "cache get value".to_string(),
      language: None,
      limit: Some(10),
      include_context: false,
      visibility: vec![],
      chunk_type: vec![],
      min_caller_count: None,
      adaptive_limit: false,
    };

    // Explicitly pass None for reranker - should work fine
    let result = search::search(
      &code_ctx,
      search_params,
      &RankingConfig::default(),
      Some(&fts_config),
      None,
    )
    .await
    .expect("search without reranker should succeed");

    assert!(
      !result.results.is_empty(),
      "search without reranker should still return results"
    );
  }
}
