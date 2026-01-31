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

    let search_result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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

    let result = search::search(&code_ctx, search_params, &RankingConfig::default())
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
    // (embedding similarity depends on model, but should be > 0.5 for exact match)
    assert!(
      confidence > 0.3,
      "Exact symbol match should have confidence > 0.3, got {}",
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

    let relevant_result = search::search(&code_ctx, relevant_params, &RankingConfig::default())
      .await
      .expect("search should succeed");

    // Search quality should be reasonable for relevant query
    let quality = &relevant_result.search_quality;
    assert!(
      quality.best_distance < 0.7,
      "Relevant query should have best_distance < 0.7, got {}",
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

    let unrelated_result = search::search(&code_ctx, unrelated_params, &RankingConfig::default())
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

    let adaptive_result = search::search(&code_ctx, adaptive_params, &RankingConfig::default())
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

    let normal_result = search::search(&code_ctx, normal_params, &RankingConfig::default())
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
}
