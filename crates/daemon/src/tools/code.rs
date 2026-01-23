//! Code indexing and search tool methods

use super::ToolHandler;
use crate::router::{Request, Response};
use db::{CheckpointType, IndexCheckpoint, ProjectDb};
use engram_core::{CodeChunk, MemoryType};
use index::{Chunker, Scanner, compute_gitignore_hash};
use parser::import_matches_file;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tracing::{debug, warn};

// ============================================================================
// QUERY EXPANSION FOR CODE EXPLORATION
// ============================================================================

/// Expand exploration queries with related terms
///
/// When exploring a codebase, users often have vague queries. This expands
/// common concepts into multiple related terms to improve recall.
fn expand_query(query: &str) -> String {
  // Common expansion maps for code exploration
  let expansions: HashMap<&str, &[&str]> = HashMap::from([
    // Authentication & Authorization
    (
      "auth",
      &[
        "authentication",
        "authorization",
        "login",
        "session",
        "token",
        "jwt",
        "oauth",
        "credential",
        "password",
        "user",
      ][..],
    ),
    (
      "authentication",
      &[
        "auth",
        "login",
        "session",
        "token",
        "jwt",
        "oauth",
        "credential",
        "password",
      ][..],
    ),
    (
      "login",
      &["auth", "authentication", "signin", "session", "credential"][..],
    ),
    // Error Handling
    (
      "error",
      &[
        "Error",
        "Result",
        "Err",
        "Ok",
        "unwrap",
        "expect",
        "anyhow",
        "thiserror",
        "exception",
        "try",
        "catch",
        "panic",
      ][..],
    ),
    (
      "errors",
      &["error", "Error", "Result", "Err", "exception", "handling"][..],
    ),
    // Database & Storage
    (
      "database",
      &[
        "db",
        "sql",
        "query",
        "connection",
        "pool",
        "migrate",
        "schema",
        "table",
        "postgres",
        "sqlite",
        "mysql",
      ][..],
    ),
    ("db", &["database", "sql", "query", "connection", "storage"][..]),
    (
      "storage",
      &["database", "db", "file", "persist", "save", "store", "cache"][..],
    ),
    // Testing
    (
      "test",
      &[
        "#[test]",
        "#[cfg(test)]",
        "assert",
        "mock",
        "fixture",
        "expect",
        "spec",
        "unit",
        "integration",
      ][..],
    ),
    ("tests", &["test", "#[test]", "assert", "mock", "spec"][..]),
    // Configuration
    (
      "config",
      &[
        "configuration",
        "settings",
        "env",
        "environment",
        "options",
        "Config",
        "configure",
      ][..],
    ),
    (
      "configuration",
      &["config", "settings", "env", "setup", "initialize"][..],
    ),
    // API & HTTP
    (
      "api",
      &[
        "endpoint",
        "route",
        "handler",
        "request",
        "response",
        "http",
        "rest",
        "controller",
      ][..],
    ),
    (
      "http",
      &[
        "api", "request", "response", "endpoint", "route", "handler", "server", "client",
      ][..],
    ),
    ("endpoint", &["api", "route", "handler", "path"][..]),
    // Async & Concurrency
    (
      "async",
      &["await", "Future", "spawn", "tokio", "async_trait", "concurrent"][..],
    ),
    (
      "concurrent",
      &["async", "thread", "spawn", "parallel", "mutex", "lock", "channel"][..],
    ),
    // Parsing & Serialization
    (
      "parse",
      &["parser", "parsing", "deserialize", "decode", "read", "extract"][..],
    ),
    (
      "serialize",
      &["serialization", "encode", "json", "serde", "write", "format"][..],
    ),
    // Validation
    (
      "validate",
      &["validation", "check", "verify", "assert", "constraint", "rule"][..],
    ),
    ("validation", &["validate", "check", "verify", "constraint"][..]),
    // Logging & Monitoring
    (
      "log",
      &["logging", "tracing", "debug", "info", "warn", "error", "trace", "span"][..],
    ),
    ("logging", &["log", "tracing", "debug", "info", "warn"][..]),
    // Cache
    (
      "cache",
      &["caching", "memoize", "store", "ttl", "expire", "invalidate"][..],
    ),
    // Search & Index
    ("search", &["find", "query", "lookup", "index", "match", "filter"][..]),
    ("index", &["indexing", "search", "lookup", "scan"][..]),
    // Memory & State
    (
      "memory",
      &["Memory", "state", "store", "persist", "salience", "recall"][..],
    ),
    ("state", &["memory", "store", "context", "session", "persist"][..]),
    // Embedding & Vector
    (
      "embedding",
      &["embed", "vector", "similarity", "semantic", "encode"][..],
    ),
    ("vector", &["embedding", "similarity", "cosine", "distance"][..]),
  ]);

  let query_lower = query.to_lowercase();
  let mut expanded_terms: HashSet<String> = HashSet::new();

  // Always include original query terms
  for term in query.split_whitespace() {
    expanded_terms.insert(term.to_string());
  }

  // Add expansions for matching terms
  for (key, related) in &expansions {
    if query_lower.contains(key) {
      for term in *related {
        expanded_terms.insert((*term).to_string());
      }
    }
  }

  // Combine into a single query string
  expanded_terms.into_iter().collect::<Vec<_>>().join(" ")
}

/// Detect query intent and return adjusted query + hints
fn detect_query_intent(query: &str) -> (String, Option<&'static str>) {
  let lower = query.to_lowercase();

  // "how does X work" -> focus on X implementation
  if lower.starts_with("how does") || lower.starts_with("how do") {
    let cleaned = lower
      .trim_start_matches("how does")
      .trim_start_matches("how do")
      .trim_end_matches("work")
      .trim_end_matches("?")
      .trim();
    return (cleaned.to_string(), Some("implementation"));
  }

  // "where is X used" / "what uses X" -> look for callers
  if lower.starts_with("where is") && lower.contains("used")
    || lower.starts_with("what uses")
    || lower.starts_with("who uses")
    || lower.starts_with("what calls")
  {
    let cleaned = lower
      .trim_start_matches("where is")
      .trim_start_matches("what uses")
      .trim_start_matches("who uses")
      .trim_start_matches("what calls")
      .trim_end_matches("used")
      .trim_end_matches("?")
      .trim();
    return (cleaned.to_string(), Some("callers"));
  }

  // "what is X" / "explain X" -> look for definition
  if lower.starts_with("what is") || lower.starts_with("explain") {
    let cleaned = lower
      .trim_start_matches("what is")
      .trim_start_matches("explain")
      .trim_end_matches("?")
      .trim();
    return (cleaned.to_string(), Some("definition"));
  }

  (query.to_string(), None)
}

// ============================================================================
// SYMBOL MATCH BOOSTING & RANKING
// ============================================================================

/// Calculate boost factor based on symbol/metadata matches
fn calculate_symbol_boost(chunk: &CodeChunk, query_terms: &[&str]) -> f32 {
  let mut boost = 0.0f32;

  for term in query_terms {
    let term_lower = term.to_lowercase();

    // Symbol match (highest boost)
    for symbol in &chunk.symbols {
      if symbol.to_lowercase() == term_lower {
        boost += 0.4; // Exact match
      } else if symbol.to_lowercase().contains(&term_lower) {
        boost += 0.2; // Partial match
      }
    }

    // Definition name match
    if let Some(ref name) = chunk.definition_name {
      if name.to_lowercase() == term_lower {
        boost += 0.35;
      } else if name.to_lowercase().contains(&term_lower) {
        boost += 0.15;
      }
    }

    // Imports match (medium boost)
    for import in &chunk.imports {
      if import.to_lowercase().contains(&term_lower) {
        boost += 0.1;
        break; // Only count once per term
      }
    }

    // Calls match (medium boost)
    for call in &chunk.calls {
      if call.to_lowercase() == term_lower {
        boost += 0.15;
        break;
      }
    }

    // File path match (low boost)
    if chunk.file_path.to_lowercase().contains(&term_lower) {
      boost += 0.05;
    }
  }

  // Cap the boost to avoid runaway scores
  boost.min(1.0)
}

/// Calculate importance factor based on definition visibility
fn calculate_importance(chunk: &CodeChunk) -> f32 {
  match chunk.visibility.as_deref() {
    Some("pub") | Some("export") | Some("export default") | Some("public") => 1.0,
    Some("pub(crate)") | Some("protected") => 0.8,
    Some("private") | Some("pub(super)") => 0.6,
    _ => 0.7, // Unknown visibility
  }
}

/// Rank code search results by combining multiple signals
fn rank_results(results: Vec<(CodeChunk, f32)>, query: &str) -> Vec<(CodeChunk, f32)> {
  let query_terms: Vec<&str> = query.split_whitespace().collect();

  let mut scored: Vec<(CodeChunk, f32)> = results
    .into_iter()
    .map(|(chunk, distance)| {
      let vector_similarity = 1.0 - distance.min(1.0);
      let symbol_boost = calculate_symbol_boost(&chunk, &query_terms);
      let importance = calculate_importance(&chunk);

      // Weighted combination
      // vector_similarity: 0.5 (semantic relevance)
      // symbol_boost: 0.30 (exact matches matter a lot for exploration)
      // importance: 0.20 (public APIs are usually more important)
      let final_score = vector_similarity * 0.50 + symbol_boost * 0.30 + importance * 0.20;

      (chunk, final_score)
    })
    .collect();

  // Sort by final score descending
  scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

  scored
}

// ============================================================================
// CONTEXT EXPANSION
// ============================================================================

/// Result with expanded context for exploration
#[derive(serde::Serialize)]
struct EnrichedCodeResult {
  // Core chunk info
  id: String,
  file_path: String,
  content: String,
  language: String,
  chunk_type: String,
  symbols: Vec<String>,
  start_line: u32,
  end_line: u32,
  similarity: f32,

  // Definition metadata
  #[serde(skip_serializing_if = "Option::is_none")]
  definition_kind: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  definition_name: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  visibility: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  signature: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  docstring_preview: Option<String>,

  // Context for exploration
  #[serde(skip_serializing_if = "Vec::is_empty")]
  file_imports: Vec<String>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  sibling_symbols: Vec<SiblingInfo>,
}

#[derive(serde::Serialize)]
struct SiblingInfo {
  symbol: String,
  kind: Option<String>,
  start_line: u32,
}

/// Create enriched result with file context
async fn enrich_with_context(
  chunk: CodeChunk,
  score: f32,
  db: &ProjectDb,
  include_siblings: bool,
) -> EnrichedCodeResult {
  // Get siblings from same file for context
  let mut sibling_symbols = Vec::new();
  let mut file_imports = Vec::new();

  if include_siblings && let Ok(siblings) = db.get_chunks_for_file(&chunk.file_path).await {
    for sib in siblings {
      if sib.id == chunk.id {
        continue;
      }

      // Collect file-level imports from import chunks
      if sib.chunk_type == engram_core::ChunkType::Import {
        file_imports.extend(sib.imports.clone());
      }

      // Collect sibling symbol info
      for symbol in &sib.symbols {
        sibling_symbols.push(SiblingInfo {
          symbol: symbol.clone(),
          kind: sib.definition_kind.clone(),
          start_line: sib.start_line,
        });
      }
    }

    // Limit siblings to avoid bloat
    sibling_symbols.truncate(10);
    file_imports.sort();
    file_imports.dedup();
  }

  // If chunk doesn't have imports, use file-level imports
  if chunk.imports.is_empty() && !file_imports.is_empty() {
    // file_imports is already populated
  } else {
    file_imports = chunk.imports.clone();
  }

  EnrichedCodeResult {
    id: chunk.id.to_string(),
    file_path: chunk.file_path,
    content: chunk.content,
    language: format!("{:?}", chunk.language).to_lowercase(),
    chunk_type: format!("{:?}", chunk.chunk_type).to_lowercase(),
    symbols: chunk.symbols,
    start_line: chunk.start_line,
    end_line: chunk.end_line,
    similarity: score,
    definition_kind: chunk.definition_kind,
    definition_name: chunk.definition_name,
    visibility: chunk.visibility,
    signature: chunk.signature.map(|s| {
      // Clean up multi-line signatures
      s.lines().map(|l| l.trim()).collect::<Vec<_>>().join(" ")
    }),
    docstring_preview: chunk.docstring.map(|d| {
      // Truncate for preview
      if d.len() > 200 { format!("{}...", &d[..200]) } else { d }
    }),
    file_imports,
    sibling_symbols,
  }
}

/// Helper to resolve a code chunk by ID or prefix
///
/// Tries exact match first, then falls back to prefix matching.
/// Returns an appropriate error response for not found, ambiguous, or invalid prefixes.
async fn resolve_code_chunk(
  db: &ProjectDb,
  id_or_prefix: &str,
  request_id: Option<serde_json::Value>,
) -> Result<CodeChunk, Response> {
  match db.get_code_chunk_by_id_or_prefix(id_or_prefix).await {
    Ok(Some(chunk)) => Ok(chunk),
    Ok(None) => Err(Response::error(
      request_id,
      -32000,
      &format!("Code chunk not found: {}", id_or_prefix),
    )),
    Err(db::DbError::AmbiguousPrefix { prefix, count }) => Err(Response::error(
      request_id,
      -32000,
      &format!(
        "Ambiguous prefix '{}' matches {} chunks. Use more characters.",
        prefix, count
      ),
    )),
    Err(db::DbError::InvalidInput(msg)) => Err(Response::error(request_id, -32602, &msg)),
    Err(e) => Err(Response::error(request_id, -32000, &format!("Database error: {}", e))),
  }
}

impl ToolHandler {
  pub async fn code_search(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      query: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      language: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
      /// Disable query expansion for precise searches
      #[serde(default)]
      exact: Option<bool>,
      /// Include file context (imports, siblings) - slightly slower but better for exploration
      #[serde(default)]
      include_context: Option<bool>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Build filter for language if provided
    let filter = args
      .language
      .as_ref()
      .map(|lang| format!("language = '{}'", lang.to_lowercase()));

    let limit = args.limit.unwrap_or(10);
    let exact = args.exact.unwrap_or(false);
    let include_context = args.include_context.unwrap_or(true); // Default to including context for exploration

    // Query processing: detect intent and expand for exploration
    let (processed_query, intent) = detect_query_intent(&args.query);
    let search_query = if exact {
      processed_query.clone()
    } else {
      expand_query(&processed_query)
    };

    debug!(
      "Code search: original='{}', processed='{}', expanded='{}', intent={:?}",
      args.query, processed_query, search_query, intent
    );

    // Try vector search if embedding provider is available
    // Oversample for ranking (fetch more, then rank and trim)
    let oversample = (limit * 3).min(50);

    if let Some(query_vec) = self.get_embedding(&search_query).await {
      debug!("Using vector search with ranking for code query");
      match db.search_code_chunks(&query_vec, oversample, filter.as_deref()).await {
        Ok(results) => {
          // Apply ranking with symbol boost and importance
          let ranked = rank_results(results, &args.query);

          // Take top results and enrich with context
          let mut enriched_results = Vec::new();
          for (chunk, score) in ranked.into_iter().take(limit) {
            let enriched = enrich_with_context(chunk, score, &db, include_context).await;
            enriched_results.push(enriched);
          }

          return Response::success(
            request.id,
            serde_json::json!({
              "results": enriched_results,
              "query_info": {
                "original": args.query,
                "expanded": if exact { None } else { Some(&search_query) },
                "intent": intent,
              }
            }),
          );
        }
        Err(e) => {
          warn!("Vector code search failed, falling back to text: {}", e);
        }
      }
    }

    // Fallback: text-based search with symbol boosting
    debug!("Using text search with ranking for code query");
    match db.list_code_chunks(filter.as_deref(), Some(oversample)).await {
      Ok(chunks) => {
        let query_terms: Vec<&str> = args.query.split_whitespace().collect();

        // Score each chunk based on text matches and symbol boost
        let mut scored: Vec<(CodeChunk, f32)> = chunks
          .into_iter()
          .filter_map(|chunk| {
            let content_lower = chunk.content.to_lowercase();
            let query_lower = args.query.to_lowercase();

            // Must have some text match
            let has_match = content_lower.contains(&query_lower)
              || chunk.symbols.iter().any(|s| s.to_lowercase().contains(&query_lower))
              || chunk
                .definition_name
                .as_ref()
                .is_some_and(|n| n.to_lowercase().contains(&query_lower));

            if !has_match {
              return None;
            }

            let symbol_boost = calculate_symbol_boost(&chunk, &query_terms);
            let importance = calculate_importance(&chunk);

            // For text search, weight symbol matches more heavily
            let score = symbol_boost * 0.5 + importance * 0.3 + 0.2; // Base score for having a match

            Some((chunk, score))
          })
          .collect();

        // Sort by score and take top results
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Enrich with context
        let mut enriched_results = Vec::new();
        for (chunk, score) in scored.into_iter().take(limit) {
          let enriched = enrich_with_context(chunk, score, &db, include_context).await;
          enriched_results.push(enriched);
        }

        Response::success(
          request.id,
          serde_json::json!({
            "results": enriched_results,
            "query_info": {
              "original": args.query,
              "search_mode": "text_fallback",
            }
          }),
        )
      }
      Err(e) => Response::error(request.id, -32000, &format!("Code search error: {}", e)),
    }
  }

  pub async fn code_index(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      force: Option<bool>,
      #[serde(default)]
      dry_run: Option<bool>,
      #[serde(default)]
      resume: Option<bool>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let force = args.force.unwrap_or(false);
    let dry_run = args.dry_run.unwrap_or(false);
    let resume = args.resume.unwrap_or(true); // Resume by default

    debug!(
      "Code index: path={:?}, force={}, dry_run={}, resume={}",
      project_path, force, dry_run, resume
    );

    let (info, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let project_id = info.id.as_str();

    // Load index config for this project
    let config = engram_core::Config::load_for_project(&project_path);

    // Scan the project directory with config
    let scanner = Scanner::new().with_max_file_size(config.index.max_file_size as u64);
    let scan_result = scanner.scan(&project_path, |progress| {
      debug!("Scanning: {} files, current: {:?}", progress.scanned, progress.path);
    });

    // Compute gitignore hash to detect config changes
    let current_gitignore_hash = Some(compute_gitignore_hash(&project_path));

    if dry_run {
      return Response::success(
        request.id,
        serde_json::json!({
            "status": "dry_run",
            "files_found": scan_result.files.len(),
            "skipped": scan_result.skipped_count,
            "total_bytes": scan_result.total_bytes,
            "scan_duration_ms": scan_result.scan_duration.as_millis(),
        }),
      );
    }

    // Check for existing checkpoint
    let mut checkpoint = if resume && !force {
      match db.get_checkpoint(project_id, CheckpointType::Code).await {
        Ok(Some(cp)) => {
          // Check if gitignore changed - if so, invalidate checkpoint
          if cp.gitignore_hash != current_gitignore_hash {
            debug!("Gitignore changed, starting fresh index");
            None
          } else if cp.is_complete {
            debug!("Previous indexing complete, starting fresh");
            None
          } else {
            debug!("Resuming from checkpoint: {}% complete", cp.progress_percent());
            Some(cp)
          }
        }
        Ok(None) => None,
        Err(e) => {
          warn!("Failed to get checkpoint: {}", e);
          None
        }
      }
    } else {
      None
    };

    // If force or no checkpoint, clear existing chunks and create new checkpoint
    if force || checkpoint.is_none() {
      if force {
        for file in &scan_result.files {
          if let Err(e) = db.delete_chunks_for_file(&file.relative_path).await {
            warn!("Failed to clear chunks for {}: {}", file.relative_path, e);
          }
        }
        // Clear any existing checkpoint
        let _ = db.clear_checkpoint(project_id, CheckpointType::Code).await;
      }

      // Create new checkpoint with all files
      let pending: Vec<String> = scan_result.files.iter().map(|f| f.relative_path.clone()).collect();
      let mut new_cp = IndexCheckpoint::new(project_id, CheckpointType::Code, pending);
      new_cp.gitignore_hash = current_gitignore_hash;
      if let Err(e) = db.save_checkpoint(&new_cp).await {
        warn!("Failed to save checkpoint: {}", e);
      }
      checkpoint = Some(new_cp);
    }

    // Safety: checkpoint is always set by this point - either from existing checkpoint
    // or from creation in the if block above
    let Some(mut checkpoint) = checkpoint else {
      return Response::error(request.id, -32603, "Internal error: checkpoint not initialized");
    };

    // Build a map of files to process for quick lookup
    let file_map: std::collections::HashMap<_, _> =
      scan_result.files.iter().map(|f| (f.relative_path.clone(), f)).collect();

    // Process only pending files
    let mut chunker = Chunker::default();
    let mut total_chunks = 0;
    let mut indexed_files = 0;
    let mut failed_files = Vec::new();
    let mut save_counter = 0;
    let mut bytes_processed: u64 = 0;

    // Clone pending files to avoid borrow issues
    let pending_to_process: Vec<String> = checkpoint.pending_files.clone();

    // Track indexing start time for performance metrics
    let index_start = std::time::Instant::now();

    for relative_path in &pending_to_process {
      let file = match file_map.get(relative_path) {
        Some(f) => *f,
        None => {
          // File no longer exists, mark as error
          checkpoint.mark_error(relative_path);
          continue;
        }
      };

      // Read file content
      let content = match std::fs::read_to_string(&file.path) {
        Ok(c) => c,
        Err(e) => {
          warn!("Failed to read {}: {}", relative_path, e);
          failed_files.push(relative_path.clone());
          checkpoint.mark_error(relative_path);
          save_counter += 1;
          continue;
        }
      };

      // Track bytes processed for metrics
      bytes_processed += file.size;

      // Chunk the file
      let chunks: Vec<_> = chunker.chunk(&content, relative_path, file.language, &file.checksum);

      // Generate embeddings in batch for better performance
      let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
      let embeddings = self.get_embeddings_batch(&texts).await;

      // Store chunks with their embeddings
      let mut file_success = true;
      for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
        let vector = embedding.unwrap_or_else(|| vec![0.0f32; db.vector_dim]);

        if let Err(e) = db.add_code_chunk(&chunk, Some(&vector)).await {
          warn!("Failed to store chunk for {}: {}", relative_path, e);
          file_success = false;
          break;
        }
        total_chunks += 1;
      }

      if file_success {
        checkpoint.mark_processed(relative_path);
        indexed_files += 1;
      } else {
        checkpoint.mark_error(relative_path);
        failed_files.push(relative_path.clone());
      }

      save_counter += 1;

      // Save checkpoint periodically (every 10 files)
      if save_counter >= 10 {
        if let Err(e) = db.save_checkpoint(&checkpoint).await {
          warn!("Failed to save checkpoint: {}", e);
        }
        save_counter = 0;
      }
    }

    // Mark complete and save final checkpoint
    checkpoint.mark_complete();
    if let Err(e) = db.save_checkpoint(&checkpoint).await {
      warn!("Failed to save final checkpoint: {}", e);
    }

    // Clear checkpoint on successful completion
    if failed_files.is_empty() {
      let _ = db.clear_checkpoint(project_id, CheckpointType::Code).await;
    }

    // Calculate performance metrics
    let index_duration = index_start.elapsed();
    let index_duration_ms = index_duration.as_millis() as u64;
    let files_per_second = if index_duration_ms > 0 && indexed_files > 0 {
      (indexed_files as f64 / index_duration_ms as f64) * 1000.0
    } else {
      0.0
    };
    let total_duration_ms = scan_result.scan_duration.as_millis() as u64 + index_duration_ms;

    Response::success(
      request.id,
      serde_json::json!({
          "status": "complete",
          "files_scanned": scan_result.files.len(),
          "files_indexed": indexed_files,
          "chunks_created": total_chunks,
          "failed_files": failed_files,
          "resumed_from_checkpoint": !pending_to_process.is_empty() && pending_to_process.len() < scan_result.files.len(),
          "scan_duration_ms": scan_result.scan_duration.as_millis(),
          "index_duration_ms": index_duration_ms,
          "total_duration_ms": total_duration_ms,
          "files_per_second": files_per_second,
          "bytes_processed": bytes_processed,
          "total_bytes": scan_result.total_bytes,
      }),
    )
  }

  /// List all code chunks for export
  pub async fn code_list(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    match db.list_code_chunks(None, args.limit).await {
      Ok(chunks) => {
        let results: Vec<_> = chunks
          .into_iter()
          .map(|chunk| {
            serde_json::json!({
                "id": chunk.id.to_string(),
                "file_path": chunk.file_path,
                "content": chunk.content,
                "language": format!("{:?}", chunk.language).to_lowercase(),
                "chunk_type": format!("{:?}", chunk.chunk_type).to_lowercase(),
                "symbols": chunk.symbols,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "file_hash": chunk.file_hash,
                "tokens_estimate": chunk.tokens_estimate,
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("List error: {}", e)),
    }
  }

  /// Import a single code chunk (used during index import)
  pub async fn code_import_chunk(&self, request: Request) -> Response {
    use engram_core::{ChunkType, CodeChunk, Language};

    #[derive(Deserialize)]
    struct ChunkData {
      file_path: String,
      content: String,
      language: String,
      chunk_type: String,
      symbols: Vec<String>,
      start_line: u32,
      end_line: u32,
      file_hash: String,
      #[serde(default)]
      tokens_estimate: Option<u32>,
    }

    #[derive(Deserialize)]
    struct Args {
      chunk: ChunkData,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Parse language from extension-like string
    let language = Language::from_extension(&args.chunk.language).unwrap_or(Language::Rust);

    // Parse chunk type
    let chunk_type = match args.chunk.chunk_type.as_str() {
      "function" => ChunkType::Function,
      "class" => ChunkType::Class,
      "module" => ChunkType::Module,
      "import" => ChunkType::Import,
      _ => ChunkType::Block,
    };

    let chunk = CodeChunk {
      id: uuid::Uuid::now_v7(),
      file_path: args.chunk.file_path,
      content: args.chunk.content.clone(),
      language,
      chunk_type,
      symbols: args.chunk.symbols,
      imports: Vec::new(), // TODO: extract from tree-sitter
      calls: Vec::new(),   // TODO: extract from tree-sitter
      start_line: args.chunk.start_line,
      end_line: args.chunk.end_line,
      file_hash: args.chunk.file_hash,
      indexed_at: chrono::Utc::now(),
      tokens_estimate: args
        .chunk
        .tokens_estimate
        .unwrap_or((args.chunk.content.len() / 4) as u32),
      definition_kind: None,
      definition_name: None,
      visibility: None,
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: None,
    };

    // Generate embedding
    let vector = match self.get_embedding(&chunk.content).await {
      Some(v) => v,
      None => vec![0.0f32; db.vector_dim],
    };

    match db.add_code_chunk(&chunk, Some(&vector)).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "id": chunk.id.to_string(),
            "status": "imported"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Import failed: {}", e)),
    }
  }

  /// Get surrounding lines for a code chunk by reading from filesystem
  pub async fn code_context(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      chunk_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      lines_before: Option<usize>,
      #[serde(default)]
      lines_after: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Cap and default context lines
    let lines_before = args.lines_before.unwrap_or(20).min(500);
    let lines_after = args.lines_after.unwrap_or(20).min(500);

    // Look up chunk by ID or prefix
    let chunk = match db.get_code_chunk_by_id_or_prefix(&args.chunk_id).await {
      Ok(Some(c)) => c,
      Ok(None) => {
        return Response::error(request.id, -32000, &format!("Code chunk not found: {}", args.chunk_id));
      }
      Err(db::DbError::AmbiguousPrefix { prefix, count }) => {
        return Response::error(
          request.id,
          -32000,
          &format!(
            "Ambiguous prefix '{}' matches {} chunks. Use more characters.",
            prefix, count
          ),
        );
      }
      Err(db::DbError::InvalidInput(msg)) => {
        return Response::error(request.id, -32602, &msg);
      }
      Err(e) => {
        return Response::error(request.id, -32000, &format!("Database error: {}", e));
      }
    };

    // Construct the full file path
    let file_path = project_path.join(&chunk.file_path);

    // Read the file
    let file_content = match std::fs::read_to_string(&file_path) {
      Ok(content) => content,
      Err(e) => {
        // File not found or not readable - return chunk content as fallback
        warn!(
          "Could not read file {} for context: {}. Returning stored chunk content.",
          file_path.display(),
          e
        );
        return Response::success(
          request.id,
          serde_json::json!({
            "chunk_id": chunk.id.to_string(),
            "file_path": chunk.file_path,
            "language": format!("{:?}", chunk.language).to_lowercase(),
            "context": {
              "before": { "content": "", "start_line": 0, "end_line": 0 },
              "target": {
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line
              },
              "after": { "content": "", "start_line": 0, "end_line": 0 }
            },
            "total_file_lines": 0,
            "warning": format!("File not readable: {}", e)
          }),
        );
      }
    };

    let lines: Vec<&str> = file_content.lines().collect();
    let total_lines = lines.len();

    // Calculate line ranges (chunk lines are 1-indexed)
    let target_start = (chunk.start_line as usize).saturating_sub(1); // Convert to 0-indexed
    let target_end = (chunk.end_line as usize).min(total_lines); // Exclusive end

    let before_start = target_start.saturating_sub(lines_before);
    let after_end = (target_end + lines_after).min(total_lines);

    // Extract content for each section
    let before_content: String = lines[before_start..target_start].join("\n");
    let target_content: String = lines[target_start..target_end].join("\n");
    let after_content: String = lines[target_end..after_end].join("\n");

    Response::success(
      request.id,
      serde_json::json!({
        "chunk_id": chunk.id.to_string(),
        "file_path": chunk.file_path,
        "language": format!("{:?}", chunk.language).to_lowercase(),
        "context": {
          "before": {
            "content": before_content,
            "start_line": before_start + 1, // Convert back to 1-indexed
            "end_line": target_start        // Exclusive, so equals target_start
          },
          "target": {
            "content": target_content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line
          },
          "after": {
            "content": after_content,
            "start_line": target_end + 1,   // Convert back to 1-indexed
            "end_line": after_end           // This is the count
          }
        },
        "total_file_lines": total_lines
      }),
    )
  }

  /// Get memories (decisions, gotchas, patterns) related to code
  ///
  /// Queries memories by:
  /// 1. File path match (files array, scope_path)
  /// 2. Semantic similarity to code content
  pub async fn code_memories(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      chunk_id: Option<String>,
      #[serde(default)]
      file_path: Option<String>,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let limit = args.limit.unwrap_or(10);

    // Resolve the file path - either from chunk_id or direct file_path
    let (file_path, chunk_content) = if let Some(ref chunk_id) = args.chunk_id {
      let chunk = match resolve_code_chunk(&db, chunk_id, request.id.clone()).await {
        Ok(c) => c,
        Err(response) => return response,
      };
      (chunk.file_path.clone(), Some(chunk.content.clone()))
    } else if let Some(ref fp) = args.file_path {
      (fp.clone(), None)
    } else {
      return Response::error(request.id, -32602, "Must provide chunk_id or file_path");
    };

    let mut memories = Vec::new();
    let mut seen_ids = HashSet::new();

    // Strategy 1: File path match via scope_path
    let scope_filter = format!(
      "is_deleted = false AND (scope_path LIKE '{}%' OR scope_path LIKE '%{}%')",
      file_path.replace('\'', "''"),
      file_path.replace('\'', "''")
    );
    if let Ok(path_matches) = db.list_memories(Some(&scope_filter), Some(limit)).await {
      for m in path_matches {
        if seen_ids.insert(m.id) {
          memories.push((m, 0.8f32, "file_path".to_string()));
        }
      }
    }

    // Strategy 2: Semantic similarity (if chunk content available)
    if let Some(content) = chunk_content
      && let Some(query_vec) = self.get_embedding(&content).await
      && let Ok(similar) = db.search_memories(&query_vec, limit, Some("is_deleted = false")).await
    {
      for (m, distance) in similar {
        if seen_ids.insert(m.id) {
          let similarity = 1.0 - distance.min(1.0);
          memories.push((m, similarity, "semantic".to_string()));
        }
      }
    }

    // Sort by memory type priority, then by score
    memories.sort_by(|a, b| {
      let type_priority = |m: &engram_core::Memory| match m.memory_type {
        Some(MemoryType::Decision) => 0,
        Some(MemoryType::Gotcha) => 1,
        Some(MemoryType::Pattern) => 2,
        Some(MemoryType::Codebase) => 3,
        _ => 4,
      };
      let pa = type_priority(&a.0);
      let pb = type_priority(&b.0);
      if pa != pb {
        pa.cmp(&pb)
      } else {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
      }
    });

    // Take top results
    memories.truncate(limit);

    let results: Vec<_> = memories
      .into_iter()
      .map(|(m, score, source)| {
        serde_json::json!({
          "id": m.id.to_string(),
          "content": m.content,
          "summary": m.summary,
          "memory_type": m.memory_type.map(|t| t.as_str()),
          "sector": m.sector.as_str(),
          "salience": m.salience,
          "score": score,
          "source": source,
          "scope_path": m.scope_path,
          "tags": m.tags,
          "created_at": m.created_at.to_rfc3339(),
        })
      })
      .collect();

    Response::success(
      request.id,
      serde_json::json!({
        "file_path": file_path,
        "memories": results
      }),
    )
  }

  /// Find all code that calls a function/method
  ///
  /// Essential for understanding impact of changes.
  pub async fn code_callers(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      chunk_id: Option<String>,
      #[serde(default)]
      symbol: Option<String>,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let limit = args.limit.unwrap_or(20);

    // Resolve the symbol to search for
    let symbol = if let Some(ref chunk_id) = args.chunk_id {
      let chunk = match resolve_code_chunk(&db, chunk_id, request.id.clone()).await {
        Ok(c) => c,
        Err(response) => return response,
      };
      // Use the first symbol from the chunk
      chunk
        .symbols
        .first()
        .cloned()
        .ok_or_else(|| Response::error(request.id.clone(), -32000, "Chunk has no symbols"))
    } else if let Some(ref sym) = args.symbol {
      Ok(sym.clone())
    } else {
      Err(Response::error(
        request.id.clone(),
        -32602,
        "Must provide chunk_id or symbol",
      ))
    };

    let symbol = match symbol {
      Ok(s) => s,
      Err(response) => return response,
    };

    // Find chunks that call this symbol
    // Note: LanceDB SQL filter with JSON array contains
    let filter = format!("calls LIKE '%\"{}%'", symbol.replace('\'', "''"));
    let callers = match db.list_code_chunks(Some(&filter), Some(limit)).await {
      Ok(c) => c,
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    let results: Vec<_> = callers
      .into_iter()
      .map(|c| {
        serde_json::json!({
          "id": c.id.to_string(),
          "file_path": c.file_path,
          "symbols": c.symbols,
          "start_line": c.start_line,
          "end_line": c.end_line,
          "language": format!("{:?}", c.language).to_lowercase(),
          "chunk_type": format!("{:?}", c.chunk_type).to_lowercase(),
        })
      })
      .collect();

    Response::success(
      request.id,
      serde_json::json!({
        "symbol": symbol,
        "callers": results,
        "count": results.len()
      }),
    )
  }

  /// Find functions that a code chunk calls
  ///
  /// Returns the calls array and attempts to resolve each call to its definition.
  pub async fn code_callees(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      chunk_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let chunk = match resolve_code_chunk(&db, &args.chunk_id, request.id.clone()).await {
      Ok(c) => c,
      Err(response) => return response,
    };

    if chunk.calls.is_empty() {
      return Response::success(
        request.id,
        serde_json::json!({
          "chunk_id": chunk.id.to_string(),
          "calls": chunk.calls,
          "callees": [],
          "unresolved": []
        }),
      );
    }

    let limit_per_call = args.limit.unwrap_or(3);
    let mut callees = Vec::new();
    let mut unresolved = Vec::new();
    let mut seen_ids = HashSet::new();

    // Try to resolve each call to its definition
    for call in &chunk.calls {
      // Search for chunks where symbols contains this call
      let filter = format!("symbols LIKE '%\"{}%'", call.replace('\'', "''"));
      match db.list_code_chunks(Some(&filter), Some(limit_per_call)).await {
        Ok(matches) => {
          if matches.is_empty() {
            unresolved.push(call.clone());
          } else {
            for m in matches {
              if seen_ids.insert(m.id) {
                callees.push(serde_json::json!({
                  "call": call,
                  "id": m.id.to_string(),
                  "file_path": m.file_path,
                  "symbols": m.symbols,
                  "start_line": m.start_line,
                  "end_line": m.end_line,
                  "language": format!("{:?}", m.language).to_lowercase(),
                }));
              }
            }
          }
        }
        Err(_) => {
          unresolved.push(call.clone());
        }
      }
    }

    Response::success(
      request.id,
      serde_json::json!({
        "chunk_id": chunk.id.to_string(),
        "calls": chunk.calls,
        "callees": callees,
        "unresolved": unresolved
      }),
    )
  }

  /// Find code related to a chunk via multiple methods
  ///
  /// Methods: same_file, shared_imports, similar, callers, callees
  pub async fn code_related(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      chunk_id: String,
      #[serde(default)]
      methods: Option<Vec<String>>,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let chunk = match resolve_code_chunk(&db, &args.chunk_id, request.id.clone()).await {
      Ok(c) => c,
      Err(response) => return response,
    };

    let methods: Vec<&str> = args
      .methods
      .as_ref()
      .map(|m| m.iter().map(|s| s.as_str()).collect())
      .unwrap_or_else(|| vec!["same_file", "shared_imports", "similar"]);

    let limit = args.limit.unwrap_or(20);
    let mut related: Vec<(CodeChunk, f32, String)> = Vec::new();
    let mut seen_ids = HashSet::new();
    seen_ids.insert(chunk.id); // Exclude the source chunk

    for method in methods {
      match method {
        "same_file" => {
          if let Ok(siblings) = db.get_chunks_for_file(&chunk.file_path).await {
            for s in siblings {
              if seen_ids.insert(s.id) {
                related.push((s, 0.9, "same_file".to_string()));
              }
            }
          }
        }
        "shared_imports" => {
          // For each import in this chunk, find other chunks that import the same thing
          // Use import resolution to handle NodeNext (.js -> .ts), bundler (extensionless), etc.
          for import in &chunk.imports {
            // First try exact match
            let filter = format!("imports LIKE '%{}%'", import.replace('\'', "''"));
            if let Ok(matches) = db.list_code_chunks(Some(&filter), Some(10)).await {
              for m in matches {
                if seen_ids.insert(m.id) {
                  related.push((m, 0.7, format!("imports:{}", import)));
                }
              }
            }

            // Also find chunks for files that this import resolves to
            // This handles the case where ./utils.js resolves to utils.ts
            if let Ok(all_chunks) = db.list_code_chunks(None, Some(100)).await {
              for m in all_chunks {
                if seen_ids.contains(&m.id) {
                  continue;
                }
                // Check if this import resolves to this chunk's file
                if import_matches_file(import, &m.file_path) && seen_ids.insert(m.id) {
                  related.push((m, 0.75, format!("imports:{} -> {}", import, chunk.file_path)));
                }
              }
            }
          }
        }
        "similar" => {
          if let Some(query_vec) = self.get_embedding(&chunk.content).await
            && let Ok(similar) = db.search_code_chunks(&query_vec, 10, None).await
          {
            for (c, distance) in similar {
              if seen_ids.insert(c.id) {
                let similarity = 1.0 - distance.min(1.0);
                related.push((c, similarity, "similar".to_string()));
              }
            }
          }
        }
        "callers" => {
          if let Some(symbol) = chunk.symbols.first() {
            let filter = format!("calls LIKE '%\"{}%'", symbol.replace('\'', "''"));
            if let Ok(callers) = db.list_code_chunks(Some(&filter), Some(10)).await {
              for c in callers {
                if seen_ids.insert(c.id) {
                  related.push((c, 0.8, "caller".to_string()));
                }
              }
            }
          }
        }
        "callees" => {
          for call in &chunk.calls {
            let filter = format!("symbols LIKE '%\"{}%'", call.replace('\'', "''"));
            if let Ok(matches) = db.list_code_chunks(Some(&filter), Some(5)).await {
              for m in matches {
                if seen_ids.insert(m.id) {
                  related.push((m, 0.8, format!("callee:{}", call)));
                }
              }
            }
          }
        }
        _ => {}
      }
    }

    // Sort by score descending and truncate
    related.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    related.truncate(limit);

    let results: Vec<_> = related
      .into_iter()
      .map(|(c, score, relationship)| {
        serde_json::json!({
          "id": c.id.to_string(),
          "file_path": c.file_path,
          "symbols": c.symbols,
          "start_line": c.start_line,
          "end_line": c.end_line,
          "language": format!("{:?}", c.language).to_lowercase(),
          "chunk_type": format!("{:?}", c.chunk_type).to_lowercase(),
          "score": score,
          "relationship": relationship,
        })
      })
      .collect();

    Response::success(
      request.id,
      serde_json::json!({
        "chunk_id": chunk.id.to_string(),
        "file_path": chunk.file_path,
        "symbols": chunk.symbols,
        "related": results,
        "count": results.len()
      }),
    )
  }

  /// Get comprehensive context for code in ONE call
  ///
  /// Returns: chunk details, callers, callees, sibling functions, related memories, and documentation.
  pub async fn code_context_full(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      chunk_id: Option<String>,
      #[serde(default)]
      file_path: Option<String>,
      #[serde(default)]
      symbol: Option<String>,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit_per_section: Option<usize>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let limit = args.limit_per_section.unwrap_or(5);

    // Resolve the target chunk
    let chunk = if let Some(ref chunk_id) = args.chunk_id {
      match resolve_code_chunk(&db, chunk_id, request.id.clone()).await {
        Ok(c) => c,
        Err(response) => return response,
      }
    } else if let Some(ref file_path) = args.file_path {
      // Try to find chunk by file path and optional symbol
      let file_chunks = match db.get_chunks_for_file(file_path).await {
        Ok(c) => c,
        Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
      };

      if file_chunks.is_empty() {
        return Response::error(request.id, -32000, &format!("No chunks found for file: {}", file_path));
      }

      if let Some(ref symbol) = args.symbol {
        // Find chunk containing this symbol
        match file_chunks.into_iter().find(|c| c.symbols.iter().any(|s| s == symbol)) {
          Some(chunk) => chunk,
          None => return Response::error(request.id, -32000, &format!("Symbol '{}' not found in file", symbol)),
        }
      } else {
        // Return first chunk
        file_chunks.into_iter().next().expect("checked not empty")
      }
    } else {
      return Response::error(request.id, -32602, "Must provide chunk_id or file_path");
    };

    // Gather all context sections
    let chunk_id = chunk.id;
    let file_path = chunk.file_path.clone();

    // 1. Callers - who calls this?
    let callers: Vec<serde_json::Value> = if let Some(symbol) = chunk.symbols.first() {
      let filter = format!("calls LIKE '%\"{}%'", symbol.replace('\'', "''"));
      db.list_code_chunks(Some(&filter), Some(limit))
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|c| c.id != chunk_id)
        .map(|c| {
          serde_json::json!({
            "id": c.id.to_string(),
            "file_path": c.file_path,
            "symbols": c.symbols,
            "start_line": c.start_line,
            "end_line": c.end_line,
          })
        })
        .collect()
    } else {
      vec![]
    };

    // 2. Callees - what does this call?
    let mut callees: Vec<serde_json::Value> = Vec::new();
    let mut unresolved_calls: Vec<String> = Vec::new();
    for call in &chunk.calls {
      let filter = format!("symbols LIKE '%\"{}%'", call.replace('\'', "''"));
      if let Ok(matches) = db.list_code_chunks(Some(&filter), Some(2)).await {
        if matches.is_empty() {
          unresolved_calls.push(call.clone());
        } else {
          for m in matches {
            callees.push(serde_json::json!({
              "call": call,
              "id": m.id.to_string(),
              "file_path": m.file_path,
              "symbols": m.symbols,
              "start_line": m.start_line,
            }));
          }
        }
      }
    }
    callees.truncate(limit);

    // 3. Same file siblings
    let same_file: Vec<serde_json::Value> = db
      .get_chunks_for_file(&file_path)
      .await
      .unwrap_or_default()
      .into_iter()
      .filter(|c| c.id != chunk_id)
      .take(limit)
      .map(|c| {
        serde_json::json!({
          "id": c.id.to_string(),
          "symbols": c.symbols,
          "chunk_type": format!("{:?}", c.chunk_type).to_lowercase(),
          "start_line": c.start_line,
          "end_line": c.end_line,
        })
      })
      .collect();

    // 4. Related memories
    let memories: Vec<serde_json::Value> = {
      let scope_filter = format!(
        "is_deleted = false AND (scope_path LIKE '{}%' OR scope_path LIKE '%{}%')",
        file_path.replace('\'', "''"),
        file_path.replace('\'', "''")
      );
      db.list_memories(Some(&scope_filter), Some(limit))
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|m| {
          serde_json::json!({
            "id": m.id.to_string(),
            "content": m.content,
            "memory_type": m.memory_type.map(|t| t.as_str()),
            "salience": m.salience,
          })
        })
        .collect()
    };

    // 5. Related documentation (semantic search if embedding available)
    let documentation: Vec<serde_json::Value> = if let Some(query_vec) = self.get_embedding(&chunk.content).await {
      db.search_documents(&query_vec, limit, None)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|(doc, distance): (engram_core::DocumentChunk, f32)| {
          serde_json::json!({
            "id": doc.id.to_string(),
            "title": doc.title,
            "content": doc.content,
            "similarity": 1.0 - distance.min(1.0),
          })
        })
        .collect()
    } else {
      vec![]
    };

    Response::success(
      request.id,
      serde_json::json!({
        "chunk": {
          "id": chunk.id.to_string(),
          "file_path": chunk.file_path,
          "content": chunk.content,
          "language": format!("{:?}", chunk.language).to_lowercase(),
          "chunk_type": format!("{:?}", chunk.chunk_type).to_lowercase(),
          "symbols": chunk.symbols,
          "imports": chunk.imports,
          "calls": chunk.calls,
          "start_line": chunk.start_line,
          "end_line": chunk.end_line,
        },
        "callers": callers,
        "callees": callees,
        "unresolved_calls": unresolved_calls,
        "same_file": same_file,
        "memories": memories,
        "documentation": documentation,
      }),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::super::create_test_handler;
  use super::*;
  use crate::router::Request;
  use engram_core::{ChunkType, Language};

  // ============================================================================
  // QUERY EXPANSION TESTS
  // ============================================================================

  #[test]
  fn test_expand_query_auth() {
    let expanded = expand_query("auth");
    assert!(expanded.contains("authentication"));
    assert!(expanded.contains("login"));
    assert!(expanded.contains("token"));
    assert!(expanded.contains("jwt"));
    // Original term should be preserved
    assert!(expanded.contains("auth"));
  }

  #[test]
  fn test_expand_query_database() {
    let expanded = expand_query("database queries");
    assert!(expanded.contains("sql"));
    assert!(expanded.contains("query"));
    assert!(expanded.contains("connection"));
  }

  #[test]
  fn test_expand_query_no_expansion() {
    let expanded = expand_query("foobar baz");
    // Should just contain original terms
    let terms: HashSet<&str> = expanded.split_whitespace().collect();
    assert!(terms.contains("foobar"));
    assert!(terms.contains("baz"));
  }

  #[test]
  fn test_expand_query_multiple_expansions() {
    let expanded = expand_query("auth error");
    assert!(expanded.contains("login"));
    assert!(expanded.contains("Result"));
    assert!(expanded.contains("unwrap"));
  }

  // ============================================================================
  // INTENT DETECTION TESTS
  // ============================================================================

  #[test]
  fn test_detect_intent_how_does() {
    let (query, intent) = detect_query_intent("how does authentication work");
    assert_eq!(query, "authentication");
    assert_eq!(intent, Some("implementation"));
  }

  #[test]
  fn test_detect_intent_how_do() {
    let (query, intent) = detect_query_intent("how do errors get handled?");
    assert_eq!(query, "errors get handled");
    assert_eq!(intent, Some("implementation"));
  }

  #[test]
  fn test_detect_intent_where_used() {
    let (query, intent) = detect_query_intent("where is DatabasePool used");
    assert_eq!(query, "databasepool");
    assert_eq!(intent, Some("callers"));
  }

  #[test]
  fn test_detect_intent_what_uses() {
    let (query, intent) = detect_query_intent("what uses the Memory struct?");
    assert_eq!(query, "the memory struct");
    assert_eq!(intent, Some("callers"));
  }

  #[test]
  fn test_detect_intent_what_is() {
    let (query, intent) = detect_query_intent("what is CodeChunk?");
    assert_eq!(query, "codechunk");
    assert_eq!(intent, Some("definition"));
  }

  #[test]
  fn test_detect_intent_plain_query() {
    let (query, intent) = detect_query_intent("search for memory functions");
    assert_eq!(query, "search for memory functions");
    assert_eq!(intent, None);
  }

  // ============================================================================
  // SYMBOL BOOST TESTS
  // ============================================================================

  fn create_test_chunk(
    symbols: Vec<&str>,
    imports: Vec<&str>,
    calls: Vec<&str>,
    file_path: &str,
    definition_name: Option<&str>,
    visibility: Option<&str>,
  ) -> CodeChunk {
    CodeChunk {
      id: uuid::Uuid::new_v4(),
      file_path: file_path.to_string(),
      content: "test content".to_string(),
      language: Language::Rust,
      chunk_type: ChunkType::Function,
      symbols: symbols.into_iter().map(String::from).collect(),
      imports: imports.into_iter().map(String::from).collect(),
      calls: calls.into_iter().map(String::from).collect(),
      start_line: 1,
      end_line: 10,
      file_hash: "hash123".to_string(),
      indexed_at: chrono::Utc::now(),
      tokens_estimate: 50,
      definition_kind: Some("function".to_string()),
      definition_name: definition_name.map(String::from),
      visibility: visibility.map(String::from),
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: None,
    }
  }

  #[test]
  fn test_symbol_boost_exact_match() {
    let chunk = create_test_chunk(
      vec!["authenticate"],
      vec![],
      vec![],
      "auth.rs",
      Some("authenticate"),
      Some("pub"),
    );

    let boost = calculate_symbol_boost(&chunk, &["authenticate"]);
    // Should get exact symbol match (0.4) + exact definition name match (0.35)
    assert!(boost >= 0.7, "Expected >= 0.7, got {}", boost);
  }

  #[test]
  fn test_symbol_boost_partial_match() {
    let chunk = create_test_chunk(
      vec!["authenticate_user"],
      vec![],
      vec![],
      "auth.rs",
      Some("authenticate_user"),
      Some("pub"),
    );

    let boost = calculate_symbol_boost(&chunk, &["auth"]);
    // Should get partial symbol match (0.2) + partial definition name match (0.15)
    assert!(boost >= 0.35, "Expected >= 0.35, got {}", boost);
  }

  #[test]
  fn test_symbol_boost_import_match() {
    let chunk = create_test_chunk(
      vec!["main"],
      vec!["std::collections::HashMap"],
      vec![],
      "main.rs",
      Some("main"),
      Some("pub"),
    );

    let boost = calculate_symbol_boost(&chunk, &["HashMap"]);
    // Should get import match (0.1)
    assert!(boost >= 0.1, "Expected >= 0.1, got {}", boost);
  }

  #[test]
  fn test_symbol_boost_call_match() {
    let chunk = create_test_chunk(
      vec!["process_data"],
      vec![],
      vec!["validate", "transform"],
      "processor.rs",
      Some("process_data"),
      Some("pub"),
    );

    let boost = calculate_symbol_boost(&chunk, &["validate"]);
    // Should get call match (0.15)
    assert!(boost >= 0.15, "Expected >= 0.15, got {}", boost);
  }

  #[test]
  fn test_symbol_boost_file_path_match() {
    let chunk = create_test_chunk(
      vec!["something"],
      vec![],
      vec![],
      "authentication/handler.rs",
      None,
      None,
    );

    let boost = calculate_symbol_boost(&chunk, &["authentication"]);
    // Should get file path match (0.05)
    assert!(boost >= 0.05, "Expected >= 0.05, got {}", boost);
  }

  #[test]
  fn test_symbol_boost_capped_at_one() {
    let chunk = create_test_chunk(
      vec!["auth", "authenticate", "authorization"],
      vec!["auth_lib"],
      vec!["auth_check"],
      "auth/auth.rs",
      Some("auth"),
      Some("pub"),
    );

    let boost = calculate_symbol_boost(&chunk, &["auth"]);
    assert!(boost <= 1.0, "Boost should be capped at 1.0, got {}", boost);
  }

  // ============================================================================
  // IMPORTANCE TESTS
  // ============================================================================

  #[test]
  fn test_importance_pub() {
    let chunk = create_test_chunk(vec![], vec![], vec![], "test.rs", None, Some("pub"));
    assert_eq!(calculate_importance(&chunk), 1.0);
  }

  #[test]
  fn test_importance_export() {
    let chunk = create_test_chunk(vec![], vec![], vec![], "test.ts", None, Some("export"));
    assert_eq!(calculate_importance(&chunk), 1.0);
  }

  #[test]
  fn test_importance_pub_crate() {
    let chunk = create_test_chunk(vec![], vec![], vec![], "test.rs", None, Some("pub(crate)"));
    assert_eq!(calculate_importance(&chunk), 0.8);
  }

  #[test]
  fn test_importance_private() {
    let chunk = create_test_chunk(vec![], vec![], vec![], "test.rs", None, Some("private"));
    assert_eq!(calculate_importance(&chunk), 0.6);
  }

  #[test]
  fn test_importance_unknown() {
    let chunk = create_test_chunk(vec![], vec![], vec![], "test.rs", None, None);
    assert_eq!(calculate_importance(&chunk), 0.7);
  }

  // ============================================================================
  // RANKING TESTS
  // ============================================================================

  #[test]
  fn test_rank_results_prefers_exact_matches() {
    let chunk_exact = create_test_chunk(
      vec!["authenticate"],
      vec![],
      vec![],
      "auth.rs",
      Some("authenticate"),
      Some("pub"),
    );

    let chunk_partial = create_test_chunk(
      vec!["user_auth_helper"],
      vec![],
      vec![],
      "helpers.rs",
      Some("user_auth_helper"),
      Some("pub"),
    );

    let results = vec![
      (chunk_partial.clone(), 0.1), // Better vector similarity
      (chunk_exact.clone(), 0.3),   // Worse vector similarity but exact symbol match
    ];

    let ranked = rank_results(results, "authenticate");

    // The exact match should be ranked higher despite worse vector similarity
    assert_eq!(ranked[0].0.symbols[0], "authenticate", "Exact match should be first");
  }

  #[test]
  fn test_rank_results_considers_visibility() {
    let chunk_pub = create_test_chunk(vec!["helper"], vec![], vec![], "lib.rs", Some("helper"), Some("pub"));

    let chunk_private = create_test_chunk(
      vec!["helper"],
      vec![],
      vec![],
      "internal.rs",
      Some("helper"),
      Some("private"),
    );

    let results = vec![
      (chunk_private.clone(), 0.2),
      (chunk_pub.clone(), 0.2), // Same vector similarity
    ];

    let ranked = rank_results(results, "helper");

    // Public should be ranked higher due to visibility importance
    assert_eq!(ranked[0].0.visibility.as_deref(), Some("pub"), "Public should be first");
  }

  #[test]
  fn test_rank_results_empty() {
    let results: Vec<(CodeChunk, f32)> = vec![];
    let ranked = rank_results(results, "anything");
    assert!(ranked.is_empty());
  }

  // ============================================================================
  // INTEGRATION TEST
  // ============================================================================

  #[tokio::test]
  async fn test_code_search_invalid_params() {
    let (_dir, handler) = create_test_handler();

    // Missing required 'query' param
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "code_search".to_string(),
      params: serde_json::json!({
          "language": "rust"
      }),
    };

    let response = handler.code_search(request).await;
    assert!(response.error.is_some());
  }
}
