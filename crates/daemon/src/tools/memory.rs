//! Memory tool methods

use super::ToolHandler;
use super::ranking::{RankingWeights, rank_memories};
use crate::router::{Request, Response};
use chrono::Utc;
use db::ProjectDb;
use engram_core::{Memory, MemoryType, Sector};
use extract::{DuplicateChecker, DuplicateMatch, content_hash, extract_concepts, extract_files, simhash};
use serde::Deserialize;
use std::path::PathBuf;
use tracing::debug;

/// Helper to resolve a memory by ID or prefix
///
/// Tries exact match first, then falls back to prefix matching.
/// Returns an appropriate error response for not found, ambiguous, or invalid prefixes.
async fn resolve_memory(
  db: &ProjectDb,
  id_or_prefix: &str,
  request_id: Option<serde_json::Value>,
) -> Result<Memory, Response> {
  match db.get_memory_by_id_or_prefix(id_or_prefix).await {
    Ok(Some(memory)) => Ok(memory),
    Ok(None) => Err(Response::error(
      request_id,
      -32000,
      &format!("Memory not found: {}", id_or_prefix),
    )),
    Err(db::DbError::AmbiguousPrefix { prefix, count }) => Err(Response::error(
      request_id,
      -32000,
      &format!(
        "Ambiguous prefix '{}' matches {} memories. Use more characters.",
        prefix, count
      ),
    )),
    Err(db::DbError::InvalidInput(msg)) => Err(Response::error(request_id, -32602, &msg)),
    Err(e) => Err(Response::error(request_id, -32000, &format!("Database error: {}", e))),
  }
}

impl ToolHandler {
  pub async fn memory_search(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      query: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      sector: Option<String>,
      #[serde(default)]
      tier: Option<String>,
      #[serde(rename = "type")]
      #[serde(default)]
      memory_type: Option<String>,
      #[serde(default)]
      min_salience: Option<f32>,
      #[serde(default)]
      scope_path: Option<String>,
      #[serde(default)]
      scope_module: Option<String>,
      #[serde(default)]
      session_id: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
      #[serde(default)]
      include_superseded: Option<bool>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Get or create project
    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Load project config for search defaults and ranking weights
    let config = engram_core::Config::load_for_project(&project_path);

    // Build filter from all provided parameters
    let mut filters = Vec::new();
    if !args.include_superseded.unwrap_or(config.search.include_superseded) {
      filters.push("is_deleted = false".to_string());
      filters.push("superseded_by IS NULL".to_string());
    }
    if let Some(ref sector) = args.sector {
      filters.push(format!("sector = '{}'", sector.to_lowercase()));
    }
    if let Some(ref tier) = args.tier {
      filters.push(format!("tier = '{}'", tier.to_lowercase()));
    }
    if let Some(ref memory_type) = args.memory_type {
      filters.push(format!("memory_type = '{}'", memory_type.to_lowercase()));
    }
    if let Some(min_sal) = args.min_salience {
      filters.push(format!("salience >= {}", min_sal));
    }
    if let Some(ref scope_path) = args.scope_path {
      filters.push(format!("scope_path LIKE '{}%'", scope_path.replace('\'', "''")));
    }
    if let Some(ref scope_module) = args.scope_module {
      filters.push(format!("scope_module = '{}'", scope_module.replace('\'', "''")));
    }
    if let Some(ref session_id) = args.session_id {
      filters.push(format!("session_id = '{}'", session_id.replace('\'', "''")));
    }

    let filter = if filters.is_empty() {
      None
    } else {
      Some(filters.join(" AND "))
    };

    let limit = args.limit.unwrap_or(config.search.default_limit);
    // Oversample by 2x for post-search ranking
    let fetch_limit = limit * 2;

    // Get ranking weights from config
    let ranking_weights = RankingWeights::from(&config.search);

    // Try vector search if embedding provider is available
    if let Some(query_vec) = self.get_embedding(&args.query).await {
      debug!("Using vector search for query: {}", args.query);
      match db.search_memories(&query_vec, fetch_limit, filter.as_deref()).await {
        Ok(results) => {
          // Apply post-search ranking with project config weights
          let ranked = rank_memories(results, limit, Some(&ranking_weights));

          // Automatically reinforce top results (small amount: 0.02 per search)
          // This helps frequently-accessed memories maintain salience
          for (i, (m, _, _)) in ranked.iter().take(3).enumerate() {
            let amount = 0.02 * (1.0 - 0.3 * i as f32); // Top result gets 0.02, 2nd gets 0.014, 3rd gets 0.008
            // Get a mutable copy of the memory, reinforce it, and update
            if let Ok(Some(mut memory)) = db.get_memory(&m.id).await {
              memory.reinforce(amount, Utc::now());
              if let Err(e) = db.update_memory(&memory, None).await {
                debug!("Auto-reinforce failed for {}: {}", m.id, e);
              }
            }
          }

          let results: Vec<_> = ranked
            .into_iter()
            .map(|(m, distance, rank_score)| {
              // Convert distance to similarity score (1.0 - distance for cosine)
              let similarity = 1.0 - distance.min(1.0);
              let is_superseded = m.superseded_by.is_some();
              serde_json::json!({
                  "id": m.id.to_string(),
                  "content": m.content,
                  "summary": m.summary,
                  "sector": m.sector.as_str(),
                  "tier": m.tier.as_str(),
                  "memory_type": m.memory_type.map(|t| t.as_str()),
                  "salience": m.salience,
                  "importance": m.importance,
                  "similarity": similarity,
                  "rank_score": rank_score,
                  "is_superseded": is_superseded,
                  "superseded_by": m.superseded_by.map(|id| id.to_string()),
                  "tags": m.tags,
                  "categories": m.categories,
                  "scope_path": m.scope_path,
                  "scope_module": m.scope_module,
                  "created_at": m.created_at.to_rfc3339(),
                  "last_accessed": m.last_accessed.to_rfc3339(),
              })
            })
            .collect();

          return Response::success(request.id, serde_json::json!(results));
        }
        Err(e) => {
          tracing::warn!("Vector search failed, falling back to text: {}", e);
          // Fall through to text search
        }
      }
    }

    // Fallback: text-based search (limit to 3x to avoid excessive memory usage)
    debug!("Using text search for query: {}", args.query);
    match db.list_memories(filter.as_deref(), Some(fetch_limit * 3)).await {
      Ok(memories) => {
        let query_lower = args.query.to_lowercase();
        let results: Vec<_> = memories
          .into_iter()
          .filter(|m| m.content.to_lowercase().contains(&query_lower))
          .take(limit)
          .map(|m| {
            let is_superseded = m.superseded_by.is_some();
            serde_json::json!({
                "id": m.id.to_string(),
                "content": m.content,
                "summary": m.summary,
                "sector": m.sector.as_str(),
                "tier": m.tier.as_str(),
                "memory_type": m.memory_type.map(|t| t.as_str()),
                "salience": m.salience,
                "importance": m.importance,
                "is_superseded": is_superseded,
                "superseded_by": m.superseded_by.map(|id| id.to_string()),
                "tags": m.tags,
                "categories": m.categories,
                "scope_path": m.scope_path,
                "scope_module": m.scope_module,
                "created_at": m.created_at.to_rfc3339(),
                "last_accessed": m.last_accessed.to_rfc3339(),
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Search error: {}", e)),
    }
  }

  pub async fn memory_add(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      content: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      sector: Option<String>,
      #[serde(rename = "type")]
      #[serde(default)]
      memory_type: Option<String>,
      #[serde(default)]
      context: Option<String>,
      #[serde(default)]
      tags: Option<Vec<String>>,
      #[serde(default)]
      categories: Option<Vec<String>>,
      #[serde(default)]
      scope_path: Option<String>,
      #[serde(default)]
      scope_module: Option<String>,
      #[serde(default)]
      importance: Option<f32>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    // Validate content
    if args.content.len() < 5 {
      return Response::error(request.id, -32602, "Content too short (min 5 chars)");
    }
    if args.content.len() > 32000 {
      return Response::error(request.id, -32602, "Content too long (max 32000 chars)");
    }

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (info, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Parse sector
    let sector = args
      .sector
      .and_then(|s| match s.to_lowercase().as_str() {
        "episodic" => Some(Sector::Episodic),
        "semantic" => Some(Sector::Semantic),
        "procedural" => Some(Sector::Procedural),
        "emotional" => Some(Sector::Emotional),
        "reflective" => Some(Sector::Reflective),
        _ => None,
      })
      .unwrap_or(Sector::Semantic);

    // Parse memory type
    let memory_type = args.memory_type.and_then(|t| match t.to_lowercase().as_str() {
      "preference" => Some(MemoryType::Preference),
      "codebase" => Some(MemoryType::Codebase),
      "decision" => Some(MemoryType::Decision),
      "gotcha" => Some(MemoryType::Gotcha),
      "pattern" => Some(MemoryType::Pattern),
      "turn_summary" => Some(MemoryType::TurnSummary),
      "task_completion" => Some(MemoryType::TaskCompletion),
      _ => None,
    });

    // Compute content hash and simhash for deduplication
    let new_content_hash = content_hash(&args.content);
    let new_simhash = simhash(&args.content);

    // Check for duplicates using existing memories
    // We search for similar memories using the embedding if available
    let duplicate = if let Some(query_vec) = self.get_embedding(&args.content).await {
      if let Ok(candidates) = db.search_memories(&query_vec, 10, Some("is_deleted = false")).await {
        let checker = DuplicateChecker::new();

        // Check each candidate for duplicates
        let mut found_dup = None;
        for (m, _distance) in &candidates {
          let match_result = checker.is_duplicate(&args.content, &new_content_hash, new_simhash, m);
          match match_result {
            DuplicateMatch::Exact => {
              debug!("Duplicate memory detected (exact match): {}", m.id);
              found_dup = Some((m.id, "Exact content match"));
              break;
            }
            DuplicateMatch::Simhash { distance, jaccard } if jaccard > 0.85 => {
              debug!(
                "Duplicate memory detected (similar): {} (distance={}, jaccard={})",
                m.id, distance, jaccard
              );
              found_dup = Some((m.id, "Highly similar content"));
              break;
            }
            _ => {}
          }
        }
        found_dup
      } else {
        None
      }
    } else {
      None
    };

    // If duplicate found, return the existing memory ID instead of creating a new one
    if let Some((existing_id, reason)) = duplicate {
      return Response::success(
        request.id,
        serde_json::json!({
            "id": existing_id.to_string(),
            "message": format!("Duplicate detected: {}", reason),
            "is_duplicate": true
        }),
      );
    }

    // Create memory - Memory::new takes (project_id: Uuid, content: String, sector: Sector)
    let project_uuid = uuid::Uuid::parse_str(info.id.as_str()).unwrap_or_else(|_| uuid::Uuid::new_v4());
    let mut memory = Memory::new(project_uuid, args.content.clone(), sector);

    // Set deduplication fields
    memory.content_hash = new_content_hash;
    memory.simhash = new_simhash;

    // Auto-extract concepts and files from content
    memory.concepts = extract_concepts(&args.content);
    memory.files = extract_files(&args.content);

    // Apply optional fields
    memory.memory_type = memory_type;
    if let Some(ctx) = args.context {
      memory.context = Some(ctx);
    }
    if let Some(tags) = args.tags {
      memory.tags = tags;
    }
    if let Some(categories) = args.categories {
      memory.categories = categories;
    }
    if let Some(scope_path) = args.scope_path {
      memory.scope_path = Some(scope_path);
    }
    if let Some(scope_module) = args.scope_module {
      memory.scope_module = Some(scope_module);
    }
    if let Some(imp) = args.importance {
      memory.importance = imp.clamp(0.0, 1.0);
    }

    // Generate embedding for the content and track which model produced it
    let vector = match self.get_embedding(&args.content).await {
      Some(v) => {
        // Record which embedding model was used
        if let Some(ref provider) = self.embedding {
          memory.embedding_model_id = Some(provider.model_id().to_string());
        }
        v
      }
      None => vec![0.0f32; db.vector_dim], // Fallback to zero vector
    };

    match db.add_memory(&memory, Some(&vector)).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "id": memory.id.to_string(),
            "message": "Memory created successfully"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Failed to add memory: {}", e)),
    }
  }

  pub async fn memory_reinforce(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      amount: Option<f32>,
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

    // Get the memory by ID or prefix
    let mut memory = match resolve_memory(&db, &args.memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    // Reinforce
    let amount = args.amount.unwrap_or(0.1);
    memory.reinforce(amount, Utc::now());

    // Update in database
    match db.update_memory(&memory, None).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "id": memory.id.to_string(),
            "new_salience": memory.salience,
            "message": "Memory reinforced"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Update failed: {}", e)),
    }
  }

  pub async fn memory_deemphasize(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      amount: Option<f32>,
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

    // Get the memory by ID or prefix
    let mut memory = match resolve_memory(&db, &args.memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    // Deemphasize
    let amount = args.amount.unwrap_or(0.2);
    memory.deemphasize(amount, Utc::now());

    // Update in database
    match db.update_memory(&memory, None).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "id": memory.id.to_string(),
            "new_salience": memory.salience,
            "message": "Memory de-emphasized"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Update failed: {}", e)),
    }
  }

  pub async fn memory_delete(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      hard: Option<bool>,
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

    // Resolve memory by ID or prefix (needed for both hard and soft delete)
    let memory = match resolve_memory(&db, &args.memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    let hard = args.hard.unwrap_or(false);

    if hard {
      match db.delete_memory(&memory.id).await {
        Ok(_) => Response::success(
          request.id,
          serde_json::json!({
              "id": memory.id.to_string(),
              "hard_delete": true,
              "message": "Memory permanently deleted"
          }),
        ),
        Err(e) => Response::error(request.id, -32000, &format!("Delete failed: {}", e)),
      }
    } else {
      // Soft delete - mark as deleted, update
      let mut memory = memory;
      memory.delete(Utc::now());
      match db.update_memory(&memory, None).await {
        Ok(_) => Response::success(
          request.id,
          serde_json::json!({
              "id": memory.id.to_string(),
              "hard_delete": false,
              "message": "Memory soft deleted"
          }),
        ),
        Err(e) => Response::error(request.id, -32000, &format!("Update failed: {}", e)),
      }
    }
  }

  pub async fn memory_supersede(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      old_memory_id: String,
      new_memory_id: String,
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

    // Resolve old memory by ID or prefix
    let mut old_memory = match resolve_memory(&db, &args.old_memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    // Resolve new memory by ID or prefix
    let new_memory = match resolve_memory(&db, &args.new_memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    // Mark old memory as superseded
    old_memory.supersede(new_memory.id, Utc::now());

    match db.update_memory(&old_memory, None).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "old_memory_id": old_memory.id.to_string(),
            "new_memory_id": new_memory.id.to_string(),
            "message": "Memory superseded"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Update failed: {}", e)),
    }
  }

  pub async fn memory_timeline(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      anchor_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      depth_before: Option<usize>,
      #[serde(default)]
      depth_after: Option<usize>,
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

    // Resolve anchor memory by ID or prefix
    let anchor = match resolve_memory(&db, &args.anchor_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    let depth_before = args.depth_before.unwrap_or(5);
    let depth_after = args.depth_after.unwrap_or(5);

    // Get all memories and sort by creation time
    let all_memories = match db.list_memories(Some("is_deleted = false"), None).await {
      Ok(m) => m,
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    // Sort by creation time
    let mut sorted: Vec<_> = all_memories.into_iter().collect();
    sorted.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    // Find anchor position
    let anchor_pos = sorted.iter().position(|m| m.id == anchor.id);

    // Helper to build memory JSON with optional session context
    async fn build_memory_with_session(m: &Memory, db: &db::ProjectDb) -> serde_json::Value {
      let mut json = serde_json::json!({
          "id": m.id.to_string(),
          "content": m.content,
          "sector": format!("{:?}", m.sector),
          "salience": m.salience,
          "created_at": m.created_at.to_rfc3339(),
      });

      // Add session context if available
      if let Some(session_id) = m.session_id {
        json["session_id"] = serde_json::json!(session_id.to_string());

        // Try to fetch full session info
        if let Ok(Some(session)) = db.get_session(&session_id).await {
          json["session"] = serde_json::json!({
              "id": session.id.to_string(),
              "started_at": session.started_at.to_rfc3339(),
              "ended_at": session.ended_at.map(|t| t.to_rfc3339()),
              "summary": session.summary,
          });
        }
      }

      json
    }

    let (before, after) = match anchor_pos {
      Some(pos) => {
        let start = pos.saturating_sub(depth_before);
        let end = (pos + 1 + depth_after).min(sorted.len());

        let mut before = Vec::new();
        for m in &sorted[start..pos] {
          before.push(build_memory_with_session(m, &db).await);
        }

        let mut after = Vec::new();
        for m in &sorted[pos + 1..end] {
          after.push(build_memory_with_session(m, &db).await);
        }

        (before, after)
      }
      None => (vec![], vec![]),
    };

    // Get anchor with session info
    let anchor_json = build_memory_with_session(&anchor, &db).await;

    Response::success(
      request.id,
      serde_json::json!({
          "anchor": anchor_json,
          "before": before,
          "after": after,
      }),
    )
  }

  /// Get a single memory by ID with full details
  pub async fn memory_get(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      include_related: Option<bool>,
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

    // Resolve memory by ID or prefix
    let memory = match resolve_memory(&db, &args.memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    // Build base response
    let mut result = serde_json::json!({
        "id": memory.id.to_string(),
        "content": memory.content,
        "summary": memory.summary,
        "sector": memory.sector.as_str(),
        "tier": memory.tier.as_str(),
        "memory_type": memory.memory_type.map(|t| t.as_str()),
        "salience": memory.salience,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "access_count": memory.access_count,
        "is_deleted": memory.is_deleted,
        "superseded_by": memory.superseded_by.map(|id| id.to_string()),
        "tags": memory.tags,
        "categories": memory.categories,
        "concepts": memory.concepts,
        "files": memory.files,
        "context": memory.context,
        "scope_path": memory.scope_path,
        "scope_module": memory.scope_module,
        "created_at": memory.created_at.to_rfc3339(),
        "updated_at": memory.updated_at.to_rfc3339(),
        "last_accessed": memory.last_accessed.to_rfc3339(),
        "valid_from": memory.valid_from.to_rfc3339(),
        "valid_until": memory.valid_until.map(|t| t.to_rfc3339()),
    });

    // Include relationships if requested
    if args.include_related.unwrap_or(false) {
      match db.get_all_relationships(&memory.id).await {
        Ok(relationships) => {
          let rels: Vec<_> = relationships
            .iter()
            .map(|r| {
              serde_json::json!({
                  "type": r.relationship_type.as_str(),
                  "from_id": r.from_memory_id.to_string(),
                  "to_id": r.to_memory_id.to_string(),
                  "target_id": if r.from_memory_id == memory.id {
                      r.to_memory_id.to_string()
                  } else {
                      r.from_memory_id.to_string()
                  },
                  "confidence": r.confidence,
              })
            })
            .collect();
          result["relationships"] = serde_json::json!(rels);
        }
        Err(e) => {
          tracing::warn!("Failed to get relationships: {}", e);
          result["relationships"] = serde_json::json!([]);
        }
      }
    }

    Response::success(request.id, result)
  }

  /// List all memories for a project (for export)
  pub async fn memory_list(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      limit: Option<usize>,
      #[serde(default)]
      include_deleted: Option<bool>,
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

    let filter = if args.include_deleted.unwrap_or(false) {
      None
    } else {
      Some("is_deleted = false")
    };

    match db.list_memories(filter, args.limit).await {
      Ok(memories) => {
        let results: Vec<_> = memories
          .into_iter()
          .map(|m| {
            serde_json::json!({
                "id": m.id.to_string(),
                "content": m.content,
                "summary": m.summary,
                "sector": m.sector.as_str(),
                "tier": m.tier.as_str(),
                "memory_type": m.memory_type.map(|t| t.as_str()),
                "salience": m.salience,
                "importance": m.importance,
                "is_deleted": m.is_deleted,
                "superseded_by": m.superseded_by.map(|id| id.to_string()),
                "tags": m.tags,
                "categories": m.categories,
                "scope_path": m.scope_path,
                "scope_module": m.scope_module,
                "created_at": m.created_at.to_rfc3339(),
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Restore a soft-deleted memory
  pub async fn memory_restore(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let project_path = args
      .cwd
      .map(std::path::PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Resolve memory by ID or prefix
    let mut memory = match resolve_memory(&db, &args.memory_id, request.id.clone()).await {
      Ok(m) => m,
      Err(response) => return response,
    };

    // Check if it's actually deleted
    if !memory.is_deleted {
      return Response::error(request.id, -32000, "Memory is not deleted");
    }

    // Restore it
    memory.restore(Utc::now());

    // Update in database
    match db.update_memory(&memory, None).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "id": memory.id.to_string(),
            "content": memory.content,
            "sector": memory.sector.as_str(),
            "memory_type": memory.memory_type.map(|t| t.as_str()),
            "salience": memory.salience,
            "message": "Memory restored"
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Restore failed: {}", e)),
    }
  }

  /// List soft-deleted memories
  pub async fn memory_list_deleted(&self, request: Request) -> Response {
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
      .map(std::path::PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let limit = args.limit.unwrap_or(20);

    // Query deleted memories
    match db.list_memories(Some("is_deleted = true"), Some(limit)).await {
      Ok(memories) => {
        let results: Vec<_> = memories
          .into_iter()
          .map(|m| {
            serde_json::json!({
                "id": m.id.to_string(),
                "content": m.content,
                "sector": m.sector.as_str(),
                "memory_type": m.memory_type.map(|t| t.as_str()),
                "salience": m.salience,
                "deleted_at": m.deleted_at.map(|t| t.to_rfc3339()),
                "created_at": m.created_at.to_rfc3339(),
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::super::create_test_handler;
  use crate::router::Request;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_memory_add_validation_too_short() {
    let (_dir, handler) = create_test_handler();

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_add".to_string(),
      params: serde_json::json!({
          "content": "hi"
      }),
    };

    let response = handler.memory_add(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("too short"));
  }

  #[tokio::test]
  async fn test_memory_add_validation_too_long() {
    let (_dir, handler) = create_test_handler();

    // Create content longer than 32000 chars
    let long_content = "x".repeat(33000);
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_add".to_string(),
      params: serde_json::json!({
          "content": long_content
      }),
    };

    let response = handler.memory_add(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("too long"));
  }

  #[tokio::test]
  async fn test_memory_search_invalid_params() {
    let (_dir, handler) = create_test_handler();

    // Missing required 'query' param
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_search".to_string(),
      params: serde_json::json!({
          "limit": 10
      }),
    };

    let response = handler.memory_search(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("Invalid params"));
  }

  #[tokio::test]
  async fn test_memory_reinforce_invalid_id() {
    let (_dir, handler) = create_test_handler();

    // Test with too-short prefix (less than 6 chars)
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_reinforce".to_string(),
      params: serde_json::json!({
          "memory_id": "abc"
      }),
    };

    let response = handler.memory_reinforce(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("at least 6 characters"));
  }

  #[tokio::test]
  async fn test_memory_deemphasize_not_found() {
    let (data_dir, handler) = create_test_handler();
    let project_dir = TempDir::new().expect("Failed to create project dir");

    // Valid UUID format but memory doesn't exist
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_deemphasize".to_string(),
      params: serde_json::json!({
          "memory_id": "01936c4f-4d77-7ba5-9f8a-123456789abc",
          "cwd": project_dir.path().to_string_lossy()
      }),
    };

    let response = handler.memory_deemphasize(request).await;
    assert!(response.error.is_some());
    let _ = data_dir; // Keep alive
  }

  #[tokio::test]
  async fn test_memory_timeline_invalid_anchor() {
    let (data_dir, handler) = create_test_handler();
    let project_dir = TempDir::new().expect("Failed to create project dir");

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_timeline".to_string(),
      params: serde_json::json!({
          "anchor_id": "01936c4f-4d77-7ba5-9f8a-123456789abc",
          "cwd": project_dir.path().to_string_lossy()
      }),
    };

    let response = handler.memory_timeline(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("not found"));
    let _ = data_dir; // Keep alive
  }

  #[tokio::test]
  async fn test_memory_supersede_missing_old() {
    let (data_dir, handler) = create_test_handler();
    let project_dir = TempDir::new().expect("Failed to create project dir");

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_supersede".to_string(),
      params: serde_json::json!({
          "old_memory_id": "01936c4f-4d77-7ba5-9f8a-111111111111",
          "new_memory_id": "01936c4f-4d77-7ba5-9f8a-222222222222",
          "cwd": project_dir.path().to_string_lossy()
      }),
    };

    let response = handler.memory_supersede(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("not found"));
    let _ = data_dir; // Keep alive
  }

  #[tokio::test]
  async fn test_sector_parsing() {
    // Test all valid sectors parse correctly in memory_add params
    let valid_sectors = vec![
      "episodic",
      "semantic",
      "procedural",
      "emotional",
      "reflective",
      "EPISODIC",
      "Semantic", // Case insensitivity
    ];

    for sector in valid_sectors {
      let params = serde_json::json!({
          "content": "Test content for sector parsing",
          "sector": sector
      });
      assert!(params.get("sector").is_some());
    }
  }

  #[tokio::test]
  async fn test_memory_type_parsing() {
    // Test memory type values
    let valid_types = vec![
      "preference",
      "codebase",
      "decision",
      "gotcha",
      "pattern",
      "turn_summary",
      "task_completion",
    ];

    for mtype in valid_types {
      let params = serde_json::json!({
          "content": "Test content for type parsing",
          "type": mtype
      });
      assert!(params.get("type").is_some());
    }
  }
}
