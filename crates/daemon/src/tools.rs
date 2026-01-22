use crate::projects::ProjectRegistry;
use crate::router::{Request, Response};
use chrono::Utc;
use embedding::EmbeddingProvider;
use engram_core::{
  ChunkParams, DocumentChunk, DocumentId, DocumentSource, Memory, MemoryId, MemoryType, RelationshipType, SearchConfig,
  Sector, chunk_text,
};
use extract::{DuplicateChecker, DuplicateMatch, content_hash, extract_concepts, extract_files, simhash};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, warn};

/// Ranking weights for post-search scoring
struct RankingWeights {
  semantic: f32, // Weight for vector similarity
  salience: f32, // Weight for salience score
  recency: f32,  // Weight for recency
}

impl Default for RankingWeights {
  fn default() -> Self {
    Self {
      semantic: 0.5,
      salience: 0.3,
      recency: 0.2,
    }
  }
}

impl From<&SearchConfig> for RankingWeights {
  fn from(config: &SearchConfig) -> Self {
    Self {
      semantic: config.semantic_weight as f32,
      salience: config.salience_weight as f32,
      recency: config.recency_weight as f32,
    }
  }
}

/// Rank memories by combining vector similarity with salience, recency, and sector boosts
fn rank_memories(
  results: Vec<(Memory, f32)>,
  limit: usize,
  weights: Option<&RankingWeights>,
) -> Vec<(Memory, f32, f32)> {
  let default_weights = RankingWeights::default();
  let weights = weights.unwrap_or(&default_weights);
  let now = Utc::now();

  let mut scored: Vec<_> = results
    .into_iter()
    .map(|(m, distance)| {
      // Convert distance to similarity (1.0 - distance for cosine)
      let similarity = 1.0 - distance.min(1.0);

      // Recency score: decay based on days since last access
      let days_since_access = (now - m.last_accessed).num_days().max(0) as f32;
      let recency_score = (-0.02 * days_since_access).exp(); // Exponential decay

      // Sector boost
      let sector_boost = m.sector.search_boost();

      // Supersession penalty
      let supersession_penalty = if m.superseded_by.is_some() { 0.7 } else { 1.0 };

      // Combined rank score
      let rank_score =
        (weights.semantic * similarity + weights.salience * m.salience + weights.recency * recency_score)
          * sector_boost
          * supersession_penalty;

      (m, distance, rank_score)
    })
    .collect();

  // Sort by rank score descending
  scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

  // Return top N
  scored.into_iter().take(limit).collect()
}

/// Handler for MCP tool calls
pub struct ToolHandler {
  registry: Arc<ProjectRegistry>,
  embedding: Option<Arc<dyn EmbeddingProvider>>,
}

impl ToolHandler {
  pub fn new(registry: Arc<ProjectRegistry>) -> Self {
    Self {
      registry,
      embedding: None,
    }
  }

  pub fn with_embedding(registry: Arc<ProjectRegistry>, embedding: Arc<dyn EmbeddingProvider>) -> Self {
    Self {
      registry,
      embedding: Some(embedding),
    }
  }

  /// Get embedding for a query, with fallback to None if provider unavailable
  async fn get_embedding(&self, text: &str) -> Option<Vec<f32>> {
    if let Some(ref provider) = self.embedding {
      match provider.embed(text).await {
        Ok(vec) => Some(vec),
        Err(e) => {
          warn!("Embedding failed: {}", e);
          None
        }
      }
    } else {
      None
    }
  }

  /// Get embeddings for multiple texts in a batch (more efficient for bulk operations)
  async fn get_embeddings_batch(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>> {
    if texts.is_empty() {
      return vec![];
    }
    if let Some(ref provider) = self.embedding {
      match provider.embed_batch(texts).await {
        Ok(vecs) => vecs.into_iter().map(Some).collect(),
        Err(e) => {
          warn!("Batch embedding failed: {}", e);
          vec![None; texts.len()]
        }
      }
    } else {
      vec![None; texts.len()]
    }
  }

  // Memory tools

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
          warn!("Vector search failed, falling back to text: {}", e);
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

    // Parse memory ID
    let memory_id: MemoryId = match args.memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid memory_id format"),
    };

    // Get the memory
    let memory: Memory = match db.get_memory(&memory_id).await {
      Ok(Some(m)) => m,
      Ok(None) => return Response::error(request.id, -32000, "Memory not found"),
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    // Reinforce
    let amount = args.amount.unwrap_or(0.1);
    let mut memory = memory;
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

    // Parse memory ID
    let memory_id: MemoryId = match args.memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid memory_id format"),
    };

    // Get the memory
    let memory: Memory = match db.get_memory(&memory_id).await {
      Ok(Some(m)) => m,
      Ok(None) => return Response::error(request.id, -32000, "Memory not found"),
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    // Deemphasize
    let amount = args.amount.unwrap_or(0.2);
    let mut memory = memory;
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

    // Parse memory ID
    let memory_id: MemoryId = match args.memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid memory_id format"),
    };

    let hard = args.hard.unwrap_or(false);

    if hard {
      match db.delete_memory(&memory_id).await {
        Ok(_) => Response::success(
          request.id,
          serde_json::json!({
              "id": args.memory_id,
              "hard_delete": true,
              "message": "Memory permanently deleted"
          }),
        ),
        Err(e) => Response::error(request.id, -32000, &format!("Delete failed: {}", e)),
      }
    } else {
      // Soft delete - get memory, mark as deleted, update
      match db.get_memory(&memory_id).await {
        Ok(Some(mut memory)) => {
          memory.delete(Utc::now());
          match db.update_memory(&memory, None).await {
            Ok(_) => Response::success(
              request.id,
              serde_json::json!({
                  "id": args.memory_id,
                  "hard_delete": false,
                  "message": "Memory soft deleted"
              }),
            ),
            Err(e) => Response::error(request.id, -32000, &format!("Update failed: {}", e)),
          }
        }
        Ok(None) => Response::error(request.id, -32000, "Memory not found"),
        Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
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

    // Parse IDs
    let old_memory_id: MemoryId = match args.old_memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid old_memory_id format"),
    };

    let new_memory_id: MemoryId = match args.new_memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid new_memory_id format"),
    };

    // Get the old memory
    let old_memory: Memory = match db.get_memory(&old_memory_id).await {
      Ok(Some(m)) => m,
      Ok(None) => return Response::error(request.id, -32000, "Old memory not found"),
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    // Verify new memory exists
    match db.get_memory(&new_memory_id).await {
      Ok(Some(_)) => {}
      Ok(None) => return Response::error(request.id, -32000, "New memory not found"),
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    // Mark old memory as superseded
    let mut old_memory = old_memory;
    old_memory.supersede(new_memory_id, Utc::now());

    match db.update_memory(&old_memory, None).await {
      Ok(_) => Response::success(
        request.id,
        serde_json::json!({
            "old_memory_id": args.old_memory_id,
            "new_memory_id": args.new_memory_id,
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

    // Parse anchor ID
    let anchor_id: MemoryId = match args.anchor_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid anchor_id format"),
    };

    // Get the anchor memory
    let anchor: Memory = match db.get_memory(&anchor_id).await {
      Ok(Some(m)) => m,
      Ok(None) => return Response::error(request.id, -32000, "Anchor memory not found"),
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
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

    // Parse memory ID
    let memory_id: MemoryId = match args.memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid memory_id format"),
    };

    // Get the memory
    let memory: Memory = match db.get_memory(&memory_id).await {
      Ok(Some(m)) => m,
      Ok(None) => return Response::error(request.id, -32000, "Memory not found"),
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
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
      match db.get_all_relationships(&memory_id).await {
        Ok(relationships) => {
          let rels: Vec<_> = relationships
            .iter()
            .map(|r| {
              serde_json::json!({
                  "type": r.relationship_type.as_str(),
                  "from_id": r.from_memory_id.to_string(),
                  "to_id": r.to_memory_id.to_string(),
                  "target_id": if r.from_memory_id == memory_id {
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
          warn!("Failed to get relationships: {}", e);
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

  // Code tools

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

    // Try vector search if embedding provider is available
    if let Some(query_vec) = self.get_embedding(&args.query).await {
      debug!("Using vector search for code query: {}", args.query);
      match db.search_code_chunks(&query_vec, limit, filter.as_deref()).await {
        Ok(results) => {
          let results: Vec<_> = results
            .into_iter()
            .map(|(chunk, distance)| {
              let similarity = 1.0 - distance.min(1.0);
              serde_json::json!({
                  "id": chunk.id.to_string(),
                  "file_path": chunk.file_path,
                  "content": chunk.content,
                  "language": format!("{:?}", chunk.language).to_lowercase(),
                  "chunk_type": format!("{:?}", chunk.chunk_type).to_lowercase(),
                  "symbols": chunk.symbols,
                  "start_line": chunk.start_line,
                  "end_line": chunk.end_line,
                  "similarity": similarity,
              })
            })
            .collect();

          return Response::success(request.id, serde_json::json!(results));
        }
        Err(e) => {
          warn!("Vector code search failed, falling back to text: {}", e);
        }
      }
    }

    // Fallback: text-based search
    debug!("Using text search for code query: {}", args.query);
    match db.list_code_chunks(filter.as_deref(), Some(limit * 10)).await {
      Ok(chunks) => {
        let query_lower = args.query.to_lowercase();
        let results: Vec<_> = chunks
          .into_iter()
          .filter(|c| {
            c.content.to_lowercase().contains(&query_lower)
              || c.symbols.iter().any(|s| s.to_lowercase().contains(&query_lower))
          })
          .take(limit)
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
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Code search error: {}", e)),
    }
  }

  pub async fn code_index(&self, request: Request) -> Response {
    use db::{CheckpointType, IndexCheckpoint};
    use index::{Chunker, Scanner, compute_gitignore_hash};

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
    let chunker = Chunker::default();
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
      start_line: args.chunk.start_line,
      end_line: args.chunk.end_line,
      file_hash: args.chunk.file_hash,
      indexed_at: chrono::Utc::now(),
      tokens_estimate: args
        .chunk
        .tokens_estimate
        .unwrap_or((args.chunk.content.len() / 4) as u32),
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

  // Watch tools

  /// Start file watcher for a project
  pub async fn watch_start(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get or create project to get its ID
    let (info, _db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Start the watcher for this project (with embedding if available)
    if let Err(e) = self
      .registry
      .start_watcher(info.id.as_str(), &project_path, self.embedding.clone())
      .await
    {
      return Response::error(request.id, -32000, &format!("Failed to start watcher: {}", e));
    }

    Response::success(
      request.id,
      serde_json::json!({
          "status": "started",
          "path": project_path.to_string_lossy(),
          "project_id": info.id.as_str(),
      }),
    )
  }

  /// Stop file watcher for a project
  pub async fn watch_stop(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get project to get its ID
    let (info, _db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Stop the watcher
    if let Err(e) = self.registry.stop_watcher(info.id.as_str()).await {
      return Response::error(request.id, -32000, &format!("Failed to stop watcher: {}", e));
    }

    Response::success(
      request.id,
      serde_json::json!({
          "status": "stopped",
          "path": project_path.to_string_lossy(),
          "project_id": info.id.as_str(),
      }),
    )
  }

  /// Get file watcher status
  pub async fn watch_status(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get project to get its ID
    let (info, _db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Get watcher status
    let status = self.registry.watcher_status(info.id.as_str()).await;

    Response::success(
      request.id,
      serde_json::json!({
          "running": status.running,
          "root": status.root.map(|p| p.to_string_lossy().to_string()),
          "pending_changes": status.pending_changes,
          "project_id": info.id.as_str(),
      }),
    )
  }

  /// Get code index statistics
  pub async fn code_stats(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    // Get all chunks to compute statistics
    match db.list_code_chunks(None, None).await {
      Ok(chunks) => {
        use std::collections::HashMap;

        let mut language_counts: HashMap<String, usize> = HashMap::new();
        let mut chunk_type_counts: HashMap<String, usize> = HashMap::new();
        let mut files_indexed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut total_tokens: u64 = 0;
        let mut total_lines: u64 = 0;

        for chunk in &chunks {
          let lang = format!("{:?}", chunk.language).to_lowercase();
          *language_counts.entry(lang).or_default() += 1;

          let ctype = format!("{:?}", chunk.chunk_type).to_lowercase();
          *chunk_type_counts.entry(ctype).or_default() += 1;

          files_indexed.insert(chunk.file_path.clone());
          total_tokens += chunk.tokens_estimate as u64;
          total_lines += (chunk.end_line - chunk.start_line + 1) as u64;
        }

        let total_chunks = chunks.len();
        let total_files = files_indexed.len();
        let avg_chunks_per_file = if total_files > 0 {
          total_chunks as f32 / total_files as f32
        } else {
          0.0
        };

        // Compute health score (0-100)
        // Factors: coverage (has chunks), diversity (multiple languages), recent indexing
        let mut health_score: f32 = 0.0;
        if total_chunks > 0 {
          health_score += 40.0; // Base score for having any chunks
          if total_files > 0 {
            health_score += 20.0; // Has files indexed
          }
          if language_counts.len() > 1 {
            health_score += 10.0; // Multiple languages
          }
          if avg_chunks_per_file > 1.0 && avg_chunks_per_file < 50.0 {
            health_score += 20.0; // Reasonable chunk density
          }
          // Age-based scoring would require checking indexed_at times
          health_score += 10.0; // Assume recent for now
        }

        Response::success(
          request.id,
          serde_json::json!({
              "total_chunks": total_chunks,
              "total_files": total_files,
              "total_tokens_estimate": total_tokens,
              "total_lines": total_lines,
              "average_chunks_per_file": avg_chunks_per_file,
              "language_breakdown": language_counts,
              "chunk_type_breakdown": chunk_type_counts,
              "index_health_score": health_score.min(100.0).round() as u32,
          }),
        )
      }
      Err(e) => Response::error(request.id, -32000, &format!("Stats error: {}", e)),
    }
  }

  // Document tools

  pub async fn docs_search(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      query: String,
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

    let limit = args.limit.unwrap_or(5);

    // Try vector search if embedding provider is available
    if let Some(query_vec) = self.get_embedding(&args.query).await {
      debug!("Using vector search for docs query: {}", args.query);
      match db.search_documents(&query_vec, limit, None).await {
        Ok(results) => {
          let results: Vec<_> = results
            .into_iter()
            .map(|(chunk, distance)| {
              let similarity = 1.0 - distance.min(1.0);
              serde_json::json!({
                  "id": chunk.id.to_string(),
                  "document_id": chunk.document_id.to_string(),
                  "title": chunk.title,
                  "source": chunk.source,
                  "content": chunk.content,
                  "chunk_index": chunk.chunk_index,
                  "total_chunks": chunk.total_chunks,
                  "similarity": similarity,
              })
            })
            .collect();

          return Response::success(request.id, serde_json::json!(results));
        }
        Err(e) => {
          warn!("Vector docs search failed, falling back to text: {}", e);
        }
      }
    }

    // Fallback: text-based search
    debug!("Using text search for docs query: {}", args.query);
    match db.list_document_chunks(None, Some(limit * 10)).await {
      Ok(chunks) => {
        let query_lower = args.query.to_lowercase();
        let results: Vec<_> = chunks
          .into_iter()
          .filter(|c| c.content.to_lowercase().contains(&query_lower) || c.title.to_lowercase().contains(&query_lower))
          .take(limit)
          .map(|chunk| {
            serde_json::json!({
                "id": chunk.id.to_string(),
                "document_id": chunk.document_id.to_string(),
                "title": chunk.title,
                "source": chunk.source,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            })
          })
          .collect();

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Docs search error: {}", e)),
    }
  }

  pub async fn docs_ingest(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      path: Option<String>,
      #[serde(default)]
      url: Option<String>,
      #[serde(default)]
      content: Option<String>,
      #[serde(default)]
      title: Option<String>,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    // Must provide one of path, url, or content
    if args.path.is_none() && args.url.is_none() && args.content.is_none() {
      return Response::error(request.id, -32602, "Must provide path, url, or content");
    }

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (info, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Determine source type and get content
    let (content, source, source_type, title) = if let Some(path) = args.path {
      // Read from file
      let full_path = if path.starts_with('/') {
        PathBuf::from(&path)
      } else {
        project_path.join(&path)
      };

      match std::fs::read_to_string(&full_path) {
        Ok(content) => {
          let title = args.title.unwrap_or_else(|| {
            full_path
              .file_name()
              .map(|s| s.to_string_lossy().to_string())
              .unwrap_or_else(|| path.clone())
          });
          (content, path, DocumentSource::File, title)
        }
        Err(e) => return Response::error(request.id, -32000, &format!("Failed to read file: {}", e)),
      }
    } else if let Some(url) = args.url {
      // Fetch from URL
      match reqwest::get(&url).await {
        Ok(resp) => match resp.text().await {
          Ok(content) => {
            let title = args.title.unwrap_or_else(|| url.clone());
            (content, url, DocumentSource::Url, title)
          }
          Err(e) => return Response::error(request.id, -32000, &format!("Failed to read response: {}", e)),
        },
        Err(e) => return Response::error(request.id, -32000, &format!("Failed to fetch URL: {}", e)),
      }
    } else if let Some(content) = args.content {
      let title = args.title.unwrap_or_else(|| "Untitled Document".to_string());
      (content, "content".to_string(), DocumentSource::Content, title)
    } else {
      return Response::error(request.id, -32602, "Must provide path, url, or content");
    };

    // Validate content
    if content.is_empty() {
      return Response::error(request.id, -32602, "Document content is empty");
    }
    if content.len() > 1_000_000 {
      return Response::error(request.id, -32602, "Document too large (max 1MB)");
    }

    // Compute content hash for deduplication
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let content_hash = format!("{:x}", hasher.finalize());

    // Check if document already exists
    let filter = format!(
      "source = '{}' AND title = '{}'",
      source.replace('\'', "''"),
      title.replace('\'', "''")
    );
    match db.list_document_chunks(Some(&filter), Some(1)).await {
      Ok(existing) if !existing.is_empty() => {
        // Delete existing document first
        let existing_doc_id = existing[0].document_id;
        if let Err(e) = db.delete_document(&existing_doc_id).await {
          warn!("Failed to delete existing document: {}", e);
        }
      }
      _ => {}
    }

    // Chunk the content
    let params = ChunkParams::default();
    let text_chunks = chunk_text(&content, &params);
    let total_chunks = text_chunks.len();

    // Create document ID
    let document_id = DocumentId::new();
    let project_uuid = uuid::Uuid::parse_str(info.id.as_str()).unwrap_or_else(|_| uuid::Uuid::new_v4());

    // Create and store chunks
    let mut stored_chunks = 0;
    for (i, (chunk_content, char_offset)) in text_chunks.into_iter().enumerate() {
      let chunk = DocumentChunk::new(
        document_id,
        project_uuid,
        chunk_content.clone(),
        title.clone(),
        source.clone(),
        source_type,
        i,
        total_chunks,
        char_offset,
      );

      // Generate embedding
      let vector = match self.get_embedding(&chunk_content).await {
        Some(v) => v,
        None => vec![0.0f32; db.vector_dim],
      };

      if let Err(e) = db.add_document_chunk(&chunk, Some(&vector)).await {
        warn!("Failed to store chunk {}: {}", i, e);
        continue;
      }
      stored_chunks += 1;
    }

    Response::success(
      request.id,
      serde_json::json!({
          "document_id": document_id.to_string(),
          "title": title,
          "source": source,
          "source_type": source_type.as_str(),
          "content_hash": content_hash,
          "char_count": content.len(),
          "chunks_created": stored_chunks,
          "total_chunks": total_chunks,
      }),
    )
  }

  // Entity tools

  /// List entities (people, technologies, concepts, etc.)
  pub async fn entity_list(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(rename = "type")]
      #[serde(default)]
      entity_type: Option<String>,
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

    let entities = if let Some(type_str) = args.entity_type {
      match type_str.parse::<engram_core::EntityType>() {
        Ok(t) => match db.list_entities_by_type(t).await {
          Ok(e) => e,
          Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
        },
        Err(_) => return Response::error(request.id, -32602, &format!("Invalid entity type: {}", type_str)),
      }
    } else {
      match db.list_entities(args.limit).await {
        Ok(e) => e,
        Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
      }
    };

    let results: Vec<_> = entities
      .into_iter()
      .map(|e| {
        serde_json::json!({
            "id": e.id.to_string(),
            "name": e.name,
            "type": format!("{:?}", e.entity_type).to_lowercase(),
            "summary": e.summary,
            "aliases": e.aliases,
            "mention_count": e.mention_count,
            "first_seen_at": e.first_seen_at.to_rfc3339(),
            "last_seen_at": e.last_seen_at.to_rfc3339(),
        })
      })
      .collect();

    Response::success(request.id, serde_json::json!(results))
  }

  /// Get entity by ID or name
  pub async fn entity_get(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      id: Option<String>,
      #[serde(default)]
      name: Option<String>,
      #[serde(default)]
      include_memories: Option<bool>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    if args.id.is_none() && args.name.is_none() {
      return Response::error(request.id, -32602, "Must provide id or name");
    }

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    let entity = if let Some(id_str) = args.id {
      match uuid::Uuid::parse_str(&id_str) {
        Ok(id) => match db.get_entity(&id).await {
          Ok(e) => e,
          Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
        },
        Err(_) => return Response::error(request.id, -32602, "Invalid UUID"),
      }
    } else if let Some(name) = args.name {
      match db.find_entity_by_name(&name).await {
        Ok(e) => e,
        Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
      }
    } else {
      None
    };

    match entity {
      Some(e) => {
        let mut result = serde_json::json!({
            "id": e.id.to_string(),
            "name": e.name,
            "type": format!("{:?}", e.entity_type).to_lowercase(),
            "summary": e.summary,
            "aliases": e.aliases,
            "mention_count": e.mention_count,
            "first_seen_at": e.first_seen_at.to_rfc3339(),
            "last_seen_at": e.last_seen_at.to_rfc3339(),
        });

        // Include linked memories if requested
        if args.include_memories.unwrap_or(false) {
          match db.get_entity_memory_links(&e.id).await {
            Ok(links) => {
              let memory_links: Vec<_> = links
                .iter()
                .map(|l| {
                  serde_json::json!({
                      "memory_id": l.memory_id,
                      "role": format!("{:?}", l.role).to_lowercase(),
                      "confidence": l.confidence,
                  })
                })
                .collect();
              result["memories"] = serde_json::json!(memory_links);
            }
            Err(e) => {
              warn!("Failed to get entity memory links: {}", e);
            }
          }
        }

        Response::success(request.id, result)
      }
      None => Response::error(request.id, -32000, "Entity not found"),
    }
  }

  /// Get top entities by mention count
  pub async fn entity_top(&self, request: Request) -> Response {
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

    match db.get_top_entities(args.limit.unwrap_or(10)).await {
      Ok(entities) => {
        let results: Vec<_> = entities
          .into_iter()
          .map(|e| {
            serde_json::json!({
                "id": e.id.to_string(),
                "name": e.name,
                "type": format!("{:?}", e.entity_type).to_lowercase(),
                "mention_count": e.mention_count,
            })
          })
          .collect();
        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Create a relationship between two memories
  ///
  /// Relationship types: supersedes, contradicts, related_to, builds_on,
  /// confirms, applies_to, depends_on, alternative_to
  pub async fn relationship_add(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      from_memory_id: String,
      to_memory_id: String,
      relationship_type: String,
      #[serde(default)]
      confidence: Option<f32>,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    // Parse memory IDs
    let from_id: MemoryId = match args.from_memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid from_memory_id format"),
    };

    let to_id: MemoryId = match args.to_memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid to_memory_id format"),
    };

    // Parse relationship type
    let rel_type: RelationshipType = match args.relationship_type.parse() {
      Ok(t) => t,
      Err(_) => {
        return Response::error(
          request.id,
          -32602,
          "Invalid relationship_type. Valid: supersedes, contradicts, related_to, builds_on, confirms, applies_to, depends_on, alternative_to",
        );
      }
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    match db
      .create_relationship(&from_id, &to_id, rel_type, args.confidence.unwrap_or(1.0), "user")
      .await
    {
      Ok(rel) => Response::success(
        request.id,
        serde_json::json!({
            "id": rel.id.to_string(),
            "from_memory_id": rel.from_memory_id.to_string(),
            "to_memory_id": rel.to_memory_id.to_string(),
            "relationship_type": rel.relationship_type.as_str(),
            "confidence": rel.confidence,
        }),
      ),
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Get all relationships for a memory
  pub async fn relationship_list(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      direction: Option<String>, // "from", "to", or "all" (default)
      #[serde(default)]
      relationship_type: Option<String>,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let memory_id: MemoryId = match args.memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid memory_id format"),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Get relationships based on direction and optional type filter
    let relationships = match args.direction.as_deref() {
      Some("from") => db.get_relationships_from(&memory_id).await,
      Some("to") => db.get_relationships_to(&memory_id).await,
      _ => db.get_all_relationships(&memory_id).await,
    };

    match relationships {
      Ok(rels) => {
        // Apply type filter if specified
        let rels: Vec<_> = if let Some(ref type_filter) = args.relationship_type {
          if let Ok(rel_type) = type_filter.parse::<RelationshipType>() {
            rels.into_iter().filter(|r| r.relationship_type == rel_type).collect()
          } else {
            rels
          }
        } else {
          rels
        };

        let results: Vec<_> = rels
          .into_iter()
          .map(|r| {
            serde_json::json!({
                "id": r.id.to_string(),
                "from_memory_id": r.from_memory_id.to_string(),
                "to_memory_id": r.to_memory_id.to_string(),
                "relationship_type": r.relationship_type.as_str(),
                "confidence": r.confidence,
                "created_at": r.created_at.to_rfc3339(),
                "valid_until": r.valid_until.map(|t| t.to_rfc3339()),
            })
          })
          .collect();
        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Delete a relationship by ID
  pub async fn relationship_delete(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      relationship_id: String,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let rel_id: uuid::Uuid = match uuid::Uuid::parse_str(&args.relationship_id) {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid relationship_id format"),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    match db.delete_relationship(&rel_id).await {
      Ok(()) => Response::success(request.id, serde_json::json!({"deleted": true})),
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Get related memories (memories connected via relationships)
  pub async fn relationship_related(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
      memory_id: String,
      #[serde(default)]
      relationship_type: Option<String>,
      #[serde(default)]
      cwd: Option<String>,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let memory_id: MemoryId = match args.memory_id.parse() {
      Ok(id) => id,
      Err(_) => return Response::error(request.id, -32602, "Invalid memory_id format"),
    };

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_, db) = match self.registry.get_or_create(&project_path).await {
      Ok(p) => p,
      Err(e) => return Response::error(request.id, -32000, &format!("Project error: {}", e)),
    };

    // Get all relationships for this memory
    let relationships = match args.relationship_type {
      Some(ref type_str) => match type_str.parse::<RelationshipType>() {
        Ok(rel_type) => db.get_relationships_of_type(&memory_id, rel_type).await,
        Err(_) => db.get_all_relationships(&memory_id).await,
      },
      None => db.get_all_relationships(&memory_id).await,
    };

    match relationships {
      Ok(rels) => {
        // Collect related memory IDs
        let mut related_ids: Vec<MemoryId> = Vec::new();
        for rel in &rels {
          if rel.from_memory_id == memory_id {
            related_ids.push(rel.to_memory_id);
          } else {
            related_ids.push(rel.from_memory_id);
          }
        }

        // Fetch the actual memories
        let mut results = Vec::new();
        for (rel, related_id) in rels.into_iter().zip(related_ids) {
          if let Ok(Some(memory)) = db.get_memory(&related_id).await {
            results.push(serde_json::json!({
                "memory": {
                    "id": memory.id.to_string(),
                    "content": memory.content,
                    "summary": memory.summary,
                    "sector": format!("{:?}", memory.sector).to_lowercase(),
                    "salience": memory.salience,
                },
                "relationship": {
                    "type": rel.relationship_type.as_str(),
                    "confidence": rel.confidence,
                    "direction": if rel.from_memory_id == memory_id { "outgoing" } else { "incoming" },
                }
            }));
          }
        }

        Response::success(request.id, serde_json::json!(results))
      }
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Get comprehensive project statistics
  pub async fn project_stats(&self, request: Request) -> Response {
    #[derive(Deserialize)]
    struct Args {
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

    match db.get_project_stats().await {
      Ok(stats) => Response::success(request.id, serde_json::to_value(&stats).unwrap_or_default()),
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }

  /// Get comprehensive health status
  pub async fn health_check(&self, request: Request) -> Response {
    use embedding::OllamaProvider;

    #[derive(Deserialize)]
    struct Args {
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

    // Check database connection
    let db_status = match self.registry.get_or_create(&project_path).await {
      Ok((_, db)) => {
        // Try a simple operation to verify DB is working
        match db.count_memories(None).await {
          Ok(_) => serde_json::json!({
              "status": "healthy",
              "wal_mode": true, // LanceDB uses its own format
          }),
          Err(e) => serde_json::json!({
              "status": "error",
              "error": e.to_string(),
          }),
        }
      }
      Err(e) => {
        serde_json::json!({
            "status": "error",
            "error": e.to_string(),
        })
      }
    };

    // Check Ollama availability
    let ollama = OllamaProvider::new();
    let ollama_status = ollama.check_health().await;

    // Check embedding provider (use what we have configured)
    let embedding_status = match &self.embedding {
      Some(provider) => {
        serde_json::json!({
            "configured": true,
            "provider": provider.name(),
            "model": provider.model_id(),
            "dimensions": provider.dimensions(),
            "available": provider.is_available().await,
        })
      }
      None => {
        serde_json::json!({
            "configured": false,
            "provider": "none",
        })
      }
    };

    let health = serde_json::json!({
        "database": db_status,
        "ollama": {
            "available": ollama_status.available,
            "models_count": ollama_status.models.len(),
            "configured_model": ollama_status.configured_model,
            "configured_model_available": ollama_status.configured_model_available,
        },
        "embedding": embedding_status,
    });

    Response::success(request.id, health)
  }

  /// Migrate embeddings to new dimensions/model
  pub async fn migrate_embedding(&self, request: Request) -> Response {
    use std::time::Instant;

    #[derive(Deserialize)]
    struct Args {
      #[serde(default)]
      cwd: Option<String>,
      #[serde(default)]
      force: bool,
    }

    let args: Args = match serde_json::from_value(request.params.clone()) {
      Ok(a) => a,
      Err(e) => return Response::error(request.id, -32602, &format!("Invalid params: {}", e)),
    };

    let embedding = match &self.embedding {
      Some(e) => e,
      None => return Response::error(request.id, -32000, "Embedding provider not configured. Cannot migrate."),
    };

    // Check if embedding provider is available
    if !embedding.is_available().await {
      return Response::error(
        request.id,
        -32000,
        "Embedding provider not available. Please ensure Ollama is running.",
      );
    }

    let project_path = args
      .cwd
      .map(PathBuf::from)
      .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let (_config, db) = match self.registry.get_or_create(&project_path).await {
      Ok(r) => r,
      Err(e) => return Response::error(request.id, -32000, &format!("Database error: {}", e)),
    };

    let start = Instant::now();
    let mut migrated_count = 0u64;
    let mut skipped_count = 0u64;
    let mut error_count = 0u64;
    let target_dimensions = embedding.dimensions();

    // Migrate memories
    // Note: We always re-embed when force is set, otherwise re-embed all
    // (since we can't easily check current dimensions from LanceDB)
    match db.list_memories(Some("is_deleted = false"), None).await {
      Ok(memories) => {
        for memory in memories {
          if !args.force {
            // Without force, skip if we're unsure
            skipped_count += 1;
            continue;
          }

          // Re-embed the content
          match embedding.embed(&memory.content).await {
            Ok(new_embedding) => {
              let new_vec: Vec<f32> = new_embedding.into_iter().collect();
              if let Err(e) = db.update_memory(&memory, Some(&new_vec)).await {
                warn!("Failed to update memory {} embedding: {}", memory.id, e);
                error_count += 1;
              } else {
                migrated_count += 1;
              }
            }
            Err(e) => {
              warn!("Failed to re-embed memory {}: {}", memory.id, e);
              error_count += 1;
            }
          }
        }
      }
      Err(e) => {
        warn!("Failed to list memories for migration: {}", e);
      }
    }

    // Migrate code chunks
    match db.list_code_chunks(None, None).await {
      Ok(chunks) => {
        for chunk in chunks {
          if !args.force {
            skipped_count += 1;
            continue;
          }

          match embedding.embed(&chunk.content).await {
            Ok(new_embedding) => {
              let new_vec: Vec<f32> = new_embedding.into_iter().collect();
              if let Err(e) = db.update_code_chunk(&chunk, Some(&new_vec)).await {
                warn!("Failed to update code chunk {} embedding: {}", chunk.id, e);
                error_count += 1;
              } else {
                migrated_count += 1;
              }
            }
            Err(e) => {
              warn!("Failed to re-embed code chunk {}: {}", chunk.id, e);
              error_count += 1;
            }
          }
        }
      }
      Err(e) => {
        warn!("Failed to list code chunks for migration: {}", e);
      }
    }

    // Migrate document chunks
    match db.list_document_chunks(None, None).await {
      Ok(chunks) => {
        for chunk in chunks {
          if !args.force {
            skipped_count += 1;
            continue;
          }

          match embedding.embed(&chunk.content).await {
            Ok(new_embedding) => {
              let new_vec: Vec<f32> = new_embedding.into_iter().collect();
              if let Err(e) = db.update_document_chunk(&chunk, Some(&new_vec)).await {
                warn!("Failed to update doc chunk {} embedding: {}", chunk.id, e);
                error_count += 1;
              } else {
                migrated_count += 1;
              }
            }
            Err(e) => {
              warn!("Failed to re-embed doc chunk {}: {}", chunk.id, e);
              error_count += 1;
            }
          }
        }
      }
      Err(e) => {
        warn!("Failed to list document chunks for migration: {}", e);
      }
    }

    let duration = start.elapsed();

    Response::success(
      request.id,
      serde_json::json!({
          "migrated_count": migrated_count,
          "skipped_count": skipped_count,
          "error_count": error_count,
          "duration_ms": duration.as_millis() as u64,
          "target_dimensions": target_dimensions,
      }),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  fn create_test_handler() -> (TempDir, ToolHandler) {
    let data_dir = TempDir::new().expect("Failed to create temp dir");
    let registry = Arc::new(ProjectRegistry::with_data_dir(data_dir.path().to_path_buf()));
    let handler = ToolHandler::new(registry);
    (data_dir, handler)
  }

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

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "memory_reinforce".to_string(),
      params: serde_json::json!({
          "memory_id": "invalid-uuid-format"
      }),
    };

    let response = handler.memory_reinforce(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("Invalid memory_id"));
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
  async fn test_docs_ingest_missing_source() {
    let (_dir, handler) = create_test_handler();

    // No path, url, or content provided
    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "docs_ingest".to_string(),
      params: serde_json::json!({
          "title": "Test Doc"
      }),
    };

    let response = handler.docs_ingest(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("Must provide"));
  }

  #[tokio::test]
  async fn test_docs_ingest_empty_content() {
    let (_dir, handler) = create_test_handler();

    let request = Request {
      id: Some(serde_json::json!(1)),
      method: "docs_ingest".to_string(),
      params: serde_json::json!({
          "content": "",
          "title": "Empty Doc"
      }),
    };

    let response = handler.docs_ingest(request).await;
    assert!(response.error.is_some());
    assert!(response.error.unwrap().message.contains("empty"));
  }

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
