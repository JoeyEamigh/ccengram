//! Entity and relationship tool methods

use super::ToolHandler;
use crate::router::{
  DeletedResult, EntityFullResult, EntityListItem, EntityMemoryLink, MemorySummary, RelatedMemoryItem,
  RelationshipInfo, RelationshipListItem, RelationshipResult, Request, Response, TopEntityItem,
};
use engram_core::{MemoryId, RelationshipType};
use serde::Deserialize;
use std::path::PathBuf;
use tracing::warn;

impl ToolHandler {
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

    let results: Vec<EntityListItem> = entities
      .into_iter()
      .map(|e| EntityListItem {
        id: e.id.to_string(),
        name: e.name,
        entity_type: format!("{:?}", e.entity_type).to_lowercase(),
        summary: e.summary,
        aliases: e.aliases,
        mention_count: e.mention_count,
        first_seen_at: e.first_seen_at.to_rfc3339(),
        last_seen_at: e.last_seen_at.to_rfc3339(),
      })
      .collect();

    Response::success(request.id, results)
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
        // Include linked memories if requested
        let memories = if args.include_memories.unwrap_or(false) {
          match db.get_entity_memory_links(&e.id).await {
            Ok(links) => Some(
              links
                .iter()
                .map(|l| EntityMemoryLink {
                  memory_id: l.memory_id.to_string(),
                  role: format!("{:?}", l.role).to_lowercase(),
                  confidence: l.confidence,
                })
                .collect(),
            ),
            Err(err) => {
              warn!("Failed to get entity memory links: {}", err);
              None
            }
          }
        } else {
          None
        };

        let result = EntityFullResult {
          id: e.id.to_string(),
          name: e.name,
          entity_type: format!("{:?}", e.entity_type).to_lowercase(),
          summary: e.summary,
          aliases: e.aliases,
          mention_count: e.mention_count,
          first_seen_at: e.first_seen_at.to_rfc3339(),
          last_seen_at: e.last_seen_at.to_rfc3339(),
          memories,
        };

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
        let results: Vec<TopEntityItem> = entities
          .into_iter()
          .map(|e| TopEntityItem {
            id: e.id.to_string(),
            name: e.name,
            entity_type: format!("{:?}", e.entity_type).to_lowercase(),
            mention_count: e.mention_count,
          })
          .collect();
        Response::success(request.id, results)
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
        RelationshipResult {
          id: rel.id.to_string(),
          from_memory_id: rel.from_memory_id.to_string(),
          to_memory_id: rel.to_memory_id.to_string(),
          relationship_type: rel.relationship_type.as_str().to_string(),
          confidence: rel.confidence,
        },
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

        let results: Vec<RelationshipListItem> = rels
          .into_iter()
          .map(|r| RelationshipListItem {
            id: r.id.to_string(),
            from_memory_id: r.from_memory_id.to_string(),
            to_memory_id: r.to_memory_id.to_string(),
            relationship_type: r.relationship_type.as_str().to_string(),
            confidence: r.confidence,
            created_at: r.created_at.to_rfc3339(),
            valid_until: r.valid_until.map(|t| t.to_rfc3339()),
          })
          .collect();
        Response::success(request.id, results)
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
      Ok(()) => Response::success(request.id, DeletedResult { deleted: true }),
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
        let mut results: Vec<RelatedMemoryItem> = Vec::new();
        for (rel, related_id) in rels.into_iter().zip(related_ids) {
          if let Ok(Some(mem)) = db.get_memory(&related_id).await {
            results.push(RelatedMemoryItem {
              memory: MemorySummary {
                id: mem.id.to_string(),
                content: mem.content,
                summary: mem.summary,
                sector: format!("{:?}", mem.sector).to_lowercase(),
                salience: mem.salience,
              },
              relationship: RelationshipInfo {
                relationship_type: rel.relationship_type.as_str().to_string(),
                confidence: rel.confidence,
                direction: if rel.from_memory_id == memory_id {
                  "outgoing".to_string()
                } else {
                  "incoming".to_string()
                },
              },
            });
          }
        }

        Response::success(request.id, results)
      }
      Err(e) => Response::error(request.id, -32000, &format!("Database error: {}", e)),
    }
  }
}
