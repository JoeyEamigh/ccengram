//! System-level tool methods (stats, health, migration)

use super::ToolHandler;
use crate::router::{Request, Response};
use embedding::OllamaProvider;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;
use tracing::warn;

impl ToolHandler {
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

    // Check Ollama availability and context length
    let ollama = OllamaProvider::new();
    let ollama_status = ollama.check_health().await;

    // Check for context length mismatch (P0-D)
    let mut context_length_warning: Option<String> = None;
    if ollama_status.available && ollama_status.configured_model_available
      && let Some(ref config) = self.embedding_config {
        // Query model's actual context length
        let model_provider = OllamaProvider::new()
          .with_url(&config.ollama_url)
          .with_model(&config.model, config.dimensions);
        if let Some(actual_context_length) = model_provider.get_model_context_length().await
          && config.context_length > actual_context_length {
            let msg = format!(
              "Configured context_length ({}) exceeds model's actual context length ({}). \
               Batch embedding may fail or produce errors. Consider reducing context_length in config.",
              config.context_length, actual_context_length
            );
            warn!("{}", msg);
            context_length_warning = Some(msg);
          }
      }

    // Check embedding provider (use what we have configured)
    let embedding_status = match &self.embedding {
      Some(provider) => {
        let mut status = serde_json::json!({
            "configured": true,
            "provider": provider.name(),
            "model": provider.model_id(),
            "dimensions": provider.dimensions(),
            "available": provider.is_available().await,
        });
        // Add context_length info if available
        if let Some(ref config) = self.embedding_config {
          status["context_length"] = serde_json::json!(config.context_length);
          if let Some(batch_size) = config.max_batch_size {
            status["max_batch_size"] = serde_json::json!(batch_size);
          }
        }
        if let Some(ref warning) = context_length_warning {
          status["warning"] = serde_json::json!(warning);
        }
        status
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
