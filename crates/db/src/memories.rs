use arrow_array::{
  Array, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
  UInt32Array, UInt64Array,
};
use chrono::{TimeZone, Utc};
use engram_core::{Memory, MemoryId, MemoryType, Sector, Tier};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::memories_schema;

impl ProjectDb {
  /// Add a new memory to the database
  pub async fn add_memory(&self, memory: &Memory, vector: Option<&[f32]>) -> Result<()> {
    let table = self.memories_table().await?;

    let batch = memory_to_batch(memory, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memories_schema(self.vector_dim));

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get a memory by ID
  pub async fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
    let table = self.memories_table().await?;
    let id_str = id.to_string();

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id_str))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() {
      return Ok(None);
    }

    let batch = &results[0];
    if batch.num_rows() == 0 {
      return Ok(None);
    }

    Ok(Some(batch_to_memory(batch, 0)?))
  }

  /// Update a memory
  pub async fn update_memory(&self, memory: &Memory, vector: Option<&[f32]>) -> Result<()> {
    let table = self.memories_table().await?;

    // LanceDB uses upsert via merge_insert
    let batch = memory_to_batch(memory, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memories_schema(self.vector_dim));

    // For upsert, delete old and insert new (merge_insert API is awkward)
    // First try to delete existing record with same id
    let _ = table.delete(&format!("id = '{}'", memory.id)).await;

    // Then insert the new record
    table.add(Box::new(batches)).execute().await?;

    Ok(())
  }

  /// Batch update multiple memories (more efficient than individual updates)
  ///
  /// This is optimized for operations like decay where many memories need their
  /// salience updated. Uses bulk delete + insert instead of individual upserts.
  pub async fn batch_update_memories(&self, memories: &[Memory]) -> Result<usize> {
    if memories.is_empty() {
      return Ok(0);
    }

    let table = self.memories_table().await?;

    // Build single delete query for all IDs
    let ids: Vec<String> = memories.iter().map(|m| format!("'{}'", m.id)).collect();
    let delete_filter = format!("id IN ({})", ids.join(", "));
    let _ = table.delete(&delete_filter).await;

    // Batch all records into a single insert
    let batches: Vec<_> = memories
      .iter()
      .map(|m| memory_to_batch(m, None, self.vector_dim))
      .collect::<Result<Vec<_>>>()?;

    let merged = arrow::compute::concat_batches(&memories_schema(self.vector_dim), &batches)?;
    let batches = RecordBatchIterator::new(vec![Ok(merged)], memories_schema(self.vector_dim));

    table.add(Box::new(batches)).execute().await?;

    Ok(memories.len())
  }

  /// Delete a memory by ID (hard delete)
  pub async fn delete_memory(&self, id: &MemoryId) -> Result<()> {
    let table = self.memories_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Search memories by vector similarity
  pub async fn search_memories(
    &self,
    query_vector: &[f32],
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(Memory, f32)>> {
    let table = self.memories_table().await?;

    let query = if let Some(f) = filter {
      table.vector_search(query_vector.to_vec())?.limit(limit).only_if(f)
    } else {
      table.vector_search(query_vector.to_vec())?.limit(limit)
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut memories = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        let memory = batch_to_memory(&batch, i)?;
        // Get distance score from _distance column if present
        let distance = batch
          .column_by_name("_distance")
          .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
          .map(|arr| arr.value(i))
          .unwrap_or(0.0);
        memories.push((memory, distance));
      }
    }

    Ok(memories)
  }

  /// List memories with optional filters
  pub async fn list_memories(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<Memory>> {
    let table = self.memories_table().await?;

    let query = match (filter, limit) {
      (Some(f), Some(l)) => table.query().only_if(f).limit(l),
      (Some(f), None) => table.query().only_if(f),
      (None, Some(l)) => table.query().limit(l),
      (None, None) => table.query(),
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut memories = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        memories.push(batch_to_memory(&batch, i)?);
      }
    }

    Ok(memories)
  }

  /// Count memories matching a filter (uses native count_rows for efficiency)
  pub async fn count_memories(&self, filter: Option<&str>) -> Result<usize> {
    let table = self.memories_table().await?;
    let count = table.count_rows(filter.map(|s| s.to_string())).await?;
    Ok(count)
  }
}

/// Convert a Memory to an Arrow RecordBatch
fn memory_to_batch(memory: &Memory, vector: Option<&[f32]>, vector_dim: usize) -> Result<RecordBatch> {
  let id = StringArray::from(vec![memory.id.to_string()]);
  let project_id = StringArray::from(vec![memory.project_id.to_string()]);
  let content = StringArray::from(vec![memory.content.clone()]);
  let summary = StringArray::from(vec![memory.summary.clone()]);
  let sector = StringArray::from(vec![memory.sector.as_str().to_string()]);
  let tier = StringArray::from(vec![memory.tier.as_str().to_string()]);
  let memory_type = StringArray::from(vec![memory.memory_type.map(|t| t.as_str().to_string())]);
  let importance = Float32Array::from(vec![memory.importance]);
  let salience = Float32Array::from(vec![memory.salience]);
  let confidence = Float32Array::from(vec![memory.confidence]);
  let access_count = UInt32Array::from(vec![memory.access_count]);
  let tags = StringArray::from(vec![serde_json::to_string(&memory.tags)?]);
  let concepts = StringArray::from(vec![serde_json::to_string(&memory.concepts)?]);
  let files = StringArray::from(vec![serde_json::to_string(&memory.files)?]);
  let categories = StringArray::from(vec![serde_json::to_string(&memory.categories)?]);
  let context = StringArray::from(vec![memory.context.clone()]);
  let session_id = StringArray::from(vec![memory.session_id.map(|id| id.to_string())]);
  let segment_id = StringArray::from(vec![memory.segment_id.map(|id| id.to_string())]);
  let scope_path = StringArray::from(vec![memory.scope_path.clone()]);
  let scope_module = StringArray::from(vec![memory.scope_module.clone()]);
  let created_at = Int64Array::from(vec![memory.created_at.timestamp_millis()]);
  let updated_at = Int64Array::from(vec![memory.updated_at.timestamp_millis()]);
  let last_accessed = Int64Array::from(vec![memory.last_accessed.timestamp_millis()]);
  let deleted_at = Int64Array::from(vec![memory.deleted_at.map(|t| t.timestamp_millis())]);
  let valid_from = Int64Array::from(vec![memory.valid_from.timestamp_millis()]);
  let valid_until = Int64Array::from(vec![memory.valid_until.map(|t| t.timestamp_millis())]);
  let is_deleted = BooleanArray::from(vec![memory.is_deleted]);
  let content_hash = StringArray::from(vec![memory.content_hash.clone()]);
  let simhash = UInt64Array::from(vec![memory.simhash]);
  let superseded_by = StringArray::from(vec![memory.superseded_by.map(|id| id.to_string())]);
  let decay_rate = Float32Array::from(vec![memory.decay_rate]);
  let next_decay_at = Int64Array::from(vec![memory.next_decay_at.map(|t| t.timestamp_millis())]);
  let embedding_model_id = StringArray::from(vec![memory.embedding_model_id.clone()]);

  // Handle vector - pad or truncate to match expected dimensions
  let vector_arr = if let Some(v) = vector {
    let mut vec_padded = v.to_vec();
    vec_padded.resize(vector_dim, 0.0);
    Some(Float32Array::from(vec_padded))
  } else {
    None
  };

  let vector_list = if let Some(v) = vector_arr {
    let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
    FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(v), None)?
  } else {
    // Create null vector
    let null_vec = Float32Array::from(vec![0.0f32; vector_dim]);
    let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
    FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(null_vec), Some(vec![false].into()))?
  };

  let batch = RecordBatch::try_new(
    memories_schema(vector_dim),
    vec![
      Arc::new(id),
      Arc::new(project_id),
      Arc::new(content),
      Arc::new(summary),
      Arc::new(sector),
      Arc::new(tier),
      Arc::new(memory_type),
      Arc::new(importance),
      Arc::new(salience),
      Arc::new(confidence),
      Arc::new(access_count),
      Arc::new(tags),
      Arc::new(concepts),
      Arc::new(files),
      Arc::new(categories),
      Arc::new(context),
      Arc::new(session_id),
      Arc::new(segment_id),
      Arc::new(scope_path),
      Arc::new(scope_module),
      Arc::new(created_at),
      Arc::new(updated_at),
      Arc::new(last_accessed),
      Arc::new(deleted_at),
      Arc::new(valid_from),
      Arc::new(valid_until),
      Arc::new(is_deleted),
      Arc::new(content_hash),
      Arc::new(simhash),
      Arc::new(superseded_by),
      Arc::new(decay_rate),
      Arc::new(next_decay_at),
      Arc::new(embedding_model_id),
      Arc::new(vector_list),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a Memory
fn batch_to_memory(batch: &RecordBatch, row: usize) -> Result<Memory> {
  let get_string = |name: &str| -> Result<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_optional_string = |name: &str| -> Option<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .and_then(|a| {
        if a.is_null(row) {
          None
        } else {
          Some(a.value(row).to_string())
        }
      })
  };

  let get_f32 = |name: &str| -> Result<f32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_u32 = |name: &str| -> Result<u32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_u64 = |name: &str| -> Result<u64> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_i64 = |name: &str| -> Result<i64> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_optional_i64 = |name: &str| -> Option<i64> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
      .and_then(|a| if a.is_null(row) { None } else { Some(a.value(row)) })
  };

  let get_bool = |name: &str| -> Result<bool> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
      .map(|a| a.value(row))
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_optional_f32 = |name: &str| -> Option<f32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
      .and_then(|a| if a.is_null(row) { None } else { Some(a.value(row)) })
  };

  let id_str = get_string("id")?;
  let project_id_str = get_string("project_id")?;
  let sector_str = get_string("sector")?;
  let tier_str = get_string("tier")?;

  let tags_json = get_string("tags")?;
  let concepts_json = get_string("concepts")?;
  let files_json = get_string("files")?;
  let categories_json = get_optional_string("categories").unwrap_or_else(|| "[]".to_string());

  let sector = sector_str.parse::<Sector>().map_err(DbError::NotFound)?;

  let tier = match tier_str.as_str() {
    "session" => Tier::Session,
    "project" => Tier::Project,
    _ => Tier::Project,
  };

  let memory_type = get_optional_string("memory_type").and_then(|s| match s.as_str() {
    "preference" => Some(MemoryType::Preference),
    "codebase" => Some(MemoryType::Codebase),
    "decision" => Some(MemoryType::Decision),
    "gotcha" => Some(MemoryType::Gotcha),
    "pattern" => Some(MemoryType::Pattern),
    "turn_summary" => Some(MemoryType::TurnSummary),
    "task_completion" => Some(MemoryType::TaskCompletion),
    _ => None,
  });

  let created_at = Utc
    .timestamp_millis_opt(get_i64("created_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid created_at timestamp".into()))?;
  let updated_at = Utc
    .timestamp_millis_opt(get_i64("updated_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid updated_at timestamp".into()))?;
  let last_accessed = Utc
    .timestamp_millis_opt(get_i64("last_accessed")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid last_accessed timestamp".into()))?;
  let valid_from = Utc
    .timestamp_millis_opt(get_i64("valid_from")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid valid_from timestamp".into()))?;
  let valid_until = get_optional_i64("valid_until").and_then(|ts| Utc.timestamp_millis_opt(ts).single());
  let deleted_at = get_optional_i64("deleted_at").and_then(|ts| Utc.timestamp_millis_opt(ts).single());
  let next_decay_at = get_optional_i64("next_decay_at").and_then(|ts| Utc.timestamp_millis_opt(ts).single());

  let superseded_by = get_optional_string("superseded_by").and_then(|s| s.parse::<MemoryId>().ok());

  Ok(Memory {
    id: id_str.parse().map_err(|_| DbError::NotFound("invalid id".into()))?,
    project_id: Uuid::parse_str(&project_id_str).map_err(|_| DbError::NotFound("invalid project_id".into()))?,
    content: get_string("content")?,
    summary: get_optional_string("summary"),
    sector,
    tier,
    memory_type,
    importance: get_f32("importance")?,
    salience: get_f32("salience")?,
    confidence: get_f32("confidence")?,
    access_count: get_u32("access_count")?,
    tags: serde_json::from_str(&tags_json)?,
    concepts: serde_json::from_str(&concepts_json)?,
    files: serde_json::from_str(&files_json)?,
    categories: serde_json::from_str(&categories_json)?,
    scope_path: get_optional_string("scope_path"),
    scope_module: get_optional_string("scope_module"),
    decay_rate: get_optional_f32("decay_rate"),
    next_decay_at,
    embedding_model_id: get_optional_string("embedding_model_id"),
    context: get_optional_string("context"),
    session_id: get_optional_string("session_id").and_then(|s| Uuid::parse_str(&s).ok()),
    segment_id: get_optional_string("segment_id").and_then(|s| Uuid::parse_str(&s).ok()),
    created_at,
    updated_at,
    last_accessed,
    valid_from,
    valid_until,
    is_deleted: get_bool("is_deleted")?,
    deleted_at,
    content_hash: get_string("content_hash")?,
    simhash: get_u64("simhash")?,
    superseded_by,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::path::Path;
  use tempfile::TempDir;

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = engram_core::ProjectId::from_path(Path::new("/test"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();
    (temp_dir, db)
  }

  fn create_test_memory() -> Memory {
    Memory::new(Uuid::new_v4(), "Test memory content".to_string(), Sector::Semantic)
  }

  #[tokio::test]
  async fn test_add_and_get_memory() {
    let (_temp, db) = create_test_db().await;
    let mut memory = create_test_memory();
    memory.content_hash = "test_hash".to_string();

    db.add_memory(&memory, None).await.unwrap();

    let retrieved = db.get_memory(&memory.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.content, memory.content);
  }

  #[tokio::test]
  async fn test_list_memories() {
    let (_temp, db) = create_test_db().await;

    let mut m1 = create_test_memory();
    m1.content_hash = "hash1".to_string();
    let mut m2 = create_test_memory();
    m2.content_hash = "hash2".to_string();

    db.add_memory(&m1, None).await.unwrap();
    db.add_memory(&m2, None).await.unwrap();

    let memories = db.list_memories(None, None).await.unwrap();
    assert_eq!(memories.len(), 2);
  }

  #[tokio::test]
  async fn test_delete_memory() {
    let (_temp, db) = create_test_db().await;
    let mut memory = create_test_memory();
    memory.content_hash = "test_hash".to_string();

    db.add_memory(&memory, None).await.unwrap();

    let before = db.get_memory(&memory.id).await.unwrap();
    assert!(before.is_some());

    db.delete_memory(&memory.id).await.unwrap();

    let after = db.get_memory(&memory.id).await.unwrap();
    assert!(after.is_none());
  }
}
