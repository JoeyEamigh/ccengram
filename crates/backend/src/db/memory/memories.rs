use std::sync::Arc;

use arrow_array::{
  Array, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
  UInt32Array, UInt64Array,
};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use tracing::debug;
use uuid::Uuid;

use crate::{
  db::{
    connection::{DbError, ProjectDb, Result},
    schema::memories_schema,
  },
  domain::memory::{Memory, MemoryId, MemoryType, Sector, Tier},
};

impl ProjectDb {
  /// Add a new memory to the database
  #[tracing::instrument(level = "trace", skip(self, memory, vector), fields(id = %memory.id))]
  pub async fn add_memory(&self, memory: &Memory, vector: &[f32]) -> Result<()> {
    let table = self.memories_table();

    debug!(
      table = "memories",
      operation = "insert",
      id = %memory.id,
      sector = %memory.sector.as_str(),
      "Adding memory"
    );

    let batch = memory_to_batch(memory, vector, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memories_schema(self.vector_dim));

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get a memory by ID
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
    let table = self.memories_table();
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
  ///
  /// If `vector` is `None`, the existing vector is preserved.
  /// This is important for operations like restore/reinforce that don't
  /// want to lose the embedding.
  #[tracing::instrument(level = "trace", skip(self, memory, vector), fields(id = %memory.id))]
  pub async fn update_memory(&self, memory: &Memory, vector: Option<&[f32]>) -> Result<()> {
    let table = self.memories_table();

    debug!(
      table = "memories",
      operation = "update",
      id = %memory.id,
      has_vector = vector.is_some(),
      "Updating memory"
    );

    // If no vector provided, fetch the existing one to preserve it
    let existing_vector = if vector.is_none() {
      Some(self.get_memory_vector(&memory.id).await?)
    } else {
      None
    };
    let vector_to_use: &[f32] = vector
      .or(existing_vector.as_deref())
      .expect("this is logically infallible");

    let batch = memory_to_batch(memory, vector_to_use, self.vector_dim)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memories_schema(self.vector_dim));

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert.when_matched_update_all(None).when_not_matched_insert_all();
    merge_insert.execute(Box::new(batches)).await?;

    Ok(())
  }

  /// Get the embedding vector for a memory by ID.
  ///
  /// Returns `Ok(Some(vec))` if the memory exists and has a vector,
  /// `Ok(None)` if the memory doesn't exist,
  /// or an error if there's a database issue.
  ///
  /// This is useful for cross-domain search (e.g., finding code related to a memory).
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_memory_embedding(&self, id: &MemoryId) -> Result<Option<Vec<f32>>> {
    let table = self.memories_table();
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

    // Extract vector from the batch
    let vector_col = batch
      .column_by_name("vector")
      .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

    let Some(arr) = vector_col else {
      return Ok(None);
    };

    Ok(
      arr
        .value(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .map(|a| a.values().to_vec()),
    )
  }

  /// Get the vector for a memory by ID (internal use, returns error if not found)
  async fn get_memory_vector(&self, id: &MemoryId) -> Result<Vec<f32>> {
    let table = self.memories_table();
    let id_str = id.to_string();

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id_str))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() {
      return Err(DbError::NotFound(format!("Memory {} not found", id)));
    }

    let batch = &results[0];
    if batch.num_rows() == 0 {
      return Err(DbError::NotFound(format!("Memory {} not found", id)));
    }

    // Extract vector from the batch
    let vector_col = batch
      .column_by_name("vector")
      .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

    let Some(arr) = vector_col else {
      return Err(DbError::NotFound(format!("Memory {} has no vector", id)));
    };

    if let Some(vec) = arr
      .value(0)
      .as_any()
      .downcast_ref::<Float32Array>()
      .map(|a| a.values().to_vec())
    {
      Ok(vec)
    } else {
      Err(DbError::NotFound(format!("Memory {} has no vector", id)))
    }
  }

  /// Get vectors for multiple memories by ID in a single query
  async fn get_memory_vectors_batch(&self, ids: &[MemoryId]) -> Result<std::collections::HashMap<MemoryId, Vec<f32>>> {
    if ids.is_empty() {
      return Ok(std::collections::HashMap::new());
    }

    let table = self.memories_table();
    let id_list: Vec<String> = ids.iter().map(|id| format!("'{}'", id)).collect();
    let filter = format!("id IN ({})", id_list.join(", "));

    let results: Vec<RecordBatch> = table.query().only_if(filter).execute().await?.try_collect().await?;

    let mut vectors = std::collections::HashMap::new();

    for batch in results {
      let id_col = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
      let vector_col = batch
        .column_by_name("vector")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

      if let (Some(ids_arr), Some(vecs_arr)) = (id_col, vector_col) {
        for i in 0..batch.num_rows() {
          if let Ok(id_str) = ids_arr.value(i).parse::<MemoryId>()
            && let Some(vec) = vecs_arr
              .value(i)
              .as_any()
              .downcast_ref::<Float32Array>()
              .map(|a| a.values().to_vec())
          {
            vectors.insert(id_str, vec);
          }
        }
      }
    }

    Ok(vectors)
  }

  /// Batch update multiple memories (more efficient than individual updates)
  ///
  /// This is optimized for operations like decay where many memories need their
  /// salience updated. Uses merge_insert for efficient batch upserts.
  /// Existing vectors are preserved.
  #[tracing::instrument(level = "trace", skip(self, memories), fields(count = memories.len()))]
  pub async fn batch_update_memories(&self, memories: &[Memory]) -> Result<usize> {
    if memories.is_empty() {
      return Ok(0);
    }

    debug!(
      table = "memories",
      operation = "batch_update",
      batch_size = memories.len(),
      "Batch updating memories"
    );

    let table = self.memories_table();

    // Fetch existing vectors for all memories being updated
    let ids: Vec<_> = memories.iter().map(|m| m.id).collect();
    let vectors = self.get_memory_vectors_batch(&ids).await?;

    // Batch all records into a single RecordBatch
    let batches: Vec<_> = memories
      .iter()
      .map(|m| {
        let vector = vectors.get(&m.id).map(|v| v.as_slice()).unwrap_or(&[]);
        memory_to_batch(m, vector, self.vector_dim)
      })
      .collect::<Result<Vec<_>>>()?;

    let merged = arrow::compute::concat_batches(&memories_schema(self.vector_dim), &batches)?;
    let batches = RecordBatchIterator::new(vec![Ok(merged)], memories_schema(self.vector_dim));

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert.when_matched_update_all(None).when_not_matched_insert_all();
    merge_insert.execute(Box::new(batches)).await?;

    debug!(
      table = "memories",
      operation = "batch_update",
      updated = memories.len(),
      "Batch update complete"
    );

    Ok(memories.len())
  }

  /// Delete a memory by ID (hard delete)
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn delete_memory(&self, id: &MemoryId) -> Result<()> {
    debug!(table = "memories", operation = "delete", id = %id, "Deleting memory");
    let table = self.memories_table();
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Reinforce a memory (increment salience with diminishing returns)
  ///
  /// Formula: new_salience = min(salience + amount * (1.0 - salience), 1.0)
  ///
  /// Note: Uses read-modify-write since LanceDB doesn't support CASE WHEN expressions.
  /// Race conditions are acceptable for salience updates.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn reinforce_memory(&self, id: &MemoryId, amount: f32) -> Result<()> {
    let amount = amount.clamp(0.01, 0.5);

    // Read current salience
    let memory = self
      .get_memory(id)
      .await?
      .ok_or_else(|| DbError::NotFound(format!("Memory {} not found", id)))?;

    // Calculate new salience with diminishing returns, capped at 1.0
    let new_salience = (memory.salience + amount * (1.0 - memory.salience)).min(1.0);

    // Write back
    let table = self.memories_table();
    let now_millis = Utc::now().timestamp_millis();

    table
      .update()
      .only_if(format!("id = '{}'", id))
      .column("salience", format!("{}", new_salience))
      .column("access_count", "access_count + 1")
      .column("last_accessed", format!("{}", now_millis))
      .column("updated_at", format!("{}", now_millis))
      .execute()
      .await?;

    Ok(())
  }

  /// Deemphasize a memory (decrease salience)
  ///
  /// Formula: new_salience = max(salience - amount, 0.05)
  ///
  /// Note: Uses read-modify-write since LanceDB doesn't support CASE WHEN expressions.
  /// Race conditions are acceptable for salience updates.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn deemphasize_memory(&self, id: &MemoryId, amount: f32) -> Result<()> {
    let amount = amount.clamp(0.01, 0.5);

    // Read current salience
    let memory = self
      .get_memory(id)
      .await?
      .ok_or_else(|| DbError::NotFound(format!("Memory {} not found", id)))?;

    // Calculate new salience, floored at 0.05 to ensure memories are never completely forgotten
    let new_salience = (memory.salience - amount).max(0.05);

    // Write back
    let table = self.memories_table();
    let now_millis = Utc::now().timestamp_millis();

    table
      .update()
      .only_if(format!("id = '{}'", id))
      .column("salience", format!("{}", new_salience))
      .column("updated_at", format!("{}", now_millis))
      .execute()
      .await?;

    Ok(())
  }

  /// Atomically supersede a memory
  ///
  /// Marks the memory as superseded by another, setting valid_until and superseded_by.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn supersede_memory(&self, id: &MemoryId, superseded_by: &MemoryId) -> Result<()> {
    let table = self.memories_table();
    let now_millis = Utc::now().timestamp_millis();

    table
      .update()
      .only_if(format!("id = '{}'", id))
      .column("valid_until", format!("{}", now_millis))
      .column("superseded_by", format!("'{}'", superseded_by))
      .column("updated_at", format!("{}", now_millis))
      .execute()
      .await?;

    Ok(())
  }

  /// Atomically set a memory's salience to a specific value
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn set_memory_salience(&self, id: &MemoryId, salience: f32) -> Result<()> {
    let table = self.memories_table();
    let now_millis = Utc::now().timestamp_millis();
    let salience = salience.clamp(0.05, 1.0);

    table
      .update()
      .only_if(format!("id = '{}'", id))
      .column("salience", format!("{}", salience))
      .column("updated_at", format!("{}", now_millis))
      .execute()
      .await?;

    Ok(())
  }

  /// Atomically promote a memory from Session to Project tier
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn promote_memory_to_project(&self, id: &MemoryId) -> Result<()> {
    let table = self.memories_table();
    let now_millis = Utc::now().timestamp_millis();

    table
      .update()
      .only_if(format!("id = '{}' AND tier = 'session'", id))
      .column("tier", "'project'")
      .column("updated_at", format!("{}", now_millis))
      .execute()
      .await?;

    Ok(())
  }

  /// Search memories by full-text search (BM25)
  ///
  /// Searches the `content` column using BM25 scoring.
  /// Results are ordered by relevance score.
  pub async fn fts_search_memories(
    &self,
    query: &str,
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(Memory, f32)>> {
    use lance_index::scalar::FullTextSearchQuery;

    debug!(
      table = "memories",
      operation = "fts_search",
      query = %query,
      limit = limit,
      has_filter = filter.is_some(),
      "FTS searching memories"
    );

    let table = self.memories_table();

    let fts_query = FullTextSearchQuery::new(query.to_owned())
      .with_column("content".to_string())
      .map_err(|e| DbError::Query(format!("FTS query construction failed: {e}")))?;

    let builder = if let Some(f) = filter {
      table.query().full_text_search(fts_query).limit(limit).only_if(f)
    } else {
      table.query().full_text_search(fts_query).limit(limit)
    };

    let results: Vec<RecordBatch> = builder.execute().await?.try_collect().await?;

    let mut memories = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        let memory = batch_to_memory(&batch, i)?;
        let score = batch
          .column_by_name("_score")
          .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
          .map(|arr| arr.value(i))
          .unwrap_or(0.0);
        memories.push((memory, score));
      }
    }

    debug!(
      table = "memories",
      operation = "fts_search",
      results = memories.len(),
      "FTS search complete"
    );

    Ok(memories)
  }

  /// Search memories by vector similarity
  #[tracing::instrument(level = "trace", skip(self, query_vector))]
  pub async fn search_memories(
    &self,
    query_vector: &[f32],
    limit: usize,
    filter: Option<&str>,
  ) -> Result<Vec<(Memory, f32)>> {
    debug!(
      table = "memories",
      operation = "search",
      query_len = query_vector.len(),
      limit = limit,
      has_filter = filter.is_some(),
      "Searching memories"
    );

    let table = self.memories_table();

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

    debug!(
      table = "memories",
      operation = "search",
      results = memories.len(),
      "Search complete"
    );

    Ok(memories)
  }

  /// List memories with optional filters
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn list_memories(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<Memory>> {
    let table = self.memories_table();

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

  /// Find memories by ID prefix
  ///
  /// Searches for memories whose ID starts with the given prefix.
  /// Requires minimum 6 characters for safety. Returns up to 10 matches.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn find_by_prefix(&self, prefix: &str) -> Result<Vec<Memory>> {
    if prefix.len() < 6 {
      return Err(DbError::InvalidInput("ID prefix must be at least 6 characters".into()));
    }

    let table = self.memories_table();

    // Use LIKE query for prefix matching
    let filter = format!("id LIKE '{}%'", prefix);
    let results: Vec<RecordBatch> = table
      .query()
      .only_if(filter)
      .limit(10)
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut memories = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        memories.push(batch_to_memory(&batch, i)?);
      }
    }

    Ok(memories)
  }

  /// Get a memory by ID or prefix
  ///
  /// First tries exact match, then falls back to prefix matching.
  /// Returns error if prefix matches multiple memories (ambiguous).
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_memory_by_id_or_prefix(&self, id_or_prefix: &str) -> Result<Option<Memory>> {
    // Try exact match first (if it looks like a full UUID)
    if let Ok(memory_id) = id_or_prefix.parse::<MemoryId>()
      && let Some(memory) = self.get_memory(&memory_id).await?
    {
      return Ok(Some(memory));
    }

    // Try prefix match if at least 6 characters
    if id_or_prefix.len() >= 6 {
      let matches = self.find_by_prefix(id_or_prefix).await?;
      match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches.into_iter().next().unwrap())),
        _ => Err(DbError::AmbiguousPrefix {
          prefix: id_or_prefix.to_string(),
          count: matches.len(),
        }),
      }
    } else if id_or_prefix.len() < 6 {
      Err(DbError::InvalidInput("ID prefix must be at least 6 characters".into()))
    } else {
      Ok(None)
    }
  }
}

/// Convert a Memory to an Arrow RecordBatch
fn memory_to_batch(memory: &Memory, vector: &[f32], vector_dim: usize) -> Result<RecordBatch> {
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
  let session_id = StringArray::from(vec![memory.session_id.clone()]);
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
  let mut vec_padded = vector.to_vec();
  vec_padded.resize(vector_dim, 0.0);
  let vector_arr = Float32Array::from(vec_padded);

  let field = Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true));
  let vector_list = FixedSizeListArray::try_new(field, vector_dim as i32, Arc::new(vector_arr), None)?;

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
    session_id: get_optional_string("session_id"),
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
  use std::path::Path;

  use tempfile::TempDir;

  use super::*;
  use crate::config::Config;

  fn dummy_vector(dim: usize) -> Vec<f32> {
    vec![0.0f32; dim]
  }

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = crate::domain::project::ProjectId::from_path(Path::new("/test")).await;
    let db = ProjectDb::open_at_path(
      project_id,
      temp_dir.path().join("test.lancedb"),
      Arc::new(Config::default()),
    )
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

    db.add_memory(&memory, &dummy_vector(db.vector_dim)).await.unwrap();

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

    db.add_memory(&m1, &dummy_vector(db.vector_dim)).await.unwrap();
    db.add_memory(&m2, &dummy_vector(db.vector_dim)).await.unwrap();

    let memories = db.list_memories(None, None).await.unwrap();
    assert_eq!(memories.len(), 2);
  }

  #[tokio::test]
  async fn test_delete_memory() {
    let (_temp, db) = create_test_db().await;
    let mut memory = create_test_memory();
    memory.content_hash = "test_hash".to_string();

    db.add_memory(&memory, &dummy_vector(db.vector_dim)).await.unwrap();

    let before = db.get_memory(&memory.id).await.unwrap();
    assert!(before.is_some());

    db.delete_memory(&memory.id).await.unwrap();

    let after = db.get_memory(&memory.id).await.unwrap();
    assert!(after.is_none());
  }
}
