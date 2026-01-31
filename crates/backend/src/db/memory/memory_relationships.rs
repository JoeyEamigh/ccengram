// Memory Relationships table operations
//
// Tracks relationships between memories beyond simple supersession:
// - Supersedes, Contradicts, RelatedTo, BuildsOn
// - Confirms, AppliesTo, DependsOn, AlternativeTo

use std::sync::Arc;

use arrow_array::{Array, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use uuid::Uuid;

use crate::{
  db::{DbError, ProjectDb, Result, schema::memory_relationships_schema},
  domain::memory::{MemoryId, MemoryRelationship, RelationshipType},
};

impl ProjectDb {
  /// Add a relationship between two memories
  #[tracing::instrument(level = "trace", skip(self, relationship))]
  pub async fn add_relationship(&self, relationship: &MemoryRelationship) -> Result<()> {
    let table = self.memory_relationships_table();

    let batch = relationship_to_batch(relationship)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memory_relationships_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Create a new relationship between memories
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn create_relationship(
    &self,
    from: &MemoryId,
    to: &MemoryId,
    rel_type: RelationshipType,
    confidence: f32,
    extracted_by: &str,
  ) -> Result<MemoryRelationship> {
    let relationship = MemoryRelationship::new(*from, *to, rel_type, confidence, extracted_by);
    self.add_relationship(&relationship).await?;
    Ok(relationship)
  }

  /// Get all relationships for a memory (both from and to)
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_all_relationships(&self, memory_id: &MemoryId) -> Result<Vec<MemoryRelationship>> {
    let table = self.memory_relationships_table();

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!(
        "from_memory_id = '{}' OR to_memory_id = '{}'",
        memory_id, memory_id
      ))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut relationships = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        relationships.push(batch_to_relationship(&batch, i)?);
      }
    }

    Ok(relationships)
  }

  /// Delete a relationship by ID
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn delete_relationship(&self, id: &Uuid) -> Result<()> {
    let table = self.memory_relationships_table();
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }
}

/// Convert a MemoryRelationship to an Arrow RecordBatch
fn relationship_to_batch(rel: &MemoryRelationship) -> Result<RecordBatch> {
  let id = StringArray::from(vec![rel.id.to_string()]);
  let from_memory_id = StringArray::from(vec![rel.from_memory_id.to_string()]);
  let to_memory_id = StringArray::from(vec![rel.to_memory_id.to_string()]);
  let relationship_type = StringArray::from(vec![rel.relationship_type.as_str().to_string()]);
  let confidence = Float32Array::from(vec![rel.confidence]);
  let valid_from = Int64Array::from(vec![rel.valid_from.timestamp_millis()]);
  let valid_until = Int64Array::from(vec![rel.valid_until.map(|t| t.timestamp_millis())]);
  let extracted_by = StringArray::from(vec![rel.extracted_by.clone()]);
  let created_at = Int64Array::from(vec![rel.created_at.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    memory_relationships_schema(),
    vec![
      Arc::new(id),
      Arc::new(from_memory_id),
      Arc::new(to_memory_id),
      Arc::new(relationship_type),
      Arc::new(confidence),
      Arc::new(valid_from),
      Arc::new(valid_until),
      Arc::new(extracted_by),
      Arc::new(created_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a MemoryRelationship
fn batch_to_relationship(batch: &RecordBatch, row: usize) -> Result<MemoryRelationship> {
  let get_string = |name: &str| -> Result<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
      .ok_or_else(|| DbError::NotFound(format!("column {}", name)))
  };

  let get_f32 = |name: &str| -> Result<f32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
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

  let id_str = get_string("id")?;
  let from_memory_id_str = get_string("from_memory_id")?;
  let to_memory_id_str = get_string("to_memory_id")?;
  let relationship_type_str = get_string("relationship_type")?;

  let relationship_type = relationship_type_str
    .parse::<RelationshipType>()
    .map_err(DbError::NotFound)?;

  let valid_from = Utc
    .timestamp_millis_opt(get_i64("valid_from")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid valid_from timestamp".into()))?;

  let valid_until = get_optional_i64("valid_until").and_then(|ts| Utc.timestamp_millis_opt(ts).single());

  let created_at = Utc
    .timestamp_millis_opt(get_i64("created_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid created_at timestamp".into()))?;

  Ok(MemoryRelationship {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    from_memory_id: from_memory_id_str
      .parse()
      .map_err(|_| DbError::NotFound("invalid from_memory_id".into()))?,
    to_memory_id: to_memory_id_str
      .parse()
      .map_err(|_| DbError::NotFound("invalid to_memory_id".into()))?,
    relationship_type,
    confidence: get_f32("confidence")?,
    valid_from,
    valid_until,
    extracted_by: get_string("extracted_by")?,
    created_at,
  })
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use tempfile::TempDir;

  use super::*;
  use crate::{config::Config, domain::project::ProjectId};

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test")).await;
    let db = ProjectDb::open_at_path(
      project_id,
      temp_dir.path().join("test.lancedb"),
      Arc::new(Config::default()),
    )
    .await
    .unwrap();
    (temp_dir, db)
  }

  #[tokio::test]
  async fn test_create_and_get_all_relationships() {
    let (_temp, db) = create_test_db().await;
    let mem = MemoryId::new();
    let other1 = MemoryId::new();
    let other2 = MemoryId::new();

    db.create_relationship(&mem, &other1, RelationshipType::RelatedTo, 0.8, "test")
      .await
      .unwrap();
    db.create_relationship(&other2, &mem, RelationshipType::BuildsOn, 0.7, "test")
      .await
      .unwrap();

    let rels = db.get_all_relationships(&mem).await.unwrap();
    assert_eq!(rels.len(), 2, "Should find both from and to relationships");
  }
}
