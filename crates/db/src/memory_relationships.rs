// Memory Relationships table operations
//
// Tracks relationships between memories beyond simple supersession:
// - Supersedes, Contradicts, RelatedTo, BuildsOn
// - Confirms, AppliesTo, DependsOn, AlternativeTo

use arrow_array::{Array, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::{TimeZone, Utc};
use engram_core::{MemoryId, MemoryRelationship, RelationshipType};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::memory_relationships_schema;

impl ProjectDb {
  /// Add a relationship between two memories
  pub async fn add_relationship(&self, relationship: &MemoryRelationship) -> Result<()> {
    let table = self.memory_relationships_table().await?;

    let batch = relationship_to_batch(relationship)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memory_relationships_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Create a new relationship between memories
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

  /// Get all relationships from a memory
  pub async fn get_relationships_from(&self, memory_id: &MemoryId) -> Result<Vec<MemoryRelationship>> {
    let table = self.memory_relationships_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("from_memory_id = '{}'", memory_id))
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

  /// Get all relationships to a memory
  pub async fn get_relationships_to(&self, memory_id: &MemoryId) -> Result<Vec<MemoryRelationship>> {
    let table = self.memory_relationships_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("to_memory_id = '{}'", memory_id))
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

  /// Get all relationships for a memory (both from and to)
  pub async fn get_all_relationships(&self, memory_id: &MemoryId) -> Result<Vec<MemoryRelationship>> {
    let table = self.memory_relationships_table().await?;

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

  /// Get relationships of a specific type from a memory
  pub async fn get_relationships_of_type(
    &self,
    memory_id: &MemoryId,
    rel_type: RelationshipType,
  ) -> Result<Vec<MemoryRelationship>> {
    let table = self.memory_relationships_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!(
        "from_memory_id = '{}' AND relationship_type = '{}'",
        memory_id,
        rel_type.as_str()
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

  /// Count relationships for a memory
  pub async fn count_relationships(&self, memory_id: &MemoryId) -> Result<usize> {
    let relationships = self.get_all_relationships(memory_id).await?;
    Ok(relationships.len())
  }

  /// Delete a relationship by ID
  pub async fn delete_relationship(&self, id: &Uuid) -> Result<()> {
    let table = self.memory_relationships_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Delete all relationships for a memory
  pub async fn delete_relationships_for_memory(&self, memory_id: &MemoryId) -> Result<()> {
    let table = self.memory_relationships_table().await?;
    table
      .delete(&format!(
        "from_memory_id = '{}' OR to_memory_id = '{}'",
        memory_id, memory_id
      ))
      .await?;
    Ok(())
  }

  /// Invalidate a relationship (set valid_until to now)
  pub async fn invalidate_relationship(&self, id: &Uuid) -> Result<()> {
    // Get the relationship first
    let table = self.memory_relationships_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() || results[0].num_rows() == 0 {
      return Err(DbError::NotFound(format!("Relationship {} not found", id)));
    }

    let mut relationship = batch_to_relationship(&results[0], 0)?;
    relationship.valid_until = Some(Utc::now());

    // Delete and re-add
    table.delete(&format!("id = '{}'", id)).await?;
    let batch = relationship_to_batch(&relationship)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memory_relationships_schema());
    table.add(Box::new(batches)).execute().await?;

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
  use super::*;
  use engram_core::ProjectId;
  use std::path::Path;
  use tempfile::TempDir;

  async fn create_test_db() -> (TempDir, ProjectDb) {
    let temp_dir = TempDir::new().unwrap();
    let project_id = ProjectId::from_path(Path::new("/test"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();
    (temp_dir, db)
  }

  #[tokio::test]
  async fn test_create_relationship() {
    let (_temp, db) = create_test_db().await;
    let from = MemoryId::new();
    let to = MemoryId::new();

    let rel = db
      .create_relationship(&from, &to, RelationshipType::RelatedTo, 0.8, "test")
      .await
      .unwrap();

    assert_eq!(rel.from_memory_id, from);
    assert_eq!(rel.to_memory_id, to);
    assert_eq!(rel.relationship_type, RelationshipType::RelatedTo);
    assert_eq!(rel.confidence, 0.8);
  }

  #[tokio::test]
  async fn test_get_relationships_from() {
    let (_temp, db) = create_test_db().await;
    let from = MemoryId::new();
    let to1 = MemoryId::new();
    let to2 = MemoryId::new();

    db.create_relationship(&from, &to1, RelationshipType::BuildsOn, 0.9, "test")
      .await
      .unwrap();
    db.create_relationship(&from, &to2, RelationshipType::Confirms, 0.7, "test")
      .await
      .unwrap();

    let rels = db.get_relationships_from(&from).await.unwrap();
    assert_eq!(rels.len(), 2);
  }

  #[tokio::test]
  async fn test_get_relationships_to() {
    let (_temp, db) = create_test_db().await;
    let from1 = MemoryId::new();
    let from2 = MemoryId::new();
    let to = MemoryId::new();

    db.create_relationship(&from1, &to, RelationshipType::DependsOn, 0.9, "test")
      .await
      .unwrap();
    db.create_relationship(&from2, &to, RelationshipType::AppliesTo, 0.6, "test")
      .await
      .unwrap();

    let rels = db.get_relationships_to(&to).await.unwrap();
    assert_eq!(rels.len(), 2);
  }

  #[tokio::test]
  async fn test_get_relationships_of_type() {
    let (_temp, db) = create_test_db().await;
    let from = MemoryId::new();
    let to1 = MemoryId::new();
    let to2 = MemoryId::new();

    db.create_relationship(&from, &to1, RelationshipType::Contradicts, 0.9, "test")
      .await
      .unwrap();
    db.create_relationship(&from, &to2, RelationshipType::Confirms, 0.7, "test")
      .await
      .unwrap();

    let contradicts = db
      .get_relationships_of_type(&from, RelationshipType::Contradicts)
      .await
      .unwrap();
    assert_eq!(contradicts.len(), 1);
    assert_eq!(contradicts[0].to_memory_id, to1);
  }

  #[tokio::test]
  async fn test_count_relationships() {
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

    let count = db.count_relationships(&mem).await.unwrap();
    assert_eq!(count, 2);
  }

  #[tokio::test]
  async fn test_delete_relationships_for_memory() {
    let (_temp, db) = create_test_db().await;
    let mem = MemoryId::new();
    let other = MemoryId::new();

    db.create_relationship(&mem, &other, RelationshipType::Supersedes, 1.0, "test")
      .await
      .unwrap();

    db.delete_relationships_for_memory(&mem).await.unwrap();

    let count = db.count_relationships(&mem).await.unwrap();
    assert_eq!(count, 0);
  }

  #[test]
  fn test_relationship_type_parsing() {
    assert_eq!(
      "supersedes".parse::<RelationshipType>().unwrap(),
      RelationshipType::Supersedes
    );
    assert_eq!(
      "related_to".parse::<RelationshipType>().unwrap(),
      RelationshipType::RelatedTo
    );
    assert_eq!(
      "builds_on".parse::<RelationshipType>().unwrap(),
      RelationshipType::BuildsOn
    );
  }
}
