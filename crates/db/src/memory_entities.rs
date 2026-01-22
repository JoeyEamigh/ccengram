// Memory-Entity junction table operations
//
// Tracks which entities are mentioned in which memories

use arrow_array::{Array, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::{TimeZone, Utc};
use engram_core::{EntityRole, MemoryEntityLink};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::memory_entities_schema;

impl ProjectDb {
  /// Link an entity to a memory
  pub async fn link_entity_to_memory(&self, link: &MemoryEntityLink) -> Result<()> {
    let table = self.memory_entities_table().await?;

    let batch = link_to_batch(link)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], memory_entities_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get all entity links for a memory
  pub async fn get_memory_entity_links(&self, memory_id: &str) -> Result<Vec<MemoryEntityLink>> {
    let table = self.memory_entities_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("memory_id = '{}'", memory_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut links = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        links.push(batch_to_link(&batch, i)?);
      }
    }

    Ok(links)
  }

  /// Get all memory links for an entity
  pub async fn get_entity_memory_links(&self, entity_id: &Uuid) -> Result<Vec<MemoryEntityLink>> {
    let table = self.memory_entities_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("entity_id = '{}'", entity_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut links = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        links.push(batch_to_link(&batch, i)?);
      }
    }

    Ok(links)
  }

  /// Check if an entity-memory link exists
  pub async fn entity_memory_link_exists(&self, memory_id: &str, entity_id: &Uuid) -> Result<bool> {
    let table = self.memory_entities_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("memory_id = '{}' AND entity_id = '{}'", memory_id, entity_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    Ok(!results.is_empty() && results[0].num_rows() > 0)
  }

  /// Delete a memory-entity link by ID
  pub async fn delete_memory_entity_link(&self, id: &Uuid) -> Result<()> {
    let table = self.memory_entities_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Delete all entity links for a memory
  pub async fn delete_memory_entity_links(&self, memory_id: &str) -> Result<()> {
    let table = self.memory_entities_table().await?;
    table.delete(&format!("memory_id = '{}'", memory_id)).await?;
    Ok(())
  }

  /// Delete all memory links for an entity
  pub async fn delete_entity_memory_links(&self, entity_id: &Uuid) -> Result<()> {
    let table = self.memory_entities_table().await?;
    table.delete(&format!("entity_id = '{}'", entity_id)).await?;
    Ok(())
  }

  /// Count memories containing an entity
  pub async fn count_entity_memories(&self, entity_id: &Uuid) -> Result<usize> {
    let links = self.get_entity_memory_links(entity_id).await?;
    Ok(links.len())
  }

  /// Get entities for a memory with their roles
  pub async fn get_memory_entities_with_roles(&self, memory_id: &str) -> Result<Vec<(Uuid, EntityRole, f32)>> {
    let links = self.get_memory_entity_links(memory_id).await?;
    Ok(links.into_iter().map(|l| (l.entity_id, l.role, l.confidence)).collect())
  }
}

/// Convert a MemoryEntityLink to an Arrow RecordBatch
fn link_to_batch(link: &MemoryEntityLink) -> Result<RecordBatch> {
  let id = StringArray::from(vec![link.id.to_string()]);
  let memory_id = StringArray::from(vec![link.memory_id.clone()]);
  let entity_id = StringArray::from(vec![link.entity_id.to_string()]);
  let role = StringArray::from(vec![link.role.as_str().to_string()]);
  let confidence = Float32Array::from(vec![link.confidence]);
  let extracted_at = Int64Array::from(vec![link.extracted_at.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    memory_entities_schema(),
    vec![
      Arc::new(id),
      Arc::new(memory_id),
      Arc::new(entity_id),
      Arc::new(role),
      Arc::new(confidence),
      Arc::new(extracted_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a MemoryEntityLink
fn batch_to_link(batch: &RecordBatch, row: usize) -> Result<MemoryEntityLink> {
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

  let id_str = get_string("id")?;
  let entity_id_str = get_string("entity_id")?;
  let role_str = get_string("role")?;

  let role = role_str.parse::<EntityRole>().map_err(DbError::NotFound)?;

  let extracted_at = Utc
    .timestamp_millis_opt(get_i64("extracted_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid extracted_at timestamp".into()))?;

  Ok(MemoryEntityLink {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    memory_id: get_string("memory_id")?,
    entity_id: Uuid::parse_str(&entity_id_str).map_err(|_| DbError::NotFound("invalid entity_id".into()))?,
    role,
    confidence: get_f32("confidence")?,
    extracted_at,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ProjectDb;
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
  async fn test_link_entity_to_memory() {
    let (_temp, db) = create_test_db().await;
    let memory_id = Uuid::new_v4().to_string();
    let entity_id = Uuid::new_v4();

    let link = MemoryEntityLink::new(memory_id.clone(), entity_id, EntityRole::Subject, 0.95);

    db.link_entity_to_memory(&link).await.unwrap();

    let links = db.get_memory_entity_links(&memory_id).await.unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].entity_id, entity_id);
    assert_eq!(links[0].role, EntityRole::Subject);
  }

  #[tokio::test]
  async fn test_get_entity_memory_links() {
    let (_temp, db) = create_test_db().await;
    let entity_id = Uuid::new_v4();
    let memory1 = Uuid::new_v4().to_string();
    let memory2 = Uuid::new_v4().to_string();

    let link1 = MemoryEntityLink::new(memory1, entity_id, EntityRole::Subject, 0.9);
    let link2 = MemoryEntityLink::new(memory2, entity_id, EntityRole::Reference, 0.8);

    db.link_entity_to_memory(&link1).await.unwrap();
    db.link_entity_to_memory(&link2).await.unwrap();

    let links = db.get_entity_memory_links(&entity_id).await.unwrap();
    assert_eq!(links.len(), 2);
  }

  #[tokio::test]
  async fn test_entity_memory_link_exists() {
    let (_temp, db) = create_test_db().await;
    let memory_id = Uuid::new_v4().to_string();
    let entity_id = Uuid::new_v4();

    // Should not exist yet
    let exists = db.entity_memory_link_exists(&memory_id, &entity_id).await.unwrap();
    assert!(!exists);

    // Create link
    let link = MemoryEntityLink::new(memory_id.clone(), entity_id, EntityRole::Mention, 0.7);
    db.link_entity_to_memory(&link).await.unwrap();

    // Should exist now
    let exists = db.entity_memory_link_exists(&memory_id, &entity_id).await.unwrap();
    assert!(exists);
  }

  #[tokio::test]
  async fn test_delete_memory_entity_links() {
    let (_temp, db) = create_test_db().await;
    let memory_id = Uuid::new_v4().to_string();
    let entity1 = Uuid::new_v4();
    let entity2 = Uuid::new_v4();

    let link1 = MemoryEntityLink::new(memory_id.clone(), entity1, EntityRole::Subject, 0.9);
    let link2 = MemoryEntityLink::new(memory_id.clone(), entity2, EntityRole::Reference, 0.8);

    db.link_entity_to_memory(&link1).await.unwrap();
    db.link_entity_to_memory(&link2).await.unwrap();

    let links = db.get_memory_entity_links(&memory_id).await.unwrap();
    assert_eq!(links.len(), 2);

    db.delete_memory_entity_links(&memory_id).await.unwrap();

    let links = db.get_memory_entity_links(&memory_id).await.unwrap();
    assert_eq!(links.len(), 0);
  }
}
