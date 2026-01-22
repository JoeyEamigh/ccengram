// Entities table operations
//
// Tracks named entities (people, projects, technologies) across memories

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use chrono::{TimeZone, Utc};
use engram_core::{Entity, EntityType};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::entities_schema;

impl ProjectDb {
  /// Add a new entity to the database
  pub async fn add_entity(&self, entity: &Entity) -> Result<()> {
    let table = self.entities_table().await?;

    let batch = entity_to_batch(entity)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], entities_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get an entity by ID
  pub async fn get_entity(&self, id: &Uuid) -> Result<Option<Entity>> {
    let table = self.entities_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id))
      .execute()
      .await?
      .try_collect()
      .await?;

    if results.is_empty() || results[0].num_rows() == 0 {
      return Ok(None);
    }

    Ok(Some(batch_to_entity(&results[0], 0)?))
  }

  /// Find entity by name (case-insensitive)
  pub async fn find_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
    let table = self.entities_table().await?;

    // LanceDB doesn't support case-insensitive queries directly, so we'll get all and filter
    let results: Vec<RecordBatch> = table.query().execute().await?.try_collect().await?;

    let name_lower = name.to_lowercase();
    for batch in results {
      for i in 0..batch.num_rows() {
        let entity = batch_to_entity(&batch, i)?;
        if entity.name.to_lowercase() == name_lower {
          return Ok(Some(entity));
        }
        // Also check aliases
        for alias in &entity.aliases {
          if alias.to_lowercase() == name_lower {
            return Ok(Some(entity));
          }
        }
      }
    }

    Ok(None)
  }

  /// Update an entity
  pub async fn update_entity(&self, entity: &Entity) -> Result<()> {
    let table = self.entities_table().await?;

    let batch = entity_to_batch(entity)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], entities_schema());

    // Delete old and insert new
    let _ = table.delete(&format!("id = '{}'", entity.id)).await;
    table.add(Box::new(batches)).execute().await?;

    Ok(())
  }

  /// Delete an entity by ID
  pub async fn delete_entity(&self, id: &Uuid) -> Result<()> {
    let table = self.entities_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// List entities by type
  pub async fn list_entities_by_type(&self, entity_type: EntityType) -> Result<Vec<Entity>> {
    let table = self.entities_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("entity_type = '{}'", entity_type.as_str()))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut entities = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        entities.push(batch_to_entity(&batch, i)?);
      }
    }

    Ok(entities)
  }

  /// List all entities
  pub async fn list_entities(&self, limit: Option<usize>) -> Result<Vec<Entity>> {
    let table = self.entities_table().await?;

    let query = if let Some(l) = limit {
      table.query().limit(l)
    } else {
      table.query()
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut entities = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        entities.push(batch_to_entity(&batch, i)?);
      }
    }

    Ok(entities)
  }

  /// Get entities sorted by mention count (most mentioned first)
  pub async fn get_top_entities(&self, limit: usize) -> Result<Vec<Entity>> {
    let mut entities = self.list_entities(None).await?;
    entities.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));
    entities.truncate(limit);
    Ok(entities)
  }

  /// Find or create an entity by name
  pub async fn find_or_create_entity(&self, name: &str, entity_type: EntityType) -> Result<Entity> {
    if let Some(existing) = self.find_entity_by_name(name).await? {
      return Ok(existing);
    }

    let entity = Entity::new(name.to_string(), entity_type);
    self.add_entity(&entity).await?;
    Ok(entity)
  }

  /// Increment mention count for an entity
  pub async fn record_entity_mention(&self, id: &Uuid) -> Result<()> {
    if let Some(mut entity) = self.get_entity(id).await? {
      entity.mention();
      self.update_entity(&entity).await?;
    }
    Ok(())
  }
}

/// Convert an Entity to an Arrow RecordBatch
fn entity_to_batch(entity: &Entity) -> Result<RecordBatch> {
  let id = StringArray::from(vec![entity.id.to_string()]);
  let name = StringArray::from(vec![entity.name.clone()]);
  let entity_type = StringArray::from(vec![entity.entity_type.as_str().to_string()]);
  let summary = StringArray::from(vec![entity.summary.clone()]);
  let aliases = StringArray::from(vec![serde_json::to_string(&entity.aliases)?]);
  let first_seen_at = Int64Array::from(vec![entity.first_seen_at.timestamp_millis()]);
  let last_seen_at = Int64Array::from(vec![entity.last_seen_at.timestamp_millis()]);
  let mention_count = UInt32Array::from(vec![entity.mention_count]);

  let batch = RecordBatch::try_new(
    entities_schema(),
    vec![
      Arc::new(id),
      Arc::new(name),
      Arc::new(entity_type),
      Arc::new(summary),
      Arc::new(aliases),
      Arc::new(first_seen_at),
      Arc::new(last_seen_at),
      Arc::new(mention_count),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to an Entity
fn batch_to_entity(batch: &RecordBatch, row: usize) -> Result<Entity> {
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

  let get_i64 = |name: &str| -> Result<i64> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
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

  let id_str = get_string("id")?;
  let entity_type_str = get_string("entity_type")?;
  let aliases_json = get_string("aliases")?;

  let entity_type = entity_type_str.parse::<EntityType>().map_err(DbError::NotFound)?;

  let first_seen_at = Utc
    .timestamp_millis_opt(get_i64("first_seen_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid first_seen_at timestamp".into()))?;

  let last_seen_at = Utc
    .timestamp_millis_opt(get_i64("last_seen_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid last_seen_at timestamp".into()))?;

  Ok(Entity {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    name: get_string("name")?,
    entity_type,
    summary: get_optional_string("summary"),
    aliases: serde_json::from_str(&aliases_json)?,
    first_seen_at,
    last_seen_at,
    mention_count: get_u32("mention_count")?,
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
  async fn test_add_and_get_entity() {
    let (_temp, db) = create_test_db().await;
    let entity = Entity::new("TypeScript".to_string(), EntityType::Technology);

    db.add_entity(&entity).await.unwrap();

    let retrieved = db.get_entity(&entity.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.name, "TypeScript");
    assert_eq!(retrieved.entity_type, EntityType::Technology);
  }

  #[tokio::test]
  async fn test_find_entity_by_name() {
    let (_temp, db) = create_test_db().await;
    let mut entity = Entity::new("React".to_string(), EntityType::Technology);
    entity.add_alias("ReactJS".to_string());

    db.add_entity(&entity).await.unwrap();

    // Find by name
    let found = db.find_entity_by_name("React").await.unwrap();
    assert!(found.is_some());

    // Find by alias
    let found = db.find_entity_by_name("ReactJS").await.unwrap();
    assert!(found.is_some());

    // Case insensitive
    let found = db.find_entity_by_name("react").await.unwrap();
    assert!(found.is_some());
  }

  #[tokio::test]
  async fn test_list_entities_by_type() {
    let (_temp, db) = create_test_db().await;

    let tech1 = Entity::new("Rust".to_string(), EntityType::Technology);
    let tech2 = Entity::new("Python".to_string(), EntityType::Technology);
    let person = Entity::new("Alice".to_string(), EntityType::Person);

    db.add_entity(&tech1).await.unwrap();
    db.add_entity(&tech2).await.unwrap();
    db.add_entity(&person).await.unwrap();

    let techs = db.list_entities_by_type(EntityType::Technology).await.unwrap();
    assert_eq!(techs.len(), 2);

    let people = db.list_entities_by_type(EntityType::Person).await.unwrap();
    assert_eq!(people.len(), 1);
  }

  #[tokio::test]
  async fn test_record_entity_mention() {
    let (_temp, db) = create_test_db().await;
    let entity = Entity::new("Docker".to_string(), EntityType::Technology);

    db.add_entity(&entity).await.unwrap();

    db.record_entity_mention(&entity.id).await.unwrap();
    db.record_entity_mention(&entity.id).await.unwrap();

    let retrieved = db.get_entity(&entity.id).await.unwrap().unwrap();
    assert_eq!(retrieved.mention_count, 3); // Initial 1 + 2 mentions
  }

  #[tokio::test]
  async fn test_find_or_create_entity() {
    let (_temp, db) = create_test_db().await;

    // Should create new entity
    let entity1 = db
      .find_or_create_entity("Kubernetes", EntityType::Technology)
      .await
      .unwrap();
    assert_eq!(entity1.name, "Kubernetes");

    // Should return existing entity
    let entity2 = db
      .find_or_create_entity("Kubernetes", EntityType::Technology)
      .await
      .unwrap();
    assert_eq!(entity1.id, entity2.id);
  }
}
