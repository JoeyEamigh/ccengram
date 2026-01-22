// Events table operations (audit log / event sourcing)

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::{DateTime, TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::events_schema;

/// Entity types that events can reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
  Memory,
  Session,
  CodeChunk,
  Project,
}

impl EntityType {
  pub fn as_str(&self) -> &'static str {
    match self {
      EntityType::Memory => "memory",
      EntityType::Session => "session",
      EntityType::CodeChunk => "code_chunk",
      EntityType::Project => "project",
    }
  }
}

impl std::str::FromStr for EntityType {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "memory" => Ok(EntityType::Memory),
      "session" => Ok(EntityType::Session),
      "code_chunk" => Ok(EntityType::CodeChunk),
      "project" => Ok(EntityType::Project),
      _ => Err(format!("Unknown entity type: {}", s)),
    }
  }
}

/// Event types for audit logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
  Created,
  Updated,
  Deleted,
  Accessed,
  Reinforced,
  Deemphasized,
  Superseded,
  Indexed,
  Searched,
}

impl EventType {
  pub fn as_str(&self) -> &'static str {
    match self {
      EventType::Created => "created",
      EventType::Updated => "updated",
      EventType::Deleted => "deleted",
      EventType::Accessed => "accessed",
      EventType::Reinforced => "reinforced",
      EventType::Deemphasized => "deemphasized",
      EventType::Superseded => "superseded",
      EventType::Indexed => "indexed",
      EventType::Searched => "searched",
    }
  }
}

impl std::str::FromStr for EventType {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "created" => Ok(EventType::Created),
      "updated" => Ok(EventType::Updated),
      "deleted" => Ok(EventType::Deleted),
      "accessed" => Ok(EventType::Accessed),
      "reinforced" => Ok(EventType::Reinforced),
      "deemphasized" => Ok(EventType::Deemphasized),
      "superseded" => Ok(EventType::Superseded),
      "indexed" => Ok(EventType::Indexed),
      "searched" => Ok(EventType::Searched),
      _ => Err(format!("Unknown event type: {}", s)),
    }
  }
}

/// An event record for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
  pub id: Uuid,
  pub entity_id: String,
  pub entity_type: EntityType,
  pub event_type: EventType,
  pub payload: serde_json::Value,
  pub timestamp: DateTime<Utc>,
}

impl Event {
  pub fn new(entity_id: String, entity_type: EntityType, event_type: EventType, payload: serde_json::Value) -> Self {
    Self {
      id: Uuid::now_v7(),
      entity_id,
      entity_type,
      event_type,
      payload,
      timestamp: Utc::now(),
    }
  }

  /// Create a memory event
  pub fn memory(memory_id: &str, event_type: EventType, payload: serde_json::Value) -> Self {
    Self::new(memory_id.to_string(), EntityType::Memory, event_type, payload)
  }

  /// Create a session event
  pub fn session(session_id: &str, event_type: EventType, payload: serde_json::Value) -> Self {
    Self::new(session_id.to_string(), EntityType::Session, event_type, payload)
  }
}

impl ProjectDb {
  /// Log an event
  pub async fn log_event(&self, event: &Event) -> Result<()> {
    let table = self.events_table().await?;

    let batch = event_to_batch(event)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], events_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Log multiple events in a batch
  pub async fn log_events(&self, events: &[Event]) -> Result<()> {
    if events.is_empty() {
      return Ok(());
    }

    let table = self.events_table().await?;

    let batches: Vec<_> = events.iter().map(event_to_batch).collect::<Result<Vec<_>>>()?;

    let iter = RecordBatchIterator::new(batches.into_iter().map(Ok), events_schema());
    table.add(Box::new(iter)).execute().await?;
    Ok(())
  }

  /// Get events for an entity
  pub async fn get_events_for_entity(
    &self,
    entity_id: &str,
    entity_type: EntityType,
    limit: Option<usize>,
  ) -> Result<Vec<Event>> {
    let table = self.events_table().await?;

    let filter = format!(
      "entity_id = '{}' AND entity_type = '{}'",
      entity_id,
      entity_type.as_str()
    );

    let query = if let Some(l) = limit {
      table.query().only_if(filter).limit(l)
    } else {
      table.query().only_if(filter)
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut events = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        events.push(batch_to_event(&batch, i)?);
      }
    }

    // Sort by timestamp descending (most recent first)
    events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(events)
  }

  /// Get recent events of a specific type
  pub async fn get_recent_events(&self, event_type: Option<EventType>, limit: usize) -> Result<Vec<Event>> {
    let table = self.events_table().await?;

    let query = if let Some(et) = event_type {
      table
        .query()
        .only_if(format!("event_type = '{}'", et.as_str()))
        .limit(limit * 2) // Over-fetch to compensate for lack of sorting
    } else {
      table.query().limit(limit * 2)
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut events = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        events.push(batch_to_event(&batch, i)?);
      }
    }

    // Sort by timestamp descending
    events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    events.truncate(limit);

    Ok(events)
  }

  /// Count events matching a filter
  pub async fn count_events(&self, filter: Option<&str>) -> Result<usize> {
    let table = self.events_table().await?;

    let query = if let Some(f) = filter {
      table.query().only_if(f)
    } else {
      table.query()
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
    Ok(results.iter().map(|b| b.num_rows()).sum())
  }
}

/// Convert an Event to an Arrow RecordBatch
fn event_to_batch(event: &Event) -> Result<RecordBatch> {
  let id = StringArray::from(vec![event.id.to_string()]);
  let entity_id = StringArray::from(vec![event.entity_id.clone()]);
  let entity_type = StringArray::from(vec![event.entity_type.as_str().to_string()]);
  let event_type = StringArray::from(vec![event.event_type.as_str().to_string()]);
  let payload = StringArray::from(vec![serde_json::to_string(&event.payload)?]);
  let timestamp = Int64Array::from(vec![event.timestamp.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    events_schema(),
    vec![
      Arc::new(id),
      Arc::new(entity_id),
      Arc::new(entity_type),
      Arc::new(event_type),
      Arc::new(payload),
      Arc::new(timestamp),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to an Event
fn batch_to_event(batch: &RecordBatch, row: usize) -> Result<Event> {
  let get_string = |name: &str| -> Result<String> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<StringArray>())
      .map(|a| a.value(row).to_string())
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
  let entity_type_str = get_string("entity_type")?;
  let event_type_str = get_string("event_type")?;
  let payload_str = get_string("payload")?;

  let entity_type = entity_type_str.parse::<EntityType>().map_err(DbError::NotFound)?;
  let event_type = event_type_str.parse::<EventType>().map_err(DbError::NotFound)?;

  let timestamp = Utc
    .timestamp_millis_opt(get_i64("timestamp")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid timestamp".into()))?;

  Ok(Event {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    entity_id: get_string("entity_id")?,
    entity_type,
    event_type,
    payload: serde_json::from_str(&payload_str)?,
    timestamp,
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
  async fn test_log_and_get_event() {
    let (_temp, db) = create_test_db().await;

    let event = Event::memory(
      "test-memory-123",
      EventType::Created,
      serde_json::json!({"content": "test"}),
    );

    db.log_event(&event).await.unwrap();

    let events = db
      .get_events_for_entity("test-memory-123", EntityType::Memory, None)
      .await
      .unwrap();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, EventType::Created);
  }

  #[tokio::test]
  async fn test_log_batch_events() {
    let (_temp, db) = create_test_db().await;

    let events = vec![
      Event::memory("mem-1", EventType::Created, serde_json::json!({})),
      Event::memory("mem-1", EventType::Accessed, serde_json::json!({})),
      Event::memory("mem-1", EventType::Reinforced, serde_json::json!({"amount": 0.1})),
    ];

    db.log_events(&events).await.unwrap();

    let retrieved = db
      .get_events_for_entity("mem-1", EntityType::Memory, None)
      .await
      .unwrap();

    assert_eq!(retrieved.len(), 3);
  }

  #[tokio::test]
  async fn test_get_recent_events() {
    let (_temp, db) = create_test_db().await;

    let events = vec![
      Event::memory("mem-1", EventType::Created, serde_json::json!({})),
      Event::session("sess-1", EventType::Created, serde_json::json!({})),
      Event::memory("mem-2", EventType::Created, serde_json::json!({})),
    ];

    db.log_events(&events).await.unwrap();

    let recent = db.get_recent_events(None, 10).await.unwrap();
    assert_eq!(recent.len(), 3);

    let memory_events = db.get_recent_events(Some(EventType::Created), 10).await.unwrap();
    assert_eq!(memory_events.len(), 3);
  }

  #[tokio::test]
  async fn test_count_events() {
    let (_temp, db) = create_test_db().await;

    let events = vec![
      Event::memory("mem-1", EventType::Created, serde_json::json!({})),
      Event::memory("mem-2", EventType::Created, serde_json::json!({})),
    ];

    db.log_events(&events).await.unwrap();

    let count = db.count_events(None).await.unwrap();
    assert_eq!(count, 2);
  }
}
