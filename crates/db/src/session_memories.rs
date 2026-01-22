// Session-Memory linkage table operations
//
// Tracks how memories are used across sessions:
// - Created: Memory was created in this session
// - Recalled: Memory was retrieved/accessed in this session
// - Updated: Memory was modified in this session
// - Reinforced: Memory was confirmed/used repeatedly

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::{DateTime, TimeZone, Utc};
use engram_core::{MemoryId, Tier};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::session_memories_schema;
use tracing::warn;

/// Usage type for session-memory linkage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageType {
  /// Memory was created in this session
  Created,
  /// Memory was recalled/accessed in this session
  Recalled,
  /// Memory was updated/modified in this session
  Updated,
  /// Memory was confirmed/reinforced in this session
  Reinforced,
}

impl UsageType {
  pub fn as_str(&self) -> &'static str {
    match self {
      UsageType::Created => "created",
      UsageType::Recalled => "recalled",
      UsageType::Updated => "updated",
      UsageType::Reinforced => "reinforced",
    }
  }
}

impl std::str::FromStr for UsageType {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "created" => Ok(UsageType::Created),
      "recalled" => Ok(UsageType::Recalled),
      "updated" => Ok(UsageType::Updated),
      "reinforced" => Ok(UsageType::Reinforced),
      _ => Err(format!("Unknown usage type: {}", s)),
    }
  }
}

/// A session-memory linkage record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMemoryLink {
  pub id: Uuid,
  pub session_id: Uuid,
  pub memory_id: String,
  pub usage_type: UsageType,
  pub linked_at: DateTime<Utc>,
}

impl SessionMemoryLink {
  pub fn new(session_id: Uuid, memory_id: String, usage_type: UsageType) -> Self {
    Self {
      id: Uuid::now_v7(),
      session_id,
      memory_id,
      usage_type,
      linked_at: Utc::now(),
    }
  }
}

/// Session statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
  pub total_memories: usize,
  pub created: usize,
  pub recalled: usize,
  pub updated: usize,
  pub reinforced: usize,
  /// Memory count by sector
  pub by_sector: HashMap<String, usize>,
  /// Average salience of memories in session
  pub average_salience: f32,
}

impl ProjectDb {
  /// Link a memory to a session
  pub async fn link_memory(&self, session_id: Uuid, memory_id: &str, usage_type: UsageType) -> Result<()> {
    let link = SessionMemoryLink::new(session_id, memory_id.to_string(), usage_type);
    self.add_session_memory_link(&link).await
  }

  /// Add a session-memory link to the database
  pub async fn add_session_memory_link(&self, link: &SessionMemoryLink) -> Result<()> {
    let table = self.session_memories_table().await?;

    let batch = link_to_batch(link)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], session_memories_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get all memory links for a session
  pub async fn get_session_memory_links(&self, session_id: &Uuid) -> Result<Vec<SessionMemoryLink>> {
    let table = self.session_memories_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("session_id = '{}'", session_id))
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

  /// Get all session links for a memory
  pub async fn get_memory_session_links(&self, memory_id: &str) -> Result<Vec<SessionMemoryLink>> {
    let table = self.session_memories_table().await?;

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

  /// Get session statistics
  pub async fn get_session_stats(&self, session_id: &Uuid) -> Result<SessionStats> {
    let links = self.get_session_memory_links(session_id).await?;

    let mut stats = SessionStats {
      total_memories: links.len(),
      ..Default::default()
    };

    // Count by usage type
    for link in &links {
      match link.usage_type {
        UsageType::Created => stats.created += 1,
        UsageType::Recalled => stats.recalled += 1,
        UsageType::Updated => stats.updated += 1,
        UsageType::Reinforced => stats.reinforced += 1,
      }
    }

    // Get memories to compute sector breakdown and average salience
    let mut total_salience = 0.0f32;
    let mut memory_count = 0usize;

    for link in &links {
      let memory_id = match link.memory_id.parse::<MemoryId>() {
        Ok(id) => id,
        Err(_) => {
          warn!("Invalid memory_id UUID in session link: {}", link.memory_id);
          continue;
        }
      };
      if let Ok(Some(memory)) = self.get_memory(&memory_id).await {
        // Count by sector
        let sector_name = memory.sector.as_str().to_string();
        *stats.by_sector.entry(sector_name).or_insert(0) += 1;

        // Accumulate salience
        total_salience += memory.salience;
        memory_count += 1;
      }
    }

    // Compute average salience
    if memory_count > 0 {
      stats.average_salience = total_salience / memory_count as f32;
    }

    Ok(stats)
  }

  /// Promote session-tier memories to project-tier based on usage count
  ///
  /// Memories that have been used across multiple sessions (usage_count >= threshold)
  /// are promoted from Session tier to Project tier, making them persistent.
  pub async fn promote_session_memories(&self, session_id: &Uuid, threshold: usize) -> Result<usize> {
    let links = self.get_session_memory_links(session_id).await?;

    let mut promoted_count = 0;

    for link in links {
      // Only consider memories created in this session
      if link.usage_type != UsageType::Created {
        continue;
      }

      // Get usage count across all sessions
      let usage_count = self.get_memory_usage_count(&link.memory_id).await?;

      if usage_count >= threshold {
        let memory_id = match link.memory_id.parse::<MemoryId>() {
          Ok(id) => id,
          Err(_) => {
            warn!("Invalid memory_id UUID in session link: {}", link.memory_id);
            continue;
          }
        };
        if let Ok(Some(mut memory)) = self.get_memory(&memory_id).await
          && memory.tier == Tier::Session
        {
          // Promote to project tier
          memory.tier = Tier::Project;
          memory.updated_at = Utc::now();
          self.update_memory(&memory, None).await?;
          promoted_count += 1;
        }
      }
    }

    Ok(promoted_count)
  }

  /// Count how many times a memory was used across sessions
  pub async fn get_memory_usage_count(&self, memory_id: &str) -> Result<usize> {
    let links = self.get_memory_session_links(memory_id).await?;
    Ok(links.len())
  }

  /// Promote high-salience session-tier memories to project tier
  ///
  /// Memories with salience >= threshold are promoted from Session tier to Project tier.
  /// This ensures valuable learnings are persisted even if used only once.
  pub async fn promote_high_salience_memories(&self, session_id: &Uuid, salience_threshold: f32) -> Result<usize> {
    let links = self.get_session_memory_links(session_id).await?;

    let mut promoted_count = 0;

    for link in links {
      // Only consider memories created in this session
      if link.usage_type != UsageType::Created {
        continue;
      }

      // Get the memory and check if it's session-tier with high salience
      let memory_id = match link.memory_id.parse::<MemoryId>() {
        Ok(id) => id,
        Err(_) => {
          warn!("Invalid memory_id UUID in session link: {}", link.memory_id);
          continue;
        }
      };
      if let Ok(Some(mut memory)) = self.get_memory(&memory_id).await
        && memory.tier == Tier::Session
        && memory.salience >= salience_threshold
      {
        // Promote to project tier
        memory.tier = Tier::Project;
        memory.updated_at = Utc::now();
        self.update_memory(&memory, None).await?;
        promoted_count += 1;
      }
    }

    Ok(promoted_count)
  }

  /// Delete all links for a session
  pub async fn delete_session_links(&self, session_id: &Uuid) -> Result<()> {
    let table = self.session_memories_table().await?;
    table.delete(&format!("session_id = '{}'", session_id)).await?;
    Ok(())
  }

  /// Delete all links for a memory
  pub async fn delete_memory_links(&self, memory_id: &str) -> Result<()> {
    let table = self.session_memories_table().await?;
    table.delete(&format!("memory_id = '{}'", memory_id)).await?;
    Ok(())
  }

  /// Delete a specific session-memory link by ID
  pub async fn delete_session_memory_link(&self, id: &Uuid) -> Result<()> {
    let table = self.session_memories_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }
}

/// Convert a SessionMemoryLink to an Arrow RecordBatch
fn link_to_batch(link: &SessionMemoryLink) -> Result<RecordBatch> {
  let id = StringArray::from(vec![link.id.to_string()]);
  let session_id = StringArray::from(vec![link.session_id.to_string()]);
  let memory_id = StringArray::from(vec![link.memory_id.clone()]);
  let usage_type = StringArray::from(vec![link.usage_type.as_str().to_string()]);
  let linked_at = Int64Array::from(vec![link.linked_at.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    session_memories_schema(),
    vec![
      Arc::new(id),
      Arc::new(session_id),
      Arc::new(memory_id),
      Arc::new(usage_type),
      Arc::new(linked_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a SessionMemoryLink
fn batch_to_link(batch: &RecordBatch, row: usize) -> Result<SessionMemoryLink> {
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
  let session_id_str = get_string("session_id")?;
  let memory_id = get_string("memory_id")?;
  let usage_type_str = get_string("usage_type")?;
  let linked_at_ts = get_i64("linked_at")?;

  let usage_type = usage_type_str.parse::<UsageType>().map_err(DbError::NotFound)?;

  let linked_at = Utc
    .timestamp_millis_opt(linked_at_ts)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid linked_at timestamp".into()))?;

  Ok(SessionMemoryLink {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    session_id: Uuid::parse_str(&session_id_str).map_err(|_| DbError::NotFound("invalid session_id".into()))?,
    memory_id,
    usage_type,
    linked_at,
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
  async fn test_link_memory() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let memory_id = "mem-123";

    db.link_memory(session_id, memory_id, UsageType::Created).await.unwrap();

    let links = db.get_session_memory_links(&session_id).await.unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].memory_id, memory_id);
    assert_eq!(links[0].usage_type, UsageType::Created);
  }

  #[tokio::test]
  async fn test_get_memory_session_links() {
    let (_temp, db) = create_test_db().await;
    let session1 = Uuid::new_v4();
    let session2 = Uuid::new_v4();
    let memory_id = "mem-456";

    db.link_memory(session1, memory_id, UsageType::Created).await.unwrap();
    db.link_memory(session2, memory_id, UsageType::Recalled).await.unwrap();

    let links = db.get_memory_session_links(memory_id).await.unwrap();
    assert_eq!(links.len(), 2);
  }

  #[tokio::test]
  async fn test_session_stats() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();

    db.link_memory(session_id, "mem-1", UsageType::Created).await.unwrap();
    db.link_memory(session_id, "mem-2", UsageType::Created).await.unwrap();
    db.link_memory(session_id, "mem-3", UsageType::Recalled).await.unwrap();

    let stats = db.get_session_stats(&session_id).await.unwrap();
    assert_eq!(stats.total_memories, 3);
    assert_eq!(stats.created, 2);
    assert_eq!(stats.recalled, 1);
  }

  #[tokio::test]
  async fn test_usage_type_parsing() {
    assert_eq!("created".parse::<UsageType>().unwrap(), UsageType::Created);
    assert_eq!("recalled".parse::<UsageType>().unwrap(), UsageType::Recalled);
    assert_eq!("updated".parse::<UsageType>().unwrap(), UsageType::Updated);
    assert_eq!("reinforced".parse::<UsageType>().unwrap(), UsageType::Reinforced);
  }

  #[tokio::test]
  async fn test_session_stats_extended() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();

    // Create memories with different sectors
    let mut m1 = engram_core::Memory::new(Uuid::new_v4(), "Memory 1".to_string(), engram_core::Sector::Semantic);
    m1.content_hash = "hash1".to_string();
    m1.salience = 0.8;

    let mut m2 = engram_core::Memory::new(Uuid::new_v4(), "Memory 2".to_string(), engram_core::Sector::Emotional);
    m2.content_hash = "hash2".to_string();
    m2.salience = 0.6;

    let mut m3 = engram_core::Memory::new(Uuid::new_v4(), "Memory 3".to_string(), engram_core::Sector::Semantic);
    m3.content_hash = "hash3".to_string();
    m3.salience = 0.4;

    // Add memories to db
    db.add_memory(&m1, None).await.unwrap();
    db.add_memory(&m2, None).await.unwrap();
    db.add_memory(&m3, None).await.unwrap();

    // Link memories to session
    db.link_memory(session_id, &m1.id.to_string(), UsageType::Created)
      .await
      .unwrap();
    db.link_memory(session_id, &m2.id.to_string(), UsageType::Created)
      .await
      .unwrap();
    db.link_memory(session_id, &m3.id.to_string(), UsageType::Recalled)
      .await
      .unwrap();

    let stats = db.get_session_stats(&session_id).await.unwrap();

    assert_eq!(stats.total_memories, 3);
    assert_eq!(stats.created, 2);
    assert_eq!(stats.recalled, 1);
    assert_eq!(*stats.by_sector.get("semantic").unwrap_or(&0), 2);
    assert_eq!(*stats.by_sector.get("emotional").unwrap_or(&0), 1);
    // Average salience: (0.8 + 0.6 + 0.4) / 3 = 0.6
    assert!((stats.average_salience - 0.6).abs() < 0.01);
  }

  #[tokio::test]
  async fn test_promote_session_memories() {
    let (_temp, db) = create_test_db().await;
    let session1 = Uuid::new_v4();
    let session2 = Uuid::new_v4();

    // Create a session-tier memory
    let mut memory = engram_core::Memory::new(
      Uuid::new_v4(),
      "Session memory".to_string(),
      engram_core::Sector::Semantic,
    );
    memory.content_hash = "hash1".to_string();
    memory.tier = engram_core::Tier::Session;

    db.add_memory(&memory, None).await.unwrap();

    // Link to session1 as created
    db.link_memory(session1, &memory.id.to_string(), UsageType::Created)
      .await
      .unwrap();

    // Promotion with threshold 2 should not promote (only 1 usage)
    let promoted = db.promote_session_memories(&session1, 2).await.unwrap();
    assert_eq!(promoted, 0);

    // Link to session2 as recalled (second usage)
    db.link_memory(session2, &memory.id.to_string(), UsageType::Recalled)
      .await
      .unwrap();

    // Now promotion should work (2 usages >= threshold 2)
    let promoted = db.promote_session_memories(&session1, 2).await.unwrap();
    assert_eq!(promoted, 1);

    // Verify memory is now project-tier
    let updated = db.get_memory(&memory.id).await.unwrap().unwrap();
    assert_eq!(updated.tier, engram_core::Tier::Project);
  }

  #[tokio::test]
  async fn test_promote_high_salience_memories() {
    let (_temp, db) = create_test_db().await;
    let session = Uuid::new_v4();

    // Create a high-salience session-tier memory
    let mut high_salience = engram_core::Memory::new(
      Uuid::new_v4(),
      "Important preference to remember".to_string(),
      engram_core::Sector::Semantic,
    );
    high_salience.content_hash = "hash_high".to_string();
    high_salience.tier = engram_core::Tier::Session;
    high_salience.salience = 0.9; // High salience

    // Create a low-salience session-tier memory
    let mut low_salience = engram_core::Memory::new(
      Uuid::new_v4(),
      "Unimportant session note".to_string(),
      engram_core::Sector::Episodic,
    );
    low_salience.content_hash = "hash_low".to_string();
    low_salience.tier = engram_core::Tier::Session;
    low_salience.salience = 0.3; // Low salience

    db.add_memory(&high_salience, None).await.unwrap();
    db.add_memory(&low_salience, None).await.unwrap();

    // Link both to session as created
    db.link_memory(session, &high_salience.id.to_string(), UsageType::Created)
      .await
      .unwrap();
    db.link_memory(session, &low_salience.id.to_string(), UsageType::Created)
      .await
      .unwrap();

    // Promote with threshold 0.8 - only high-salience should be promoted
    let promoted = db.promote_high_salience_memories(&session, 0.8).await.unwrap();
    assert_eq!(promoted, 1);

    // Verify high-salience is now project-tier
    let updated_high = db.get_memory(&high_salience.id).await.unwrap().unwrap();
    assert_eq!(updated_high.tier, engram_core::Tier::Project);

    // Verify low-salience is still session-tier
    let updated_low = db.get_memory(&low_salience.id).await.unwrap().unwrap();
    assert_eq!(updated_low.tier, engram_core::Tier::Session);
  }
}
