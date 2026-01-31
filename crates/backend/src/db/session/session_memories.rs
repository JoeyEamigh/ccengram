// Session-Memory linkage table operations
//
// Tracks how memories are used across sessions:
// - Created: Memory was created in this session
// - Recalled: Memory was retrieved/accessed in this session
// - Updated: Memory was modified in this session
// - Reinforced: Memory was confirmed/used repeatedly

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use chrono::{DateTime, TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::{
  db::{DbError, ProjectDb, Result},
  domain::memory::{MemoryId, Tier},
};

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
  /// Claude session ID string
  pub session_id: String,
  pub memory_id: String,
  pub usage_type: UsageType,
  pub linked_at: DateTime<Utc>,
}

impl ProjectDb {
  /// Get all memory links for a session
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_session_memory_links(&self, session_id: &str) -> Result<Vec<SessionMemoryLink>> {
    let table = self.session_memories_table();

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
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_memory_session_links(&self, memory_id: &str) -> Result<Vec<SessionMemoryLink>> {
    let table = self.session_memories_table();

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

  /// Promote session-tier memories to project-tier based on usage count
  ///
  /// Memories that have been used across multiple sessions (usage_count >= threshold)
  /// are promoted from Session tier to Project tier, making them persistent.
  ///
  /// Uses atomic updates to avoid race conditions when multiple promotions
  /// happen concurrently.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn promote_session_memories(&self, session_id: &str, threshold: usize) -> Result<usize> {
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
        // Atomic promotion - only promotes if tier is still 'session'
        // This avoids race conditions if another thread promotes first
        self.promote_memory_to_project(&memory_id).await?;
        promoted_count += 1;
      }
    }

    Ok(promoted_count)
  }

  /// Count how many times a memory was used across sessions
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn get_memory_usage_count(&self, memory_id: &str) -> Result<usize> {
    let links = self.get_memory_session_links(memory_id).await?;
    Ok(links.len())
  }

  /// Promote high-salience session-tier memories to project tier
  ///
  /// Memories with salience >= threshold are promoted from Session tier to Project tier.
  /// This ensures valuable learnings are persisted even if used only once.
  ///
  /// Uses atomic updates to avoid race conditions.
  #[tracing::instrument(level = "trace", skip(self))]
  pub async fn promote_high_salience_memories(&self, session_id: &str, salience_threshold: f32) -> Result<usize> {
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

      // Check salience threshold - we still need to read to check the threshold
      // But the promotion itself is atomic
      if let Ok(Some(memory)) = self.get_memory(&memory_id).await
        && memory.tier == Tier::Session
        && memory.salience >= salience_threshold
      {
        // Atomic promotion - only promotes if tier is still 'session'
        self.promote_memory_to_project(&memory_id).await?;
        promoted_count += 1;
      }
    }

    Ok(promoted_count)
  }
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
    session_id: session_id_str,
    memory_id,
    usage_type,
    linked_at,
  })
}

// Tests removed - the methods they tested (link_memory, get_session_stats) were dead code
