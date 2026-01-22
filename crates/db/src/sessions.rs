// Sessions table operations

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::{DateTime, TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::sessions_schema;

/// A session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
  pub id: Uuid,
  pub project_id: Uuid,
  pub started_at: DateTime<Utc>,
  pub ended_at: Option<DateTime<Utc>>,
  pub summary: Option<String>,
  pub user_prompt: Option<String>,
  pub context: Option<serde_json::Value>,
}

impl Session {
  pub fn new(project_id: Uuid) -> Self {
    Self {
      id: Uuid::now_v7(),
      project_id,
      started_at: Utc::now(),
      ended_at: None,
      summary: None,
      user_prompt: None,
      context: None,
    }
  }
}

impl ProjectDb {
  /// Add a new session to the database
  pub async fn add_session(&self, session: &Session) -> Result<()> {
    let table = self.sessions_table().await?;

    let batch = session_to_batch(session)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], sessions_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get a session by ID
  pub async fn get_session(&self, id: &Uuid) -> Result<Option<Session>> {
    let table = self.sessions_table().await?;
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

    Ok(Some(batch_to_session(batch, 0)?))
  }

  /// Update a session
  pub async fn update_session(&self, session: &Session) -> Result<()> {
    let table = self.sessions_table().await?;

    let batch = session_to_batch(session)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], sessions_schema());

    // Delete old and insert new
    let _ = table.delete(&format!("id = '{}'", session.id)).await;
    table.add(Box::new(batches)).execute().await?;

    Ok(())
  }

  /// End a session
  pub async fn end_session(&self, id: &Uuid, summary: Option<String>) -> Result<()> {
    if let Some(mut session) = self.get_session(id).await? {
      session.ended_at = Some(Utc::now());
      session.summary = summary;
      self.update_session(&session).await?;
    }
    Ok(())
  }

  /// List sessions for a project
  pub async fn list_sessions(&self, filter: Option<&str>, limit: Option<usize>) -> Result<Vec<Session>> {
    let table = self.sessions_table().await?;

    let query = match (filter, limit) {
      (Some(f), Some(l)) => table.query().only_if(f).limit(l),
      (Some(f), None) => table.query().only_if(f),
      (None, Some(l)) => table.query().limit(l),
      (None, None) => table.query(),
    };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut sessions = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        sessions.push(batch_to_session(&batch, i)?);
      }
    }

    Ok(sessions)
  }

  /// Get recent sessions
  pub async fn recent_sessions(&self, limit: usize) -> Result<Vec<Session>> {
    // List and sort by started_at descending
    let mut sessions = self.list_sessions(None, Some(limit * 2)).await?;
    sessions.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    sessions.truncate(limit);
    Ok(sessions)
  }

  /// Get active session (most recent without end time)
  pub async fn active_session(&self) -> Result<Option<Session>> {
    let sessions = self.list_sessions(Some("ended_at IS NULL"), Some(1)).await?;
    Ok(sessions.into_iter().next())
  }

  /// Get active session for a project, ending any stale sessions first
  pub async fn get_or_end_active_session(&self, project_id: &Uuid, max_age_hours: u64) -> Result<Option<Session>> {
    // Get all active sessions for this project
    let filter = format!("project_id = '{}' AND ended_at IS NULL", project_id);
    let sessions = self.list_sessions(Some(&filter), None).await?;

    if sessions.is_empty() {
      return Ok(None);
    }

    let now = Utc::now();
    let max_age = chrono::Duration::hours(max_age_hours as i64);

    // Find the most recent non-stale session, ending any stale ones
    let mut active: Option<Session> = None;
    for session in sessions {
      let age = now.signed_duration_since(session.started_at);
      if age > max_age {
        // End stale session
        self
          .end_session(&session.id, Some("Session timed out".to_string()))
          .await?;
      } else if active.is_none() || session.started_at > active.as_ref().unwrap().started_at {
        active = Some(session);
      }
    }

    Ok(active)
  }

  /// Cleanup stale sessions (those without end time older than max_age_hours)
  pub async fn cleanup_stale_sessions(&self, max_age_hours: u64) -> Result<usize> {
    let sessions = self.list_sessions(Some("ended_at IS NULL"), None).await?;

    let now = Utc::now();
    let max_age = chrono::Duration::hours(max_age_hours as i64);

    let mut cleaned = 0;
    for session in sessions {
      let age = now.signed_duration_since(session.started_at);
      if age > max_age {
        self
          .end_session(&session.id, Some("Session timed out (cleanup)".to_string()))
          .await?;
        cleaned += 1;
      }
    }

    Ok(cleaned)
  }

  /// Delete a session and its memory links
  pub async fn delete_session(&self, id: &Uuid) -> Result<()> {
    // Delete session-memory links first
    let links = self.get_session_memory_links(id).await?;
    for link in links {
      self.delete_session_memory_link(&link.id).await?;
    }

    // Delete the session
    let table = self.sessions_table().await?;
    table.delete(&format!("id = '{}'", id)).await?;
    Ok(())
  }

  /// Count sessions for a project
  pub async fn count_sessions(&self, project_id: &Uuid) -> Result<usize> {
    let filter = format!("project_id = '{}'", project_id);
    let sessions = self.list_sessions(Some(&filter), None).await?;
    Ok(sessions.len())
  }
}

/// Convert a Session to an Arrow RecordBatch
fn session_to_batch(session: &Session) -> Result<RecordBatch> {
  let id = StringArray::from(vec![session.id.to_string()]);
  let project_id = StringArray::from(vec![session.project_id.to_string()]);
  let started_at = Int64Array::from(vec![session.started_at.timestamp_millis()]);
  let ended_at = Int64Array::from(vec![session.ended_at.map(|t| t.timestamp_millis())]);
  let summary = StringArray::from(vec![session.summary.clone()]);
  let user_prompt = StringArray::from(vec![session.user_prompt.clone()]);
  let context = StringArray::from(vec![
    session
      .context
      .as_ref()
      .map(|c| serde_json::to_string(c).unwrap_or_default()),
  ]);

  let batch = RecordBatch::try_new(
    sessions_schema(),
    vec![
      Arc::new(id),
      Arc::new(project_id),
      Arc::new(started_at),
      Arc::new(ended_at),
      Arc::new(summary),
      Arc::new(user_prompt),
      Arc::new(context),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a Session
fn batch_to_session(batch: &RecordBatch, row: usize) -> Result<Session> {
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

  let get_optional_i64 = |name: &str| -> Option<i64> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
      .and_then(|a| if a.is_null(row) { None } else { Some(a.value(row)) })
  };

  let id_str = get_string("id")?;
  let project_id_str = get_string("project_id")?;

  let started_at = Utc
    .timestamp_millis_opt(get_i64("started_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid started_at timestamp".into()))?;
  let ended_at = get_optional_i64("ended_at").and_then(|ts| Utc.timestamp_millis_opt(ts).single());

  let context = get_optional_string("context").and_then(|s| serde_json::from_str(&s).ok());

  Ok(Session {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    project_id: Uuid::parse_str(&project_id_str).map_err(|_| DbError::NotFound("invalid project_id".into()))?,
    started_at,
    ended_at,
    summary: get_optional_string("summary"),
    user_prompt: get_optional_string("user_prompt"),
    context,
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
  async fn test_add_and_get_session() {
    let (_temp, db) = create_test_db().await;
    let session = Session::new(Uuid::new_v4());

    db.add_session(&session).await.unwrap();

    let retrieved = db.get_session(&session.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, session.id);
  }

  #[tokio::test]
  async fn test_end_session() {
    let (_temp, db) = create_test_db().await;
    let session = Session::new(Uuid::new_v4());

    db.add_session(&session).await.unwrap();
    db.end_session(&session.id, Some("Test summary".to_string()))
      .await
      .unwrap();

    let retrieved = db.get_session(&session.id).await.unwrap().unwrap();
    assert!(retrieved.ended_at.is_some());
    assert_eq!(retrieved.summary, Some("Test summary".to_string()));
  }

  #[tokio::test]
  async fn test_list_sessions() {
    let (_temp, db) = create_test_db().await;
    let project_id = Uuid::new_v4();

    let s1 = Session::new(project_id);
    let s2 = Session::new(project_id);

    db.add_session(&s1).await.unwrap();
    db.add_session(&s2).await.unwrap();

    let sessions = db.list_sessions(None, None).await.unwrap();
    assert_eq!(sessions.len(), 2);
  }
}
