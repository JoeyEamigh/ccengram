// Segment accumulator persistence for extraction pipeline
//
// Tracks work context during a session segment:
// - User prompts (classified with signals)
// - Files read/modified
// - Commands run
// - Errors encountered
// - Searches performed
// - Completed tasks
// - Last assistant message

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use chrono::{DateTime, TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::segment_accumulators_schema;

/// A user prompt with signal classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPrompt {
  pub prompt: String,
  pub category: Option<String>, // correction, preference, context, task, question, feedback
  pub is_extractable: bool,
  pub timestamp: i64,
}

/// A command that was run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRecord {
  pub command: String,
  pub exit_code: i32,
}

/// Accumulated context from a session segment (persisted version)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentAccumulator {
  pub id: Uuid,
  pub session_id: Uuid,
  pub project_id: Uuid,
  pub segment_start: DateTime<Utc>,
  pub user_prompts: Vec<UserPrompt>,
  pub files_read: Vec<String>,
  pub files_modified: Vec<String>,
  pub commands_run: Vec<CommandRecord>,
  pub errors_encountered: Vec<String>,
  pub searches_performed: Vec<String>,
  pub completed_tasks: Vec<String>,
  pub last_assistant_message: Option<String>,
  pub tool_call_count: u32,
  pub updated_at: DateTime<Utc>,
}

impl SegmentAccumulator {
  /// Create a new segment accumulator
  pub fn new(session_id: Uuid, project_id: Uuid) -> Self {
    let now = Utc::now();
    Self {
      id: Uuid::now_v7(),
      session_id,
      project_id,
      segment_start: now,
      user_prompts: Vec::new(),
      files_read: Vec::new(),
      files_modified: Vec::new(),
      commands_run: Vec::new(),
      errors_encountered: Vec::new(),
      searches_performed: Vec::new(),
      completed_tasks: Vec::new(),
      last_assistant_message: None,
      tool_call_count: 0,
      updated_at: now,
    }
  }

  /// Add a user prompt
  pub fn add_user_prompt(&mut self, prompt: &str, category: Option<String>, is_extractable: bool) {
    self.user_prompts.push(UserPrompt {
      prompt: prompt.to_string(),
      category,
      is_extractable,
      timestamp: Utc::now().timestamp_millis(),
    });
    self.updated_at = Utc::now();
  }

  /// Add a file that was read (deduplicated)
  pub fn add_file_read(&mut self, path: &str) {
    if !self.files_read.contains(&path.to_string()) && self.files_read.len() < 100 {
      self.files_read.push(path.to_string());
      self.updated_at = Utc::now();
    }
  }

  /// Add a file that was modified (deduplicated)
  pub fn add_file_modified(&mut self, path: &str) {
    if !self.files_modified.contains(&path.to_string()) && self.files_modified.len() < 100 {
      self.files_modified.push(path.to_string());
      self.updated_at = Utc::now();
    }
  }

  /// Add a command that was run
  pub fn add_command(&mut self, command: &str, exit_code: i32) {
    if self.commands_run.len() < 50 {
      // Truncate long commands
      let cmd_display = if command.len() > 200 {
        format!("{}...", &command[..200])
      } else {
        command.to_string()
      };
      self.commands_run.push(CommandRecord {
        command: cmd_display,
        exit_code,
      });
      self.updated_at = Utc::now();
    }
  }

  /// Add an error that was encountered
  pub fn add_error(&mut self, error: &str) {
    if self.errors_encountered.len() < 20 {
      self.errors_encountered.push(error.to_string());
      self.updated_at = Utc::now();
    }
  }

  /// Add a search pattern (deduplicated)
  pub fn add_search(&mut self, pattern: &str) {
    if !self.searches_performed.contains(&pattern.to_string()) && self.searches_performed.len() < 50 {
      self.searches_performed.push(pattern.to_string());
      self.updated_at = Utc::now();
    }
  }

  /// Add a completed task (deduplicated)
  pub fn add_completed_task(&mut self, task: &str) {
    if !self.completed_tasks.contains(&task.to_string()) && self.completed_tasks.len() < 50 {
      self.completed_tasks.push(task.to_string());
      self.updated_at = Utc::now();
    }
  }

  /// Set the last assistant message (truncated to 10KB)
  pub fn set_last_assistant_message(&mut self, message: &str) {
    self.last_assistant_message = Some(if message.len() > 10240 {
      format!("{}...", &message[..10240])
    } else {
      message.to_string()
    });
    self.updated_at = Utc::now();
  }

  /// Increment tool call count
  pub fn increment_tool_calls(&mut self) {
    self.tool_call_count += 1;
    self.updated_at = Utc::now();
  }

  /// Check if this segment has meaningful work to extract
  pub fn has_meaningful_work(&self) -> bool {
    self.tool_call_count >= 3
      || !self.files_modified.is_empty()
      || !self.completed_tasks.is_empty()
      || !self.errors_encountered.is_empty()
  }

  /// Check for todo_completion trigger: ≥3 tasks AND ≥5 tool calls
  pub fn should_trigger_todo_extraction(&self) -> bool {
    self.completed_tasks.len() >= 3 && self.tool_call_count >= 5
  }

  /// Reset for a new segment while preserving IDs
  pub fn reset(&mut self) {
    let now = Utc::now();
    self.id = Uuid::now_v7();
    self.segment_start = now;
    self.user_prompts.clear();
    self.files_read.clear();
    self.files_modified.clear();
    self.commands_run.clear();
    self.errors_encountered.clear();
    self.searches_performed.clear();
    self.completed_tasks.clear();
    self.last_assistant_message = None;
    self.tool_call_count = 0;
    self.updated_at = now;
  }
}

impl ProjectDb {
  /// Save or update a segment accumulator
  pub async fn save_accumulator(&self, accumulator: &SegmentAccumulator) -> Result<()> {
    let table = self.segment_accumulators_table().await?;

    let batch = accumulator_to_batch(accumulator)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], segment_accumulators_schema());

    // Delete existing accumulator with same session_id first (only one active per session)
    let _ = table
      .delete(&format!("session_id = '{}'", accumulator.session_id))
      .await;

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get the accumulator for a session
  pub async fn get_accumulator(&self, session_id: &Uuid) -> Result<Option<SegmentAccumulator>> {
    let table = self.segment_accumulators_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("session_id = '{}'", session_id))
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

    Ok(Some(batch_to_accumulator(batch, 0)?))
  }

  /// Delete a segment accumulator (called after extraction completes)
  pub async fn clear_accumulator(&self, session_id: &Uuid) -> Result<()> {
    let table = self.segment_accumulators_table().await?;
    table.delete(&format!("session_id = '{}'", session_id)).await?;
    Ok(())
  }

  /// List all active accumulators (for cleanup/resume)
  pub async fn list_accumulators(&self) -> Result<Vec<SegmentAccumulator>> {
    let table = self.segment_accumulators_table().await?;

    let results: Vec<RecordBatch> = table.query().execute().await?.try_collect().await?;

    let mut accumulators = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        accumulators.push(batch_to_accumulator(&batch, i)?);
      }
    }

    Ok(accumulators)
  }
}

/// Convert a SegmentAccumulator to an Arrow RecordBatch
fn accumulator_to_batch(acc: &SegmentAccumulator) -> Result<RecordBatch> {
  let id = StringArray::from(vec![acc.id.to_string()]);
  let session_id = StringArray::from(vec![acc.session_id.to_string()]);
  let project_id = StringArray::from(vec![acc.project_id.to_string()]);
  let segment_start = Int64Array::from(vec![acc.segment_start.timestamp_millis()]);
  let user_prompts = StringArray::from(vec![serde_json::to_string(&acc.user_prompts)?]);
  let files_read = StringArray::from(vec![serde_json::to_string(&acc.files_read)?]);
  let files_modified = StringArray::from(vec![serde_json::to_string(&acc.files_modified)?]);
  let commands_run = StringArray::from(vec![serde_json::to_string(&acc.commands_run)?]);
  let errors_encountered = StringArray::from(vec![serde_json::to_string(&acc.errors_encountered)?]);
  let searches_performed = StringArray::from(vec![serde_json::to_string(&acc.searches_performed)?]);
  let completed_tasks = StringArray::from(vec![serde_json::to_string(&acc.completed_tasks)?]);
  let last_assistant_message = StringArray::from(vec![acc.last_assistant_message.clone()]);
  let tool_call_count = UInt32Array::from(vec![acc.tool_call_count]);
  let updated_at = Int64Array::from(vec![acc.updated_at.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    segment_accumulators_schema(),
    vec![
      Arc::new(id),
      Arc::new(session_id),
      Arc::new(project_id),
      Arc::new(segment_start),
      Arc::new(user_prompts),
      Arc::new(files_read),
      Arc::new(files_modified),
      Arc::new(commands_run),
      Arc::new(errors_encountered),
      Arc::new(searches_performed),
      Arc::new(completed_tasks),
      Arc::new(last_assistant_message),
      Arc::new(tool_call_count),
      Arc::new(updated_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to a SegmentAccumulator
fn batch_to_accumulator(batch: &RecordBatch, row: usize) -> Result<SegmentAccumulator> {
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
  let session_id_str = get_string("session_id")?;
  let project_id_str = get_string("project_id")?;

  let segment_start = Utc
    .timestamp_millis_opt(get_i64("segment_start")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid segment_start timestamp".into()))?;
  let updated_at = Utc
    .timestamp_millis_opt(get_i64("updated_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid updated_at timestamp".into()))?;

  let user_prompts: Vec<UserPrompt> = serde_json::from_str(&get_string("user_prompts")?)?;
  let files_read: Vec<String> = serde_json::from_str(&get_string("files_read")?)?;
  let files_modified: Vec<String> = serde_json::from_str(&get_string("files_modified")?)?;
  let commands_run: Vec<CommandRecord> = serde_json::from_str(&get_string("commands_run")?)?;
  let errors_encountered: Vec<String> = serde_json::from_str(&get_string("errors_encountered")?)?;
  let searches_performed: Vec<String> = serde_json::from_str(&get_string("searches_performed")?)?;
  let completed_tasks: Vec<String> = serde_json::from_str(&get_string("completed_tasks")?)?;

  Ok(SegmentAccumulator {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    session_id: Uuid::parse_str(&session_id_str).map_err(|_| DbError::NotFound("invalid session_id".into()))?,
    project_id: Uuid::parse_str(&project_id_str).map_err(|_| DbError::NotFound("invalid project_id".into()))?,
    segment_start,
    user_prompts,
    files_read,
    files_modified,
    commands_run,
    errors_encountered,
    searches_performed,
    completed_tasks,
    last_assistant_message: get_optional_string("last_assistant_message"),
    tool_call_count: get_u32("tool_call_count")?,
    updated_at,
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
  async fn test_save_and_get_accumulator() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut acc = SegmentAccumulator::new(session_id, project_id);
    acc.add_user_prompt("Hello", Some("task".to_string()), false);
    acc.add_file_read("/src/main.rs");
    acc.add_file_modified("/src/lib.rs");
    acc.add_command("cargo build", 0);
    acc.increment_tool_calls();

    db.save_accumulator(&acc).await.unwrap();

    let retrieved = db.get_accumulator(&session_id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.session_id, session_id);
    assert_eq!(retrieved.user_prompts.len(), 1);
    assert_eq!(retrieved.files_read.len(), 1);
    assert_eq!(retrieved.files_modified.len(), 1);
    assert_eq!(retrieved.commands_run.len(), 1);
    assert_eq!(retrieved.tool_call_count, 1);
  }

  #[tokio::test]
  async fn test_accumulator_limits() {
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut acc = SegmentAccumulator::new(session_id, project_id);

    // Files should be deduplicated
    acc.add_file_read("/src/main.rs");
    acc.add_file_read("/src/main.rs");
    acc.add_file_read("/src/lib.rs");
    assert_eq!(acc.files_read.len(), 2);

    // Commands should be limited
    for i in 0..60 {
      acc.add_command(&format!("command {}", i), 0);
    }
    assert_eq!(acc.commands_run.len(), 50); // Limited to 50
  }

  #[tokio::test]
  async fn test_accumulator_reset() {
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut acc = SegmentAccumulator::new(session_id, project_id);
    let original_id = acc.id;

    acc.add_file_read("/src/main.rs");
    acc.increment_tool_calls();

    acc.reset();

    assert_ne!(acc.id, original_id); // New ID
    assert!(acc.files_read.is_empty());
    assert_eq!(acc.tool_call_count, 0);
    assert_eq!(acc.session_id, session_id); // Session preserved
    assert_eq!(acc.project_id, project_id); // Project preserved
  }

  #[tokio::test]
  async fn test_meaningful_work_detection() {
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut acc = SegmentAccumulator::new(session_id, project_id);
    assert!(!acc.has_meaningful_work());

    // File modifications count
    acc.add_file_modified("/src/main.rs");
    assert!(acc.has_meaningful_work());

    acc.reset();

    // Completed tasks count
    acc.add_completed_task("Fix bug");
    assert!(acc.has_meaningful_work());

    acc.reset();

    // 3+ tool calls count
    acc.increment_tool_calls();
    acc.increment_tool_calls();
    assert!(!acc.has_meaningful_work());
    acc.increment_tool_calls();
    assert!(acc.has_meaningful_work());
  }

  #[tokio::test]
  async fn test_todo_extraction_trigger() {
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut acc = SegmentAccumulator::new(session_id, project_id);

    // Not triggered with insufficient tasks/tools
    acc.add_completed_task("Task 1");
    acc.add_completed_task("Task 2");
    for _ in 0..5 {
      acc.increment_tool_calls();
    }
    assert!(!acc.should_trigger_todo_extraction());

    // Triggered with 3+ tasks and 5+ tools
    acc.add_completed_task("Task 3");
    assert!(acc.should_trigger_todo_extraction());
  }

  #[tokio::test]
  async fn test_clear_accumulator() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let acc = SegmentAccumulator::new(session_id, project_id);
    db.save_accumulator(&acc).await.unwrap();

    // Verify exists
    assert!(db.get_accumulator(&session_id).await.unwrap().is_some());

    // Clear
    db.clear_accumulator(&session_id).await.unwrap();

    // Verify gone
    assert!(db.get_accumulator(&session_id).await.unwrap().is_none());
  }
}
