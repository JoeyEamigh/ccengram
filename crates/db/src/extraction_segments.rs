// Extraction segment persistence - records extraction runs for auditing and debugging
//
// Each extraction run (triggered by user_prompt, pre_compact, stop, or todo_completion)
// is recorded with metadata about what was extracted and how long it took.

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use chrono::{DateTime, TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{DbError, ProjectDb, Result};
use crate::schema::extraction_segments_schema;

/// Extraction trigger type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionTrigger {
  /// New user prompt submitted (segment boundary)
  UserPrompt,
  /// Before LLM context compaction
  PreCompact,
  /// End of session/conversation
  Stop,
  /// 3+ tasks completed AND 5+ tool calls
  TodoCompletion,
}

impl ExtractionTrigger {
  pub fn as_str(&self) -> &'static str {
    match self {
      ExtractionTrigger::UserPrompt => "user_prompt",
      ExtractionTrigger::PreCompact => "pre_compact",
      ExtractionTrigger::Stop => "stop",
      ExtractionTrigger::TodoCompletion => "todo_completion",
    }
  }
}

impl std::str::FromStr for ExtractionTrigger {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "user_prompt" => Ok(ExtractionTrigger::UserPrompt),
      "pre_compact" => Ok(ExtractionTrigger::PreCompact),
      "stop" => Ok(ExtractionTrigger::Stop),
      "todo_completion" => Ok(ExtractionTrigger::TodoCompletion),
      _ => Err(format!("Unknown extraction trigger: {}", s)),
    }
  }
}

/// Record of an extraction run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSegment {
  pub id: Uuid,
  pub session_id: Uuid,
  pub project_id: Uuid,
  pub trigger: ExtractionTrigger,
  pub user_prompts_json: String,
  pub files_read_count: u32,
  pub files_modified_count: u32,
  pub tool_call_count: u32,
  pub memories_extracted: u32,
  pub extraction_duration_ms: u32,
  pub input_tokens: Option<u32>,
  pub output_tokens: Option<u32>,
  pub model_used: Option<String>,
  pub error: Option<String>,
  pub created_at: DateTime<Utc>,
}

impl ExtractionSegment {
  /// Create a new extraction segment record
  pub fn new(
    session_id: Uuid,
    project_id: Uuid,
    trigger: ExtractionTrigger,
    user_prompts: &[String],
    files_read_count: u32,
    files_modified_count: u32,
    tool_call_count: u32,
  ) -> Self {
    Self {
      id: Uuid::now_v7(),
      session_id,
      project_id,
      trigger,
      user_prompts_json: serde_json::to_string(user_prompts).unwrap_or_else(|_| "[]".to_string()),
      files_read_count,
      files_modified_count,
      tool_call_count,
      memories_extracted: 0,
      extraction_duration_ms: 0,
      input_tokens: None,
      output_tokens: None,
      model_used: None,
      error: None,
      created_at: Utc::now(),
    }
  }

  /// Record successful extraction results
  pub fn record_success(
    &mut self,
    memories_extracted: u32,
    duration_ms: u32,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    model: Option<&str>,
  ) {
    self.memories_extracted = memories_extracted;
    self.extraction_duration_ms = duration_ms;
    self.input_tokens = input_tokens;
    self.output_tokens = output_tokens;
    self.model_used = model.map(|s| s.to_string());
  }

  /// Record extraction failure
  pub fn record_failure(&mut self, error: &str, duration_ms: u32) {
    self.error = Some(error.to_string());
    self.extraction_duration_ms = duration_ms;
  }
}

impl ProjectDb {
  /// Save an extraction segment record
  pub async fn save_extraction_segment(&self, segment: &ExtractionSegment) -> Result<()> {
    let table = self.extraction_segments_table().await?;

    let batch = segment_to_batch(segment)?;
    let batches = RecordBatchIterator::new(vec![Ok(batch)], extraction_segments_schema());

    table.add(Box::new(batches)).execute().await?;
    Ok(())
  }

  /// Get an extraction segment by ID
  pub async fn get_extraction_segment(&self, id: &Uuid) -> Result<Option<ExtractionSegment>> {
    let table = self.extraction_segments_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("id = '{}'", id))
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

    Ok(Some(batch_to_segment(batch, 0)?))
  }

  /// List extraction segments for a session
  pub async fn list_extraction_segments(
    &self,
    session_id: &Uuid,
    limit: Option<usize>,
  ) -> Result<Vec<ExtractionSegment>> {
    let table = self.extraction_segments_table().await?;

    let query = table.query().only_if(format!("session_id = '{}'", session_id));

    let query = if let Some(l) = limit { query.limit(l) } else { query };

    let results: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

    let mut segments = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        segments.push(batch_to_segment(&batch, i)?);
      }
    }

    // Sort by created_at descending
    segments.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(segments)
  }

  /// Get extraction statistics for a project
  pub async fn extraction_stats(&self, project_id: &Uuid) -> Result<ExtractionStats> {
    let table = self.extraction_segments_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("project_id = '{}'", project_id))
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut stats = ExtractionStats::default();

    for batch in results {
      for i in 0..batch.num_rows() {
        let segment = batch_to_segment(&batch, i)?;
        stats.total_extractions += 1;
        stats.total_memories_extracted += segment.memories_extracted as u64;

        if segment.error.is_some() {
          stats.failed_extractions += 1;
        }

        match segment.trigger {
          ExtractionTrigger::UserPrompt => stats.user_prompt_triggers += 1,
          ExtractionTrigger::PreCompact => stats.pre_compact_triggers += 1,
          ExtractionTrigger::Stop => stats.stop_triggers += 1,
          ExtractionTrigger::TodoCompletion => stats.todo_completion_triggers += 1,
        }
      }
    }

    Ok(stats)
  }
}

/// Statistics about extraction runs
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ExtractionStats {
  pub total_extractions: u64,
  pub failed_extractions: u64,
  pub total_memories_extracted: u64,
  pub user_prompt_triggers: u64,
  pub pre_compact_triggers: u64,
  pub stop_triggers: u64,
  pub todo_completion_triggers: u64,
}

/// Convert an ExtractionSegment to an Arrow RecordBatch
fn segment_to_batch(segment: &ExtractionSegment) -> Result<RecordBatch> {
  let id = StringArray::from(vec![segment.id.to_string()]);
  let session_id = StringArray::from(vec![segment.session_id.to_string()]);
  let project_id = StringArray::from(vec![segment.project_id.to_string()]);
  let trigger = StringArray::from(vec![segment.trigger.as_str().to_string()]);
  let user_prompts_json = StringArray::from(vec![segment.user_prompts_json.clone()]);
  let files_read_count = UInt32Array::from(vec![segment.files_read_count]);
  let files_modified_count = UInt32Array::from(vec![segment.files_modified_count]);
  let tool_call_count = UInt32Array::from(vec![segment.tool_call_count]);
  let memories_extracted = UInt32Array::from(vec![segment.memories_extracted]);
  let extraction_duration_ms = UInt32Array::from(vec![segment.extraction_duration_ms]);
  let input_tokens = UInt32Array::from(vec![segment.input_tokens]);
  let output_tokens = UInt32Array::from(vec![segment.output_tokens]);
  let model_used = StringArray::from(vec![segment.model_used.clone()]);
  let error = StringArray::from(vec![segment.error.clone()]);
  let created_at = Int64Array::from(vec![segment.created_at.timestamp_millis()]);

  let batch = RecordBatch::try_new(
    extraction_segments_schema(),
    vec![
      Arc::new(id),
      Arc::new(session_id),
      Arc::new(project_id),
      Arc::new(trigger),
      Arc::new(user_prompts_json),
      Arc::new(files_read_count),
      Arc::new(files_modified_count),
      Arc::new(tool_call_count),
      Arc::new(memories_extracted),
      Arc::new(extraction_duration_ms),
      Arc::new(input_tokens),
      Arc::new(output_tokens),
      Arc::new(model_used),
      Arc::new(error),
      Arc::new(created_at),
    ],
  )?;

  Ok(batch)
}

/// Convert a RecordBatch row to an ExtractionSegment
fn batch_to_segment(batch: &RecordBatch, row: usize) -> Result<ExtractionSegment> {
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

  let get_optional_u32 = |name: &str| -> Option<u32> {
    batch
      .column_by_name(name)
      .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
      .and_then(|a| if a.is_null(row) { None } else { Some(a.value(row)) })
  };

  let id_str = get_string("id")?;
  let session_id_str = get_string("session_id")?;
  let project_id_str = get_string("project_id")?;
  let trigger_str = get_string("trigger")?;

  let created_at = Utc
    .timestamp_millis_opt(get_i64("created_at")?)
    .single()
    .ok_or_else(|| DbError::NotFound("invalid created_at timestamp".into()))?;

  let trigger = trigger_str.parse::<ExtractionTrigger>().map_err(DbError::NotFound)?;

  Ok(ExtractionSegment {
    id: Uuid::parse_str(&id_str).map_err(|_| DbError::NotFound("invalid id".into()))?,
    session_id: Uuid::parse_str(&session_id_str).map_err(|_| DbError::NotFound("invalid session_id".into()))?,
    project_id: Uuid::parse_str(&project_id_str).map_err(|_| DbError::NotFound("invalid project_id".into()))?,
    trigger,
    user_prompts_json: get_string("user_prompts_json")?,
    files_read_count: get_u32("files_read_count")?,
    files_modified_count: get_u32("files_modified_count")?,
    tool_call_count: get_u32("tool_call_count")?,
    memories_extracted: get_u32("memories_extracted")?,
    extraction_duration_ms: get_u32("extraction_duration_ms")?,
    input_tokens: get_optional_u32("input_tokens"),
    output_tokens: get_optional_u32("output_tokens"),
    model_used: get_optional_string("model_used"),
    error: get_optional_string("error"),
    created_at,
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
  async fn test_save_and_get_segment() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut segment = ExtractionSegment::new(
      session_id,
      project_id,
      ExtractionTrigger::Stop,
      &["Hello".to_string()],
      5,
      2,
      10,
    );

    segment.record_success(3, 1500, Some(100), Some(200), Some("haiku"));

    db.save_extraction_segment(&segment).await.unwrap();

    let retrieved = db.get_extraction_segment(&segment.id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.session_id, session_id);
    assert_eq!(retrieved.trigger, ExtractionTrigger::Stop);
    assert_eq!(retrieved.memories_extracted, 3);
    assert_eq!(retrieved.extraction_duration_ms, 1500);
    assert_eq!(retrieved.input_tokens, Some(100));
    assert_eq!(retrieved.model_used, Some("haiku".to_string()));
  }

  #[tokio::test]
  async fn test_segment_with_error() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    let mut segment = ExtractionSegment::new(session_id, project_id, ExtractionTrigger::PreCompact, &[], 0, 0, 5);

    segment.record_failure("LLM timeout", 60000);

    db.save_extraction_segment(&segment).await.unwrap();

    let retrieved = db.get_extraction_segment(&segment.id).await.unwrap().unwrap();
    assert!(retrieved.error.is_some());
    assert_eq!(retrieved.error.unwrap(), "LLM timeout");
    assert_eq!(retrieved.extraction_duration_ms, 60000);
  }

  #[tokio::test]
  async fn test_list_segments_for_session() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    // Create multiple segments
    for trigger in [
      ExtractionTrigger::UserPrompt,
      ExtractionTrigger::PreCompact,
      ExtractionTrigger::Stop,
    ] {
      let segment = ExtractionSegment::new(session_id, project_id, trigger, &[], 0, 0, 0);
      db.save_extraction_segment(&segment).await.unwrap();
    }

    let segments = db.list_extraction_segments(&session_id, None).await.unwrap();
    assert_eq!(segments.len(), 3);
  }

  #[tokio::test]
  async fn test_extraction_stats() {
    let (_temp, db) = create_test_db().await;
    let session_id = Uuid::new_v4();
    let project_id = Uuid::new_v4();

    // Create segments with varying success/failure
    let mut s1 = ExtractionSegment::new(session_id, project_id, ExtractionTrigger::Stop, &[], 0, 0, 5);
    s1.record_success(3, 1000, None, None, None);
    db.save_extraction_segment(&s1).await.unwrap();

    let mut s2 = ExtractionSegment::new(session_id, project_id, ExtractionTrigger::UserPrompt, &[], 0, 0, 2);
    s2.record_success(1, 500, None, None, None);
    db.save_extraction_segment(&s2).await.unwrap();

    let mut s3 = ExtractionSegment::new(session_id, project_id, ExtractionTrigger::PreCompact, &[], 0, 0, 8);
    s3.record_failure("Timeout", 60000);
    db.save_extraction_segment(&s3).await.unwrap();

    let stats = db.extraction_stats(&project_id).await.unwrap();
    assert_eq!(stats.total_extractions, 3);
    assert_eq!(stats.failed_extractions, 1);
    assert_eq!(stats.total_memories_extracted, 4);
    assert_eq!(stats.stop_triggers, 1);
    assert_eq!(stats.user_prompt_triggers, 1);
    assert_eq!(stats.pre_compact_triggers, 1);
  }

  #[test]
  fn test_trigger_parsing() {
    assert_eq!(
      "user_prompt".parse::<ExtractionTrigger>().unwrap(),
      ExtractionTrigger::UserPrompt
    );
    assert_eq!(
      "pre_compact".parse::<ExtractionTrigger>().unwrap(),
      ExtractionTrigger::PreCompact
    );
    assert_eq!("stop".parse::<ExtractionTrigger>().unwrap(), ExtractionTrigger::Stop);
    assert_eq!(
      "todo_completion".parse::<ExtractionTrigger>().unwrap(),
      ExtractionTrigger::TodoCompletion
    );
    assert!("invalid".parse::<ExtractionTrigger>().is_err());
  }
}
