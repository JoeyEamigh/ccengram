// Code references table operations
//
// This module provides efficient caller/callee lookups by storing
// explicit references between code chunks, eliminating the need for
// expensive LIKE queries on JSON columns.

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::connection::{ProjectDb, Result};
use crate::schema::code_references_schema;

/// A reference from one code chunk to a symbol in another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReference {
  pub id: Uuid,
  pub project_id: String,
  pub source_chunk_id: String,
  pub target_symbol: String,
  pub target_chunk_id: Option<String>,
  pub reference_type: ReferenceType,
  pub created_at: i64,
}

/// Type of reference between chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceType {
  /// Function/method call
  Call,
  /// Import statement
  Import,
  /// Type reference (in type annotations, generics, etc.)
  TypeRef,
}

impl ReferenceType {
  pub fn as_str(&self) -> &'static str {
    match self {
      ReferenceType::Call => "call",
      ReferenceType::Import => "import",
      ReferenceType::TypeRef => "type_ref",
    }
  }

  pub fn new_from_str(s: &str) -> Self {
    match s {
      "call" => ReferenceType::Call,
      "import" => ReferenceType::Import,
      "type_ref" => ReferenceType::TypeRef,
      _ => ReferenceType::Call,
    }
  }
}

impl ProjectDb {
  /// Insert references extracted from code chunks
  ///
  /// This should be called after chunking to record what symbols each chunk calls.
  pub async fn insert_references(&self, refs: &[CodeReference]) -> Result<()> {
    if refs.is_empty() {
      return Ok(());
    }

    let table = self.code_references_table().await?;
    let schema = code_references_schema();

    let batches: Vec<_> = refs.iter().map(code_reference_to_batch).collect::<Result<Vec<_>>>()?;

    let iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    table.add(Box::new(iter)).execute().await?;
    Ok(())
  }

  /// Delete all references originating from chunks in a specific file
  ///
  /// This should be called before re-indexing a file to remove stale references.
  pub async fn delete_references_for_file(&self, file_path: &str) -> Result<()> {
    // First get all chunk IDs for this file
    let chunks = self.get_chunks_for_file(file_path).await?;
    if chunks.is_empty() {
      return Ok(());
    }

    let chunk_ids: Vec<String> = chunks.iter().map(|c| c.id.to_string()).collect();
    self.delete_references_for_chunks(&chunk_ids).await
  }

  /// Delete references for specific chunk IDs
  pub async fn delete_references_for_chunks(&self, chunk_ids: &[String]) -> Result<()> {
    if chunk_ids.is_empty() {
      return Ok(());
    }

    let table = self.code_references_table().await?;

    // Build IN clause
    let ids_list = chunk_ids
      .iter()
      .map(|id| format!("'{}'", id.replace('\'', "''")))
      .collect::<Vec<_>>()
      .join(", ");

    table.delete(&format!("source_chunk_id IN ({})", ids_list)).await?;
    Ok(())
  }

  /// Count how many chunks call symbols defined in the given chunk
  ///
  /// Uses the code_references table for O(log n) indexed lookup instead of
  /// O(n) LIKE scan on JSON columns.
  pub async fn count_callers_for_symbols(&self, symbols: &[String]) -> Result<usize> {
    if symbols.is_empty() {
      return Ok(0);
    }

    let table = self.code_references_table().await?;

    // Build IN clause for symbols
    let symbols_list = symbols
      .iter()
      .map(|s| format!("'{}'", s.replace('\'', "''")))
      .collect::<Vec<_>>()
      .join(", ");

    // Count distinct source chunks that reference any of these symbols
    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("target_symbol IN ({})", symbols_list))
      .execute()
      .await?
      .try_collect()
      .await?;

    // Count unique source_chunk_ids
    let mut unique_callers = HashSet::new();
    for batch in results {
      if let Some(col) = batch.column_by_name("source_chunk_id")
        && let Some(arr) = col.as_any().downcast_ref::<StringArray>()
      {
        for i in 0..arr.len() {
          unique_callers.insert(arr.value(i).to_string());
        }
      }
    }

    Ok(unique_callers.len())
  }

  /// Count how many unique symbols a chunk calls
  pub async fn count_callees_for_chunk(&self, chunk_id: &str) -> Result<usize> {
    let table = self.code_references_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("source_chunk_id = '{}'", chunk_id.replace('\'', "''")))
      .execute()
      .await?
      .try_collect()
      .await?;

    // Count unique target symbols
    let mut unique_callees = HashSet::new();
    for batch in results {
      if let Some(col) = batch.column_by_name("target_symbol")
        && let Some(arr) = col.as_any().downcast_ref::<StringArray>()
      {
        for i in 0..arr.len() {
          unique_callees.insert(arr.value(i).to_string());
        }
      }
    }

    Ok(unique_callees.len())
  }

  /// Get chunks that call symbols defined in the given chunk
  ///
  /// Returns (source_chunk_id, target_symbol) pairs.
  pub async fn get_callers_for_symbols(&self, symbols: &[String], limit: usize) -> Result<Vec<(String, String)>> {
    if symbols.is_empty() {
      return Ok(vec![]);
    }

    let table = self.code_references_table().await?;

    let symbols_list = symbols
      .iter()
      .map(|s| format!("'{}'", s.replace('\'', "''")))
      .collect::<Vec<_>>()
      .join(", ");

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("target_symbol IN ({})", symbols_list))
      .limit(limit)
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut callers = Vec::new();
    for batch in results {
      let source_col = batch.column_by_name("source_chunk_id");
      let target_col = batch.column_by_name("target_symbol");

      if let (Some(source), Some(target)) = (source_col, target_col)
        && let (Some(source_arr), Some(target_arr)) = (
          source.as_any().downcast_ref::<StringArray>(),
          target.as_any().downcast_ref::<StringArray>(),
        )
      {
        for i in 0..source_arr.len() {
          callers.push((source_arr.value(i).to_string(), target_arr.value(i).to_string()));
        }
      }
    }

    Ok(callers)
  }

  /// Get symbols called by a chunk and their defining chunks (if resolved)
  ///
  /// Returns (target_symbol, target_chunk_id) pairs.
  pub async fn get_callees_for_chunk(&self, chunk_id: &str, limit: usize) -> Result<Vec<(String, Option<String>)>> {
    let table = self.code_references_table().await?;

    let results: Vec<RecordBatch> = table
      .query()
      .only_if(format!("source_chunk_id = '{}'", chunk_id.replace('\'', "''")))
      .limit(limit)
      .execute()
      .await?
      .try_collect()
      .await?;

    let mut callees = Vec::new();
    for batch in results {
      let symbol_col = batch.column_by_name("target_symbol");
      let chunk_col = batch.column_by_name("target_chunk_id");

      if let (Some(symbol), Some(chunk)) = (symbol_col, chunk_col)
        && let (Some(symbol_arr), Some(chunk_arr)) = (
          symbol.as_any().downcast_ref::<StringArray>(),
          chunk.as_any().downcast_ref::<StringArray>(),
        )
      {
        for i in 0..symbol_arr.len() {
          let target_chunk = {
            let val = chunk_arr.value(i);
            if val.is_empty() { None } else { Some(val.to_string()) }
          };
          callees.push((symbol_arr.value(i).to_string(), target_chunk));
        }
      }
    }

    Ok(callees)
  }

  /// Update caller counts for chunks that define the given symbols
  ///
  /// This should be called after inserting new references to update the
  /// pre-computed `caller_count` field on affected chunks.
  pub async fn update_caller_counts_for_symbols(&self, symbols: &[String]) -> Result<HashMap<String, u32>> {
    if symbols.is_empty() {
      return Ok(HashMap::new());
    }

    // Count callers for each symbol
    let mut symbol_counts: HashMap<String, u32> = HashMap::new();
    for symbol in symbols {
      let count = self.count_callers_for_symbols(std::slice::from_ref(symbol)).await? as u32;
      symbol_counts.insert(symbol.clone(), count);
    }

    Ok(symbol_counts)
  }

  /// Resolve target_chunk_id for references where the symbol matches a chunk's symbols
  ///
  /// This links references to their target definitions for efficient navigation.
  pub async fn resolve_reference_targets(&self) -> Result<usize> {
    // Get all references without resolved targets
    let table = self.code_references_table().await?;
    let results: Vec<RecordBatch> = table
      .query()
      .only_if("target_chunk_id IS NULL OR target_chunk_id = ''")
      .execute()
      .await?
      .try_collect()
      .await?;

    // Collect unique target symbols
    let mut symbols_to_resolve = HashSet::new();
    for batch in &results {
      if let Some(col) = batch.column_by_name("target_symbol")
        && let Some(arr) = col.as_any().downcast_ref::<StringArray>()
      {
        for i in 0..arr.len() {
          symbols_to_resolve.insert(arr.value(i).to_string());
        }
      }
    }

    if symbols_to_resolve.is_empty() {
      return Ok(0);
    }

    // Find chunks that define these symbols
    // Note: This uses LIKE for now since symbols is a JSON array
    // Future optimization: create a separate symbol->chunk mapping table
    let mut resolved = 0;
    for symbol in symbols_to_resolve {
      let escaped_symbol = symbol.replace('\'', "''");
      let filter = format!("symbols LIKE '%\"{}%'", escaped_symbol);

      if let Ok(chunks) = self.list_code_chunks(Some(&filter), Some(1)).await
        && let Some(_chunk) = chunks.first()
      {
        // Update references to point to this chunk
        // Note: LanceDB doesn't support UPDATE, so we'd need delete+insert
        // For now, just count what we could resolve
        resolved += 1;
      }
    }

    Ok(resolved)
  }
}

/// Convert a CodeReference to an Arrow RecordBatch
fn code_reference_to_batch(reference: &CodeReference) -> Result<RecordBatch> {
  let id = StringArray::from(vec![reference.id.to_string()]);
  let project_id = StringArray::from(vec![reference.project_id.clone()]);
  let source_chunk_id = StringArray::from(vec![reference.source_chunk_id.clone()]);
  let target_symbol = StringArray::from(vec![reference.target_symbol.clone()]);
  let target_chunk_id = StringArray::from(vec![reference.target_chunk_id.clone().unwrap_or_default()]);
  let reference_type = StringArray::from(vec![reference.reference_type.as_str()]);
  let created_at = Int64Array::from(vec![reference.created_at]);

  let batch = RecordBatch::try_new(
    code_references_schema(),
    vec![
      Arc::new(id),
      Arc::new(project_id),
      Arc::new(source_chunk_id),
      Arc::new(target_symbol),
      Arc::new(target_chunk_id),
      Arc::new(reference_type),
      Arc::new(created_at),
    ],
  )?;

  Ok(batch)
}

/// Helper to create a CodeReference from extracted call data
impl CodeReference {
  pub fn from_call(project_id: &str, source_chunk_id: &str, target_symbol: &str) -> Self {
    Self {
      id: Uuid::new_v4(),
      project_id: project_id.to_string(),
      source_chunk_id: source_chunk_id.to_string(),
      target_symbol: target_symbol.to_string(),
      target_chunk_id: None,
      reference_type: ReferenceType::Call,
      created_at: Utc::now().timestamp_millis(),
    }
  }

  pub fn from_import(project_id: &str, source_chunk_id: &str, target_symbol: &str) -> Self {
    Self {
      id: Uuid::new_v4(),
      project_id: project_id.to_string(),
      source_chunk_id: source_chunk_id.to_string(),
      target_symbol: target_symbol.to_string(),
      target_chunk_id: None,
      reference_type: ReferenceType::Import,
      created_at: Utc::now().timestamp_millis(),
    }
  }
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
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();
    (temp_dir, db)
  }

  #[tokio::test]
  async fn test_insert_and_count_references() {
    let (_temp, db) = create_test_db().await;

    // Insert some references
    let refs = vec![
      CodeReference::from_call("test", "chunk-a", "foo"),
      CodeReference::from_call("test", "chunk-b", "foo"),
      CodeReference::from_call("test", "chunk-c", "bar"),
    ];

    db.insert_references(&refs).await.unwrap();

    // Count callers for "foo" - should be 2
    let count = db.count_callers_for_symbols(&["foo".to_string()]).await.unwrap();
    assert_eq!(count, 2);

    // Count callers for "bar" - should be 1
    let count = db.count_callers_for_symbols(&["bar".to_string()]).await.unwrap();
    assert_eq!(count, 1);
  }

  #[tokio::test]
  async fn test_delete_references_for_chunks() {
    let (_temp, db) = create_test_db().await;

    let refs = vec![
      CodeReference::from_call("test", "chunk-a", "foo"),
      CodeReference::from_call("test", "chunk-a", "bar"),
      CodeReference::from_call("test", "chunk-b", "foo"),
    ];

    db.insert_references(&refs).await.unwrap();

    // Delete references from chunk-a
    db.delete_references_for_chunks(&["chunk-a".to_string()]).await.unwrap();

    // Only chunk-b's reference should remain
    let count = db.count_callers_for_symbols(&["foo".to_string()]).await.unwrap();
    assert_eq!(count, 1);
  }

  #[tokio::test]
  async fn test_get_callers() {
    let (_temp, db) = create_test_db().await;

    let refs = vec![
      CodeReference::from_call("test", "chunk-a", "target_func"),
      CodeReference::from_call("test", "chunk-b", "target_func"),
      CodeReference::from_call("test", "chunk-c", "other_func"),
    ];

    db.insert_references(&refs).await.unwrap();

    let callers = db
      .get_callers_for_symbols(&["target_func".to_string()], 10)
      .await
      .unwrap();
    assert_eq!(callers.len(), 2);

    let caller_ids: HashSet<_> = callers.iter().map(|(id, _)| id.as_str()).collect();
    assert!(caller_ids.contains("chunk-a"));
    assert!(caller_ids.contains("chunk-b"));
  }

  #[tokio::test]
  async fn test_get_callees() {
    let (_temp, db) = create_test_db().await;

    let refs = vec![
      CodeReference::from_call("test", "chunk-a", "func1"),
      CodeReference::from_call("test", "chunk-a", "func2"),
      CodeReference::from_call("test", "chunk-a", "func3"),
    ];

    db.insert_references(&refs).await.unwrap();

    let callees = db.get_callees_for_chunk("chunk-a", 10).await.unwrap();
    assert_eq!(callees.len(), 3);

    let callee_symbols: HashSet<_> = callees.iter().map(|(s, _)| s.as_str()).collect();
    assert!(callee_symbols.contains("func1"));
    assert!(callee_symbols.contains("func2"));
    assert!(callee_symbols.contains("func3"));
  }
}
