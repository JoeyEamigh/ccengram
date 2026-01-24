use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// Schema for the memories table
pub fn memories_schema(vector_dim: usize) -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("project_id", DataType::Utf8, false),
    Field::new("content", DataType::Utf8, false),
    Field::new("summary", DataType::Utf8, true),
    Field::new("sector", DataType::Utf8, false),
    Field::new("tier", DataType::Utf8, false),
    Field::new("memory_type", DataType::Utf8, true),
    Field::new("importance", DataType::Float32, false),
    Field::new("salience", DataType::Float32, false),
    Field::new("confidence", DataType::Float32, false),
    Field::new("access_count", DataType::UInt32, false),
    Field::new("tags", DataType::Utf8, false),       // JSON array
    Field::new("concepts", DataType::Utf8, false),   // JSON array
    Field::new("files", DataType::Utf8, false),      // JSON array
    Field::new("categories", DataType::Utf8, false), // JSON array
    Field::new("context", DataType::Utf8, true),
    Field::new("session_id", DataType::Utf8, true),
    Field::new("segment_id", DataType::Utf8, true), // Conversation segment ID
    Field::new("scope_path", DataType::Utf8, true), // Code path context
    Field::new("scope_module", DataType::Utf8, true), // Logical module context
    Field::new("created_at", DataType::Int64, false), // Unix timestamp ms
    Field::new("updated_at", DataType::Int64, false),
    Field::new("last_accessed", DataType::Int64, false),
    Field::new("deleted_at", DataType::Int64, true), // Soft delete timestamp
    Field::new("valid_from", DataType::Int64, false),
    Field::new("valid_until", DataType::Int64, true),
    Field::new("is_deleted", DataType::Boolean, false),
    Field::new("content_hash", DataType::Utf8, false),
    Field::new("simhash", DataType::UInt64, false),
    Field::new("superseded_by", DataType::Utf8, true),
    Field::new("decay_rate", DataType::Float32, true), // Cached decay rate
    Field::new("next_decay_at", DataType::Int64, true), // Next scheduled decay
    Field::new("embedding_model_id", DataType::Utf8, true), // Model used for embedding
    Field::new(
      "vector",
      DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), vector_dim as i32),
      true,
    ),
  ]))
}

/// Schema for the code_chunks table
pub fn code_chunks_schema(vector_dim: usize) -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("project_id", DataType::Utf8, false),
    Field::new("file_path", DataType::Utf8, false),
    Field::new("content", DataType::Utf8, false),
    Field::new("language", DataType::Utf8, false),
    Field::new("chunk_type", DataType::Utf8, false),
    Field::new("symbols", DataType::Utf8, false), // JSON array
    Field::new("imports", DataType::Utf8, false), // JSON array of import paths
    Field::new("calls", DataType::Utf8, false),   // JSON array of function/method calls
    Field::new("start_line", DataType::UInt32, false),
    Field::new("end_line", DataType::UInt32, false),
    Field::new("file_hash", DataType::Utf8, false),
    Field::new("indexed_at", DataType::Int64, false),
    // Definition metadata for AST-level chunking
    Field::new("definition_kind", DataType::Utf8, true), // function, struct, impl, trait, etc.
    Field::new("definition_name", DataType::Utf8, true), // Primary symbol name
    Field::new("visibility", DataType::Utf8, true),      // pub, pub(crate), private
    Field::new("signature", DataType::Utf8, true),       // Full signature for display
    Field::new("docstring", DataType::Utf8, true),       // Documentation comments
    Field::new("parent_definition", DataType::Utf8, true), // Parent for nested items
    Field::new("embedding_text", DataType::Utf8, true),  // Enriched text for embedding
    Field::new("content_hash", DataType::Utf8, true),    // Hash for detecting unchanged chunks
    // Pre-computed relationship counts for fast hint computation
    Field::new("caller_count", DataType::UInt32, false), // Chunks calling symbols in this chunk
    Field::new("callee_count", DataType::UInt32, false), // Unique symbols this chunk calls
    Field::new(
      "vector",
      DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), vector_dim as i32),
      true,
    ),
  ]))
}

/// Schema for code references table (tracks call relationships between chunks)
/// Used for efficient caller/callee lookups without LIKE queries on JSON columns
pub fn code_references_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),              // Unique reference ID
    Field::new("project_id", DataType::Utf8, false),      // Project UUID
    Field::new("source_chunk_id", DataType::Utf8, false), // Chunk containing the call
    Field::new("target_symbol", DataType::Utf8, false),   // Symbol being called
    Field::new("target_chunk_id", DataType::Utf8, true),  // Resolved chunk (nullable until resolved)
    Field::new("reference_type", DataType::Utf8, false),  // 'call', 'import', 'type_ref'
    Field::new("created_at", DataType::Int64, false),     // Unix timestamp ms
  ]))
}

/// Schema for the sessions table
pub fn sessions_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("project_id", DataType::Utf8, false),
    Field::new("started_at", DataType::Int64, false),
    Field::new("ended_at", DataType::Int64, true),
    Field::new("summary", DataType::Utf8, true),
    Field::new("user_prompt", DataType::Utf8, true),
    Field::new("context", DataType::Utf8, true), // JSON object
  ]))
}

/// Schema for the events table (for event sourcing)
pub fn events_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("entity_id", DataType::Utf8, false),
    Field::new("entity_type", DataType::Utf8, false),
    Field::new("event_type", DataType::Utf8, false),
    Field::new("payload", DataType::Utf8, false), // JSON
    Field::new("timestamp", DataType::Int64, false),
  ]))
}

/// Schema for the documents table (ingested docs for search)
pub fn documents_schema(vector_dim: usize) -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("document_id", DataType::Utf8, false),
    Field::new("project_id", DataType::Utf8, false),
    Field::new("content", DataType::Utf8, false),
    Field::new("title", DataType::Utf8, false),
    Field::new("source", DataType::Utf8, false),
    Field::new("source_type", DataType::Utf8, false),
    Field::new("chunk_index", DataType::UInt32, false),
    Field::new("total_chunks", DataType::UInt32, false),
    Field::new("char_offset", DataType::UInt32, false),
    Field::new("created_at", DataType::Int64, false),
    Field::new("updated_at", DataType::Int64, false),
    Field::new(
      "vector",
      DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), vector_dim as i32),
      true,
    ),
  ]))
}

/// Schema for the session_memories junction table
pub fn session_memories_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("session_id", DataType::Utf8, false),
    Field::new("memory_id", DataType::Utf8, false),
    Field::new("usage_type", DataType::Utf8, false), // created, recalled, updated, reinforced
    Field::new("linked_at", DataType::Int64, false), // Unix timestamp ms
  ]))
}

/// Schema for the memory_relationships table
pub fn memory_relationships_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("from_memory_id", DataType::Utf8, false),
    Field::new("to_memory_id", DataType::Utf8, false),
    Field::new("relationship_type", DataType::Utf8, false),
    Field::new("confidence", DataType::Float32, false),
    Field::new("valid_from", DataType::Int64, false),
    Field::new("valid_until", DataType::Int64, true),
    Field::new("extracted_by", DataType::Utf8, false),
    Field::new("created_at", DataType::Int64, false),
  ]))
}

/// Schema for the document_metadata table (tracks documents for update detection)
pub fn document_metadata_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),            // DocumentId
    Field::new("project_id", DataType::Utf8, false),    // Project UUID
    Field::new("title", DataType::Utf8, false),         // Document title
    Field::new("source", DataType::Utf8, false),        // Source path/URL
    Field::new("source_type", DataType::Utf8, false),   // file, url, content
    Field::new("content_hash", DataType::Utf8, false),  // SHA-256 hash for update detection
    Field::new("char_count", DataType::UInt32, false),  // Total character count
    Field::new("chunk_count", DataType::UInt32, false), // Number of chunks created
    Field::new("full_content", DataType::Utf8, true),   // Full document content (optional)
    Field::new("created_at", DataType::Int64, false),   // Unix timestamp ms
    Field::new("updated_at", DataType::Int64, false),   // Unix timestamp ms
  ]))
}

/// Schema for the entities table
pub fn entities_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("entity_type", DataType::Utf8, false),
    Field::new("summary", DataType::Utf8, true),
    Field::new("aliases", DataType::Utf8, false), // JSON array
    Field::new("first_seen_at", DataType::Int64, false),
    Field::new("last_seen_at", DataType::Int64, false),
    Field::new("mention_count", DataType::UInt32, false),
  ]))
}

/// Schema for the memory_entities junction table
pub fn memory_entities_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("memory_id", DataType::Utf8, false),
    Field::new("entity_id", DataType::Utf8, false),
    Field::new("role", DataType::Utf8, false),
    Field::new("confidence", DataType::Float32, false),
    Field::new("extracted_at", DataType::Int64, false),
  ]))
}

/// Schema for the index_checkpoints table (for resuming interrupted indexing)
pub fn index_checkpoints_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false), // Checkpoint ID (project_id + type)
    Field::new("project_id", DataType::Utf8, false), // Project UUID
    Field::new("checkpoint_type", DataType::Utf8, false), // "code" or "document"
    Field::new("processed_files", DataType::Utf8, false), // JSON array of processed file paths
    Field::new("pending_files", DataType::Utf8, false), // JSON array of pending file paths
    Field::new("total_files", DataType::UInt32, false), // Total files to process
    Field::new("processed_count", DataType::UInt32, false), // Files successfully processed
    Field::new("error_count", DataType::UInt32, false), // Files with errors
    Field::new("gitignore_hash", DataType::Utf8, true), // Hash of gitignore rules at start
    Field::new("started_at", DataType::Int64, false), // Unix timestamp ms
    Field::new("updated_at", DataType::Int64, false), // Unix timestamp ms
    Field::new("is_complete", DataType::Boolean, false), // Whether indexing finished
  ]))
}

/// Schema for the segment_accumulators table (tracks work context during extraction)
pub fn segment_accumulators_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),                    // segment_id
    Field::new("session_id", DataType::Utf8, false),            // Session UUID
    Field::new("project_id", DataType::Utf8, false),            // Project UUID
    Field::new("segment_start", DataType::Int64, false),        // Unix timestamp ms
    Field::new("user_prompts", DataType::Utf8, false),          // JSON array of UserPrompt objects
    Field::new("files_read", DataType::Utf8, false),            // JSON array of paths
    Field::new("files_modified", DataType::Utf8, false),        // JSON array of paths
    Field::new("commands_run", DataType::Utf8, false),          // JSON array of {command, exit_code}
    Field::new("errors_encountered", DataType::Utf8, false),    // JSON array
    Field::new("searches_performed", DataType::Utf8, false),    // JSON array of patterns
    Field::new("completed_tasks", DataType::Utf8, false),       // JSON array of task content
    Field::new("last_assistant_message", DataType::Utf8, true), // Truncated to 10KB
    Field::new("tool_call_count", DataType::UInt32, false),
    Field::new("updated_at", DataType::Int64, false), // Unix timestamp ms
  ]))
}

/// Schema for the extraction_segments table (records extraction runs)
pub fn extraction_segments_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),                // Segment ID
    Field::new("session_id", DataType::Utf8, false),        // Session UUID
    Field::new("project_id", DataType::Utf8, false),        // Project UUID
    Field::new("trigger", DataType::Utf8, false),           // user_prompt, pre_compact, stop, todo_completion
    Field::new("user_prompts_json", DataType::Utf8, false), // Snapshot of prompts at extraction
    Field::new("files_read_count", DataType::UInt32, false),
    Field::new("files_modified_count", DataType::UInt32, false),
    Field::new("tool_call_count", DataType::UInt32, false),
    Field::new("memories_extracted", DataType::UInt32, false),
    Field::new("extraction_duration_ms", DataType::UInt32, false),
    Field::new("input_tokens", DataType::UInt32, true),
    Field::new("output_tokens", DataType::UInt32, true),
    Field::new("model_used", DataType::Utf8, true),
    Field::new("error", DataType::Utf8, true), // Error message if extraction failed
    Field::new("created_at", DataType::Int64, false), // Unix timestamp ms
  ]))
}

/// Default vector dimensions for embedding models
pub const DEFAULT_VECTOR_DIM: usize = 4096; // qwen3-embedding

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_memories_schema() {
    let schema = memories_schema(4096);
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("content").is_ok());
    assert!(schema.field_with_name("vector").is_ok());
  }

  #[test]
  fn test_code_chunks_schema() {
    let schema = code_chunks_schema(4096);
    assert!(schema.field_with_name("file_path").is_ok());
    assert!(schema.field_with_name("vector").is_ok());
  }

  #[test]
  fn test_sessions_schema() {
    let schema = sessions_schema();
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("project_id").is_ok());
  }

  #[test]
  fn test_documents_schema() {
    let schema = documents_schema(4096);
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("content").is_ok());
    assert!(schema.field_with_name("title").is_ok());
    assert!(schema.field_with_name("vector").is_ok());
  }

  #[test]
  fn test_document_metadata_schema() {
    let schema = document_metadata_schema();
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("source").is_ok());
    assert!(schema.field_with_name("content_hash").is_ok());
  }
}
