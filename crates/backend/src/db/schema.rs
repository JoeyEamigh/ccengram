use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};

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
      false,
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
      false,
    ),
  ]))
}

/// Schema for the sessions table
///
/// The `id` field is the Claude Code session ID string, which is stable
/// across thread resumes. This allows tying memories back to sessions.
pub fn sessions_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false), // Claude session ID string
    Field::new("project_id", DataType::Utf8, false),
    Field::new("started_at", DataType::Int64, false),
    Field::new("ended_at", DataType::Int64, true),
    Field::new("summary", DataType::Utf8, true),
    Field::new("user_prompt", DataType::Utf8, true),
    Field::new("context", DataType::Utf8, true), // JSON object
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
    Field::new("content_hash", DataType::Utf8, false), // Hash for merge_insert key
    Field::new("created_at", DataType::Int64, false),
    Field::new("updated_at", DataType::Int64, false),
    Field::new(
      "vector",
      DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), vector_dim as i32),
      false,
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

/// Schema for the indexed_files table (tracks file metadata for startup scan)
///
/// This table stores metadata about indexed files to enable detection of:
/// - Added files: file exists on disk but not in table
/// - Deleted files: file in table but not on disk
/// - Modified files: mtime changed -> verify with content_hash
/// - Moved files: same content_hash, different file_path
pub fn indexed_files_schema() -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("file_path", DataType::Utf8, false), // Relative path from project root
    Field::new("project_id", DataType::Utf8, false), // Project UUID
    Field::new("mtime", DataType::Int64, false),    // Unix timestamp (seconds) for quick change detection
    Field::new("content_hash", DataType::Utf8, false), // SHA-256 hash for content verification
    Field::new("file_size", DataType::UInt64, false), // File size in bytes
    Field::new("last_indexed_at", DataType::Int64, false), // Unix timestamp ms when file was last indexed
  ]))
}
