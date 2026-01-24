pub mod accumulators;
pub mod code;
pub mod connection;
pub mod document_metadata;
pub mod documents;
pub mod entities;
pub mod events;
pub mod extraction_segments;
pub mod index_checkpoints;
pub mod memories;
pub mod memory_entities;
pub mod memory_relationships;
pub mod migrations;
pub mod references;
pub mod schema;
pub mod session_memories;
pub mod sessions;
pub mod stats;

pub use accumulators::{CommandRecord, SegmentAccumulator, UserPrompt};
pub use connection::{
  DbError, ProjectDb, Result, default_cache_dir, default_config_dir, default_data_dir, default_port,
};
pub use document_metadata::{DocumentUpdateCheck, compute_content_hash};
pub use events::{EntityType as EventEntityType, Event, EventType};
pub use extraction_segments::{ExtractionSegment, ExtractionStats, ExtractionTrigger};
pub use index_checkpoints::{CheckpointType, IndexCheckpoint};
pub use migrations::{CURRENT_SCHEMA_VERSION, MIGRATIONS, Migration, MigrationRecord};
pub use references::{CodeReference, ReferenceType};
pub use schema::{
  DEFAULT_VECTOR_DIM, code_chunks_schema, code_references_schema, document_metadata_schema, documents_schema,
  entities_schema, events_schema, extraction_segments_schema, index_checkpoints_schema, memories_schema,
  memory_entities_schema, memory_relationships_schema, segment_accumulators_schema, session_memories_schema,
  sessions_schema,
};
pub use session_memories::{SessionMemoryLink, SessionStats, UsageType};
pub use sessions::Session;
pub use stats::{
  CodeStats, DocumentStats, EntityStats, MemoryStats, ProjectStats, RecentIndexActivity, SalienceDistribution,
};
