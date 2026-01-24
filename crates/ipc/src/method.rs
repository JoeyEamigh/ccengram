use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
  // Meta
  Ping,
  Status,
  Metrics,
  Shutdown,
  // Memory
  MemorySearch,
  MemoryGet,
  MemoryAdd,
  MemoryList,
  MemoryReinforce,
  MemoryDeemphasize,
  MemoryDelete,
  MemorySupersede,
  MemoryTimeline,
  MemoryRelated,
  MemoryRestore,
  MemoryListDeleted,
  // Code
  CodeSearch,
  CodeContext,
  CodeIndex,
  CodeList,
  CodeImportChunk,
  CodeStats,
  CodeMemories,
  CodeCallers,
  CodeCallees,
  CodeRelated,
  CodeContextFull,
  // Watch
  WatchStart,
  WatchStop,
  WatchStatus,
  // Documents
  DocsSearch,
  DocContext,
  DocsIngest,
  // Entity
  EntityList,
  EntityGet,
  EntityTop,
  // Relationship
  RelationshipAdd,
  RelationshipList,
  RelationshipDelete,
  RelationshipRelated,
  // Stats
  ProjectStats,
  HealthCheck,
  // Unified
  Explore,
  Context,
  // Migration
  MigrateEmbedding,
  // Project management
  ProjectsList,
  ProjectInfo,
  ProjectClean,
  ProjectsCleanAll,
  // Hooks
  Hook,
}
