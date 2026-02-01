use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a memory (newtype for type safety)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(Uuid);

impl MemoryId {
  pub fn new() -> Self {
    Self(Uuid::new_v4())
  }

  pub fn from_uuid(id: Uuid) -> Self {
    Self(id)
  }

  pub fn as_uuid(&self) -> Uuid {
    self.0
  }
}

impl Default for MemoryId {
  fn default() -> Self {
    Self::new()
  }
}

impl std::fmt::Display for MemoryId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl std::str::FromStr for MemoryId {
  type Err = uuid::Error;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(Self(Uuid::parse_str(s)?))
  }
}

/// Memory sector determines decay rate and search boosting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sector {
  /// Session events, tool observations - fastest decay
  Episodic,
  /// Facts, codebase knowledge - medium decay
  Semantic,
  /// How-to instructions, workflows - medium decay
  Procedural,
  /// Preferences, frustrations - slowest decay
  Emotional,
  /// Insights, summaries - slow decay
  Reflective,
}

impl Sector {
  /// Base decay rate per day (higher = faster decay)
  pub fn decay_rate(&self) -> f32 {
    match self {
      Sector::Episodic => 0.02,
      Sector::Procedural => 0.01,
      Sector::Reflective => 0.008,
      Sector::Semantic => 0.005,
      Sector::Emotional => 0.003,
    }
  }

  /// Search ranking boost multiplier
  pub fn search_boost(&self) -> f32 {
    match self {
      Sector::Reflective => 1.2,
      Sector::Semantic => 1.1,
      Sector::Procedural => 1.0,
      Sector::Emotional => 0.9,
      Sector::Episodic => 0.8,
    }
  }

  pub fn as_str(&self) -> &'static str {
    match self {
      Sector::Episodic => "episodic",
      Sector::Semantic => "semantic",
      Sector::Procedural => "procedural",
      Sector::Emotional => "emotional",
      Sector::Reflective => "reflective",
    }
  }
}

impl std::str::FromStr for Sector {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "episodic" => Ok(Sector::Episodic),
      "semantic" => Ok(Sector::Semantic),
      "procedural" => Ok(Sector::Procedural),
      "emotional" => Ok(Sector::Emotional),
      "reflective" => Ok(Sector::Reflective),
      _ => Err(format!("Unknown sector: {}", s)),
    }
  }
}

/// Memory persistence tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
  /// Temporary, per-session memories
  Session,
  /// Persistent, cross-session memories
  Project,
}

impl Tier {
  pub fn as_str(&self) -> &'static str {
    match self {
      Tier::Session => "session",
      Tier::Project => "project",
    }
  }
}

// Re-export MemoryType from llm crate
pub use llm::MemoryType;

impl Sector {
  /// Derive sector from memory type
  pub fn from_memory_type(memory_type: MemoryType) -> Self {
    match memory_type {
      MemoryType::Preference => Sector::Emotional,
      MemoryType::Codebase => Sector::Semantic,
      MemoryType::Decision => Sector::Reflective,
      MemoryType::Gotcha => Sector::Procedural,
      MemoryType::Pattern => Sector::Procedural,
      MemoryType::TurnSummary => Sector::Reflective,
      MemoryType::TaskCompletion => Sector::Episodic,
    }
  }
}

/// Core memory entity
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
  pub id: MemoryId,
  pub project_id: Uuid,
  pub content: String,
  pub summary: Option<String>,
  pub sector: Sector,
  pub tier: Tier,
  #[serde(rename = "type")]
  pub memory_type: Option<MemoryType>,

  // Importance and relevance
  pub importance: f32,   // User-assigned (0-1)
  pub salience: f32,     // Computed, decays over time (0-1)
  pub confidence: f32,   // Extraction confidence (0-1)
  pub access_count: u32, // How many times recalled

  // Content metadata
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub tags: Vec<String>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub concepts: Vec<String>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub files: Vec<String>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub categories: Vec<String>,

  // Scope (for codebase memories)
  pub scope_path: Option<String>,
  pub scope_module: Option<String>,

  // Decay scheduling (internal)
  pub decay_rate: Option<f32>,              // Cached decay rate
  pub next_decay_at: Option<DateTime<Utc>>, // Next scheduled decay time

  // Embedding tracking (internal)
  pub embedding_model_id: Option<String>, // Model used to generate embedding

  // Context
  pub context: Option<String>,
  /// Claude session ID - stable across thread resumes
  pub session_id: Option<String>,
  pub segment_id: Option<Uuid>, // Conversation segment this memory came from

  // Timestamps
  pub created_at: DateTime<Utc>,
  pub updated_at: DateTime<Utc>,
  pub last_accessed: DateTime<Utc>,

  // Validity window (bi-temporal)
  pub valid_from: DateTime<Utc>,
  pub valid_until: Option<DateTime<Utc>>,

  // Soft delete
  #[serde(default)]
  pub is_deleted: bool,
  pub deleted_at: Option<DateTime<Utc>>,

  // Deduplication hashes
  pub content_hash: String, // SHA-256
  pub simhash: u64,         // Locality-sensitive hash

  // Supersession
  pub superseded_by: Option<MemoryId>,
}

impl Memory {
  /// Create a new memory with default values
  pub fn new(project_id: Uuid, content: String, sector: Sector) -> Self {
    let now = Utc::now();
    Self {
      id: MemoryId::new(),
      project_id,
      content,
      summary: None,
      sector,
      tier: Tier::Project,
      memory_type: None,
      importance: 0.5,
      salience: 1.0,
      confidence: 0.5,
      access_count: 0,
      tags: Vec::new(),
      concepts: Vec::new(),
      files: Vec::new(),
      categories: Vec::new(),
      scope_path: None,
      scope_module: None,
      decay_rate: None,
      next_decay_at: None,
      embedding_model_id: None,
      context: None,
      session_id: None,
      segment_id: None,
      created_at: now,
      updated_at: now,
      last_accessed: now,
      valid_from: now,
      valid_until: None,
      is_deleted: false,
      deleted_at: None,
      content_hash: String::new(),
      simhash: 0,
      superseded_by: None,
    }
  }

  /// Check if memory is superseded
  pub fn is_superseded(&self) -> bool {
    self.valid_until.is_some_and(|until| until <= Utc::now()) || self.superseded_by.is_some()
  }

  /// Check if memory is active (not deleted, not superseded)
  pub fn is_active(&self) -> bool {
    !self.is_deleted && !self.is_superseded()
  }

  /// Apply decay based on time since last access
  pub fn apply_decay(&mut self, now: DateTime<Utc>) {
    let days_since_access = (now - self.last_accessed).num_days() as f32;
    let effective_rate = self.sector.decay_rate() / (self.importance + 0.1);
    let decay_factor = (-effective_rate * days_since_access).exp();

    // Access protection: frequently accessed memories decay slower
    let access_protection = (1.0 + self.access_count as f32).ln() * 0.02;
    let access_protection = access_protection.min(0.1);

    self.salience = (self.salience * decay_factor + access_protection).clamp(0.05, 1.0);
    self.updated_at = now;
  }

  /// Reinforce memory (increases salience)
  pub fn reinforce(&mut self, amount: f32, now: DateTime<Utc>) {
    // Diminishing returns as salience approaches 1.0
    self.salience += amount * (1.0 - self.salience);
    self.salience = self.salience.min(1.0);
    self.access_count += 1;
    self.last_accessed = now;
    self.updated_at = now;
  }

  /// Deemphasize memory (decreases salience)
  pub fn deemphasize(&mut self, amount: f32, now: DateTime<Utc>) {
    self.salience = (self.salience - amount).max(0.05);
    self.updated_at = now;
  }

  /// Mark as superseded by another memory
  pub fn supersede(&mut self, new_id: MemoryId, now: DateTime<Utc>) {
    self.valid_until = Some(now);
    self.superseded_by = Some(new_id);
    self.updated_at = now;
  }

  /// Soft delete
  pub fn delete(&mut self, now: DateTime<Utc>) {
    self.is_deleted = true;
    self.deleted_at = Some(now);
    self.updated_at = now;
  }

  /// Restore from soft delete
  pub fn restore(&mut self, now: DateTime<Utc>) {
    self.is_deleted = false;
    self.deleted_at = None;
    self.updated_at = now;
  }

  /// Calculate effective score for ranking (combines salience, importance, sector boost)
  pub fn effective_score(&self) -> f32 {
    self.salience * self.importance * self.sector.search_boost()
  }
}

/// Relationship types between memories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipType {
  /// New memory replaces old (already tracked via superseded_by)
  Supersedes,
  /// New memory contradicts old
  Contradicts,
  /// Memories are related but neither supersedes
  RelatedTo,
  /// New memory extends/builds on old
  BuildsOn,
  /// New memory confirms/validates old
  Confirms,
  /// Memory applies to a specific context
  AppliesTo,
  /// Memory depends on another for context
  DependsOn,
  /// Alternative approach to same problem
  AlternativeTo,
}

impl RelationshipType {
  pub fn as_str(&self) -> &'static str {
    match self {
      RelationshipType::Supersedes => "supersedes",
      RelationshipType::Contradicts => "contradicts",
      RelationshipType::RelatedTo => "related_to",
      RelationshipType::BuildsOn => "builds_on",
      RelationshipType::Confirms => "confirms",
      RelationshipType::AppliesTo => "applies_to",
      RelationshipType::DependsOn => "depends_on",
      RelationshipType::AlternativeTo => "alternative_to",
    }
  }
}

impl std::str::FromStr for RelationshipType {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "supersedes" => Ok(RelationshipType::Supersedes),
      "contradicts" => Ok(RelationshipType::Contradicts),
      "related_to" | "relatedto" => Ok(RelationshipType::RelatedTo),
      "builds_on" | "buildson" => Ok(RelationshipType::BuildsOn),
      "confirms" => Ok(RelationshipType::Confirms),
      "applies_to" | "appliesto" => Ok(RelationshipType::AppliesTo),
      "depends_on" | "dependson" => Ok(RelationshipType::DependsOn),
      "alternative_to" | "alternativeto" => Ok(RelationshipType::AlternativeTo),
      _ => Err(format!("Unknown relationship type: {}", s)),
    }
  }
}

/// A relationship between two memories
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationship {
  pub id: Uuid,
  pub from_memory_id: MemoryId,
  pub to_memory_id: MemoryId,
  #[serde(rename = "type")]
  pub relationship_type: RelationshipType,
  pub confidence: f32,
  pub valid_from: DateTime<Utc>,
  pub valid_until: Option<DateTime<Utc>>,
  pub extracted_by: String, // "llm", "user", "system"
  pub created_at: DateTime<Utc>,
}

impl MemoryRelationship {
  pub fn new(from: MemoryId, to: MemoryId, rel_type: RelationshipType, confidence: f32, extracted_by: &str) -> Self {
    let now = Utc::now();
    Self {
      id: Uuid::new_v4(),
      from_memory_id: from,
      to_memory_id: to,
      relationship_type: rel_type,
      confidence,
      valid_from: now,
      valid_until: None,
      extracted_by: extracted_by.to_string(),
      created_at: now,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_memory_id_roundtrip() {
    let id = MemoryId::new();
    let s = id.to_string();
    let parsed: MemoryId = s.parse().unwrap();
    assert_eq!(id, parsed);
  }

  #[test]
  fn test_memory_reinforce() {
    let mut memory = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    memory.salience = 0.5;

    let now = Utc::now();
    memory.reinforce(0.2, now);

    // Salience should increase but with diminishing returns
    assert!(memory.salience > 0.5);
    assert!(memory.salience < 0.7); // 0.5 + 0.2 * (1 - 0.5) = 0.6
    assert_eq!(memory.access_count, 1);
  }

  #[test]
  fn test_memory_deemphasize() {
    let mut memory = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    memory.salience = 0.5;

    let now = Utc::now();
    memory.deemphasize(0.2, now);

    assert_eq!(memory.salience, 0.3);
  }

  #[test]
  fn test_memory_decay() {
    let mut memory = Memory::new(Uuid::new_v4(), "test".into(), Sector::Episodic);
    memory.salience = 1.0;
    memory.importance = 0.5;

    // Simulate 30 days passing
    let future = Utc::now() + chrono::Duration::days(30);
    memory.apply_decay(future);

    // Salience should have decayed
    assert!(memory.salience < 1.0);
    // But should stay above minimum
    assert!(memory.salience >= 0.05);
  }

  #[test]
  fn test_memory_supersede() {
    let mut memory = Memory::new(Uuid::new_v4(), "old content".into(), Sector::Semantic);
    let new_id = MemoryId::new();
    let now = Utc::now();

    memory.supersede(new_id, now);

    assert!(memory.is_superseded());
    assert!(!memory.is_active());
    assert_eq!(memory.superseded_by, Some(new_id));
  }

  #[test]
  fn test_memory_delete_restore() {
    let mut memory = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    let now = Utc::now();

    memory.delete(now);
    assert!(memory.is_deleted);
    assert!(!memory.is_active());

    memory.restore(now);
    assert!(!memory.is_deleted);
    assert!(memory.is_active());
  }
}
