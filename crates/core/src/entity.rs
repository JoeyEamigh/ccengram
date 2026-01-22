use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Entity types that can be tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
  /// A person (user, teammate, etc.)
  Person,
  /// A project or repository
  Project,
  /// A technology, framework, or tool
  Technology,
  /// An organization or company
  Organization,
  /// A concept or pattern
  Concept,
  /// A file or path
  File,
  /// A function, class, or other code symbol
  Symbol,
  /// Other entity type
  Other,
}

impl EntityType {
  pub fn as_str(&self) -> &'static str {
    match self {
      EntityType::Person => "person",
      EntityType::Project => "project",
      EntityType::Technology => "technology",
      EntityType::Organization => "organization",
      EntityType::Concept => "concept",
      EntityType::File => "file",
      EntityType::Symbol => "symbol",
      EntityType::Other => "other",
    }
  }
}

impl std::str::FromStr for EntityType {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "person" => Ok(EntityType::Person),
      "project" => Ok(EntityType::Project),
      "technology" => Ok(EntityType::Technology),
      "organization" => Ok(EntityType::Organization),
      "concept" => Ok(EntityType::Concept),
      "file" => Ok(EntityType::File),
      "symbol" => Ok(EntityType::Symbol),
      "other" => Ok(EntityType::Other),
      _ => Err(format!("Unknown entity type: {}", s)),
    }
  }
}

/// A named entity tracked across memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
  pub id: Uuid,
  pub name: String,
  pub entity_type: EntityType,
  pub summary: Option<String>,
  pub aliases: Vec<String>,
  pub first_seen_at: DateTime<Utc>,
  pub last_seen_at: DateTime<Utc>,
  pub mention_count: u32,
}

impl Entity {
  pub fn new(name: String, entity_type: EntityType) -> Self {
    let now = Utc::now();
    Self {
      id: Uuid::now_v7(),
      name,
      entity_type,
      summary: None,
      aliases: Vec::new(),
      first_seen_at: now,
      last_seen_at: now,
      mention_count: 1,
    }
  }

  /// Record another mention of this entity
  pub fn mention(&mut self) {
    self.mention_count += 1;
    self.last_seen_at = Utc::now();
  }

  /// Add an alias for this entity
  pub fn add_alias(&mut self, alias: String) {
    if !self.aliases.contains(&alias) && alias != self.name {
      self.aliases.push(alias);
    }
  }
}

/// Role of an entity in a memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityRole {
  /// Entity is the main subject
  Subject,
  /// Entity is a secondary reference
  Reference,
  /// Entity is part of the context
  Context,
  /// Entity is mentioned incidentally
  Mention,
}

impl EntityRole {
  pub fn as_str(&self) -> &'static str {
    match self {
      EntityRole::Subject => "subject",
      EntityRole::Reference => "reference",
      EntityRole::Context => "context",
      EntityRole::Mention => "mention",
    }
  }
}

impl std::str::FromStr for EntityRole {
  type Err = String;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "subject" => Ok(EntityRole::Subject),
      "reference" => Ok(EntityRole::Reference),
      "context" => Ok(EntityRole::Context),
      "mention" => Ok(EntityRole::Mention),
      _ => Err(format!("Unknown entity role: {}", s)),
    }
  }
}

/// Link between a memory and an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntityLink {
  pub id: Uuid,
  pub memory_id: String,
  pub entity_id: Uuid,
  pub role: EntityRole,
  pub confidence: f32,
  pub extracted_at: DateTime<Utc>,
}

impl MemoryEntityLink {
  pub fn new(memory_id: String, entity_id: Uuid, role: EntityRole, confidence: f32) -> Self {
    Self {
      id: Uuid::now_v7(),
      memory_id,
      entity_id,
      role,
      confidence,
      extracted_at: Utc::now(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_entity_creation() {
    let entity = Entity::new("Rust".to_string(), EntityType::Technology);
    assert_eq!(entity.name, "Rust");
    assert_eq!(entity.entity_type, EntityType::Technology);
    assert_eq!(entity.mention_count, 1);
  }

  #[test]
  fn test_entity_mention() {
    let mut entity = Entity::new("TypeScript".to_string(), EntityType::Technology);
    entity.mention();
    assert_eq!(entity.mention_count, 2);
  }

  #[test]
  fn test_entity_alias() {
    let mut entity = Entity::new("TypeScript".to_string(), EntityType::Technology);
    entity.add_alias("TS".to_string());
    entity.add_alias("TypeScript".to_string()); // Should not add duplicate of name
    assert_eq!(entity.aliases, vec!["TS".to_string()]);
  }

  #[test]
  fn test_entity_type_parsing() {
    assert_eq!("person".parse::<EntityType>().unwrap(), EntityType::Person);
    assert_eq!("technology".parse::<EntityType>().unwrap(), EntityType::Technology);
    assert!("unknown".parse::<EntityType>().is_err());
  }

  #[test]
  fn test_entity_role_parsing() {
    assert_eq!("subject".parse::<EntityRole>().unwrap(), EntityRole::Subject);
    assert_eq!("reference".parse::<EntityRole>().unwrap(), EntityRole::Reference);
  }
}
