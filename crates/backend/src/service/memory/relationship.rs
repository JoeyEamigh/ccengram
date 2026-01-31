//! Memory relationship service.
//!
//! Provides operations for managing relationships between memories.

use uuid::Uuid;

use crate::{
  db::ProjectDb,
  domain::memory::RelationshipType,
  ipc::types::relationship::{
    DeletedResult, RelationshipAddParams, RelationshipDeleteParams, RelationshipListItem, RelationshipResult,
  },
  service::util::{Resolver, ServiceError},
};

/// Add a relationship between two memories.
///
/// # Arguments
/// * `db` - Project database
/// * `params` - Relationship add parameters
///
/// # Returns
/// * `Ok(RelationshipResult)` - Created relationship
/// * `Err(ServiceError)` - If creation fails
pub async fn add(db: &ProjectDb, params: RelationshipAddParams) -> Result<RelationshipResult, ServiceError> {
  // Resolve both memories in parallel
  let (from_result, to_result) = tokio::join!(
    Resolver::memory(db, &params.from_memory_id),
    Resolver::memory(db, &params.to_memory_id)
  );
  let from_memory = from_result?;
  let to_memory = to_result?;

  // Parse relationship type
  let rel_type = params
    .relationship_type
    .parse::<RelationshipType>()
    .map_err(ServiceError::Validation)?;

  let confidence = params.confidence.unwrap_or(1.0);

  // Create the relationship
  let relationship = db
    .create_relationship(&from_memory.id, &to_memory.id, rel_type, confidence, "user")
    .await?;

  Ok(RelationshipResult {
    id: relationship.id.to_string(),
    from_memory_id: relationship.from_memory_id.to_string(),
    to_memory_id: relationship.to_memory_id.to_string(),
    relationship_type: relationship.relationship_type.as_str().to_string(),
    confidence: relationship.confidence,
  })
}

/// Delete a relationship by ID.
///
/// # Arguments
/// * `db` - Project database
/// * `params` - Relationship delete parameters
///
/// # Returns
/// * `Ok(DeletedResult)` - Deletion result
/// * `Err(ServiceError)` - If deletion fails
pub async fn delete(db: &ProjectDb, params: RelationshipDeleteParams) -> Result<DeletedResult, ServiceError> {
  // Parse relationship ID as UUID
  let id = Uuid::parse_str(&params.relationship_id)
    .map_err(|_| ServiceError::Validation(format!("Invalid relationship ID: {}", params.relationship_id)))?;

  db.delete_relationship(&id).await?;

  Ok(DeletedResult { deleted: true })
}

/// List all relationships for a memory.
///
/// # Arguments
/// * `db` - Project database
/// * `memory_id` - Memory ID or prefix
///
/// # Returns
/// * `Ok(Vec<RelationshipListItem>)` - List of relationships
/// * `Err(ServiceError)` - If query fails
pub async fn list(db: &ProjectDb, memory_id: &str) -> Result<Vec<RelationshipListItem>, ServiceError> {
  let memory = Resolver::memory(db, memory_id).await?;

  let relationships = db.get_all_relationships(&memory.id).await?;

  let items: Vec<RelationshipListItem> = relationships
    .iter()
    .map(|r| RelationshipListItem {
      id: r.id.to_string(),
      from_memory_id: r.from_memory_id.to_string(),
      to_memory_id: r.to_memory_id.to_string(),
      relationship_type: r.relationship_type.as_str().to_string(),
      confidence: r.confidence,
      created_at: r.created_at.to_rfc3339(),
      valid_until: r.valid_until.map(|t| t.to_rfc3339()),
    })
    .collect();

  Ok(items)
}
