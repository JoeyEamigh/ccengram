//! Memory lifecycle operations.
//!
//! Provides operations to manage memory salience over time:
//! - `reinforce` - Increase salience when memory is accessed/useful
//! - `deemphasize` - Decrease salience when memory is less relevant
//! - `supersede` - Mark a memory as replaced by a newer one

use super::MemoryContext;
use crate::{
  ipc::types::memory::{MemorySupersedeResult, MemoryUpdateResult},
  service::util::{Resolver, ServiceError},
};

/// Reinforce a memory, increasing its salience.
///
/// This should be called when:
/// - A memory is accessed and found useful
/// - The user explicitly marks a memory as important
/// - A memory contributes to a successful task completion
///
/// # Arguments
/// * `ctx` - Memory context with database
/// * `memory_id` - ID or prefix of the memory to reinforce
/// * `amount` - Amount to reinforce (default 0.1, clamped to reasonable range)
///
/// # Returns
/// * `Ok(MemoryUpdateResult)` - Result with new salience value
/// * `Err(ServiceError)` - If memory not found or update fails
///
/// # Reinforcement Algorithm
///
/// Uses diminishing returns: `new_salience = salience + amount * (1.0 - salience)`
///
/// This means:
/// - Low-salience memories get boosted more
/// - High-salience memories approach 1.0 asymptotically
/// - You can never over-reinforce a memory past 1.0
pub async fn reinforce(
  ctx: &MemoryContext<'_>,
  memory_id: &str,
  amount: Option<f32>,
) -> Result<MemoryUpdateResult, ServiceError> {
  // Resolve to get the ID (handles prefixes) and verify existence
  let memory = Resolver::memory(ctx.db, memory_id).await?;
  let amount = amount.unwrap_or(0.1).clamp(0.01, 0.5);

  // Atomic update - no read-modify-write race
  ctx.db.reinforce_memory(&memory.id, amount).await?;

  // Calculate expected new salience for response (approximate, may differ slightly due to race)
  let new_salience = (memory.salience + amount * (1.0 - memory.salience)).min(1.0);

  Ok(MemoryUpdateResult {
    id: memory.id.to_string(),
    new_salience,
    message: "Memory reinforced".to_string(),
  })
}

/// Deemphasize a memory, decreasing its salience.
///
/// This should be called when:
/// - A memory is marked as less relevant by the user
/// - A memory contributed to an incorrect answer
/// - The user explicitly deprioritizes a memory
///
/// # Arguments
/// * `ctx` - Memory context with database
/// * `memory_id` - ID or prefix of the memory to deemphasize
/// * `amount` - Amount to deemphasize (default 0.2, clamped to reasonable range)
///
/// # Returns
/// * `Ok(MemoryUpdateResult)` - Result with new salience value
/// * `Err(ServiceError)` - If memory not found or update fails
///
/// # Deemphasis Algorithm
///
/// Simple subtraction with floor: `new_salience = max(salience - amount, 0.05)`
///
/// The minimum salience of 0.05 ensures memories are never completely forgotten
/// and can still be found if explicitly searched for.
pub async fn deemphasize(
  ctx: &MemoryContext<'_>,
  memory_id: &str,
  amount: Option<f32>,
) -> Result<MemoryUpdateResult, ServiceError> {
  // Resolve to get the ID (handles prefixes) and verify existence
  let memory = Resolver::memory(ctx.db, memory_id).await?;
  let amount = amount.unwrap_or(0.2).clamp(0.01, 0.5);

  // Atomic update - no read-modify-write race
  ctx.db.deemphasize_memory(&memory.id, amount).await?;

  // Calculate expected new salience for response (approximate, may differ slightly due to race)
  let new_salience = (memory.salience - amount).max(0.05);

  Ok(MemoryUpdateResult {
    id: memory.id.to_string(),
    new_salience,
    message: "Memory de-emphasized".to_string(),
  })
}

/// Mark a memory as superseded by another.
///
/// This is used when:
/// - Information has been updated and a new memory replaces the old
/// - The user corrects a previous memory
/// - A more complete version of information is recorded
///
/// The old memory remains in the database but is marked as superseded,
/// which applies a penalty in search ranking. This preserves the history
/// while ensuring the newer memory is preferred.
///
/// # Arguments
/// * `ctx` - Memory context with database
/// * `old_memory_id` - ID or prefix of the memory being superseded
/// * `new_memory_id` - ID or prefix of the new memory
///
/// # Returns
/// * `Ok(MemorySupersedeResult)` - Result with both memory IDs
/// * `Err(ServiceError)` - If either memory not found or update fails
pub async fn supersede(
  ctx: &MemoryContext<'_>,
  old_memory_id: &str,
  new_memory_id: &str,
) -> Result<MemorySupersedeResult, ServiceError> {
  // Resolve both memories in parallel to verify existence and handle prefixes
  let (old_result, new_result) = tokio::join!(
    Resolver::memory(ctx.db, old_memory_id),
    Resolver::memory(ctx.db, new_memory_id)
  );
  let old_memory = old_result?;
  let new_memory = new_result?;

  // Atomic update - marks old memory as superseded
  ctx.db.supersede_memory(&old_memory.id, &new_memory.id).await?;

  Ok(MemorySupersedeResult {
    old_id: old_memory.id.to_string(),
    new_id: new_memory.id.to_string(),
    message: "Memory superseded".to_string(),
  })
}

#[allow(dead_code)] // leaving this because i may want to allow most-used reinforcement in the future
/// Batch reinforce multiple memories.
///
/// This is useful for reinforcing all top results from a search
/// or batch processing from analytics.
///
/// # Arguments
/// * `ctx` - Memory context with database
/// * `memory_ids` - List of memory IDs or prefixes
/// * `amounts` - Reinforcement amounts (same length as memory_ids, or single value for all)
///
/// # Returns
/// * `Ok(Vec<MemoryUpdateResult>)` - Results for each memory
/// * `Err(ServiceError)` - If any operation fails (partial results not returned)
pub async fn reinforce_batch(
  ctx: &MemoryContext<'_>,
  memory_ids: &[String],
  amounts: &[f32],
) -> Result<Vec<MemoryUpdateResult>, ServiceError> {
  let mut results = Vec::with_capacity(memory_ids.len());

  for (i, memory_id) in memory_ids.iter().enumerate() {
    let amount = amounts.get(i).or(amounts.first()).copied();
    let result = reinforce(ctx, memory_id, amount).await?;
    results.push(result);
  }

  Ok(results)
}

/// Update the salience of a memory to a specific value.
///
/// This is a more direct operation than reinforce/deemphasize,
/// useful for administrative corrections or bulk updates.
///
/// # Arguments
/// * `ctx` - Memory context with database
/// * `memory_id` - ID or prefix of the memory
/// * `salience` - New salience value (clamped to 0.05 - 1.0)
///
/// # Returns
/// * `Ok(MemoryUpdateResult)` - Result with new salience value
/// * `Err(ServiceError)` - If memory not found or update fails
pub async fn set_salience(
  ctx: &MemoryContext<'_>,
  memory_id: &str,
  salience: f32,
) -> Result<MemoryUpdateResult, ServiceError> {
  // Resolve to get the ID (handles prefixes) and verify existence
  let memory = Resolver::memory(ctx.db, memory_id).await?;
  let salience = salience.clamp(0.05, 1.0);

  // Atomic update - no read-modify-write race
  ctx.db.set_memory_salience(&memory.id, salience).await?;

  Ok(MemoryUpdateResult {
    id: memory.id.to_string(),
    new_salience: salience,
    message: "Salience updated".to_string(),
  })
}
