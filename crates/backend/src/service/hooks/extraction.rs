//! Memory extraction service for hooks.
//!
//! This module handles extracting memories from session context using
//! either LLM-based extraction or basic summary fallback.

use llm::{ExtractedMemory, LlmProvider, SignalClassification};
use tracing::{debug, warn};
use uuid::Uuid;

use super::context::SegmentContext;
use crate::{
  context::memory::extract::{
    classifier::{extract_concepts, extract_files},
    dedup::compute_hashes,
  },
  db::ProjectDb,
  domain::memory::{Memory, Sector},
  embedding::EmbeddingProvider,
  service::util::ServiceError,
};

/// Context for memory extraction operations.
pub struct ExtractionContext<'a> {
  /// Project database connection
  pub db: &'a ProjectDb,
  /// Optional embedding provider for vector creation
  pub embedding: &'a dyn EmbeddingProvider,
  /// Optional LLM provider for intelligent extraction
  pub llm: Option<&'a dyn LlmProvider>,
  /// Project UUID for new memories
  pub project_id: Uuid,
}

impl<'a> ExtractionContext<'a> {
  /// Create a new extraction context
  pub fn new(
    db: &'a ProjectDb,
    embedding: &'a dyn EmbeddingProvider,
    llm: Option<&'a dyn LlmProvider>,
    project_id: Uuid,
  ) -> Self {
    Self {
      db,
      embedding,
      llm,
      project_id,
    }
  }

  /// Get an embedding for the given text
  async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
    // Document mode - we're embedding memory content for storage
    Ok(
      self
        .embedding
        .embed(text, crate::embedding::EmbeddingMode::Document)
        .await?,
    )
  }
}

/// Result of memory extraction
pub struct ExtractMemoryResult {
  /// ID of the created memory, if any
  pub memory_id: Option<String>,
}

/// Extract and store a memory from content.
///
/// # Arguments
/// * `ctx` - Extraction context with database and providers
/// * `content` - The content to create a memory from
/// * `seen_hashes` - Set of already-seen content hashes for deduplication
///
/// # Returns
/// * `Ok(ExtractMemoryResult)` - Result with optional memory ID
/// * `Err(ServiceError)` - If storage fails
pub async fn extract_memory(
  ctx: &ExtractionContext<'_>,
  content: &str,
  seen_hashes: &mut std::collections::HashSet<String>,
) -> Result<ExtractMemoryResult, ServiceError> {
  // Skip if content is too short
  if content.len() < 20 {
    debug!(
      content_len = content.len(),
      "Skipping memory extraction: content too short"
    );
    return Ok(ExtractMemoryResult { memory_id: None });
  }

  // Compute hashes for dedup
  let (content_hash, simhash) = compute_hashes(content);

  // Check for duplicates
  if seen_hashes.contains(&content_hash) {
    debug!("Skipping duplicate memory (exact hash match)");
    return Ok(ExtractMemoryResult { memory_id: None });
  }

  // Default to Semantic sector for fallback extraction (no LLM to determine type)
  let sector = Sector::Semantic;

  // Create memory
  let mut memory = Memory::new(ctx.project_id, content.to_string(), sector);
  memory.content_hash = content_hash.clone();
  memory.simhash = simhash;
  memory.concepts = extract_concepts(content);
  memory.files = extract_files(content);

  // Generate embedding
  let vector = ctx.get_embedding(content).await?;

  // Store memory
  ctx.db.add_memory(&memory, &vector).await?;

  // Track hash
  seen_hashes.insert(content_hash);

  debug!("Extracted memory: {} ({:?})", memory.id, sector);
  Ok(ExtractMemoryResult {
    memory_id: Some(memory.id.to_string()),
  })
}

/// Store an extracted memory from LLM extraction.
///
/// # Arguments
/// * `ctx` - Extraction context with database and providers
/// * `extracted` - The LLM-extracted memory data
/// * `seen_hashes` - Set of already-seen content hashes for deduplication
///
/// # Returns
/// * `Ok(ExtractMemoryResult)` - Result with optional memory ID
/// * `Err(ServiceError)` - If storage fails
pub async fn store_extracted_memory(
  ctx: &ExtractionContext<'_>,
  extracted: &ExtractedMemory,
  seen_hashes: &mut std::collections::HashSet<String>,
) -> Result<ExtractMemoryResult, ServiceError> {
  // Skip if content is too short
  if extracted.content.len() < 20 {
    debug!(
      content_len = extracted.content.len(),
      "Skipping LLM memory storage: content too short"
    );
    return Ok(ExtractMemoryResult { memory_id: None });
  }

  // Compute hashes for dedup
  let (content_hash, simhash) = compute_hashes(&extracted.content);

  // Check for duplicates
  if seen_hashes.contains(&content_hash) {
    debug!("Skipping duplicate extracted memory (exact hash match)");
    return Ok(ExtractMemoryResult { memory_id: None });
  }

  // Derive sector from memory type
  let sector = Sector::from_memory_type(extracted.memory_type);

  // Create memory
  let mut memory = Memory::new(ctx.project_id, extracted.content.clone(), sector);
  memory.content_hash = content_hash.clone();
  memory.simhash = simhash;
  memory.concepts = extract_concepts(&extracted.content);
  memory.files = extract_files(&extracted.content);
  memory.tags = extracted.tags.clone();
  memory.salience = extracted.confidence;
  memory.memory_type = Some(extracted.memory_type);
  if let Some(ref summary) = extracted.summary {
    memory.summary = Some(summary.clone());
  }

  // Generate embedding
  let vector = ctx.get_embedding(&extracted.content).await?;

  // Store memory
  ctx.db.add_memory(&memory, &vector).await?;

  // Track hash
  seen_hashes.insert(content_hash);

  debug!(
    "Stored LLM-extracted memory: {} ({:?}, {:?}, confidence: {:.2})",
    memory.id, sector, memory.memory_type, extracted.confidence
  );
  Ok(ExtractMemoryResult {
    memory_id: Some(memory.id.to_string()),
  })
}

/// Extract memories using LLM from segment context.
///
/// # Arguments
/// * `ctx` - Extraction context with database and providers
/// * `segment` - The segment context to extract from
/// * `seen_hashes` - Set of already-seen content hashes for deduplication
///
/// # Returns
/// * `Ok(Vec<String>)` - List of created memory IDs
/// * `Err(ServiceError)` - If extraction fails
pub async fn extract_with_llm(
  ctx: &ExtractionContext<'_>,
  segment: &SegmentContext,
  seen_hashes: &mut std::collections::HashSet<String>,
) -> Result<Vec<String>, ServiceError> {
  if !segment.has_meaningful_work() {
    return Ok(Vec::new());
  }

  let Some(llm) = ctx.llm else {
    // No LLM provider, fall back to basic extraction
    return extract_with_summary(ctx, segment, seen_hashes).await;
  };

  let extraction_context = segment.to_extraction_context();
  let mut memories_created = Vec::new();

  match llm::extraction::extract_memories(llm, &extraction_context).await {
    Ok(result) => {
      for extracted in &result.memories {
        if let Ok(res) = store_extracted_memory(ctx, extracted, seen_hashes).await
          && let Some(id) = res.memory_id
        {
          memories_created.push(id);
        }
      }
      debug!(
        "LLM extraction completed: {} memories created from {} candidates",
        memories_created.len(),
        result.memories.len()
      );
    }
    Err(e) => {
      warn!("LLM extraction failed, falling back to basic summary: {}", e);
      // Fall back to basic summary extraction
      return extract_with_summary(ctx, segment, seen_hashes).await;
    }
  }

  Ok(memories_created)
}

/// Extract memories using basic summary (fallback when LLM unavailable).
async fn extract_with_summary(
  ctx: &ExtractionContext<'_>,
  segment: &SegmentContext,
  seen_hashes: &mut std::collections::HashSet<String>,
) -> Result<Vec<String>, ServiceError> {
  let mut memories_created = Vec::new();

  if segment.has_meaningful_work()
    && let Some(summary) = segment.summary()
    && let Ok(res) = extract_memory(ctx, &summary, seen_hashes).await
    && let Some(id) = res.memory_id
  {
    memories_created.push(id);
  }

  Ok(memories_created)
}

/// Extract high-priority memories (corrections/preferences) immediately.
///
/// # Arguments
/// * `ctx` - Extraction context with database and providers
/// * `user_message` - The user's message containing the signal
/// * `classification` - The signal classification result
/// * `seen_hashes` - Set of already-seen content hashes for deduplication
///
/// # Returns
/// * `Ok(Vec<String>)` - List of created memory IDs
/// * `Err(ServiceError)` - If extraction fails
pub async fn extract_high_priority(
  ctx: &ExtractionContext<'_>,
  user_message: &str,
  classification: &SignalClassification,
  seen_hashes: &mut std::collections::HashSet<String>,
) -> Result<Vec<String>, ServiceError> {
  let Some(llm) = ctx.llm else {
    // No LLM provider, can't do high-priority extraction
    return Ok(Vec::new());
  };

  if !classification.is_extractable {
    return Ok(Vec::new());
  }

  debug!("High-priority signal detected: {:?}", classification.category);

  let mut memories_created = Vec::new();

  match llm::extraction::extract_high_priority(llm, user_message, classification).await {
    Ok(result) => {
      for extracted in &result.memories {
        if let Ok(res) = store_extracted_memory(ctx, extracted, seen_hashes).await
          && let Some(id) = res.memory_id
        {
          memories_created.push(id);
        }
      }
      if !memories_created.is_empty() {
        debug!("High-priority extraction: {} memories", memories_created.len());
      }
    }
    Err(e) => {
      debug!("High-priority extraction failed: {}", e);
    }
  }

  Ok(memories_created)
}

/// Classify a signal from user input.
///
/// # Arguments
/// * `llm` - LLM provider for classification
/// * `user_message` - The user's message to classify
///
/// # Returns
/// * `Ok(SignalClassification)` - Classification result
/// * `Err(ServiceError)` - If classification fails
pub async fn classify_signal(llm: &dyn LlmProvider, user_message: &str) -> Result<SignalClassification, ServiceError> {
  Ok(llm::extraction::classify_signal(llm, user_message).await?)
}
