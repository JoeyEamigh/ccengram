//! High-level extraction functions using LLM inference
//!
//! This module provides functions for:
//! - Signal classification (detecting extractable user inputs)
//! - Memory extraction (extracting memories from conversation context)
//! - Superseding detection (finding memories that should be marked superseded)

use crate::{
  ExtractionResult, InferenceRequest, Model, Result, SignalCategory, SignalClassification, SupersedingResult, infer,
  parse_json,
};

use crate::prompts::{
  EXTRACTION_SYSTEM_PROMPT, ExtractionContext, build_extraction_prompt, build_signal_classification_prompt,
  build_superseding_prompt,
};

use tracing::{debug, info, trace, warn};

/// Classify a user message to determine if it contains extractable signals
pub async fn classify_signal(user_message: &str) -> Result<SignalClassification> {
  debug!(
    message_len = user_message.len(),
    message_preview = %user_message.chars().take(100).collect::<String>(),
    "Starting signal classification"
  );

  let prompt = build_signal_classification_prompt(user_message);

  let request = InferenceRequest::new(prompt)
    .with_model(Model::Haiku) // Use fastest model for classification
    .with_timeout(30);

  let response = infer(request).await?;
  let classification: SignalClassification = parse_json(&response.text)?;

  debug!(
    category = ?classification.category,
    is_extractable = classification.is_extractable,
    is_high_priority = classification.category.is_high_priority(),
    summary = ?classification.summary,
    "Signal classification complete"
  );

  if classification.category.is_high_priority() {
    debug!(
      category = ?classification.category,
      "Detected high-priority signal requiring immediate extraction"
    );
  }

  Ok(classification)
}

/// Extract memories from a conversation segment
pub async fn extract_memories(context: &ExtractionContext) -> Result<ExtractionResult> {
  debug!(
    tool_call_count = context.tool_call_count,
    files_read = context.files_read.len(),
    files_modified = context.files_modified.len(),
    commands_run = context.commands_run.len(),
    errors_encountered = context.errors_encountered.len(),
    completed_tasks = context.completed_tasks.len(),
    has_user_prompt = context.user_prompt.is_some(),
    has_assistant_message = context.last_assistant_message.is_some(),
    "Starting memory extraction"
  );

  // Check if we have enough content to extract
  if !context.has_meaningful_content() {
    debug!(
      tool_call_count = context.tool_call_count,
      files_modified = context.files_modified.len(),
      "Skipping extraction - insufficient content for meaningful memories"
    );
    return Ok(ExtractionResult { memories: Vec::new() });
  }

  let prompt = build_extraction_prompt(context);
  trace!(prompt_len = prompt.len(), "Built extraction prompt");

  let request = InferenceRequest::new(prompt)
    .with_system_prompt(EXTRACTION_SYSTEM_PROMPT)
    .with_model(Model::Haiku) // Haiku is good enough for extraction
    .with_timeout(60);

  debug!("Calling LLM for memory extraction");
  let response = infer(request).await?;
  let result: ExtractionResult = parse_json(&response.text)?;

  if result.memories.is_empty() {
    debug!(
      input_tokens = response.input_tokens,
      output_tokens = response.output_tokens,
      "No memories extracted from context"
    );
  } else {
    // Log summary of extracted memory types
    let memory_types: Vec<&str> = result.memories.iter().map(|m| m.memory_type.as_str()).collect();
    let avg_confidence: f32 = result.memories.iter().map(|m| m.confidence).sum::<f32>() / result.memories.len() as f32;

    info!(
      memories_extracted = result.memories.len(),
      memory_types = ?memory_types,
      avg_confidence = format!("{:.2}", avg_confidence),
      input_tokens = response.input_tokens,
      output_tokens = response.output_tokens,
      "Memory extraction completed"
    );

    // Log individual memories at trace level
    for (i, memory) in result.memories.iter().enumerate() {
      trace!(
        index = i,
        memory_type = %memory.memory_type,
        sector = ?memory.sector,
        confidence = memory.confidence,
        tags = ?memory.tags,
        content_len = memory.content.len(),
        "Extracted memory"
      );
    }
  }

  Ok(result)
}

/// Detect if a new memory supersedes any existing memories
///
/// Takes the new memory content and a list of candidate existing memories
/// (typically found via embedding similarity search).
pub async fn detect_superseding(
  new_memory: &str,
  existing_memories: &[(String, String)], // (id, content)
) -> Result<SupersedingResult> {
  debug!(
    new_memory_len = new_memory.len(),
    candidate_count = existing_memories.len(),
    "Starting superseding detection"
  );

  if existing_memories.is_empty() {
    debug!("No existing memories to check for superseding");
    return Ok(SupersedingResult {
      supersedes: false,
      superseded_memory_id: None,
      reason: None,
      confidence: 1.0,
    });
  }

  // Log candidate IDs at trace level
  trace!(
    candidate_ids = ?existing_memories.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
    "Checking candidates for superseding"
  );

  let prompt = build_superseding_prompt(new_memory, existing_memories);
  trace!(prompt_len = prompt.len(), "Built superseding prompt");

  let request = InferenceRequest::new(prompt).with_model(Model::Haiku).with_timeout(30);

  debug!("Calling LLM for superseding detection");
  let response = infer(request).await?;
  let result: SupersedingResult = parse_json(&response.text)?;

  if result.supersedes {
    info!(
      superseded_id = ?result.superseded_memory_id,
      reason = ?result.reason,
      confidence = result.confidence,
      candidates_checked = existing_memories.len(),
      "Detected memory supersession"
    );
  } else {
    debug!(
      candidates_checked = existing_memories.len(),
      confidence = result.confidence,
      "No superseding relationship detected"
    );
  }

  Ok(result)
}

/// High-priority extraction for corrections and preferences
///
/// Triggered immediately when a high-priority signal is detected.
pub async fn extract_high_priority(
  user_message: &str,
  classification: &SignalClassification,
) -> Result<ExtractionResult> {
  debug!(
    category = ?classification.category,
    is_extractable = classification.is_extractable,
    message_len = user_message.len(),
    "Starting high-priority extraction"
  );

  if !classification.is_extractable {
    debug!(
      category = ?classification.category,
      "Skipping high-priority extraction - signal not extractable"
    );
    return Ok(ExtractionResult { memories: Vec::new() });
  }

  // Build a minimal context just for this message
  let context = ExtractionContext {
    user_prompt: Some(user_message.to_string()),
    tool_call_count: 1, // Force meaningful content check to pass
    ..Default::default()
  };

  let signal_type = match classification.category {
    SignalCategory::Correction => "CORRECTION",
    SignalCategory::Preference => "PREFERENCE",
    _ => "SIGNAL",
  };

  debug!(signal_type = signal_type, "Building high-priority extraction prompt");

  let prompt = format!(
    "This is a high-priority {} signal. Extract the memory immediately.\n\n{}",
    signal_type,
    build_extraction_prompt(&context)
  );
  trace!(prompt_len = prompt.len(), "Built high-priority extraction prompt");

  let request = InferenceRequest::new(prompt)
    .with_system_prompt(EXTRACTION_SYSTEM_PROMPT)
    .with_model(Model::Haiku)
    .with_timeout(30);

  debug!("Calling LLM for high-priority extraction");
  let response = infer(request).await?;
  let result: ExtractionResult = parse_json(&response.text)?;

  if result.memories.is_empty() {
    warn!(
      category = ?classification.category,
      signal_type = signal_type,
      "High-priority extraction yielded no memories"
    );
  } else {
    let memory_types: Vec<&str> = result.memories.iter().map(|m| m.memory_type.as_str()).collect();
    info!(
      memories_extracted = result.memories.len(),
      memory_types = ?memory_types,
      category = ?classification.category,
      signal_type = signal_type,
      input_tokens = response.input_tokens,
      output_tokens = response.output_tokens,
      "High-priority extraction completed"
    );
  }

  Ok(result)
}

#[cfg(test)]
mod tests {
  use super::*;

  // These tests require the claude CLI to be available

  #[tokio::test]
  // #[ignore = "requires claude CLI"]
  async fn test_classify_correction_signal() {
    let result = classify_signal("No, use spaces not tabs for indentation")
      .await
      .unwrap();

    assert!(result.category == SignalCategory::Correction || result.category == SignalCategory::Preference);
    assert!(result.is_extractable);
  }

  #[tokio::test]
  // #[ignore = "requires claude CLI"]
  async fn test_classify_task_signal() {
    let result = classify_signal("Please implement the login feature").await.unwrap();

    assert_eq!(result.category, SignalCategory::Task);
  }

  #[tokio::test]
  // #[ignore = "requires claude CLI"]
  async fn test_extract_from_context() {
    let context = ExtractionContext {
      user_prompt: Some("I always prefer using Result over panicking".into()),
      files_modified: vec!["src/lib.rs".into()],
      tool_call_count: 5,
      ..Default::default()
    };

    let result = extract_memories(&context).await.unwrap();

    // Should extract at least one memory about error handling preference
    assert!(!result.memories.is_empty());
  }

  #[tokio::test]
  // #[ignore = "requires claude CLI"]
  async fn test_detect_superseding_yes() {
    let existing = vec![("mem1".to_string(), "The project uses tabs for indentation".to_string())];

    let result = detect_superseding("The project now uses spaces for indentation (2 spaces)", &existing)
      .await
      .unwrap();

    assert!(result.supersedes);
    assert_eq!(result.superseded_memory_id, Some("mem1".to_string()));
  }

  #[tokio::test]
  // #[ignore = "requires claude CLI"]
  async fn test_detect_superseding_no() {
    let existing = vec![("mem1".to_string(), "The project uses tabs for indentation".to_string())];

    let result = detect_superseding("The database uses PostgreSQL", &existing)
      .await
      .unwrap();

    assert!(!result.supersedes);
  }
}
