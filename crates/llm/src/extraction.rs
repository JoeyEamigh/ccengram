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

/// Classify a user message to determine if it contains extractable signals
pub async fn classify_signal(user_message: &str) -> Result<SignalClassification> {
  let prompt = build_signal_classification_prompt(user_message);

  let request = InferenceRequest::new(prompt)
    .with_model(Model::Haiku) // Use fastest model for classification
    .with_timeout(30);

  let response = infer(request).await?;
  let classification: SignalClassification = parse_json(&response.text)?;

  tracing::debug!(
      category = ?classification.category,
      extractable = classification.is_extractable,
      "Classified user signal"
  );

  Ok(classification)
}

/// Extract memories from a conversation segment
pub async fn extract_memories(context: &ExtractionContext) -> Result<ExtractionResult> {
  // Check if we have enough content to extract
  if !context.has_meaningful_content() {
    tracing::debug!("Skipping extraction - insufficient content");
    return Ok(ExtractionResult { memories: Vec::new() });
  }

  let prompt = build_extraction_prompt(context);

  let request = InferenceRequest::new(prompt)
    .with_system_prompt(EXTRACTION_SYSTEM_PROMPT)
    .with_model(Model::Haiku) // Haiku is good enough for extraction
    .with_timeout(60);

  let response = infer(request).await?;
  let result: ExtractionResult = parse_json(&response.text)?;

  tracing::info!(
    memories_extracted = result.memories.len(),
    input_tokens = response.input_tokens,
    output_tokens = response.output_tokens,
    "Memory extraction completed"
  );

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
  if existing_memories.is_empty() {
    return Ok(SupersedingResult {
      supersedes: false,
      superseded_memory_id: None,
      reason: None,
      confidence: 1.0,
    });
  }

  let prompt = build_superseding_prompt(new_memory, existing_memories);

  let request = InferenceRequest::new(prompt).with_model(Model::Haiku).with_timeout(30);

  let response = infer(request).await?;
  let result: SupersedingResult = parse_json(&response.text)?;

  if result.supersedes {
    tracing::info!(
        superseded_id = ?result.superseded_memory_id,
        reason = ?result.reason,
        confidence = result.confidence,
        "Detected memory supersession"
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
  if !classification.is_extractable {
    return Ok(ExtractionResult { memories: Vec::new() });
  }

  // Build a minimal context just for this message
  let context = ExtractionContext {
    user_prompt: Some(user_message.to_string()),
    tool_call_count: 1, // Force meaningful content check to pass
    ..Default::default()
  };

  let prompt = format!(
    "This is a high-priority {} signal. Extract the memory immediately.\n\n{}",
    match classification.category {
      SignalCategory::Correction => "CORRECTION",
      SignalCategory::Preference => "PREFERENCE",
      _ => "SIGNAL",
    },
    build_extraction_prompt(&context)
  );

  let request = InferenceRequest::new(prompt)
    .with_system_prompt(EXTRACTION_SYSTEM_PROMPT)
    .with_model(Model::Haiku)
    .with_timeout(30);

  let response = infer(request).await?;
  let result: ExtractionResult = parse_json(&response.text)?;

  tracing::info!(
      memories_extracted = result.memories.len(),
      category = ?classification.category,
      "High-priority extraction completed"
  );

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
