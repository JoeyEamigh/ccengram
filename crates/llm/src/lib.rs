//! LLM inference for CCEngram via Claude CLI
//!
//! This crate provides LLM inference capabilities by invoking the `claude` CLI
//! in print mode with JSON output. It disables all hooks and plugins to avoid
//! recursive calls when invoked from within CCEngram hooks.

pub mod extraction;
pub mod prompts;

pub use extraction::{classify_signal, detect_superseding, extract_high_priority, extract_memories};
pub use prompts::ExtractionContext;

use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

/// Errors that can occur during LLM inference
#[derive(Debug, Error)]
pub enum LlmError {
  #[error("Claude executable not found. Ensure 'claude' is in your PATH.")]
  ClaudeNotFound,

  #[error("Failed to spawn claude process: {0}")]
  SpawnFailed(#[from] std::io::Error),

  #[error("Claude process timed out after {0} seconds")]
  Timeout(u64),

  #[error("Claude process exited with non-zero status: {0}")]
  ProcessFailed(i32),

  #[error("Failed to parse JSON response: {0}")]
  ParseError(#[from] serde_json::Error),

  #[error("No assistant message in response")]
  NoResponse,

  #[error("Claude returned an error: {0}")]
  ClaudeError(String),
}

pub type Result<T> = std::result::Result<T, LlmError>;

/// Model selection for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
  /// Claude 3.5 Haiku - fastest, cheapest
  #[default]
  Haiku,
  /// Claude Sonnet 4 - balanced
  Sonnet,
  /// Claude Opus 4 - most capable
  Opus,
}

impl Model {
  pub fn as_str(&self) -> &'static str {
    match self {
      Model::Haiku => "haiku",
      Model::Sonnet => "sonnet",
      Model::Opus => "opus",
    }
  }
}

/// Request for LLM inference
#[derive(Debug, Clone)]
pub struct InferenceRequest {
  /// The prompt to send
  pub prompt: String,
  /// Optional system prompt
  pub system_prompt: Option<String>,
  /// Model to use (default: Haiku)
  pub model: Model,
  /// Timeout in seconds (default: 60)
  pub timeout_secs: u64,
}

impl InferenceRequest {
  pub fn new(prompt: impl Into<String>) -> Self {
    Self {
      prompt: prompt.into(),
      system_prompt: None,
      model: Model::default(),
      timeout_secs: 60,
    }
  }

  pub fn with_system_prompt(mut self, system: impl Into<String>) -> Self {
    self.system_prompt = Some(system.into());
    self
  }

  pub fn with_model(mut self, model: Model) -> Self {
    self.model = model;
    self
  }

  pub fn with_timeout(mut self, secs: u64) -> Self {
    self.timeout_secs = secs;
    self
  }
}

/// Response from LLM inference
#[derive(Debug, Clone)]
pub struct InferenceResponse {
  /// The text response
  pub text: String,
  /// Input tokens used
  pub input_tokens: u32,
  /// Output tokens generated
  pub output_tokens: u32,
  /// Cost in USD (if available)
  pub cost_usd: Option<f64>,
  /// Duration in milliseconds
  pub duration_ms: u64,
}

// Internal types for parsing Claude CLI JSON output

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum ClaudeMessage {
  System {},
  Assistant(AssistantMessage),
  Result(ResultMessage),
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
  message: MessageContent,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
  content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum ContentBlock {
  Text {
    text: String,
  },
  #[serde(other)]
  Other,
}

#[derive(Debug, Deserialize)]
struct ResultMessage {
  #[serde(default)]
  is_error: bool,
  #[serde(default)]
  duration_ms: u64,
  #[serde(default)]
  total_cost_usd: f64,
  usage: Option<Usage>,
  result: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
  input_tokens: u32,
  output_tokens: u32,
}

/// Find the claude executable in PATH
fn find_claude() -> Result<String> {
  let which_cmd = if cfg!(windows) { "where" } else { "which" };

  let output = std::process::Command::new(which_cmd)
    .arg("claude")
    .output()
    .map_err(|_| LlmError::ClaudeNotFound)?;

  if !output.status.success() {
    return Err(LlmError::ClaudeNotFound);
  }

  let path = String::from_utf8_lossy(&output.stdout)
    .lines()
    .next()
    .map(|s| s.trim().to_string())
    .ok_or(LlmError::ClaudeNotFound)?;

  if path.is_empty() {
    return Err(LlmError::ClaudeNotFound);
  }

  Ok(path)
}

/// Perform LLM inference using the Claude CLI
///
/// This function:
/// 1. Spawns the `claude` CLI with `--print` mode
/// 2. Disables all hooks/plugins to avoid recursion
/// 3. Parses the JSON output
/// 4. Returns the text response and usage stats
pub async fn infer(request: InferenceRequest) -> Result<InferenceResponse> {
  let claude_path = find_claude()?;

  let full_prompt = if let Some(system) = &request.system_prompt {
    format!("{}\n\n{}", system, request.prompt)
  } else {
    request.prompt.clone()
  };

  tracing::debug!(
    model = request.model.as_str(),
    prompt_len = full_prompt.len(),
    "Starting LLM inference"
  );

  let mut cmd = Command::new(&claude_path);
  cmd
    .arg("-p") // Print mode
    .arg("--model")
    .arg(request.model.as_str())
    .arg("--output-format")
    .arg("json")
    .arg("--no-session-persistence")
    // Disable all hooks and plugins to avoid recursion
    .arg("--settings")
    .arg(r#"{"hooks":{}}"#)
    .arg("--setting-sources")
    .arg("")
    // Disable all tools - we only want text generation
    .arg("--tools")
    .arg("")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

  let mut child = cmd.spawn()?;

  // Write prompt to stdin
  if let Some(mut stdin) = child.stdin.take() {
    use tokio::io::AsyncWriteExt;
    stdin.write_all(full_prompt.as_bytes()).await?;
    drop(stdin); // Close stdin to signal end of input
  }

  let stdout = child
    .stdout
    .take()
    .ok_or_else(|| std::io::Error::other("stdout not piped"))?;
  let mut reader = tokio::io::BufReader::new(stdout);

  // Read with timeout
  let read_future = async {
    let mut output = String::new();
    reader.read_to_string(&mut output).await?;
    Ok::<_, std::io::Error>(output)
  };

  let output = timeout(Duration::from_secs(request.timeout_secs), read_future)
    .await
    .map_err(|_| LlmError::Timeout(request.timeout_secs))??;

  // Wait for process to complete
  let status = child.wait().await?;
  if !status.success() {
    return Err(LlmError::ProcessFailed(status.code().unwrap_or(-1)));
  }

  // Parse JSON array output
  // The output is a JSON array: [{system}, {assistant}, {result}]
  let messages: Vec<ClaudeMessage> = serde_json::from_str(&output)?;

  let mut response_text = String::new();
  let mut input_tokens = 0u32;
  let mut output_tokens = 0u32;
  let mut cost_usd = None;
  let mut duration_ms = 0u64;

  for msg in messages {
    match msg {
      ClaudeMessage::System {} => {
        // Session init, nothing to extract
      }
      ClaudeMessage::Assistant(assistant) => {
        for block in assistant.message.content {
          if let ContentBlock::Text { text } = block {
            response_text.push_str(&text);
          }
        }
      }
      ClaudeMessage::Result(result) => {
        if result.is_error {
          let error_msg = result.result.unwrap_or_else(|| "Unknown error".to_string());
          return Err(LlmError::ClaudeError(error_msg));
        }

        duration_ms = result.duration_ms;
        cost_usd = Some(result.total_cost_usd);

        if let Some(usage) = result.usage {
          input_tokens = usage.input_tokens;
          output_tokens = usage.output_tokens;
        }
      }
    }
  }

  if response_text.is_empty() {
    return Err(LlmError::NoResponse);
  }

  tracing::debug!(
    response_len = response_text.len(),
    input_tokens,
    output_tokens,
    duration_ms,
    "LLM inference completed"
  );

  Ok(InferenceResponse {
    text: response_text,
    input_tokens,
    output_tokens,
    cost_usd,
    duration_ms,
  })
}

/// Parse JSON from an LLM response text
///
/// Handles responses that may be wrapped in markdown code blocks:
/// - ```json ... ```
/// - ``` ... ```
/// - Raw JSON
pub fn parse_json<T: for<'de> Deserialize<'de>>(text: &str) -> std::result::Result<T, serde_json::Error> {
  // Try to extract JSON from markdown code blocks
  let json_str = if let Some(captures) = extract_code_block(text) {
    captures
  } else {
    text.trim()
  };

  serde_json::from_str(json_str)
}

fn extract_code_block(text: &str) -> Option<&str> {
  // Match ```json ... ``` or ``` ... ```
  let text = text.trim();

  if !text.starts_with("```") {
    return None;
  }

  // Find the end of the first line (after ```)
  let first_newline = text.find('\n')?;
  let after_fence = &text[first_newline + 1..];

  // Find closing fence
  let end = after_fence.rfind("```")?;
  Some(after_fence[..end].trim())
}

/// Structured extraction result for memory extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
  pub memories: Vec<ExtractedMemory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMemory {
  pub content: String,
  pub summary: Option<String>,
  pub memory_type: String,
  pub sector: Option<String>,
  pub tags: Vec<String>,
  pub confidence: f32,
}

/// Signal classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalClassification {
  pub category: SignalCategory,
  pub is_extractable: bool,
  pub summary: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalCategory {
  /// User correcting previous behavior/output
  Correction,
  /// User expressing a preference
  Preference,
  /// User providing context/information
  Context,
  /// User requesting a task
  Task,
  /// User asking a question
  Question,
  /// User giving feedback
  Feedback,
  /// Unclassified
  Other,
}

impl SignalCategory {
  /// Whether this signal type should trigger immediate extraction
  pub fn is_high_priority(&self) -> bool {
    matches!(self, SignalCategory::Correction | SignalCategory::Preference)
  }
}

/// Superseding detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupersedingResult {
  pub supersedes: bool,
  pub superseded_memory_id: Option<String>,
  pub reason: Option<String>,
  pub confidence: f32,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_extract_code_block_json() {
    let text = r#"```json
{"key": "value"}
```"#;
    assert_eq!(extract_code_block(text), Some(r#"{"key": "value"}"#));
  }

  #[test]
  fn test_extract_code_block_plain() {
    let text = r#"```
{"key": "value"}
```"#;
    assert_eq!(extract_code_block(text), Some(r#"{"key": "value"}"#));
  }

  #[test]
  fn test_extract_code_block_none() {
    let text = r#"{"key": "value"}"#;
    assert_eq!(extract_code_block(text), None);
  }

  #[test]
  fn test_parse_json_raw() {
    let text = r#"{"key": "value"}"#;
    let result: serde_json::Value = parse_json(text).unwrap();
    assert_eq!(result["key"], "value");
  }

  #[test]
  fn test_parse_json_code_block() {
    let text = r#"```json
{"key": "value"}
```"#;
    let result: serde_json::Value = parse_json(text).unwrap();
    assert_eq!(result["key"], "value");
  }

  #[test]
  fn test_signal_category_high_priority() {
    assert!(SignalCategory::Correction.is_high_priority());
    assert!(SignalCategory::Preference.is_high_priority());
    assert!(!SignalCategory::Task.is_high_priority());
    assert!(!SignalCategory::Question.is_high_priority());
  }

  // Integration test - requires `claude` CLI to be available
  #[tokio::test]
  #[ignore = "requires claude CLI"]
  async fn test_infer_real() {
    let request = InferenceRequest::new("Say 'hello' and nothing else")
      .with_model(Model::Haiku)
      .with_timeout(30);

    let response = infer(request).await.unwrap();
    assert!(response.text.to_lowercase().contains("hello"));
    assert!(response.output_tokens > 0);
  }
}
