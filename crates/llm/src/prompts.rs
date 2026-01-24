//! LLM prompts for extraction, classification, and analysis
//!
//! All prompts are designed to return JSON for structured parsing.

use tracing::trace;

/// Prompt for classifying user input signals
pub const SIGNAL_CLASSIFICATION_PROMPT: &str = r#"You are a signal classifier. Analyze the user's message and classify it.

Categories:
- correction: User correcting previous behavior or output
- preference: User expressing a preference about how things should be done
- context: User providing information or context
- task: User requesting a task to be performed
- question: User asking a question
- feedback: User giving feedback about results
- other: None of the above

Respond with JSON only:
{
  "category": "<category>",
  "is_extractable": <boolean>,
  "summary": "<brief summary if extractable, null otherwise>"
}

User message to classify:
"#;

/// Prompt for extracting memories from conversation context
pub const MEMORY_EXTRACTION_PROMPT: &str = r#"You are a memory extraction system. Extract valuable long-term memories from this conversation segment.

Extract memories that would be useful to recall in future sessions:
- User preferences (coding style, tools, workflows)
- Project-specific knowledge (architecture decisions, gotchas, patterns)
- Important decisions and their rationale
- Learned patterns or best practices
- Task completions and outcomes

Memory types:
- preference: User's stated preference
- codebase: Knowledge about the codebase
- decision: Design or implementation decision
- gotcha: Something to watch out for
- pattern: Recurring pattern or best practice
- turn_summary: Summary of what was accomplished
- task_completion: Task that was completed

Respond with JSON only:
{
  "memories": [
    {
      "content": "<full memory content>",
      "summary": "<brief 1-line summary>",
      "memory_type": "<type>",
      "sector": "<user|assistant|system or null>",
      "tags": ["<relevant>", "<tags>"],
      "confidence": <0.0-1.0>
    }
  ]
}

Only extract memories with confidence >= 0.6. Return empty array if nothing worth extracting.

Conversation segment:
"#;

/// Prompt for detecting if new memory supersedes existing ones
pub const SUPERSEDING_DETECTION_PROMPT: &str = r#"You are comparing memories to detect supersession.

A new memory SUPERSEDES an existing memory when:
- It updates or replaces the same information
- It contradicts and provides a newer truth
- It refines or clarifies the same concept with more detail
- It explicitly overrides a previous decision

It does NOT supersede when:
- It's about a different topic
- It adds new information without contradicting
- It's a related but distinct piece of knowledge

Respond with JSON only:
{
  "supersedes": <boolean>,
  "superseded_memory_id": "<id if supersedes, null otherwise>",
  "reason": "<brief explanation>",
  "confidence": <0.0-1.0>
}

New memory:
{new_memory}

Candidate existing memories to check:
{existing_memories}
"#;

/// System prompt for extraction context
pub const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are CCEngram's memory extraction system. Your role is to identify and extract valuable information from Claude Code conversation segments that would be useful to recall in future sessions.

Guidelines:
1. Focus on information with lasting value - not ephemeral task details
2. Prefer explicit statements over inferred knowledge
3. Include enough context to be useful standalone
4. Use clear, concise language
5. Assign appropriate confidence scores
6. Tag memories with relevant keywords

Always respond with valid JSON matching the requested schema."#;

/// Prompt for analyzing code changes and extracting patterns
pub const CODE_ANALYSIS_PROMPT: &str = r#"You are analyzing code changes to extract patterns and learnings.

Look for:
- Architectural patterns being established
- Coding conventions being followed
- Bug patterns that were fixed
- Performance considerations
- Testing patterns

Respond with JSON only:
{
  "patterns": [
    {
      "content": "<pattern description>",
      "pattern_type": "<architectural|convention|bug_fix|performance|testing>",
      "files_involved": ["<file paths>"],
      "confidence": <0.0-1.0>
    }
  ]
}

Code changes:
"#;

/// Build a signal classification prompt for a specific user message
pub fn build_signal_classification_prompt(user_message: &str) -> String {
  let prompt = format!("{}{}", SIGNAL_CLASSIFICATION_PROMPT, user_message);
  trace!(
    template_len = SIGNAL_CLASSIFICATION_PROMPT.len(),
    message_len = user_message.len(),
    total_len = prompt.len(),
    "Built signal classification prompt"
  );
  prompt
}

/// Build a memory extraction prompt for a conversation segment
pub fn build_extraction_prompt(context: &ExtractionContext) -> String {
  let mut prompt = String::new();
  prompt.push_str(MEMORY_EXTRACTION_PROMPT);

  if let Some(user_prompt) = &context.user_prompt {
    prompt.push_str("\nUser prompt: ");
    prompt.push_str(user_prompt);
  }

  if !context.files_read.is_empty() {
    prompt.push_str("\nFiles read: ");
    prompt.push_str(&context.files_read.join(", "));
  }

  if !context.files_modified.is_empty() {
    prompt.push_str("\nFiles modified: ");
    prompt.push_str(&context.files_modified.join(", "));
  }

  if !context.commands_run.is_empty() {
    prompt.push_str("\nCommands run: ");
    for (cmd, exit_code) in &context.commands_run {
      prompt.push_str(&format!("\n  - {} (exit: {})", cmd, exit_code));
    }
  }

  if !context.errors_encountered.is_empty() {
    prompt.push_str("\nErrors encountered: ");
    for err in &context.errors_encountered {
      prompt.push_str(&format!("\n  - {}", err));
    }
  }

  if !context.completed_tasks.is_empty() {
    prompt.push_str("\nCompleted tasks: ");
    for task in &context.completed_tasks {
      prompt.push_str(&format!("\n  - {}", task));
    }
  }

  if let Some(assistant_msg) = &context.last_assistant_message {
    prompt.push_str("\nLast assistant response (truncated): ");
    let truncated: String = assistant_msg.chars().take(1000).collect();
    prompt.push_str(&truncated);
  }

  trace!(
    template_len = MEMORY_EXTRACTION_PROMPT.len(),
    total_len = prompt.len(),
    has_user_prompt = context.user_prompt.is_some(),
    files_read_count = context.files_read.len(),
    files_modified_count = context.files_modified.len(),
    commands_run_count = context.commands_run.len(),
    errors_count = context.errors_encountered.len(),
    tasks_count = context.completed_tasks.len(),
    has_assistant_message = context.last_assistant_message.is_some(),
    "Built memory extraction prompt"
  );

  prompt
}

/// Build a superseding detection prompt
pub fn build_superseding_prompt(new_memory: &str, existing_memories: &[(String, String)]) -> String {
  let mut existing_json = String::from("[\n");
  for (i, (id, content)) in existing_memories.iter().enumerate() {
    if i > 0 {
      existing_json.push_str(",\n");
    }
    existing_json.push_str(&format!(
      r#"  {{"id": "{}", "content": "{}"}}"#,
      id,
      content.replace('"', "\\\"").replace('\n', "\\n")
    ));
  }
  existing_json.push_str("\n]");

  let prompt = SUPERSEDING_DETECTION_PROMPT
    .replace("{new_memory}", new_memory)
    .replace("{existing_memories}", &existing_json);

  trace!(
    template_len = SUPERSEDING_DETECTION_PROMPT.len(),
    new_memory_len = new_memory.len(),
    existing_memories_count = existing_memories.len(),
    existing_json_len = existing_json.len(),
    total_len = prompt.len(),
    "Built superseding detection prompt"
  );

  prompt
}

/// Context for memory extraction
#[derive(Debug, Default, Clone)]
pub struct ExtractionContext {
  /// The user's prompt that started this segment
  pub user_prompt: Option<String>,
  /// Files that were read during this segment
  pub files_read: Vec<String>,
  /// Files that were modified during this segment
  pub files_modified: Vec<String>,
  /// Commands that were run with exit codes
  pub commands_run: Vec<(String, i32)>,
  /// Errors that were encountered
  pub errors_encountered: Vec<String>,
  /// Searches that were performed
  pub searches_performed: Vec<String>,
  /// Tasks that were completed
  pub completed_tasks: Vec<String>,
  /// The last assistant message
  pub last_assistant_message: Option<String>,
  /// Total tool calls in this segment
  pub tool_call_count: usize,
}

impl ExtractionContext {
  pub fn new() -> Self {
    Self::default()
  }

  /// Check if this segment has enough content to warrant extraction
  pub fn has_meaningful_content(&self) -> bool {
    // At least 3 tool calls OR explicit signals
    self.tool_call_count >= 3
      || !self.files_modified.is_empty()
      || !self.completed_tasks.is_empty()
      || !self.errors_encountered.is_empty()
  }

  /// Check if this segment has high-priority signals requiring immediate extraction
  pub fn has_high_priority_signals(&self) -> bool {
    // Corrections or preferences trigger immediate extraction
    // This will be determined by signal classification
    false // Placeholder - actual check happens via LLM classification
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_build_signal_classification_prompt() {
    let prompt = build_signal_classification_prompt("I prefer tabs over spaces");
    assert!(prompt.contains("I prefer tabs over spaces"));
    assert!(prompt.contains("correction"));
    assert!(prompt.contains("preference"));
  }

  #[test]
  fn test_build_extraction_prompt_minimal() {
    let ctx = ExtractionContext::new();
    let prompt = build_extraction_prompt(&ctx);
    assert!(prompt.contains("memory_type"));
  }

  #[test]
  fn test_build_extraction_prompt_full() {
    let ctx = ExtractionContext {
      user_prompt: Some("Fix the bug".into()),
      files_read: vec!["src/main.rs".into()],
      files_modified: vec!["src/lib.rs".into()],
      commands_run: vec![("cargo test".into(), 0)],
      errors_encountered: vec!["type error on line 5".into()],
      searches_performed: vec!["error handling".into()],
      completed_tasks: vec!["Fix type error".into()],
      last_assistant_message: Some("I fixed the bug".into()),
      tool_call_count: 10,
    };
    let prompt = build_extraction_prompt(&ctx);
    assert!(prompt.contains("Fix the bug"));
    assert!(prompt.contains("src/main.rs"));
    assert!(prompt.contains("cargo test"));
  }

  #[test]
  fn test_build_superseding_prompt() {
    let existing = vec![
      ("mem1".into(), "Use tabs for indentation".into()),
      ("mem2".into(), "Project uses React".into()),
    ];
    let prompt = build_superseding_prompt("Use spaces for indentation", &existing);
    assert!(prompt.contains("Use spaces for indentation"));
    assert!(prompt.contains("mem1"));
    assert!(prompt.contains("Use tabs for indentation"));
  }

  #[test]
  fn test_extraction_context_meaningful() {
    let mut ctx = ExtractionContext::new();
    assert!(!ctx.has_meaningful_content());

    ctx.tool_call_count = 3;
    assert!(ctx.has_meaningful_content());

    ctx.tool_call_count = 0;
    ctx.files_modified.push("test.rs".into());
    assert!(ctx.has_meaningful_content());
  }
}
