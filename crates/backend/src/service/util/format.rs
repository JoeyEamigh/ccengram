//! Response formatting for MCP tools.
//!
//! Formats tool responses as human-readable text with code blocks and XML-style
//! metadata tags, optimized for agent consumption.

use crate::service::explore::{
  CallInfo, CodeContext, ContextResponse, DocChunkEntry, DocContext, ExploreHints, ExploreResponse, ExploreResult,
  MemoryContext, RelatedMemoryInfo, SiblingInfo, TimelineEntry,
};

/// Format an explore response as human-readable text.
pub fn format_explore_response(response: &ExploreResponse) -> String {
  let mut output = String::new();

  // Summary line
  let total: usize = response.counts.values().sum();
  output.push_str(&format!("Found {} results", total));
  if !response.counts.is_empty() {
    let counts: Vec<String> = response.counts.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
    output.push_str(&format!(" ({})", counts.join(", ")));
  }
  output.push_str("\n\n");

  // Results
  for (i, result) in response.results.iter().enumerate() {
    output.push_str(&format_explore_result(result, i + 1));
    output.push('\n');
  }

  // Suggestions
  if !response.suggestions.is_empty() {
    output.push_str("---\n");
    output.push_str(&format!("Suggested queries: {}\n", response.suggestions.join(", ")));
  }

  output
}

/// Format a single explore result.
fn format_explore_result(result: &ExploreResult, index: usize) -> String {
  let mut output = String::new();

  // Result header with metadata
  output.push_str(&format!(
    "<result index=\"{}\" type=\"{}\" id=\"{}\"",
    index, result.result_type, result.id
  ));

  if let Some(ref file) = result.file {
    output.push_str(&format!(" file=\"{}\"", file));
  }
  if let Some((start, end)) = result.lines {
    output.push_str(&format!(" lines=\"{}-{}\"", start, end));
  }
  if let Some(ref lang) = result.language {
    output.push_str(&format!(" language=\"{}\"", lang));
  }
  output.push_str(&format!(" score=\"{:.2}\"", result.score));
  output.push_str(">\n");

  // Definition kind and parent context (helps LLM evaluate relevance quickly)
  if let Some(ref kind) = result.definition_kind {
    let mut def_line = format!("Definition: {}", kind);
    if let Some(ref parent) = result.parent {
      def_line.push_str(&format!(" (in {})", parent));
    }
    output.push_str(&def_line);
    output.push('\n');
  }

  // Symbols
  if !result.symbols.is_empty() {
    output.push_str(&format!("Symbols: {}\n", result.symbols.join(", ")));
  }

  // Signature (most important for relevance - shows types, params, return)
  if let Some(ref sig) = result.signature {
    let clean_sig = sig.lines().map(|l| l.trim()).collect::<Vec<_>>().join(" ");
    output.push_str(&format!("Signature: {}\n", truncate(&clean_sig, 150)));
  }

  // Docstring (natural language description)
  if let Some(ref doc) = result.docstring {
    output.push_str(&format!("Doc: {}\n", truncate(doc, 120)));
  }

  // Imports and calls (helps understand dependencies and behavior)
  if !result.imports.is_empty() {
    output.push_str(&format!("Uses: {}\n", result.imports.join(", ")));
  }
  if !result.calls.is_empty() {
    output.push_str(&format!("Calls: {}\n", result.calls.join(", ")));
  }

  // Hints (navigation counts)
  let hints = format_hints(&result.hints);
  if !hints.is_empty() {
    output.push_str(&format!("Hints: {}\n", hints.join(" | ")));
  }

  // Preview (now contains semantic content, not just truncated code)
  output.push('\n');
  output.push_str(&format_code_block(&result.preview, result.language.as_deref()));

  // Expanded context if present
  if let Some(ref ctx) = result.context {
    output.push_str("\n<expanded>\n");

    // Full content
    output.push_str("Content:\n");
    output.push_str(&format_code_block(&ctx.content, result.language.as_deref()));

    // Callers
    if !ctx.callers.is_empty() {
      output.push_str(&format!("\nCallers ({}):\n", ctx.callers.len()));
      for caller in &ctx.callers {
        output.push_str(&format_call_info(caller));
      }
    }

    // Callees
    if !ctx.callees.is_empty() {
      output.push_str(&format!("\nCallees ({}):\n", ctx.callees.len()));
      for callee in &ctx.callees {
        output.push_str(&format_call_info(callee));
      }
    }

    // Siblings
    if !ctx.siblings.is_empty() {
      output.push_str(&format!("\nSiblings ({}):\n", ctx.siblings.len()));
      for sibling in &ctx.siblings {
        output.push_str(&format!(
          "  - {} ({}) at line {}\n",
          sibling.symbol, sibling.kind, sibling.line
        ));
      }
    }

    // Memories
    if !ctx.memories.is_empty() {
      output.push_str(&format!("\nRelated memories ({}):\n", ctx.memories.len()));
      for mem in &ctx.memories {
        output.push_str(&format_related_memory(mem));
      }
    }

    output.push_str("</expanded>\n");
  }

  output.push_str("</result>\n");
  output
}

/// Format hints as a list of strings.
fn format_hints(hints: &ExploreHints) -> Vec<String> {
  let mut result = Vec::new();

  if let Some(c) = hints.callers {
    result.push(format!("{} callers", c));
  }
  if let Some(c) = hints.callees {
    result.push(format!("{} callees", c));
  }
  if let Some(c) = hints.siblings {
    result.push(format!("{} siblings", c));
  }
  if let Some(c) = hints.related_memories {
    result.push(format!("{} memories", c));
  }
  if let Some(c) = hints.total_chunks {
    result.push(format!("1/{} chunks", c));
  }

  result
}

/// Format a context response as human-readable text.
pub fn format_context_response(response: &ContextResponse) -> String {
  match response {
    ContextResponse::Code { items } => {
      let mut output = format!("Code context ({} items)\n\n", items.len());
      for item in items {
        output.push_str(&format_code_context(item));
        output.push_str("\n---\n\n");
      }
      output
    }
    ContextResponse::Memory { items } => {
      let mut output = format!("Memory context ({} items)\n\n", items.len());
      for item in items {
        output.push_str(&format_memory_context(item));
        output.push_str("\n---\n\n");
      }
      output
    }
    ContextResponse::Doc { items } => {
      let mut output = format!("Document context ({} items)\n\n", items.len());
      for item in items {
        output.push_str(&format_doc_context(item));
        output.push_str("\n---\n\n");
      }
      output
    }
    ContextResponse::Mixed { code, memories, docs } => {
      let mut output = String::from("Mixed context\n\n");

      if !code.is_empty() {
        output.push_str(&format!("## Code ({} items)\n\n", code.len()));
        for item in code {
          output.push_str(&format_code_context(item));
          output.push('\n');
        }
      }

      if !memories.is_empty() {
        output.push_str(&format!("## Memories ({} items)\n\n", memories.len()));
        for item in memories {
          output.push_str(&format_memory_context(item));
          output.push('\n');
        }
      }

      if !docs.is_empty() {
        output.push_str(&format!("## Documents ({} items)\n\n", docs.len()));
        for item in docs {
          output.push_str(&format_doc_context(item));
          output.push('\n');
        }
      }

      output
    }
  }
}

/// Format a code context item.
fn format_code_context(ctx: &CodeContext) -> String {
  let mut output = String::new();

  // Header
  output.push_str(&format!(
    "<code id=\"{}\" file=\"{}\" lines=\"{}-{}\" language=\"{}\">\n",
    ctx.id, ctx.file, ctx.lines.0, ctx.lines.1, ctx.language
  ));

  // Symbols and imports
  if !ctx.symbols.is_empty() {
    output.push_str(&format!("Symbols: {}\n", ctx.symbols.join(", ")));
  }
  if !ctx.imports.is_empty() {
    output.push_str(&format!("Imports: {}\n", ctx.imports.join(", ")));
  }
  if let Some(ref sig) = ctx.signature {
    output.push_str(&format!("Signature: {}\n", sig));
  }

  // Content
  output.push('\n');
  output.push_str(&format_code_block(&ctx.content, Some(&ctx.language)));

  // Callers
  if !ctx.callers.is_empty() {
    output.push_str(&format!("\nCallers ({}):\n", ctx.callers.len()));
    for caller in &ctx.callers {
      output.push_str(&format_call_info(caller));
    }
  }

  // Callees
  if !ctx.callees.is_empty() {
    output.push_str(&format!("\nCallees ({}):\n", ctx.callees.len()));
    for callee in &ctx.callees {
      output.push_str(&format_call_info(callee));
    }
  }

  // Siblings
  if !ctx.siblings.is_empty() {
    output.push_str(&format!("\nSiblings ({}):\n", ctx.siblings.len()));
    for sibling in &ctx.siblings {
      output.push_str(&format_sibling_info(sibling));
    }
  }

  // Memories
  if !ctx.memories.is_empty() {
    output.push_str(&format!("\nRelated memories ({}):\n", ctx.memories.len()));
    for mem in &ctx.memories {
      output.push_str(&format_related_memory(mem));
    }
  }

  output.push_str("</code>\n");
  output
}

/// Format a memory context item.
fn format_memory_context(ctx: &MemoryContext) -> String {
  let mut output = String::new();

  // Header
  output.push_str(&format!(
    "<memory id=\"{}\" type=\"{}\" sector=\"{}\" salience=\"{:.2}\">\n",
    ctx.id, ctx.memory_type, ctx.sector, ctx.salience
  ));
  output.push_str(&format!("Created: {}\n\n", ctx.created_at));

  // Content
  output.push_str(&ctx.content);
  output.push('\n');

  // Timeline
  if !ctx.timeline.before.is_empty() || !ctx.timeline.after.is_empty() {
    output.push_str("\nTimeline:\n");

    for entry in ctx.timeline.before.iter().rev() {
      output.push_str(&format_timeline_entry(entry, "before"));
    }

    output.push_str("  >>> [THIS MEMORY] <<<\n");

    for entry in &ctx.timeline.after {
      output.push_str(&format_timeline_entry(entry, "after"));
    }
  }

  // Related
  if !ctx.related.is_empty() {
    output.push_str(&format!("\nRelated memories ({}):\n", ctx.related.len()));
    for rel in &ctx.related {
      output.push_str(&format_related_memory(rel));
    }
  }

  output.push_str("</memory>\n");
  output
}

/// Format a document context item.
fn format_doc_context(ctx: &DocContext) -> String {
  let mut output = String::new();

  // Header
  output.push_str(&format!(
    "<doc id=\"{}\" source=\"{}\" chunk=\"{}/{}\">\n",
    ctx.id,
    ctx.source,
    ctx.chunk_index + 1,
    ctx.total_chunks
  ));
  output.push_str(&format!("Title: {}\n\n", ctx.title));

  // Before chunks
  if !ctx.before.is_empty() {
    output.push_str("--- Previous chunks ---\n");
    for chunk in &ctx.before {
      output.push_str(&format_doc_chunk_entry(chunk));
    }
    output.push('\n');
  }

  // Main content
  output.push_str("--- Current chunk ---\n");
  output.push_str(&ctx.content);
  output.push('\n');

  // After chunks
  if !ctx.after.is_empty() {
    output.push_str("\n--- Following chunks ---\n");
    for chunk in &ctx.after {
      output.push_str(&format_doc_chunk_entry(chunk));
    }
  }

  output.push_str("</doc>\n");
  output
}

/// Format a code block with language tag.
fn format_code_block(content: &str, language: Option<&str>) -> String {
  let lang = language.unwrap_or("");
  format!("```{}\n{}\n```\n", lang, content.trim())
}

/// Format a caller/callee info.
fn format_call_info(info: &CallInfo) -> String {
  let mut output = format!(
    "  - [{}] {}:{}-{}\n",
    &info.id[..8.min(info.id.len())],
    info.file,
    info.lines.0,
    info.lines.1
  );
  if let Some(ref symbols) = info.symbols
    && !symbols.is_empty()
  {
    output.push_str(&format!("    Symbols: {}\n", symbols.join(", ")));
  }
  output.push_str(&format!("    {}\n", truncate(&info.preview, 80)));
  output
}

/// Format a sibling info.
fn format_sibling_info(info: &SiblingInfo) -> String {
  format!("  - {} ({}) at line {}\n", info.symbol, info.kind, info.line)
}

/// Format a related memory info.
fn format_related_memory(info: &RelatedMemoryInfo) -> String {
  format!(
    "  - [{}] ({}/{}) {}\n",
    &info.id[..8.min(info.id.len())],
    info.memory_type,
    info.sector,
    truncate(&info.content, 100)
  )
}

/// Format a timeline entry.
fn format_timeline_entry(entry: &TimelineEntry, direction: &str) -> String {
  let arrow = if direction == "before" { "^" } else { "v" };
  format!(
    "  {} [{}] ({}) {}\n",
    arrow,
    &entry.id[..8.min(entry.id.len())],
    entry.memory_type,
    truncate(&entry.content, 60)
  )
}

/// Format a document chunk entry.
fn format_doc_chunk_entry(entry: &DocChunkEntry) -> String {
  format!("[Chunk {}]\n{}\n", entry.chunk_index + 1, truncate(&entry.content, 200))
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
  let s = s.trim().replace('\n', " ");
  if s.len() <= max_len {
    s
  } else {
    format!("{}...", &s[..max_len.saturating_sub(3)])
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashMap;

  use super::*;
  use crate::service::explore::ExpandedContext;

  #[test]
  fn test_format_code_block() {
    let result = format_code_block("fn main() {}", Some("rust"));
    assert!(result.contains("```rust"));
    assert!(result.contains("fn main()"));
    assert!(result.contains("```\n"));
  }

  #[test]
  fn test_format_explore_result() {
    let result = ExploreResult {
      id: "abc12345".to_string(),
      result_type: "code".to_string(),
      file: Some("src/main.rs".to_string()),
      lines: Some((10, 20)),
      preview: "fn test() {}".to_string(),
      symbols: vec!["test".to_string()],
      language: Some("rust".to_string()),
      hints: ExploreHints {
        callers: Some(5),
        callees: Some(3),
        siblings: Some(2),
        related_memories: Some(1),
        related_code: None,
        timeline_depth: None,
        total_chunks: None,
      },
      context: None,
      score: 0.95,
      definition_kind: Some("function".to_string()),
      signature: Some("fn test() -> Result<()>".to_string()),
      docstring: Some("A test function".to_string()),
      parent: Some("TestModule".to_string()),
      imports: vec!["std::io".to_string()],
      calls: vec!["helper".to_string()],
    };

    let output = format_explore_result(&result, 1);
    assert!(output.contains("type=\"code\""));
    assert!(output.contains("file=\"src/main.rs\""));
    assert!(output.contains("lines=\"10-20\""));
    assert!(output.contains("5 callers"));
    assert!(output.contains("```rust"));
    // New semantic fields
    assert!(output.contains("Definition: function (in TestModule)"));
    assert!(output.contains("Signature: fn test() -> Result<()>"));
    assert!(output.contains("Doc: A test function"));
    assert!(output.contains("Uses: std::io"));
    assert!(output.contains("Calls: helper"));
  }

  #[test]
  fn test_format_explore_response() {
    let response = ExploreResponse {
      results: vec![],
      counts: {
        let mut m = HashMap::new();
        m.insert("code".to_string(), 5);
        m.insert("memory".to_string(), 2);
        m
      },
      suggestions: vec!["auth".to_string(), "login".to_string()],
    };

    let output = format_explore_response(&response);
    assert!(output.contains("Found 7 results"));
    assert!(output.contains("Suggested queries: auth, login"));
  }

  #[test]
  fn test_truncate() {
    assert_eq!(truncate("short", 10), "short");
    assert_eq!(truncate("this is longer", 10), "this is...");
    assert_eq!(truncate("line1\nline2", 20), "line1 line2");
  }

  #[test]
  fn test_format_hints() {
    let hints = ExploreHints {
      callers: Some(5),
      callees: Some(3),
      siblings: None,
      related_memories: Some(2),
      related_code: None,
      timeline_depth: None,
      total_chunks: None,
    };

    let formatted = format_hints(&hints);
    assert!(formatted.contains(&"5 callers".to_string()));
    assert!(formatted.contains(&"3 callees".to_string()));
    assert!(formatted.contains(&"2 memories".to_string()));
  }

  #[test]
  fn test_format_expanded_context() {
    let result = ExploreResult {
      id: "abc12345".to_string(),
      result_type: "code".to_string(),
      file: Some("src/main.rs".to_string()),
      lines: Some((10, 20)),
      preview: "fn test() {}".to_string(),
      symbols: vec!["test".to_string()],
      language: Some("rust".to_string()),
      hints: ExploreHints::default(),
      context: Some(ExpandedContext {
        content: "fn test() { println!(\"hello\"); }".to_string(),
        callers: vec![CallInfo {
          id: "caller123".to_string(),
          file: "caller.rs".to_string(),
          lines: (5, 15),
          preview: "fn caller()".to_string(),
          symbols: Some(vec!["caller".to_string()]),
        }],
        callees: vec![],
        siblings: vec![],
        memories: vec![],
      }),
      score: 0.95,
      definition_kind: None,
      signature: None,
      docstring: None,
      parent: None,
      imports: vec![],
      calls: vec![],
    };

    let output = format_explore_result(&result, 1);
    assert!(output.contains("<expanded>"));
    assert!(output.contains("Callers (1):"));
    assert!(output.contains("caller.rs"));
    assert!(output.contains("</expanded>"));
  }
}
