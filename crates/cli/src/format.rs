//! MCP tool response formatting for LLM consumption.
//!
//! Formats tool responses as human-readable text with code blocks and XML-style
//! metadata tags. Designed to be token-efficient while providing structured context.

use ccengram::ipc::{
  code::{
    CodeCalleesResponse, CodeCallersResponse, CodeContextFullResponse, CodeContextResponse, CodeIndexResult, CodeItem,
    CodeMemoriesResponse, CodeRelatedResponse, CodeSearchResult, CodeStatsResult,
  },
  docs::{DocContextResult, DocSearchItem, DocsIngestFullResult},
  memory::{
    MemoryAddResult, MemoryDeleteResult, MemoryFullDetail, MemoryItem, MemoryRelatedResult, MemorySearchResult,
    MemorySupersedeResult, MemoryTimelineResult, MemoryUpdateResult,
  },
  project::{ProjectCleanAllResult, ProjectCleanResult, ProjectInfoResult, ProjectStatsResult},
  relationship::{DeletedResult, RelatedMemoryItem, RelationshipListItem, RelationshipResult},
  search::{ContextItem, ExploreResult},
  system::HealthCheckResult,
  watch::{WatchStartResult, WatchStatusResult, WatchStopResult},
};

// ============================================================================
// Public formatting API
// ============================================================================

/// Format any tool result by tool name
pub fn format_tool_result(tool_name: &str, result: &serde_json::Value) -> Option<String> {
  match tool_name {
    // Explore tools
    "explore" => serde_json::from_value(result.clone()).ok().map(|r| format_explore(&r)),
    "context" => serde_json::from_value(result.clone())
      .ok()
      .map(|r: Vec<ContextItem>| format_context(&r)),

    // Code tools
    "code_search" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_search(&r)),
    "code_context" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_context(&r)),
    "code_index" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_index(&r)),
    "code_list" => serde_json::from_value(result.clone())
      .ok()
      .map(|r: Vec<CodeItem>| format_code_list(&r)),
    "code_stats" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_stats(&r)),
    "code_memories" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_memories(&r)),
    "code_callers" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_callers(&r)),
    "code_callees" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_callees(&r)),
    "code_related" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_related(&r)),
    "code_context_full" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_code_context_full(&r)),

    // Memory tools
    "memory_search" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_search(&r)),
    "memory_get" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_get(&r)),
    "memory_add" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_add(&r)),
    "memory_list" => serde_json::from_value(result.clone())
      .ok()
      .map(|r: Vec<MemoryItem>| format_memory_list(&r)),
    "memory_reinforce" | "memory_deemphasize" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_update(&r)),
    "memory_delete" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_delete(&r)),
    "memory_supersede" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_supersede(&r)),
    "memory_timeline" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_timeline(&r)),
    "memory_related" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_memory_related(&r)),

    // Doc tools
    "docs_search" => serde_json::from_value(result.clone())
      .ok()
      .map(|r: Vec<DocSearchItem>| format_docs_search(&r)),
    "doc_context" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_doc_context(&r)),
    "docs_ingest" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_docs_ingest(&r)),

    // Relationship tools
    "relationship_add" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_relationship_add(&r)),
    "relationship_list" => serde_json::from_value(result.clone())
      .ok()
      .map(|r: Vec<RelationshipListItem>| format_relationship_list(&r)),
    "relationship_delete" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_relationship_delete(&r)),
    "relationship_related" => serde_json::from_value(result.clone())
      .ok()
      .map(|r: Vec<RelatedMemoryItem>| format_relationship_related(&r)),

    // Watch tools
    "watch_start" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_watch_start(&r)),
    "watch_stop" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_watch_stop(&r)),
    "watch_status" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_watch_status(&r)),

    // Project tools
    "project_info" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_project_info(&r)),
    "project_clean" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_project_clean(&r)),
    "project_clean_all" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_project_clean_all(&r)),

    // System tools
    "project_stats" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_project_stats(&r)),
    "health_check" => serde_json::from_value(result.clone())
      .ok()
      .map(|r| format_health_check(&r)),

    // Unknown - return None to fall back to JSON
    _ => None,
  }
}

// ============================================================================
// Explore formatters
// ============================================================================

fn format_explore(result: &ExploreResult) -> String {
  let mut out = String::new();

  // Header
  out.push_str(&format!("# Explore: {}\n\n", result.query));
  out.push_str(&format!("Found {} results\n\n", result.results.len()));

  // Results
  for (i, item) in result.results.iter().enumerate() {
    out.push_str(&format!(
      "<result index=\"{}\" type=\"{}\" id=\"{}\"",
      i + 1,
      item.result_type,
      &item.id[..8.min(item.id.len())]
    ));

    if let Some(ref file) = item.file_path {
      out.push_str(&format!(" file=\"{}\"", file));
    }
    if let Some(line) = item.line {
      out.push_str(&format!(" line=\"{}\"", line));
    }
    out.push_str(&format!(" score=\"{:.2}\"", item.similarity));
    out.push_str(">\n");

    // Symbols
    if !item.symbols.is_empty() {
      out.push_str(&format!("Symbols: {}\n", item.symbols.join(", ")));
    }

    // Hints
    if let Some(ref hints) = item.hints {
      let mut hint_parts = Vec::new();
      if hints.caller_count > 0 {
        hint_parts.push(format!("{} callers", hints.caller_count));
      }
      if hints.callee_count > 0 {
        hint_parts.push(format!("{} callees", hints.callee_count));
      }
      if hints.related_memory_count > 0 {
        hint_parts.push(format!("{} memories", hints.related_memory_count));
      }
      if !hint_parts.is_empty() {
        out.push_str(&format!("Hints: {}\n", hint_parts.join(" | ")));
      }
    }

    // Preview
    out.push('\n');
    out.push_str(&format_preview(&item.preview, None));

    // Expanded context
    if let Some(ref ctx) = item.context {
      out.push_str("\n<expanded>\n");

      if !ctx.callers.is_empty() {
        out.push_str(&format!("Callers ({}):\n", ctx.callers.len()));
        for caller in &ctx.callers {
          out.push_str(&format!(
            "  - [{}] {}:{}-{}",
            &caller.id[..8.min(caller.id.len())],
            caller.file,
            caller.start_line,
            caller.end_line
          ));
          if let Some(ref sig) = caller.signature {
            out.push_str(&format!(" `{}`", sig.lines().next().unwrap_or(sig).trim()));
          }
          out.push('\n');
        }
      }

      if !ctx.callees.is_empty() {
        out.push_str(&format!("Callees ({}):\n", ctx.callees.len()));
        for callee in &ctx.callees {
          out.push_str(&format!(
            "  - [{}] {}:{}-{}",
            &callee.id[..8.min(callee.id.len())],
            callee.file,
            callee.start_line,
            callee.end_line
          ));
          if let Some(ref sig) = callee.signature {
            out.push_str(&format!(" `{}`", sig.lines().next().unwrap_or(sig).trim()));
          }
          out.push('\n');
        }
      }

      if !ctx.siblings.is_empty() {
        out.push_str(&format!("Siblings ({}):\n", ctx.siblings.len()));
        for sib in &ctx.siblings {
          out.push_str(&format!("  - {} ({}) line {}\n", sib.symbol, sib.kind, sib.line));
        }
      }

      out.push_str("</expanded>\n");
    }

    out.push_str("</result>\n\n");
  }

  out
}

fn format_context(items: &[ContextItem]) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Context ({} items)\n\n", items.len()));

  for item in items {
    out.push_str(&format!(
      "<{} id=\"{}\">\n",
      item.item_type,
      &item.id[..8.min(item.id.len())]
    ));

    // Content
    out.push_str(&item.content);
    out.push_str("\n\n");

    // Callers
    if let Some(ref callers) = item.callers
      && !callers.is_empty()
    {
      out.push_str(&format!("Callers ({}):\n", callers.len()));
      for c in callers {
        out.push_str(&format!(
          "  - [{}] {}:{}-{}",
          &c.id[..8.min(c.id.len())],
          c.file_path,
          c.start_line,
          c.end_line
        ));
        if let Some(ref sig) = c.signature {
          out.push_str(&format!(" `{}`", sig.lines().next().unwrap_or(sig).trim()));
        }
        out.push('\n');
      }
    }

    // Callees
    if let Some(ref callees) = item.callees
      && !callees.is_empty()
    {
      out.push_str(&format!("Callees ({}):\n", callees.len()));
      for c in callees {
        out.push_str(&format!(
          "  - [{}] {}:{}-{}",
          &c.id[..8.min(c.id.len())],
          c.file_path,
          c.start_line,
          c.end_line
        ));
        if let Some(ref sig) = c.signature {
          out.push_str(&format!(" `{}`", sig.lines().next().unwrap_or(sig).trim()));
        }
        out.push('\n');
      }
    }

    // Related memories
    if let Some(ref memories) = item.related_memories
      && !memories.is_empty()
    {
      out.push_str(&format!("Related memories ({}):\n", memories.len()));
      for m in memories {
        out.push_str(&format!(
          "  - [{}] ({}) {}\n",
          &m.id[..8.min(m.id.len())],
          m.sector,
          truncate(&m.content, 80)
        ));
      }
    }

    out.push_str(&format!("</{}>\n\n", item.item_type));
  }

  out
}

// ============================================================================
// Code formatters
// ============================================================================

fn format_code_search(result: &CodeSearchResult) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Code Search: {}\n\n", result.query));

  // Quality indicator
  if let Some(ref q) = result.search_quality
    && q.low_confidence
  {
    out.push_str(&format!(
      "⚠️ Low confidence results (best distance: {:.2})\n",
      q.best_distance
    ));
    if let Some(ref action) = q.suggested_action {
      out.push_str(&format!("Suggestion: {}\n", action));
    }
    out.push('\n');
  }

  out.push_str(&format!("Found {} results\n\n", result.chunks.len()));

  for (i, chunk) in result.chunks.iter().enumerate() {
    out.push_str(&format_code_item(chunk, i + 1));
    out.push('\n');
  }

  out
}

fn format_code_item(item: &CodeItem, index: usize) -> String {
  let mut out = String::new();

  out.push_str(&format!(
    "<code index=\"{}\" id=\"{}\" file=\"{}\" lines=\"{}-{}\"",
    index,
    &item.id[..8.min(item.id.len())],
    item.file_path,
    item.start_line,
    item.end_line
  ));

  if let Some(ref lang) = item.language {
    out.push_str(&format!(" language=\"{}\"", lang));
  }
  if let Some(sim) = item.similarity {
    out.push_str(&format!(" score=\"{:.2}\"", sim));
  }
  out.push_str(">\n");

  // Symbols
  if !item.symbols.is_empty() {
    out.push_str(&format!("Symbols: {}\n", item.symbols.join(", ")));
  }

  // Relationship hints
  if let (Some(callers), Some(callees)) = (item.caller_count, item.callee_count) {
    out.push_str(&format!("References: {} callers, {} callees\n", callers, callees));
  }

  // Content
  out.push('\n');
  out.push_str(&format_code_block(&item.content, item.language.as_deref()));

  out.push_str("</code>\n");
  out
}

fn format_code_context(result: &CodeContextResponse) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Code Context: {}\n", result.file_path));
  out.push_str(&format!(
    "Language: {} | Total lines: {}\n",
    result.language, result.total_file_lines
  ));

  if let Some(ref warning) = result.warning {
    out.push_str(&format!("⚠️ {}\n", warning));
  }
  out.push('\n');

  // Before
  if !result.context.before.content.is_empty() {
    out.push_str(&format!("--- Before (line {}) ---\n", result.context.before.start_line));
    out.push_str(&format_code_block(
      &result.context.before.content,
      Some(&result.language),
    ));
  }

  // Target
  out.push_str(&format!(
    ">>> Target (lines {}-{}) <<<\n",
    result.context.target.start_line, result.context.target.end_line
  ));
  out.push_str(&format_code_block(
    &result.context.target.content,
    Some(&result.language),
  ));

  // After
  if !result.context.after.content.is_empty() {
    out.push_str(&format!("--- After (line {}) ---\n", result.context.after.start_line));
    out.push_str(&format_code_block(
      &result.context.after.content,
      Some(&result.language),
    ));
  }

  out
}

fn format_code_index(result: &CodeIndexResult) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Code Index: {}\n\n", result.status));

  out.push_str(&format!(
    "Files: {} scanned, {} indexed\n",
    result.files_scanned, result.files_indexed
  ));
  out.push_str(&format!("Chunks created: {}\n", result.chunks_created));

  if result.failed_files > 0 {
    out.push_str(&format!("⚠️ Failed files: {}\n", result.failed_files));
  }

  out.push_str(&format!(
    "\nPerformance: {:.1} files/sec, {} bytes processed\n",
    result.files_per_second, result.bytes_processed
  ));
  out.push_str(&format!(
    "Duration: scan {}ms, index {}ms, total {}ms\n",
    result.scan_duration_ms, result.index_duration_ms, result.total_duration_ms
  ));

  out
}

fn format_code_list(items: &[CodeItem]) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Code Chunks ({} items)\n\n", items.len()));

  for (i, item) in items.iter().enumerate() {
    out.push_str(&format!(
      "{}. [{}] {}:{}-{} ({})\n",
      i + 1,
      &item.id[..8.min(item.id.len())],
      item.file_path,
      item.start_line,
      item.end_line,
      item.language.as_deref().unwrap_or("unknown")
    ));
    if !item.symbols.is_empty() {
      out.push_str(&format!("   Symbols: {}\n", item.symbols.join(", ")));
    }
  }

  out
}

fn format_code_stats(result: &CodeStatsResult) -> String {
  let mut out = String::new();

  out.push_str("# Code Statistics\n\n");

  out.push_str(&format!("Total chunks: {}\n", result.total_chunks));
  out.push_str(&format!("Total files: {}\n", result.total_files));
  out.push_str(&format!("Total lines: {}\n", result.total_lines));
  out.push_str(&format!("Tokens estimate: {}\n", result.total_tokens_estimate));
  out.push_str(&format!("Avg chunks/file: {:.1}\n", result.average_chunks_per_file));
  out.push_str(&format!("Index health: {}%\n\n", result.index_health_score));

  if !result.language_breakdown.is_empty() {
    out.push_str("Languages:\n");
    for (lang, count) in &result.language_breakdown {
      out.push_str(&format!("  - {}: {}\n", lang, count));
    }
  }

  if !result.chunk_type_breakdown.is_empty() {
    out.push_str("\nChunk types:\n");
    for (typ, count) in &result.chunk_type_breakdown {
      out.push_str(&format!("  - {}: {}\n", typ, count));
    }
  }

  out
}

fn format_code_memories(result: &CodeMemoriesResponse) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Memories for: {}\n\n", result.file_path));

  if result.memories.is_empty() {
    out.push_str("No related memories found.\n");
  } else {
    for mem in &result.memories {
      out.push_str(&format!(
        "<memory id=\"{}\" sector=\"{}\" salience=\"{:.2}\">\n",
        &mem.id[..8.min(mem.id.len())],
        mem.sector,
        mem.salience
      ));
      out.push_str(&mem.content);
      out.push_str("\n</memory>\n\n");
    }
  }

  out
}

fn format_code_callers(result: &CodeCallersResponse) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Callers of: {}\n\n", result.symbol));
  out.push_str(&format!("Found {} callers\n\n", result.count));

  for (i, caller) in result.callers.iter().enumerate() {
    out.push_str(&format_code_item(caller, i + 1));
    out.push('\n');
  }

  out
}

fn format_code_callees(result: &CodeCalleesResponse) -> String {
  let mut out = String::new();

  out.push_str(&format!(
    "# Callees of: {}\n\n",
    &result.chunk_id[..8.min(result.chunk_id.len())]
  ));

  if !result.calls.is_empty() {
    out.push_str(&format!("Calls: {}\n\n", result.calls.join(", ")));
  }

  out.push_str(&format!("Resolved ({}):\n", result.callees.len()));
  for callee in &result.callees {
    out.push_str(&format!(
      "  - {} → [{}] {}:{}\n",
      callee.call,
      &callee.id[..8.min(callee.id.len())],
      callee.file_path,
      callee.start_line
    ));
  }

  if !result.unresolved.is_empty() {
    out.push_str(&format!("\nUnresolved ({}):\n", result.unresolved.len()));
    for call in &result.unresolved {
      out.push_str(&format!("  - {}\n", call));
    }
  }

  out
}

fn format_code_related(result: &CodeRelatedResponse) -> String {
  let mut out = String::new();

  out.push_str(&format!(
    "# Related to: {} ({})\n\n",
    result.symbols.join(", "),
    result.file_path
  ));
  out.push_str(&format!("Found {} related\n\n", result.count));

  for item in &result.related {
    out.push_str(&format!(
      "<related id=\"{}\" file=\"{}\" lines=\"{}-{}\" relation=\"{}\" score=\"{:.2}\">\n",
      &item.id[..8.min(item.id.len())],
      item.file_path,
      item.start_line,
      item.end_line,
      item.relationship,
      item.score
    ));
    out.push_str(&format!("Symbols: {}\n", item.symbols.join(", ")));
    out.push_str("</related>\n\n");
  }

  out
}

fn format_code_context_full(result: &CodeContextFullResponse) -> String {
  let mut out = String::new();

  // Main chunk
  out.push_str("# Full Code Context\n\n");
  out.push_str(&format_code_item(&result.chunk, 0));
  out.push('\n');

  // Callers
  if !result.callers.is_empty() {
    out.push_str(&format!("## Callers ({})\n\n", result.callers.len()));
    for (i, c) in result.callers.iter().enumerate() {
      out.push_str(&format!(
        "{}. [{}] {}:{}-{}\n",
        i + 1,
        &c.id[..8.min(c.id.len())],
        c.file_path,
        c.start_line,
        c.end_line
      ));
    }
    out.push('\n');
  }

  // Callees
  if !result.callees.is_empty() {
    out.push_str(&format!("## Callees ({})\n\n", result.callees.len()));
    for c in &result.callees {
      out.push_str(&format!(
        "  - {} → [{}] {}:{}\n",
        c.call,
        &c.id[..8.min(c.id.len())],
        c.file_path,
        c.start_line
      ));
    }
    out.push('\n');
  }

  // Unresolved calls
  if !result.unresolved_calls.is_empty() {
    out.push_str(&format!("## Unresolved calls ({})\n\n", result.unresolved_calls.len()));
    out.push_str(&format!("{}\n\n", result.unresolved_calls.join(", ")));
  }

  // Same file
  if !result.same_file.is_empty() {
    out.push_str(&format!("## Same file ({})\n\n", result.same_file.len()));
    for c in &result.same_file {
      out.push_str(&format!(
        "  - [{}] lines {}-{}: {}\n",
        &c.id[..8.min(c.id.len())],
        c.start_line,
        c.end_line,
        c.symbols.join(", ")
      ));
    }
    out.push('\n');
  }

  // Memories
  if !result.memories.is_empty() {
    out.push_str(&format!("## Related memories ({})\n\n", result.memories.len()));
    for m in &result.memories {
      out.push_str(&format!(
        "  - [{}] ({}) {}\n",
        &m.id[..8.min(m.id.len())],
        m.sector,
        truncate(&m.content, 80)
      ));
    }
    out.push('\n');
  }

  // Documentation
  if !result.documentation.is_empty() {
    out.push_str(&format!("## Documentation ({})\n\n", result.documentation.len()));
    for doc in &result.documentation {
      out.push_str(&format!("### {}\n\n{}\n\n", doc.title, doc.content));
    }
  }

  out
}

// ============================================================================
// Memory formatters
// ============================================================================

fn format_memory_search(result: &MemorySearchResult) -> String {
  let mut out = String::new();

  out.push_str("# Memory Search\n\n");

  // Quality indicator
  if let Some(ref q) = result.search_quality
    && q.low_confidence
  {
    out.push_str(&format!(
      "⚠️ Low confidence results (best distance: {:.2})\n",
      q.best_distance
    ));
    if let Some(ref action) = q.suggested_action {
      out.push_str(&format!("Suggestion: {}\n", action));
    }
    out.push('\n');
  }

  out.push_str(&format!("Found {} memories\n\n", result.items.len()));

  for (i, mem) in result.items.iter().enumerate() {
    out.push_str(&format_memory_item(mem, i + 1));
    out.push('\n');
  }

  out
}

fn format_memory_item(item: &MemoryItem, index: usize) -> String {
  let mut out = String::new();

  out.push_str(&format!(
    "<memory index=\"{}\" id=\"{}\" sector=\"{}\" salience=\"{:.2}\"",
    index,
    &item.id[..8.min(item.id.len())],
    item.sector,
    item.salience
  ));

  if let Some(sim) = item.similarity {
    out.push_str(&format!(" score=\"{:.2}\"", sim));
  }
  if item.is_superseded {
    out.push_str(" superseded=\"true\"");
  }
  out.push_str(">\n");

  // Type and tier
  if let Some(ref t) = item.memory_type {
    out.push_str(&format!("Type: {} | Tier: {}\n", t, item.tier));
  }

  // Tags
  if !item.tags.is_empty() {
    out.push_str(&format!("Tags: {}\n", item.tags.join(", ")));
  }

  // Content
  out.push('\n');
  out.push_str(&item.content);
  out.push_str("\n</memory>\n");

  out
}

fn format_memory_get(result: &MemoryFullDetail) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Memory: {}\n\n", &result.id[..8.min(result.id.len())]));

  out.push_str(&format!("Sector: {} | Tier: {}\n", result.sector, result.tier));
  if let Some(ref t) = result.memory_type {
    out.push_str(&format!("Type: {}\n", t));
  }
  out.push_str(&format!(
    "Salience: {:.2} | Importance: {:.2} | Confidence: {:.2}\n",
    result.salience, result.importance, result.confidence
  ));
  out.push_str(&format!("Access count: {}\n", result.access_count));

  if result.is_deleted {
    out.push_str("⚠️ DELETED\n");
  }
  if let Some(ref by) = result.superseded_by {
    out.push_str(&format!("Superseded by: {}\n", &by[..8.min(by.len())]));
  }

  out.push_str(&format!("\nCreated: {}\n", result.created_at));
  out.push_str(&format!("Updated: {}\n", result.updated_at));
  out.push_str(&format!("Last accessed: {}\n", result.last_accessed));

  // Tags and categories
  if !result.tags.is_empty() {
    out.push_str(&format!("\nTags: {}\n", result.tags.join(", ")));
  }
  if !result.categories.is_empty() {
    out.push_str(&format!("Categories: {}\n", result.categories.join(", ")));
  }

  // Scope
  if let Some(ref path) = result.scope_path {
    out.push_str(&format!("Scope path: {}\n", path));
  }
  if let Some(ref module) = result.scope_module {
    out.push_str(&format!("Scope module: {}\n", module));
  }

  // Content
  out.push_str("\n---\n\n");
  out.push_str(&result.content);
  out.push('\n');

  // Context
  if let Some(ref ctx) = result.context {
    out.push_str(&format!("\nContext: {}\n", ctx));
  }

  // Relationships
  if let Some(ref rels) = result.relationships
    && !rels.is_empty()
  {
    out.push_str(&format!("\nRelationships ({}):\n", rels.len()));
    for r in rels {
      out.push_str(&format!(
        "  - {} → {} ({})\n",
        &r.from_id[..8.min(r.from_id.len())],
        &r.to_id[..8.min(r.to_id.len())],
        r.relationship_type
      ));
    }
  }

  out
}

fn format_memory_add(result: &MemoryAddResult) -> String {
  let mut out = format!("✓ Memory added: {}\n", &result.id[..8.min(result.id.len())]);
  if result.is_duplicate {
    out.push_str("(duplicate detected)\n");
  }
  out
}

fn format_memory_update(result: &MemoryUpdateResult) -> String {
  format!(
    "✓ Memory updated: {} (salience: {:.2})\n{}",
    &result.id[..8.min(result.id.len())],
    result.new_salience,
    result.message
  )
}

fn format_memory_delete(result: &MemoryDeleteResult) -> String {
  let mut out = format!("✓ Memory deleted: {}\n", &result.id[..8.min(result.id.len())]);
  if result.hard_delete {
    out.push_str("(permanently removed)\n");
  }
  out
}

fn format_memory_supersede(result: &MemorySupersedeResult) -> String {
  format!(
    "✓ Memory superseded\nOld: {} → New: {}\n",
    &result.old_id[..8.min(result.old_id.len())],
    &result.new_id[..8.min(result.new_id.len())]
  )
}

fn format_memory_list(items: &[MemoryItem]) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Memories ({} items)\n\n", items.len()));

  for (i, mem) in items.iter().enumerate() {
    out.push_str(&format!(
      "{}. [{}] ({}/{}) salience={:.2}\n",
      i + 1,
      &mem.id[..8.min(mem.id.len())],
      mem.sector,
      mem.tier,
      mem.salience
    ));
    out.push_str(&format!("   {}\n", truncate(&mem.content, 100)));
  }

  out
}

fn format_memory_timeline(result: &MemoryTimelineResult) -> String {
  let mut out = String::new();

  out.push_str("# Memory Timeline\n\n");

  // Before
  for item in result.before.iter().rev() {
    out.push_str(&format!(
      "  ↑ [{}] ({}) {}\n",
      &item.id[..8.min(item.id.len())],
      item.sector,
      truncate(&item.content, 60)
    ));
  }

  // Anchor
  out.push_str(&format!(
    ">>> [{}] ({}) {} <<<\n",
    &result.anchor.id[..8.min(result.anchor.id.len())],
    result.anchor.sector,
    truncate(&result.anchor.content, 60)
  ));

  // After
  for item in &result.after {
    out.push_str(&format!(
      "  ↓ [{}] ({}) {}\n",
      &item.id[..8.min(item.id.len())],
      item.sector,
      truncate(&item.content, 60)
    ));
  }

  out
}

fn format_memory_related(result: &MemoryRelatedResult) -> String {
  let mut out = String::new();

  out.push_str(&format!(
    "# Related to: {}\n\n",
    &result.memory_id[..8.min(result.memory_id.len())]
  ));
  out.push_str(&format!("Content: {}\n\n", truncate(&result.content, 100)));
  out.push_str(&format!("Found {} related\n\n", result.count));

  for item in &result.related {
    out.push_str(&format!(
      "<related id=\"{}\" sector=\"{}\" relation=\"{}\" score=\"{:.2}\">\n",
      &item.id[..8.min(item.id.len())],
      item.sector,
      item.relationship,
      item.score
    ));
    out.push_str(&truncate(&item.content, 150));
    out.push_str("\n</related>\n\n");
  }

  out
}

// ============================================================================
// Document formatters
// ============================================================================

fn format_docs_search(items: &[DocSearchItem]) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Document Search ({} results)\n\n", items.len()));

  for (i, doc) in items.iter().enumerate() {
    out.push_str(&format!(
      "<doc index=\"{}\" id=\"{}\" chunk=\"{}/{}\"",
      i + 1,
      &doc.id[..8.min(doc.id.len())],
      doc.chunk_index + 1,
      doc.total_chunks
    ));
    if let Some(sim) = doc.similarity {
      out.push_str(&format!(" score=\"{:.2}\"", sim));
    }
    out.push_str(">\n");

    out.push_str(&format!("Title: {}\n", doc.title));
    out.push_str(&format!("Source: {}\n\n", doc.source));
    out.push_str(&doc.content);
    out.push_str("\n</doc>\n\n");
  }

  out
}

fn format_doc_context(result: &DocContextResult) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Document: {}\n", result.title));
  out.push_str(&format!(
    "Source: {} | Total chunks: {}\n\n",
    result.source, result.total_chunks
  ));

  // Before
  for chunk in &result.context.before {
    out.push_str(&format!("--- Chunk {} ---\n", chunk.chunk_index + 1));
    out.push_str(&chunk.content);
    out.push_str("\n\n");
  }

  // Target
  out.push_str(&format!(
    ">>> Chunk {} (target) <<<\n",
    result.context.target.chunk_index + 1
  ));
  out.push_str(&result.context.target.content);
  out.push_str("\n\n");

  // After
  for chunk in &result.context.after {
    out.push_str(&format!("--- Chunk {} ---\n", chunk.chunk_index + 1));
    out.push_str(&chunk.content);
    out.push_str("\n\n");
  }

  out
}

fn format_docs_ingest(result: &DocsIngestFullResult) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Document Ingest: {}\n\n", result.status));

  out.push_str(&format!(
    "Files: {} scanned, {} ingested\n",
    result.files_scanned, result.files_ingested
  ));
  out.push_str(&format!("Chunks created: {}\n", result.chunks_created));

  if result.failed_files > 0 {
    out.push_str(&format!("⚠️ Failed files: {}\n", result.failed_files));
  }

  out.push_str(&format!(
    "\nPerformance: {:.1} files/sec, {} bytes processed\n",
    result.files_per_second, result.bytes_processed
  ));
  out.push_str(&format!(
    "Duration: scan {}ms, ingest {}ms, total {}ms\n",
    result.scan_duration_ms, result.ingest_duration_ms, result.total_duration_ms
  ));

  if !result.results.is_empty() {
    out.push_str(&format!("\nIngested documents ({}):\n", result.results.len()));
    for doc in &result.results {
      out.push_str(&format!("  - {} ({} chunks)\n", doc.title, doc.chunks_created));
    }
  }

  out
}

// ============================================================================
// Relationship formatters
// ============================================================================

fn format_relationship_add(result: &RelationshipResult) -> String {
  format!(
    "✓ Relationship added: {} → {} ({})\n",
    &result.from_memory_id[..8.min(result.from_memory_id.len())],
    &result.to_memory_id[..8.min(result.to_memory_id.len())],
    result.relationship_type
  )
}

fn format_relationship_list(items: &[RelationshipListItem]) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Relationships ({} items)\n\n", items.len()));

  for item in items {
    out.push_str(&format!(
      "  [{}] {} → {} ({}) confidence={:.2}\n",
      &item.id[..8.min(item.id.len())],
      &item.from_memory_id[..8.min(item.from_memory_id.len())],
      &item.to_memory_id[..8.min(item.to_memory_id.len())],
      item.relationship_type,
      item.confidence
    ));
  }

  out
}

fn format_relationship_delete(result: &DeletedResult) -> String {
  if result.deleted {
    "✓ Relationship deleted\n".to_string()
  } else {
    "✗ Relationship not found\n".to_string()
  }
}

fn format_relationship_related(items: &[RelatedMemoryItem]) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Related Memories ({} items)\n\n", items.len()));

  for item in items {
    out.push_str(&format!(
      "<memory id=\"{}\" sector=\"{}\" relation=\"{}\" confidence=\"{:.2}\">\n",
      &item.memory.id[..8.min(item.memory.id.len())],
      item.memory.sector,
      item.relationship.relationship_type,
      item.relationship.confidence
    ));
    out.push_str(&truncate(&item.memory.content, 150));
    out.push_str("\n</memory>\n\n");
  }

  out
}

// ============================================================================
// Watch formatters
// ============================================================================

fn format_watch_start(result: &WatchStartResult) -> String {
  format!(
    "✓ Watcher started\nPath: {}\nProject: {}\n",
    result.path, result.project_id
  )
}

fn format_watch_stop(result: &WatchStopResult) -> String {
  format!(
    "✓ Watcher stopped\nPath: {}\nProject: {}\n",
    result.path, result.project_id
  )
}

fn format_watch_status(result: &WatchStatusResult) -> String {
  let mut out = String::new();

  out.push_str("# Watcher Status\n\n");

  out.push_str(&format!("Running: {}\n", if result.running { "yes" } else { "no" }));
  if let Some(ref root) = result.root {
    out.push_str(&format!("Root: {}\n", root));
  }
  out.push_str(&format!("Project: {}\n", result.project_id));
  out.push_str(&format!("Pending changes: {}\n", result.pending_changes));

  if result.scanning {
    out.push_str("⏳ Scanning in progress");
    if let Some([current, total]) = result.scan_progress {
      out.push_str(&format!(" ({}/{})", current, total));
    }
    out.push('\n');
  }

  out
}

// ============================================================================
// Project formatters
// ============================================================================

fn format_project_info(result: &ProjectInfoResult) -> String {
  let mut out = String::new();

  out.push_str(&format!("# Project: {}\n\n", result.name));

  out.push_str(&format!("ID: {}\n", result.id));
  out.push_str(&format!("Path: {}\n", result.path));
  out.push_str(&format!("Database: {}\n\n", result.db_path));

  out.push_str(&format!("Memories: {}\n", result.memory_count));
  out.push_str(&format!("Code chunks: {}\n", result.code_chunk_count));
  out.push_str(&format!("Documents: {}\n", result.document_count));
  out.push_str(&format!("Sessions: {}\n", result.session_count));

  out
}

fn format_project_clean(result: &ProjectCleanResult) -> String {
  let mut out = String::new();

  out.push_str(&format!("✓ Project cleaned: {}\n\n", result.path));
  out.push_str(&format!("Memories deleted: {}\n", result.memories_deleted));
  out.push_str(&format!("Code chunks deleted: {}\n", result.code_chunks_deleted));
  out.push_str(&format!("Documents deleted: {}\n", result.documents_deleted));

  out
}

fn format_project_clean_all(result: &ProjectCleanAllResult) -> String {
  format!("✓ {} projects removed\n", result.projects_removed)
}

fn format_project_stats(result: &ProjectStatsResult) -> String {
  let mut out = String::new();

  out.push_str("# Project Statistics\n\n");

  out.push_str(&format!("Project: {}\n", result.project_id));
  out.push_str(&format!("Path: {}\n\n", result.path));

  out.push_str(&format!("Memories: {}\n", result.memories));
  out.push_str(&format!("Code chunks: {}\n", result.code_chunks));
  out.push_str(&format!("Documents: {}\n", result.documents));
  out.push_str(&format!("Sessions: {}\n", result.sessions));

  out
}

// ============================================================================
// System formatters
// ============================================================================

fn format_health_check(result: &HealthCheckResult) -> String {
  let mut out = String::new();

  out.push_str("# Health Check\n\n");

  out.push_str(&format!(
    "Status: {}\n\n",
    if result.healthy { "✓ Healthy" } else { "✗ Unhealthy" }
  ));

  for check in &result.checks {
    let icon = if check.status == "ok" { "✓" } else { "✗" };
    out.push_str(&format!("{} {}: {}", icon, check.name, check.status));
    if let Some(ref msg) = check.message {
      out.push_str(&format!(" - {}", msg));
    }
    out.push('\n');
  }

  out
}

// ============================================================================
// Helpers
// ============================================================================

fn format_code_block(content: &str, language: Option<&str>) -> String {
  let lang = language.unwrap_or("");
  format!("```{}\n{}\n```\n", lang, content.trim())
}

fn format_preview(content: &str, language: Option<&str>) -> String {
  let trimmed = content.trim();
  if trimmed.lines().count() > 1 || trimmed.len() > 80 {
    format_code_block(trimmed, language)
  } else {
    format!("`{}`\n", trimmed)
  }
}

fn truncate(s: &str, max_len: usize) -> String {
  let s = s.trim().replace('\n', " ");
  if s.len() <= max_len {
    s
  } else {
    format!("{}...", &s[..max_len.saturating_sub(3)])
  }
}
