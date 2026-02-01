//! MCP tool definitions with config-based filtering.

use std::collections::HashMap;

use ccengram::config::Config;
use serde_json::{Value, json};

/// Get all tool definitions as a map of name -> definition
pub fn all_tool_definitions() -> HashMap<&'static str, Value> {
  let mut tools = HashMap::new();

  // ============================================================================
  // Unified Exploration Tools (NEW - minimal preset)
  // ============================================================================

  tools.insert(
    "explore",
    json!({
        "name": "explore",
        "description": "Search code and docs by natural language or symbol name. Returns ranked results with navigation hints. Use expand_top to include full context (callers, callees, related memories) for top results in one call. By default searches code+docs only; use scope='memory' or scope='all' to include memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query, symbol name, or file path"
                },
                "scope": {
                    "type": "string",
                    "enum": ["code", "memory", "docs", "all"],
                    "description": "Where to search: 'code' for implementation, 'memory' for past decisions/patterns, 'docs' for documentation, 'all' for everything including memory. Default searches code+docs only; use 'memory' or 'all' to include memories."
                },
                "expand_top": {
                    "type": "number",
                    "description": "Include full context (callers, callees, siblings, memories) for top N results (default: 3)"
                },
                "limit": {
                    "type": "number",
                    "description": "Max results per scope (default: 10)"
                },
            },
            "required": ["query"]
        }
    }),
  );

  tools.insert(
    "context",
    json!({
        "name": "context",
        "description": "Get full context for item(s) from explore results. For code: callers, callees, siblings, related memories. For memory: timeline, related. For docs: surrounding chunks. Accepts single ID or array of IDs for batch retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Single ID from explore results (prefix match supported, min 6 chars)"
                },
                "ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of IDs for batch context (max 5). Use this OR id, not both."
                },
                "depth": {
                    "type": "number",
                    "description": "Items per section - callers, callees, etc. (default: 5)"
                },
            }
        }
    }),
  );

  // ============================================================================
  // Memory tools
  // ============================================================================

  tools.insert(
        "memory_search",
        json!({
            "name": "memory_search",
            "description": "Search memories by semantic similarity. Returns relevant memories with salience scores.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "sector": { "type": "string", "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"], "description": "Filter by memory sector" },
                    "limit": { "type": "number", "description": "Max results (default: 10)" },
                    "include_superseded": { "type": "boolean", "description": "Include superseded memories (default: false)" }
                },
                "required": ["query"]
            }
        }),
    );

  tools.insert(
    "memory_get",
    json!({
        "name": "memory_get",
        "description": "Get a specific memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to retrieve" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  tools.insert(
        "memory_list",
        json!({
            "name": "memory_list",
            "description": "List memories with optional filtering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "number", "description": "Max results (default: 50)" },
                    "offset": { "type": "number", "description": "Offset for pagination" },
                    "sector": { "type": "string", "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"], "description": "Filter by sector" }
                }
            }
        }),
    );

  tools.insert(
    "memory_timeline",
    json!({
        "name": "memory_timeline",
        "description": "Get chronological context around a memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "anchor_id": { "type": "string", "description": "Memory ID to center timeline on" },
                "depth_before": { "type": "number", "description": "Memories before (default: 5)" },
                "depth_after": { "type": "number", "description": "Memories after (default: 5)" }
            },
            "required": ["anchor_id"]
        }
    }),
  );

  tools.insert(
        "memory_add",
        json!({
            "name": "memory_add",
            "description": "Manually add a memory. Use for explicit notes, decisions, preferences.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Memory content" },
                    "sector": { "type": "string", "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"], "description": "Memory sector" },
                    "type": { "type": "string", "enum": ["preference", "codebase", "decision", "gotcha", "pattern", "turn_summary", "task_completion"], "description": "Memory type" },
                    "context": { "type": "string", "description": "Context of discovery" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags" },
                    "importance": { "type": "number", "description": "Importance 0-1 (default: 0.5)" }
                },
                "required": ["content"]
            }
        }),
    );

  tools.insert(
    "memory_reinforce",
    json!({
        "name": "memory_reinforce",
        "description": "Reinforce a memory, increasing its salience.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to reinforce" },
                "amount": { "type": "number", "description": "Reinforcement amount 0-1 (default: 0.1)" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  tools.insert(
    "memory_deemphasize",
    json!({
        "name": "memory_deemphasize",
        "description": "De-emphasize a memory, reducing its salience.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to de-emphasize" },
                "amount": { "type": "number", "description": "De-emphasis amount 0-1 (default: 0.2)" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  tools.insert(
    "memory_delete",
    json!({
        "name": "memory_delete",
        "description": "Delete a memory. Soft delete preserves history, hard delete removes completely.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to delete" },
                "hard": { "type": "boolean", "description": "Permanently delete (default: false)" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  tools.insert(
    "memory_supersede",
    json!({
        "name": "memory_supersede",
        "description": "Mark one memory as superseding another.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "old_memory_id": { "type": "string", "description": "ID of memory being superseded" },
                "new_memory_id": { "type": "string", "description": "ID of newer memory that supersedes it" }
            },
            "required": ["old_memory_id", "new_memory_id"]
        }
    }),
  );

  tools.insert(
    "memory_related",
    json!({
        "name": "memory_related",
        "description": "Find memories related to a given memory via relationship graph, shared entities, semantic similarity, or supersession chain.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to find related memories for" },
                "methods": {
                    "type": "array",
                    "items": { "type": "string", "enum": ["relationships", "entities", "similar", "supersedes"] },
                    "description": "Relationship methods to use (default: all)"
                },
                "limit": { "type": "number", "description": "Max results (default: 10)" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  // Code tools
  tools.insert(
    "code_search",
    json!({
        "name": "code_search",
        "description": "Semantic code search with file paths and line numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "language": { "type": "string", "description": "Filter by programming language" },
                "limit": { "type": "number", "description": "Max results (default: 10)" }
            },
            "required": ["query"]
        }
    }),
  );

  tools.insert(
    "code_index",
    json!({
        "name": "code_index",
        "description": "Trigger indexing/re-indexing of project code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": { "type": "boolean", "description": "Force re-index all files" },
                "dry_run": { "type": "boolean", "description": "Scan only, don't index" }
            }
        }
    }),
  );

  tools.insert(
    "code_list",
    json!({
        "name": "code_list",
        "description": "List indexed code chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": { "type": "number", "description": "Max results (default: 50)" },
                "offset": { "type": "number", "description": "Offset for pagination" },
                "language": { "type": "string", "description": "Filter by language" },
                "file_path": { "type": "string", "description": "Filter by file path prefix" }
            }
        }
    }),
  );

  tools.insert(
    "code_stats",
    json!({
        "name": "code_stats",
        "description": "Get code index statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }),
  );

  tools.insert(
    "code_context",
    json!({
        "name": "code_context",
        "description": "Get surrounding lines for a code chunk by reading from filesystem. Use after code_search to see more context around a result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Code chunk ID from search results (can use ID prefix)" },
                "lines_before": { "type": "number", "description": "Lines to include before chunk (default: 20, max: 500)" },
                "lines_after": { "type": "number", "description": "Lines to include after chunk (default: 20, max: 500)" }
            },
            "required": ["chunk_id"]
        }
    }),
  );

  tools.insert(
    "code_memories",
    json!({
        "name": "code_memories",
        "description": "Get memories (decisions, gotchas, patterns) related to code. Use to understand why code is written a certain way before modifying it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Code chunk ID" },
                "file_path": { "type": "string", "description": "File path (alternative to chunk_id)" },
                "limit": { "type": "number", "description": "Max results (default: 10)" }
            }
        }
    }),
  );

  tools.insert(
    "code_callers",
    json!({
        "name": "code_callers",
        "description": "Find all code that calls a function/method. Essential for understanding impact of changes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Code chunk ID defining the function" },
                "symbol": { "type": "string", "description": "Function/method name to search for" },
                "limit": { "type": "number", "description": "Max results (default: 20)" }
            }
        }
    }),
  );

  tools.insert(
    "code_callees",
    json!({
        "name": "code_callees",
        "description": "Find functions that a code chunk calls. Returns resolved definitions and unresolved external calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Code chunk ID to analyze" },
                "limit": { "type": "number", "description": "Max results per call (default: 3)" }
            },
            "required": ["chunk_id"]
        }
    }),
  );

  tools.insert(
    "code_related",
    json!({
        "name": "code_related",
        "description": "Find code related to a chunk via multiple methods: same file, shared imports, semantic similarity, callers, callees.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Code chunk ID" },
                "methods": {
                    "type": "array",
                    "items": { "type": "string", "enum": ["same_file", "shared_imports", "similar", "callers", "callees"] },
                    "description": "Relationship methods to use (default: all)"
                },
                "limit": { "type": "number", "description": "Max results (default: 20)" }
            },
            "required": ["chunk_id"]
        }
    }),
  );

  tools.insert(
    "code_context_full",
    json!({
        "name": "code_context_full",
        "description": "Get comprehensive context for code in ONE call: the chunk itself, callers, callees, sibling functions, related memories (decisions/gotchas), and documentation. Use this first when examining unfamiliar code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Code chunk ID" },
                "file_path": { "type": "string", "description": "File path (alternative)" },
                "symbol": { "type": "string", "description": "Symbol name (used with file_path)" },
                "limit_per_section": { "type": "number", "description": "Max items per section (default: 5)" }
            }
        }
    }),
  );

  // Watch tools
  tools.insert(
    "watch_start",
    json!({
        "name": "watch_start",
        "description": "Start the file watcher for automatic re-indexing.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }),
  );

  tools.insert(
    "watch_stop",
    json!({
        "name": "watch_stop",
        "description": "Stop the file watcher.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }),
  );

  tools.insert(
    "watch_status",
    json!({
        "name": "watch_status",
        "description": "Check file watcher status.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }),
  );

  // Document tools
  tools.insert(
    "docs_search",
    json!({
        "name": "docs_search",
        "description": "Search ingested documents. Separate from memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "limit": { "type": "number", "description": "Max results (default: 5)" }
            },
            "required": ["query"]
        }
    }),
  );

  tools.insert(
    "docs_ingest",
    json!({
        "name": "docs_ingest",
        "description": "Ingest a document for searchable reference.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to ingest" },
                "url": { "type": "string", "description": "URL to fetch and ingest" },
                "content": { "type": "string", "description": "Raw content to ingest" },
                "title": { "type": "string", "description": "Document title" }
            }
        }
    }),
  );

  tools.insert(
    "doc_context",
    json!({
        "name": "doc_context",
        "description": "Get adjacent chunks from the same document. Use after docs_search to see more context around a result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string", "description": "Document chunk ID from search results (can use ID prefix)" },
                "chunks_before": { "type": "number", "description": "Number of chunks before target (default: 1, max: 10)" },
                "chunks_after": { "type": "number", "description": "Number of chunks after target (default: 1, max: 10)" }
            },
            "required": ["chunk_id"]
        }
    }),
  );

  // Entity tools
  tools.insert(
        "entity_list",
        json!({
            "name": "entity_list",
            "description": "List known entities (people, technologies, concepts).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_type": { "type": "string", "enum": ["person", "technology", "concept", "organization", "project"], "description": "Filter by entity type" },
                    "limit": { "type": "number", "description": "Max results (default: 50)" }
                }
            }
        }),
    );

  tools.insert(
    "entity_get",
    json!({
        "name": "entity_get",
        "description": "Get details about a specific entity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": { "type": "string", "description": "Entity ID to retrieve" }
            },
            "required": ["entity_id"]
        }
    }),
  );

  tools.insert(
        "entity_top",
        json!({
            "name": "entity_top",
            "description": "Get top entities by mention count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_type": { "type": "string", "enum": ["person", "technology", "concept", "organization", "project"], "description": "Filter by entity type" },
                    "limit": { "type": "number", "description": "Max results (default: 10)" }
                }
            }
        }),
    );

  // Relationship tools
  tools.insert(
        "relationship_add",
        json!({
            "name": "relationship_add",
            "description": "Add a relationship between two memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": { "type": "string", "description": "Source memory ID" },
                    "target_id": { "type": "string", "description": "Target memory ID" },
                    "relationship_type": { "type": "string", "enum": ["supersedes", "contradicts", "related_to", "elaborates", "causes", "derived_from", "supports", "opposes"], "description": "Type of relationship" },
                    "confidence": { "type": "number", "description": "Confidence score 0-1" }
                },
                "required": ["source_id", "target_id", "relationship_type"]
            }
        }),
    );

  tools.insert(
    "relationship_list",
    json!({
        "name": "relationship_list",
        "description": "List relationships for a memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to list relationships for" },
                "relationship_type": { "type": "string", "description": "Filter by relationship type" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  tools.insert(
    "relationship_delete",
    json!({
        "name": "relationship_delete",
        "description": "Delete a relationship.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "relationship_id": { "type": "string", "description": "Relationship ID to delete" }
            },
            "required": ["relationship_id"]
        }
    }),
  );

  tools.insert(
    "relationship_related",
    json!({
        "name": "relationship_related",
        "description": "Get related memories via relationships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": { "type": "string", "description": "Memory ID to find related memories for" },
                "max_depth": { "type": "number", "description": "Maximum relationship traversal depth (default: 1)" }
            },
            "required": ["memory_id"]
        }
    }),
  );

  // Statistics tools
  tools.insert(
    "project_stats",
    json!({
        "name": "project_stats",
        "description": "Get project statistics (memory counts, code stats, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }),
  );

  tools.insert(
    "health_check",
    json!({
        "name": "health_check",
        "description": "Check system health (database, embedding service, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }),
  );

  tools
}

/// Get filtered tool definitions based on config
pub fn get_filtered_tool_definitions(config: &Config) -> Value {
  let all_tools = all_tool_definitions();
  let enabled = config.enabled_tool_set();

  let filtered: Vec<Value> = all_tools
    .into_iter()
    .filter(|(name, _)| enabled.contains(*name))
    .map(|(_, def)| def)
    .collect();

  json!(filtered)
}

/// Get tool definitions filtered by the config loaded from current directory
pub async fn get_tool_definitions_for_cwd() -> Value {
  let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
  let config = Config::load_for_project(&cwd).await;
  get_filtered_tool_definitions(&config)
}

#[cfg(test)]
mod tests {
  use ccengram::config::{ToolConfig, ToolPreset};

  use super::*;

  #[test]
  fn test_all_tools_defined() {
    let tools = all_tool_definitions();
    // Verify all tools from ALL_TOOLS constant are defined
    for tool_name in ccengram::config::ALL_TOOLS {
      assert!(
        tools.contains_key(tool_name),
        "Missing definition for tool: {}",
        tool_name
      );
    }
  }

  #[test]
  fn test_minimal_preset_filtering() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Minimal,
        ..Default::default()
      },
      ..Default::default()
    };

    let filtered = get_filtered_tool_definitions(&config);
    let arr = filtered.as_array().unwrap();

    assert_eq!(arr.len(), 2);

    let names: Vec<&str> = arr.iter().filter_map(|t| t.get("name")?.as_str()).collect();
    assert!(names.contains(&"explore"));
    assert!(names.contains(&"context"));
  }

  #[test]
  fn test_standard_preset_filtering() {
    let config = Config::default();

    let filtered = get_filtered_tool_definitions(&config);
    let arr = filtered.as_array().unwrap();

    assert_eq!(arr.len(), 11);
  }

  #[test]
  fn test_full_preset_filtering() {
    let config = Config {
      tools: ToolConfig {
        preset: ToolPreset::Full,
        ..Default::default()
      },
      ..Default::default()
    };

    let filtered = get_filtered_tool_definitions(&config);
    let arr = filtered.as_array().unwrap();

    assert_eq!(arr.len(), ccengram::config::ALL_TOOLS.len());
  }
}
