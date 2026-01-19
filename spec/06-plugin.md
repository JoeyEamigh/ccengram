# Claude Code Plugin Specification

## Overview

The plugin integrates CCMemory with Claude Code through hooks for capture/summarization and an MCP server for on-demand search.

## Files to Create

- `plugin/plugin.json` - Plugin manifest
- `plugin/hooks/hooks.json` - Hook configuration
- `plugin/.mcp.json` - MCP server configuration
- `scripts/capture.ts` - PostToolUse hook script
- `scripts/summarize.ts` - Stop hook script (uses claude-agent-sdk)
- `scripts/cleanup.ts` - SessionEnd hook script
- `src/mcp/server.ts` - stdio MCP server

## Plugin Manifest

```json
// plugin/plugin.json
{
  "name": "ccmemory",
  "version": "1.0.0",
  "description": "Self-contained memory system for Claude Code with vector search",
  "author": "Your Name",
  "homepage": "https://github.com/user/ccmemory",
  "repository": "https://github.com/user/ccmemory"
}
```

## Hook Configuration

```json
// plugin/hooks/hooks.json
{
  "description": "CCMemory: Silent capture and on-demand search",
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "bun \"${CLAUDE_PLUGIN_ROOT}/../scripts/capture.js\"",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bun \"${CLAUDE_PLUGIN_ROOT}/../scripts/summarize.js\"",
            "timeout": 120
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bun \"${CLAUDE_PLUGIN_ROOT}/../scripts/cleanup.js\"",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

## MCP Server Configuration

```json
// plugin/.mcp.json
{
  "ccmemory": {
    "type": "stdio",
    "command": "bun",
    "args": ["${CLAUDE_PLUGIN_ROOT}/../dist/mcp/server.js"]
  }
}
```

## Hook Scripts

### PostToolUse Capture Hook

Captures all tool observations silently (no context injection).

```typescript
// scripts/capture.ts
import { getDatabase } from "../src/db/database";
import { createMemoryStore } from "../src/services/memory/store";
import { getOrCreateProject } from "../src/services/project";
import { log } from "../src/utils/log";

interface HookInput {
  session_id: string;
  cwd: string;
  tool_name: string;
  tool_input: Record<string, any>;
  tool_result: any;
}

async function main() {
  // Read stdin
  const input: HookInput = JSON.parse(await Bun.stdin.text());

  const { session_id, cwd, tool_name, tool_input, tool_result } = input;
  log.debug("capture", "Processing tool observation", { session_id, tool_name });

  // Skip if tool_result is too large (> 10KB)
  const resultStr = JSON.stringify(tool_result);
  if (resultStr.length > 10000) {
    log.debug("capture", "Skipping large tool result", { tool_name, bytes: resultStr.length });
    process.exit(0);
  }

  // Get or create project
  const project = await getOrCreateProject(cwd);

  // Build memory content
  const content = formatToolObservation(tool_name, tool_input, tool_result);

  // Store memory (async, fire-and-forget style but we wait for it)
  const store = createMemoryStore();
  const memory = await store.create({
    content,
    sector: "episodic",  // Tool observations are episodic memories
    tier: "session",
    sessionId: session_id,
    files: extractFilePaths(tool_input, tool_result)
  }, project.id, session_id);

  log.debug("capture", "Captured tool observation", { session_id, tool_name, memoryId: memory.id });

  // Exit silently (no output to avoid context pollution)
  process.exit(0);
}

function formatToolObservation(
  toolName: string,
  input: Record<string, any>,
  result: any
): string {
  const lines: string[] = [`Tool: ${toolName}`];

  // Format input based on tool type
  switch (toolName) {
    case "Read":
      lines.push(`Read file: ${input.file_path}`);
      break;
    case "Write":
      lines.push(`Wrote file: ${input.file_path}`);
      break;
    case "Edit":
      lines.push(`Edited file: ${input.file_path}`);
      break;
    case "Bash":
      lines.push(`Command: ${input.command?.slice(0, 200)}`);
      if (typeof result === "string" && result.length < 500) {
        lines.push(`Output: ${result}`);
      }
      break;
    case "Grep":
    case "Glob":
      lines.push(`Pattern: ${input.pattern}`);
      break;
    default:
      // Generic formatting
      lines.push(`Input: ${JSON.stringify(input).slice(0, 300)}`);
  }

  return lines.join("\n");
}

function extractFilePaths(input: Record<string, any>, result: any): string[] {
  const paths: string[] = [];

  // Check common input fields
  if (input.file_path) paths.push(input.file_path);
  if (input.path) paths.push(input.path);

  // Check result for file paths
  if (Array.isArray(result)) {
    const filePaths = result.filter(r =>
      typeof r === "string" && (r.includes("/") || r.includes("\\"))
    );
    paths.push(...filePaths.slice(0, 10));
  }

  return [...new Set(paths)];
}

main().catch(err => {
  log.error("capture", "Capture hook failed", { error: err.message });
  process.exit(0);  // Exit 0 to avoid blocking Claude Code
});
```

### Stop Hook (Session Summary)

Uses `@anthropic-ai/claude-agent-sdk` to generate session summary.

```typescript
// scripts/summarize.ts
import { query } from "@anthropic-ai/claude-agent-sdk";
import { getDatabase } from "../src/db/database";
import { createMemoryStore } from "../src/services/memory/store";
import { log } from "../src/utils/log";

interface HookInput {
  session_id: string;
  cwd: string;
  transcript_path?: string;
}

// AbortController for cleanup
const abortController = new AbortController();

// Track child process for cleanup
let sdkProcess: any = null;

async function main() {
  const input: HookInput = JSON.parse(await Bun.stdin.text());
  const { session_id, cwd } = input;

  log.info("summarize", "Starting session summary", { session_id });

  // Get session memories
  const db = getDatabase();
  const memories = await db.execute(`
    SELECT content FROM memories
    WHERE session_id = ?
    ORDER BY created_at ASC
    LIMIT 50
  `, [session_id]);

  if (memories.rows.length === 0) {
    log.debug("summarize", "No memories to summarize", { session_id });
    process.exit(0);
  }

  log.debug("summarize", "Found session memories", { session_id, count: memories.rows.length });

  // Build summary prompt
  const observations = memories.rows.map(r => r[0]).join("\n---\n");
  const prompt = `Summarize this Claude Code session. Focus on:
1. What was accomplished
2. Key decisions made
3. Important discoveries about the codebase
4. Any patterns or preferences observed

Session observations:
${observations}

Provide a concise summary (2-4 paragraphs).`;

  try {
    // Use SDK agent to generate summary
    const messageGenerator = async function* () {
      yield {
        type: "user" as const,
        message: { role: "user" as const, content: prompt }
      };
    };

    const result = query({
      prompt: messageGenerator(),
      options: {
        model: "claude-sonnet-4-20250514",
        disallowedTools: [
          "Bash", "Read", "Write", "Edit", "Grep", "Glob",
          "WebFetch", "WebSearch", "Task", "NotebookEdit",
          "AskUserQuestion", "TodoWrite"
        ],
        abortController,
        pathToClaudeCodeExecutable: findClaudeExecutable()
      }
    });

    let summary = "";
    for await (const message of result) {
      if (message.type === "assistant") {
        const content = message.message.content;
        summary = Array.isArray(content)
          ? content.filter((c: any) => c.type === "text").map((c: any) => c.text).join("\n")
          : typeof content === "string" ? content : "";
      }
    }

    if (summary) {
      log.info("summarize", "Summary generated", { session_id, length: summary.length });

      // Store summary as reflective memory (long-lived)
      const store = createMemoryStore();
      const project = await getOrCreateProject(cwd);

      await store.create({
        content: `Session Summary:\n${summary}`,
        sector: "reflective",  // Session summaries are reflective
        tier: "project",
        sessionId: session_id
      }, project.id, session_id);

      // Update session record
      await db.execute(
        "UPDATE sessions SET summary = ?, ended_at = ? WHERE id = ?",
        [summary, Date.now(), session_id]
      );

      log.info("summarize", "Session summary stored", { session_id });
    }
  } catch (err) {
    log.error("summarize", "Summary generation failed", { session_id, error: err.message });
  }

  process.exit(0);
}

function findClaudeExecutable(): string {
  // Similar to claude-mem's approach
  try {
    const result = Bun.spawnSync({
      cmd: ["which", "claude"],
      stdout: "pipe"
    });
    return new TextDecoder().decode(result.stdout).trim();
  } catch {
    return "claude";  // Hope it's in PATH
  }
}

// Handle cleanup
process.on("SIGTERM", () => {
  abortController.abort();
  process.exit(0);
});

process.on("SIGINT", () => {
  abortController.abort();
  process.exit(0);
});

main().catch(err => {
  log.error("summarize", "Summarize hook failed", { error: err.message });
  process.exit(0);
});
```

### SessionEnd Cleanup Hook

Ensures no zombie processes.

```typescript
// scripts/cleanup.ts
import { getDatabase } from "../src/db/database";
import { log } from "../src/utils/log";

interface HookInput {
  session_id: string;
}

async function main() {
  const input: HookInput = JSON.parse(await Bun.stdin.text());
  const { session_id } = input;

  log.info("cleanup", "Starting session cleanup", { session_id });

  // Mark session as ended
  const db = getDatabase();
  await db.execute(
    "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
    [Date.now(), session_id]
  );

  // Promote valuable session memories to project tier
  const promoted = await db.execute(`
    UPDATE memories
    SET tier = 'project'
    WHERE session_id = ? AND tier = 'session' AND salience > 0.7
  `, [session_id]);

  log.debug("cleanup", "Promoted high-salience memories", { session_id, count: promoted.rowsAffected });

  // Close database connection
  db.close();

  log.info("cleanup", "Session cleanup complete", { session_id });
  process.exit(0);
}

main().catch(err => {
  log.error("cleanup", "Cleanup hook failed", { error: err.message });
  process.exit(0);
});
```

## MCP Server

### Server Implementation

```typescript
// src/mcp/server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { createSearchService } from "../services/search/hybrid";
import { createMemoryStore } from "../services/memory/store";
import { createDocumentService } from "../services/documents/ingest";
import { getOrCreateProject } from "../services/project";
import { log } from "../utils/log";

// Redirect console.log to stderr (MCP protocol uses stdout)
console.log = console.error;

const tools = [
  {
    name: "memory_search",
    description: "Search memories by semantic similarity and keywords. Returns relevant memories with session context and superseded status.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        sector: { type: "string", enum: ["episodic", "semantic", "procedural", "emotional", "reflective"], description: "Filter by memory sector" },
        limit: { type: "number", description: "Max results (default: 10)" },
        mode: { type: "string", enum: ["hybrid", "semantic", "keyword"], description: "Search mode" },
        include_superseded: { type: "boolean", description: "Include memories that have been superseded (default: false)" }
      },
      required: ["query"]
    }
  },
  {
    name: "memory_timeline",
    description: "Get chronological context around a memory with session info. Use after search to understand sequence of events.",
    inputSchema: {
      type: "object",
      properties: {
        anchor_id: { type: "string", description: "Memory ID to center timeline on" },
        depth_before: { type: "number", description: "Memories before (default: 5)" },
        depth_after: { type: "number", description: "Memories after (default: 5)" }
      },
      required: ["anchor_id"]
    }
  },
  {
    name: "memory_add",
    description: "Manually add a memory. Use for explicit notes, decisions, or procedures.",
    inputSchema: {
      type: "object",
      properties: {
        content: { type: "string", description: "Memory content" },
        sector: { type: "string", enum: ["episodic", "semantic", "procedural", "emotional", "reflective"], description: "Memory sector (auto-classified if not provided)" },
        tags: { type: "array", items: { type: "string" }, description: "Tags for categorization" },
        importance: { type: "number", description: "Base importance 0-1 (default: 0.5)" }
      },
      required: ["content"]
    }
  },
  {
    name: "memory_reinforce",
    description: "Reinforce a memory, increasing its salience. Use when a memory is relevant and should be remembered longer.",
    inputSchema: {
      type: "object",
      properties: {
        memory_id: { type: "string", description: "Memory ID to reinforce" },
        amount: { type: "number", description: "Reinforcement amount 0-1 (default: 0.1)" }
      },
      required: ["memory_id"]
    }
  },
  {
    name: "memory_deemphasize",
    description: "De-emphasize a memory, reducing its salience. Use when a memory is less relevant or partially incorrect.",
    inputSchema: {
      type: "object",
      properties: {
        memory_id: { type: "string", description: "Memory ID to de-emphasize" },
        amount: { type: "number", description: "De-emphasis amount 0-1 (default: 0.2)" }
      },
      required: ["memory_id"]
    }
  },
  {
    name: "memory_delete",
    description: "Delete a memory. Use soft delete (default) to preserve history, or hard delete to remove completely.",
    inputSchema: {
      type: "object",
      properties: {
        memory_id: { type: "string", description: "Memory ID to delete" },
        hard: { type: "boolean", description: "Permanently delete (default: false, soft delete)" }
      },
      required: ["memory_id"]
    }
  },
  {
    name: "memory_supersede",
    description: "Mark one memory as superseding another. Use when new information replaces old.",
    inputSchema: {
      type: "object",
      properties: {
        old_memory_id: { type: "string", description: "ID of the memory being superseded" },
        new_memory_id: { type: "string", description: "ID of the newer memory that supersedes it" }
      },
      required: ["old_memory_id", "new_memory_id"]
    }
  },
  {
    name: "docs_search",
    description: "Search ingested documents (txt, md files). Separate from memories.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" }
      },
      required: ["query"]
    }
  },
  {
    name: "docs_ingest",
    description: "Ingest a document for searchable reference. Chunks and embeds the content.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to ingest" },
        url: { type: "string", description: "URL to fetch and ingest" },
        content: { type: "string", description: "Raw content to ingest" },
        title: { type: "string", description: "Document title" }
      }
    }
  }
];

async function handleToolCall(name: string, args: any, cwd: string) {
  const start = Date.now();
  log.debug("mcp", "Tool call received", { name, cwd });

  const project = await getOrCreateProject(cwd);
  const search = createSearchService();
  const store = createMemoryStore();
  const docs = createDocumentService();

  switch (name) {
    case "memory_search": {
      const results = await search.search({
        query: args.query,
        projectId: project.id,
        sector: args.sector,
        limit: args.limit || 10,
        mode: args.mode || "hybrid",
        includeSuperseded: args.include_superseded || false,
      });

      return formatSearchResults(results);
    }

    case "memory_timeline": {
      const timeline = await search.timeline(
        args.anchor_id,
        args.depth_before || 5,
        args.depth_after || 5
      );

      return formatTimeline(timeline);
    }

    case "memory_add": {
      const memory = await store.create({
        content: args.content,
        sector: args.sector,
        tags: args.tags,
        importance: args.importance,
        tier: "project",
      }, project.id);

      return `Memory created: ${memory.id} (sector: ${memory.sector}, salience: ${memory.salience})`;
    }

    case "memory_reinforce": {
      const memory = await store.reinforce(args.memory_id, args.amount || 0.1);
      return `Memory reinforced: ${memory.id} (new salience: ${memory.salience.toFixed(2)})`;
    }

    case "memory_deemphasize": {
      const memory = await store.deemphasize(args.memory_id, args.amount || 0.2);
      return `Memory de-emphasized: ${memory.id} (new salience: ${memory.salience.toFixed(2)})`;
    }

    case "memory_delete": {
      await store.delete(args.memory_id, args.hard || false);
      return args.hard
        ? `Memory permanently deleted: ${args.memory_id}`
        : `Memory soft-deleted: ${args.memory_id} (can be restored)`;
    }

    case "memory_supersede": {
      await supersede(args.old_memory_id, args.new_memory_id);
      return `Memory ${args.old_memory_id} marked as superseded by ${args.new_memory_id}`;
    }

    case "docs_search": {
      const results = await docs.search(args.query, project.id, args.limit || 5);
      return formatDocResults(results);
    }

    case "docs_ingest": {
      const doc = await docs.ingest({
        projectId: project.id,
        path: args.path,
        url: args.url,
        content: args.content,
        title: args.title,
      });

      return `Document ingested: ${doc.title || doc.id}`;
    }

    default:
      log.warn("mcp", "Unknown tool requested", { name });
      throw new Error(`Unknown tool: ${name}`);
  }
}

function logToolResult(name: string, start: number, resultPreview: string) {
  log.info("mcp", "Tool call completed", { name, ms: Date.now() - start, resultPreview: resultPreview.slice(0, 50) });
}

function formatSearchResults(results: SearchResult[]): string {
  if (results.length === 0) return "No memories found.";

  return results.map((r, i) => {
    const mem = r.memory;
    const lines = [
      `[${i + 1}] (${mem.sector}, score: ${r.score.toFixed(2)}, salience: ${mem.salience.toFixed(2)})`,
      `ID: ${mem.id}`,
    ];

    if (r.isSuperseded && r.supersededBy) {
      lines.push(`⚠️ SUPERSEDED by: ${r.supersededBy.id}`);
    }

    if (r.sourceSession) {
      const sessionDate = new Date(r.sourceSession.startedAt).toISOString().slice(0, 16);
      lines.push(`Session: ${sessionDate}${r.sourceSession.summary ? ` - ${r.sourceSession.summary.slice(0, 50)}...` : ''}`);
    }

    if (r.relatedMemoryCount > 0) {
      lines.push(`Related: ${r.relatedMemoryCount} memories`);
    }

    lines.push(`Content: ${mem.content.slice(0, 300)}${mem.content.length > 300 ? '...' : ''}`);

    return lines.join('\n');
  }).join('\n\n---\n\n');
}

function formatTimeline(timeline: TimelineResult): string {
  const { anchor, before, after, sessions } = timeline;
  const allMemories = [...before, anchor, ...after];

  const lines = ['Timeline:', ''];

  for (const m of allMemories) {
    const marker = m.id === anchor.id ? '>>>' : '   ';
    const date = new Date(m.createdAt).toISOString().slice(0, 16);
    const supersededMark = m.validUntil ? ' [SUPERSEDED]' : '';
    lines.push(`${marker} [${date}] (${m.sector})${supersededMark}`);
    lines.push(`    ${m.content.slice(0, 200)}`);
    lines.push('');
  }

  if (sessions.size > 0) {
    lines.push('Sessions in timeline:');
    for (const [id, session] of sessions) {
      const sessionDate = new Date(session.startedAt).toISOString().slice(0, 16);
      lines.push(`  - ${sessionDate}: ${session.summary || 'No summary'}`);
    }
  }

  return lines.join('\n');
}

function formatDocResults(results: DocumentSearchResult[]): string {
  if (results.length === 0) return "No documents found.";

  return results.map((r, i) => {
    return `[${i + 1}] ${r.document.title || "Untitled"} (score: ${r.score.toFixed(2)})
Source: ${r.document.sourcePath || r.document.sourceUrl || "inline"}
Match: ${r.chunk.content.slice(0, 200)}...`;
  }).join("\n\n");
}

// Create server
const server = new Server(
  { name: "ccmemory", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// List tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: tools.map(t => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema
  }))
}));

// Call tools
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const cwd = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const start = Date.now();

  try {
    const result = await handleToolCall(name, args || {}, cwd);
    log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
    return {
      content: [{ type: "text", text: result }]
    };
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    log.error("mcp", "Tool call failed", { name, error: errMsg, ms: Date.now() - start });
    return {
      content: [{ type: "text", text: `Error: ${errMsg}` }],
      isError: true
    };
  }
});

// Start server
async function main() {
  log.info("mcp", "Starting MCP server", { pid: process.pid });
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log.info("mcp", "MCP server connected");
}

main().catch(err => {
  log.error("mcp", "MCP server error", { error: err.message });
  process.exit(1);
});
```

## Test Specifications

### Hook Tests (Integration - tests/hooks/)

```typescript
// tests/hooks/capture.test.ts
import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import { getDatabase } from "../../src/db/database";

describe("Capture Hook", () => {
  beforeEach(async () => {
    // Setup test database
  });

  afterEach(async () => {
    // Cleanup test data
  });

  test("captures tool observations", async () => {
    const input = {
      session_id: "test-session",
      cwd: "/test/project",
      tool_name: "Read",
      tool_input: { file_path: "/test/file.ts" },
      tool_result: "file contents"
    };

    const proc = Bun.spawn({
      cmd: ["bun", "scripts/capture.ts"],
      stdin: "pipe",
      stdout: "pipe"
    });

    proc.stdin.write(JSON.stringify(input));
    proc.stdin.end();
    await proc.exited;

    const db = getDatabase();
    const memories = await db.execute(
      "SELECT * FROM memories WHERE session_id = ?",
      ["test-session"]
    );
    expect(memories.rows.length).toBe(1);
  });

  test("skips large tool results", async () => {
    const input = {
      session_id: "test-session",
      cwd: "/test/project",
      tool_name: "Read",
      tool_input: { file_path: "/test/file.ts" },
      tool_result: "x".repeat(20000)  // > 10KB
    };

    const proc = Bun.spawn({
      cmd: ["bun", "scripts/capture.ts"],
      stdin: "pipe",
      stdout: "pipe"
    });

    proc.stdin.write(JSON.stringify(input));
    proc.stdin.end();
    await proc.exited;

    const db = getDatabase();
    const memories = await db.execute(
      "SELECT * FROM memories WHERE session_id = ?",
      ["test-session"]
    );
    expect(memories.rows.length).toBe(0);
  });

  test("links memory to session", async () => {
    const input = {
      session_id: "test-session-link",
      cwd: "/test/project",
      tool_name: "Write",
      tool_input: { file_path: "/test/new.ts" },
      tool_result: "ok"
    };

    const proc = Bun.spawn({
      cmd: ["bun", "scripts/capture.ts"],
      stdin: "pipe",
      stdout: "pipe"
    });

    proc.stdin.write(JSON.stringify(input));
    proc.stdin.end();
    await proc.exited;

    const db = getDatabase();
    const links = await db.execute(
      "SELECT * FROM session_memories WHERE session_id = ?",
      ["test-session-link"]
    );
    expect(links.rows.length).toBe(1);
    expect(links.rows[0].usage_type).toBe("created");
  });
});

// tests/hooks/summarize.test.ts
describe("Summarize Hook", () => {
  test("generates session summary", async () => {
    // This would require mocking the SDK agent
    // Focus on integration test with actual SDK
  });

  test("handles abort signal", async () => {
    // Verify cleanup on SIGTERM
  });
});
```

### MCP Server Tests (Colocated - src/mcp/server.test.ts)

```typescript
// src/mcp/__test__/server.test.ts
import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import { getDatabase } from "../db/database";
import { createMemoryStore } from "../services/memory/store";

describe("MCP Server", () => {
  let store: ReturnType<typeof createMemoryStore>;

  beforeEach(async () => {
    store = createMemoryStore();
    // Setup test database
  });

  afterEach(async () => {
    // Cleanup test data
  });

  test("lists all tools", async () => {
    const response = await server.handleRequest({
      method: "tools/list",
      params: {}
    });

    expect(response.tools.length).toBe(9);  // Updated for new tools
    expect(response.tools.map(t => t.name)).toEqual([
      "memory_search",
      "memory_timeline",
      "memory_add",
      "memory_reinforce",
      "memory_deemphasize",
      "memory_delete",
      "memory_supersede",
      "docs_search",
      "docs_ingest"
    ]);
  });

  test("memory_search returns results with session context", async () => {
    await store.create({ content: "Test memory about React", sector: "semantic" }, "proj1");

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_search",
        arguments: { query: "React", limit: 5 }
      }
    });

    expect(response.content[0].text).toContain("React");
    expect(response.content[0].text).toContain("semantic");
  });

  test("memory_search excludes superseded by default", async () => {
    const old = await store.create({ content: "Old API endpoint" }, "proj1");
    const newer = await store.create({ content: "New API endpoint" }, "proj1");
    await supersede(old.id, newer.id);

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_search",
        arguments: { query: "API endpoint" }
      }
    });

    expect(response.content[0].text).not.toContain(old.id);
  });

  test("memory_search includes superseded when requested", async () => {
    const old = await store.create({ content: "Old API endpoint" }, "proj1");
    const newer = await store.create({ content: "New API endpoint" }, "proj1");
    await supersede(old.id, newer.id);

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_search",
        arguments: { query: "API endpoint", include_superseded: true }
      }
    });

    expect(response.content[0].text).toContain("SUPERSEDED");
  });

  test("memory_add creates memory with sector", async () => {
    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_add",
        arguments: {
          content: "User prefers TypeScript",
          sector: "emotional"
        }
      }
    });

    expect(response.content[0].text).toContain("Memory created");
    expect(response.content[0].text).toContain("emotional");
  });

  test("memory_reinforce increases salience", async () => {
    const memory = await store.create({ content: "Important fact" }, "proj1");
    const initialSalience = memory.salience;

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_reinforce",
        arguments: { memory_id: memory.id, amount: 0.2 }
      }
    });

    expect(response.content[0].text).toContain("reinforced");
    const updated = await store.get(memory.id);
    expect(updated.salience).toBeGreaterThan(initialSalience);
  });

  test("memory_reinforce has diminishing returns", async () => {
    const memory = await store.create({ content: "High salience fact" }, "proj1");

    // Reinforce multiple times
    for (let i = 0; i < 5; i++) {
      await server.handleRequest({
        method: "tools/call",
        params: {
          name: "memory_reinforce",
          arguments: { memory_id: memory.id, amount: 0.2 }
        }
      });
    }

    const updated = await store.get(memory.id);
    expect(updated.salience).toBeLessThan(1.0);  // Never reaches 1.0
    expect(updated.salience).toBeGreaterThan(0.8);  // But gets close
  });

  test("memory_deemphasize reduces salience", async () => {
    const memory = await store.create({ content: "Less important" }, "proj1");
    const initialSalience = memory.salience;

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_deemphasize",
        arguments: { memory_id: memory.id, amount: 0.3 }
      }
    });

    expect(response.content[0].text).toContain("de-emphasized");
    const updated = await store.get(memory.id);
    expect(updated.salience).toBeLessThan(initialSalience);
  });

  test("memory_delete soft deletes by default", async () => {
    const memory = await store.create({ content: "To be deleted" }, "proj1");

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_delete",
        arguments: { memory_id: memory.id }
      }
    });

    expect(response.content[0].text).toContain("soft-deleted");
    expect(response.content[0].text).toContain("can be restored");

    // Memory should still exist but be marked deleted
    const db = getDatabase();
    const result = await db.execute(
      "SELECT is_deleted FROM memories WHERE id = ?",
      [memory.id]
    );
    expect(result.rows[0].is_deleted).toBe(1);
  });

  test("memory_delete hard deletes when requested", async () => {
    const memory = await store.create({ content: "To be permanently deleted" }, "proj1");

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_delete",
        arguments: { memory_id: memory.id, hard: true }
      }
    });

    expect(response.content[0].text).toContain("permanently deleted");

    const db = getDatabase();
    const result = await db.execute(
      "SELECT * FROM memories WHERE id = ?",
      [memory.id]
    );
    expect(result.rows.length).toBe(0);
  });

  test("memory_supersede creates relationship", async () => {
    const old = await store.create({ content: "Old approach" }, "proj1");
    const newer = await store.create({ content: "New approach" }, "proj1");

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_supersede",
        arguments: { old_memory_id: old.id, new_memory_id: newer.id }
      }
    });

    expect(response.content[0].text).toContain("superseded");

    // Verify relationship created
    const db = getDatabase();
    const rels = await db.execute(
      "SELECT * FROM memory_relationships WHERE source_id = ? AND target_id = ? AND relationship_type = 'SUPERSEDES'",
      [newer.id, old.id]
    );
    expect(rels.rows.length).toBe(1);

    // Verify old memory has valid_until set
    const oldMem = await db.execute("SELECT valid_until FROM memories WHERE id = ?", [old.id]);
    expect(oldMem.rows[0].valid_until).not.toBeNull();
  });

  test("memory_timeline includes session info", async () => {
    // Create session
    const db = getDatabase();
    await db.execute(
      "INSERT INTO sessions (id, project_id, started_at, summary) VALUES (?, ?, ?, ?)",
      ["sess1", "proj1", Date.now() - 3600000, "Test session"]
    );

    // Create memories in session
    const m1 = await store.create({ content: "First action" }, "proj1", "sess1");
    const m2 = await store.create({ content: "Second action" }, "proj1", "sess1");

    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "memory_timeline",
        arguments: { anchor_id: m2.id }
      }
    });

    expect(response.content[0].text).toContain("Timeline:");
    expect(response.content[0].text).toContain("Sessions in timeline:");
    expect(response.content[0].text).toContain("Test session");
  });

  test("docs_ingest chunks document", async () => {
    const response = await server.handleRequest({
      method: "tools/call",
      params: {
        name: "docs_ingest",
        arguments: {
          content: "Long document content...",
          title: "Test Doc"
        }
      }
    });

    expect(response.content[0].text).toContain("Document ingested");
  });
});
```

## Process Lifecycle

### Avoiding Zombie Processes

Key differences from claude-mem:

1. **No daemon**: Hooks are short-lived scripts that exit
2. **Direct DB access**: No HTTP worker to manage
3. **AbortController**: SDK agent can be aborted on cleanup
4. **Exit 0 always**: Never block Claude Code

### Concurrent Instance Safety

1. **WAL mode**: Multiple sessions can read/write simultaneously
2. **No ports**: stdio MCP server, no port conflicts
3. **Session isolation**: Memories tagged with session_id

## Acceptance Criteria

### Hook Scripts

- [ ] PostToolUse captures tool observations silently (no context pollution)
- [ ] PostToolUse links memories to sessions with usage_type
- [ ] PostToolUse skips tool results > 10KB
- [ ] Stop hook generates session summary using claude-agent-sdk
- [ ] Stop hook respects AbortController for cleanup
- [ ] SessionEnd hook promotes valuable memories (salience > 0.7) to project tier
- [ ] SessionEnd hook closes database connections
- [ ] No zombie processes after session end

### MCP Server

- [ ] MCP server starts with Claude Code via stdio
- [ ] memory_search finds relevant memories with hybrid search
- [ ] memory_search returns session context (date, summary)
- [ ] memory_search excludes superseded memories by default
- [ ] memory_search includes superseded when include_superseded=true
- [ ] memory_timeline shows chronological context around anchor
- [ ] memory_timeline includes session summaries in results
- [ ] memory_add creates manual memories with auto-classification
- [ ] memory_reinforce increases salience with diminishing returns
- [ ] memory_deemphasize reduces salience
- [ ] memory_delete soft deletes by default (preserves history)
- [ ] memory_delete hard deletes when requested
- [ ] memory_supersede creates SUPERSEDES relationship
- [ ] memory_supersede sets valid_until on old memory
- [ ] docs_search finds document content
- [ ] docs_ingest chunks and embeds documents

### Concurrency & Lifecycle

- [ ] Multiple concurrent sessions work (WAL mode)
- [ ] No port conflicts (stdio transport)
- [ ] Session isolation via session_id
