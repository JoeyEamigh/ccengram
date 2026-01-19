import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { createSearchService, type SearchResult } from "../services/search/hybrid.js";
import { createMemoryStore } from "../services/memory/store.js";
import { createDocumentService, type DocumentSearchResult } from "../services/documents/ingest.js";
import { createEmbeddingService } from "../services/embedding/index.js";
import { getOrCreateProject } from "../services/project.js";
import { supersede } from "../services/memory/relationships.js";
import { getToolsConfig, filterTools, type ToolsConfig } from "./tools-config.js";
import { log } from "../utils/log.js";
import type { TimelineResult } from "../services/search/hybrid.js";

console.log = console.error;

let cachedToolsConfig: ToolsConfig | null = null;

const TOOLS = [
  {
    name: "memory_search",
    description:
      "Search memories by semantic similarity and keywords. Returns relevant memories with session context and superseded status.",
    inputSchema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "Search query" },
        sector: {
          type: "string",
          enum: ["episodic", "semantic", "procedural", "emotional", "reflective"],
          description: "Filter by memory sector",
        },
        limit: { type: "number", description: "Max results (default: 10)" },
        mode: {
          type: "string",
          enum: ["hybrid", "semantic", "keyword"],
          description: "Search mode",
        },
        include_superseded: {
          type: "boolean",
          description: "Include memories that have been superseded (default: false)",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "memory_timeline",
    description:
      "Get chronological context around a memory with session info. Use after search to understand sequence of events.",
    inputSchema: {
      type: "object" as const,
      properties: {
        anchor_id: {
          type: "string",
          description: "Memory ID to center timeline on",
        },
        depth_before: {
          type: "number",
          description: "Memories before (default: 5)",
        },
        depth_after: {
          type: "number",
          description: "Memories after (default: 5)",
        },
      },
      required: ["anchor_id"],
    },
  },
  {
    name: "memory_add",
    description:
      "Manually add a memory. Use for explicit notes, decisions, or procedures.",
    inputSchema: {
      type: "object" as const,
      properties: {
        content: { type: "string", description: "Memory content" },
        sector: {
          type: "string",
          enum: ["episodic", "semantic", "procedural", "emotional", "reflective"],
          description: "Memory sector (auto-classified if not provided)",
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Tags for categorization",
        },
        importance: {
          type: "number",
          description: "Base importance 0-1 (default: 0.5)",
        },
      },
      required: ["content"],
    },
  },
  {
    name: "memory_reinforce",
    description:
      "Reinforce a memory, increasing its salience. Use when a memory is relevant and should be remembered longer.",
    inputSchema: {
      type: "object" as const,
      properties: {
        memory_id: { type: "string", description: "Memory ID to reinforce" },
        amount: {
          type: "number",
          description: "Reinforcement amount 0-1 (default: 0.1)",
        },
      },
      required: ["memory_id"],
    },
  },
  {
    name: "memory_deemphasize",
    description:
      "De-emphasize a memory, reducing its salience. Use when a memory is less relevant or partially incorrect.",
    inputSchema: {
      type: "object" as const,
      properties: {
        memory_id: { type: "string", description: "Memory ID to de-emphasize" },
        amount: {
          type: "number",
          description: "De-emphasis amount 0-1 (default: 0.2)",
        },
      },
      required: ["memory_id"],
    },
  },
  {
    name: "memory_delete",
    description:
      "Delete a memory. Use soft delete (default) to preserve history, or hard delete to remove completely.",
    inputSchema: {
      type: "object" as const,
      properties: {
        memory_id: { type: "string", description: "Memory ID to delete" },
        hard: {
          type: "boolean",
          description: "Permanently delete (default: false, soft delete)",
        },
      },
      required: ["memory_id"],
    },
  },
  {
    name: "memory_supersede",
    description:
      "Mark one memory as superseding another. Use when new information replaces old.",
    inputSchema: {
      type: "object" as const,
      properties: {
        old_memory_id: {
          type: "string",
          description: "ID of the memory being superseded",
        },
        new_memory_id: {
          type: "string",
          description: "ID of the newer memory that supersedes it",
        },
      },
      required: ["old_memory_id", "new_memory_id"],
    },
  },
  {
    name: "docs_search",
    description:
      "Search ingested documents (txt, md files). Separate from memories.",
    inputSchema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "docs_ingest",
    description:
      "Ingest a document for searchable reference. Chunks and embeds the content.",
    inputSchema: {
      type: "object" as const,
      properties: {
        path: { type: "string", description: "File path to ingest" },
        url: { type: "string", description: "URL to fetch and ingest" },
        content: { type: "string", description: "Raw content to ingest" },
        title: { type: "string", description: "Document title" },
      },
    },
  },
];

type ToolArgs = {
  query?: string;
  sector?: string;
  limit?: number;
  mode?: string;
  include_superseded?: boolean;
  anchor_id?: string;
  depth_before?: number;
  depth_after?: number;
  content?: string;
  tags?: string[];
  importance?: number;
  memory_id?: string;
  amount?: number;
  hard?: boolean;
  old_memory_id?: string;
  new_memory_id?: string;
  path?: string;
  url?: string;
  title?: string;
};

async function handleToolCall(
  name: string,
  args: ToolArgs,
  cwd: string
): Promise<string> {
  const start = Date.now();
  log.debug("mcp", "Tool call received", { name, cwd });

  const project = await getOrCreateProject(cwd);
  const embeddingService = await createEmbeddingService();
  const search = createSearchService(embeddingService);
  const store = createMemoryStore();
  const docs = createDocumentService(embeddingService);

  switch (name) {
    case "memory_search": {
      if (!args.query) throw new Error("query is required");
      const results = await search.search({
        query: args.query,
        projectId: project.id,
        sector: args.sector as
          | "episodic"
          | "semantic"
          | "procedural"
          | "emotional"
          | "reflective"
          | undefined,
        limit: args.limit ?? 10,
        mode: (args.mode as "hybrid" | "semantic" | "keyword") ?? "hybrid",
        includeSuperseded: args.include_superseded ?? false,
      });
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return formatSearchResults(results);
    }

    case "memory_timeline": {
      if (!args.anchor_id) throw new Error("anchor_id is required");
      const timeline = await search.timeline(
        args.anchor_id,
        args.depth_before ?? 5,
        args.depth_after ?? 5
      );
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return formatTimeline(timeline);
    }

    case "memory_add": {
      if (!args.content) throw new Error("content is required");
      const memory = await store.create(
        {
          content: args.content,
          sector: args.sector as
            | "episodic"
            | "semantic"
            | "procedural"
            | "emotional"
            | "reflective"
            | undefined,
          tags: args.tags,
          importance: args.importance,
          tier: "project",
        },
        project.id
      );
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return `Memory created: ${memory.id} (sector: ${memory.sector}, salience: ${memory.salience})`;
    }

    case "memory_reinforce": {
      if (!args.memory_id) throw new Error("memory_id is required");
      const memory = await store.reinforce(args.memory_id, args.amount ?? 0.1);
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return `Memory reinforced: ${memory.id} (new salience: ${memory.salience.toFixed(2)})`;
    }

    case "memory_deemphasize": {
      if (!args.memory_id) throw new Error("memory_id is required");
      const memory = await store.deemphasize(args.memory_id, args.amount ?? 0.2);
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return `Memory de-emphasized: ${memory.id} (new salience: ${memory.salience.toFixed(2)})`;
    }

    case "memory_delete": {
      if (!args.memory_id) throw new Error("memory_id is required");
      await store.delete(args.memory_id, args.hard ?? false);
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return args.hard
        ? `Memory permanently deleted: ${args.memory_id}`
        : `Memory soft-deleted: ${args.memory_id} (can be restored)`;
    }

    case "memory_supersede": {
      if (!args.old_memory_id) throw new Error("old_memory_id is required");
      if (!args.new_memory_id) throw new Error("new_memory_id is required");
      await supersede(args.old_memory_id, args.new_memory_id);
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return `Memory ${args.old_memory_id} marked as superseded by ${args.new_memory_id}`;
    }

    case "docs_search": {
      if (!args.query) throw new Error("query is required");
      const results = await docs.search(args.query, project.id, args.limit ?? 5);
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
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
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return `Document ingested: ${doc.title ?? doc.id}`;
    }

    default:
      log.warn("mcp", "Unknown tool requested", { name });
      throw new Error(`Unknown tool: ${name}`);
  }
}

function formatSearchResults(results: SearchResult[]): string {
  if (results.length === 0) return "No memories found.";

  return results
    .map((r, i) => {
      const mem = r.memory;
      const lines = [
        `[${i + 1}] (${mem.sector}, score: ${r.score.toFixed(2)}, salience: ${mem.salience.toFixed(2)})`,
        `ID: ${mem.id}`,
      ];

      if (r.isSuperseded && r.supersededBy) {
        lines.push(`⚠️ SUPERSEDED by: ${r.supersededBy.id}`);
      }

      if (r.sourceSession) {
        const sessionDate = new Date(r.sourceSession.startedAt)
          .toISOString()
          .slice(0, 16);
        lines.push(
          `Session: ${sessionDate}${r.sourceSession.summary ? ` - ${r.sourceSession.summary.slice(0, 50)}...` : ""}`
        );
      }

      if (r.relatedMemoryCount > 0) {
        lines.push(`Related: ${r.relatedMemoryCount} memories`);
      }

      lines.push(
        `Content: ${mem.content.slice(0, 300)}${mem.content.length > 300 ? "..." : ""}`
      );

      return lines.join("\n");
    })
    .join("\n\n---\n\n");
}

function formatTimeline(timeline: TimelineResult): string {
  const { anchor, before, after, sessions } = timeline;
  const allMemories = [...before, anchor, ...after];

  const lines = ["Timeline:", ""];

  for (const m of allMemories) {
    const marker = m.id === anchor.id ? ">>>" : "   ";
    const date = new Date(m.createdAt).toISOString().slice(0, 16);
    const supersededMark = m.validUntil ? " [SUPERSEDED]" : "";
    lines.push(`${marker} [${date}] (${m.sector})${supersededMark}`);
    lines.push(`    ${m.content.slice(0, 200)}`);
    lines.push("");
  }

  if (sessions.size > 0) {
    lines.push("Sessions in timeline:");
    for (const [, session] of sessions) {
      const sessionDate = new Date(session.startedAt).toISOString().slice(0, 16);
      lines.push(`  - ${sessionDate}: ${session.summary ?? "No summary"}`);
    }
  }

  return lines.join("\n");
}

function formatDocResults(results: DocumentSearchResult[]): string {
  if (results.length === 0) return "No documents found.";

  return results
    .map((r, i) => {
      return `[${i + 1}] ${r.document.title ?? "Untitled"} (score: ${r.score.toFixed(2)})
Source: ${r.document.sourcePath ?? r.document.sourceUrl ?? "inline"}
Match: ${r.chunk.content.slice(0, 200)}...`;
    })
    .join("\n\n");
}

export async function runMcpServer(): Promise<void> {
  const cwd = process.env["CLAUDE_PROJECT_DIR"] ?? process.cwd();

  // Load tool configuration (cached for the session)
  if (!cachedToolsConfig) {
    cachedToolsConfig = await getToolsConfig(cwd);
    log.info("mcp", "Tool configuration loaded", {
      mode: cachedToolsConfig.mode,
      enabledTools:
        cachedToolsConfig.mode === "custom"
          ? cachedToolsConfig.enabledTools
          : undefined,
    });
  }

  const server = new Server(
    { name: "ccmemory", version: "1.0.0" },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const filteredTools = filterTools(TOOLS, cachedToolsConfig!);
    return {
      tools: filteredTools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema,
      })),
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const start = Date.now();

    // Check if tool is enabled
    const filteredTools = filterTools(TOOLS, cachedToolsConfig!);
    const isEnabled = filteredTools.some((t) => t.name === name);

    if (!isEnabled) {
      log.warn("mcp", "Tool call rejected - not enabled", {
        name,
        mode: cachedToolsConfig!.mode,
      });
      return {
        content: [
          {
            type: "text" as const,
            text: `Error: Tool '${name}' is not enabled in current configuration (mode: ${cachedToolsConfig!.mode})`,
          },
        ],
        isError: true,
      };
    }

    try {
      const result = await handleToolCall(name, (args ?? {}) as ToolArgs, cwd);
      log.info("mcp", "Tool call completed", { name, ms: Date.now() - start });
      return {
        content: [{ type: "text" as const, text: result }],
      };
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      log.error("mcp", "Tool call failed", {
        name,
        error: errMsg,
        ms: Date.now() - start,
      });
      return {
        content: [{ type: "text" as const, text: `Error: ${errMsg}` }],
        isError: true,
      };
    }
  });

  log.info("mcp", "Starting MCP server", { pid: process.pid });
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log.info("mcp", "MCP server connected");
}
