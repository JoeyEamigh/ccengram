import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { createCodeIndexService } from '../services/codeindex/index.js';
import type { CodeLanguage, CodeSearchResult } from '../services/codeindex/types.js';
import { createDocumentService, type DocumentSearchResult } from '../services/documents/ingest.js';
import { createEmbeddingService } from '../services/embedding/index.js';
import { supersede } from '../services/memory/relationships.js';
import { createMemoryStore } from '../services/memory/store.js';
import { getOrCreateProject } from '../services/project.js';
import type { TimelineResult } from '../services/search/hybrid.js';
import { createSearchService, type SearchResult } from '../services/search/hybrid.js';
import { log } from '../utils/log.js';
import {
  validateArray,
  validateOptionalEnum,
  validateOptionalNumber,
  validateOptionalString,
  validateString,
} from '../utils/validate.js';

console.log = console.error;

const TOOLS = [
  {
    name: 'memory_search',
    description:
      'Search memories by semantic similarity and keywords. Returns relevant memories with session context and superseded status.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        query: { type: 'string', description: 'Search query' },
        sector: {
          type: 'string',
          enum: ['episodic', 'semantic', 'procedural', 'emotional', 'reflective'],
          description: 'Filter by memory sector',
        },
        limit: { type: 'number', description: 'Max results (default: 10)' },
        include_superseded: {
          type: 'boolean',
          description: 'Include memories that have been superseded (default: false)',
        },
        scope_path: { type: 'string', description: 'Filter by scope path (e.g., "src/services")' },
        scope_module: { type: 'string', description: 'Filter by scope module (e.g., "auth")' },
        memory_type: {
          type: 'string',
          enum: ['preference', 'codebase', 'decision', 'gotcha', 'pattern', 'turn_summary', 'task_completion'],
          description: 'Filter by extracted memory type',
        },
        mode: {
          type: 'string',
          enum: ['hybrid', 'semantic', 'keyword'],
          description: 'Search mode',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'memory_timeline',
    description:
      'Get chronological context around a memory with session info. Use after search to understand sequence of events.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        anchor_id: {
          type: 'string',
          description: 'Memory ID to center timeline on',
        },
        depth_before: {
          type: 'number',
          description: 'Memories before (default: 5)',
        },
        depth_after: {
          type: 'number',
          description: 'Memories after (default: 5)',
        },
      },
      required: ['anchor_id'],
    },
  },
  {
    name: 'memory_add',
    description: 'Manually add a memory. Use for explicit notes, decisions, preferences, or procedures.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        content: { type: 'string', description: 'Memory content' },
        sector: {
          type: 'string',
          enum: ['episodic', 'semantic', 'procedural', 'emotional', 'reflective'],
          description: 'Memory sector (auto-classified if not provided)',
        },
        type: {
          type: 'string',
          enum: ['preference', 'codebase', 'decision', 'gotcha', 'pattern', 'turn_summary', 'task_completion'],
          description: 'Memory type (determines sector automatically)',
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Tags for categorization',
        },
        importance: {
          type: 'number',
          description: 'Base importance 0-1 (default: 0.5)',
        },
        context: { type: 'string', description: 'Context of how this was discovered or why it matters' },
        scope_path: { type: 'string', description: 'Scope path for this memory (e.g., "src/services")' },
        scope_module: { type: 'string', description: 'Scope module for this memory (e.g., "auth")' },
      },
      required: ['content'],
    },
  },
  {
    name: 'memory_reinforce',
    description:
      'Reinforce a memory, increasing its salience. Use when a memory is relevant and should be remembered longer.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        memory_id: { type: 'string', description: 'Memory ID to reinforce' },
        amount: {
          type: 'number',
          description: 'Reinforcement amount 0-1 (default: 0.1)',
        },
      },
      required: ['memory_id'],
    },
  },
  {
    name: 'memory_deemphasize',
    description:
      'De-emphasize a memory, reducing its salience. Use when a memory is less relevant or partially incorrect.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        memory_id: { type: 'string', description: 'Memory ID to de-emphasize' },
        amount: {
          type: 'number',
          description: 'De-emphasis amount 0-1 (default: 0.2)',
        },
      },
      required: ['memory_id'],
    },
  },
  {
    name: 'memory_delete',
    description: 'Delete a memory. Use soft delete (default) to preserve history, or hard delete to remove completely.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        memory_id: { type: 'string', description: 'Memory ID to delete' },
        hard: {
          type: 'boolean',
          description: 'Permanently delete (default: false, soft delete)',
        },
      },
      required: ['memory_id'],
    },
  },
  {
    name: 'memory_supersede',
    description: 'Mark one memory as superseding another. Use when new information replaces old.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        old_memory_id: {
          type: 'string',
          description: 'ID of the memory being superseded',
        },
        new_memory_id: {
          type: 'string',
          description: 'ID of the newer memory that supersedes it',
        },
      },
      required: ['old_memory_id', 'new_memory_id'],
    },
  },
  {
    name: 'docs_search',
    description: 'Search ingested documents (txt, md files). Separate from memories.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        query: { type: 'string', description: 'Search query' },
        limit: { type: 'number', description: 'Max results (default: 5)' },
      },
      required: ['query'],
    },
  },
  {
    name: 'docs_ingest',
    description: 'Ingest a document for searchable reference. Chunks and embeds the content.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: { type: 'string', description: 'File path to ingest' },
        url: { type: 'string', description: 'URL to fetch and ingest' },
        content: { type: 'string', description: 'Raw content to ingest' },
        title: { type: 'string', description: 'Document title' },
      },
    },
  },
  {
    name: 'code_search',
    description:
      'Search indexed code by semantic similarity. Returns snippets with file paths and line numbers. Project code must be indexed first using code_index or the watcher.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        query: { type: 'string', description: 'Search query describing what code you are looking for' },
        language: {
          type: 'string',
          description: 'Filter by programming language (ts, js, py, go, rs, java, etc.)',
        },
        limit: { type: 'number', description: 'Max results (default: 10)' },
      },
      required: ['query'],
    },
  },
  {
    name: 'code_index',
    description: 'Index or re-index project code files for semantic search. Respects .gitignore.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        force: { type: 'boolean', description: 'Re-index all files even if unchanged (default: false)' },
        dry_run: { type: 'boolean', description: 'Scan only, report files without indexing (default: false)' },
      },
    },
  },
];

type ToolArgs = {
  query?: string;
  sector?: string;
  limit?: number;
  include_superseded?: boolean;
  scope_path?: string;
  scope_module?: string;
  memory_type?: string;
  mode?: string;
  anchor_id?: string;
  depth_before?: number;
  depth_after?: number;
  content?: string;
  type?: string;
  tags?: string[];
  importance?: number;
  context?: string;
  memory_id?: string;
  amount?: number;
  hard?: boolean;
  old_memory_id?: string;
  new_memory_id?: string;
  path?: string;
  url?: string;
  title?: string;
  language?: string;
  force?: boolean;
  dry_run?: boolean;
};

const MEMORY_SECTORS = ['episodic', 'semantic', 'procedural', 'emotional', 'reflective'] as const;
type MemorySector = (typeof MEMORY_SECTORS)[number];

const MEMORY_TYPES = ['preference', 'codebase', 'decision', 'gotcha', 'pattern', 'turn_summary', 'task_completion'] as const;
type MemoryType = (typeof MEMORY_TYPES)[number];

const SEARCH_MODES = ['hybrid', 'semantic', 'keyword'] as const;
type SearchMode = (typeof SEARCH_MODES)[number];

const MAX_QUERY_LENGTH = 10000;
const MAX_CONTENT_LENGTH = 100000;
const MAX_LIMIT = 1000;
const MAX_DEPTH = 100;

type ValidatedArgs = {
  query?: string;
  sector?: MemorySector;
  limit?: number;
  includeSuperseded?: boolean;
  scopePath?: string;
  scopeModule?: string;
  memoryTypeFilter?: MemoryType;
  mode?: SearchMode;
  anchorId?: string;
  depthBefore?: number;
  depthAfter?: number;
  content?: string;
  memoryType?: MemoryType;
  context?: string;
  tags?: string[];
  importance?: number;
  memoryId?: string;
  amount?: number;
  hard?: boolean;
  oldMemoryId?: string;
  newMemoryId?: string;
  path?: string;
  url?: string;
  title?: string;
  language?: string;
  force?: boolean;
  dryRun?: boolean;
};

function validateToolArgs(name: string, args: ToolArgs): ValidatedArgs {
  const validated: ValidatedArgs = {};

  switch (name) {
    case 'memory_search':
      validated.query = validateString(args.query, 'query', { maxLength: MAX_QUERY_LENGTH });
      validated.sector = validateOptionalEnum(args.sector, 'sector', MEMORY_SECTORS);
      validated.limit = validateOptionalNumber(args.limit, 'limit', { min: 1, max: MAX_LIMIT });
      validated.includeSuperseded = args.include_superseded ?? false;
      validated.scopePath = validateOptionalString(args.scope_path, 'scope_path', { maxLength: 500 });
      validated.scopeModule = validateOptionalString(args.scope_module, 'scope_module', { maxLength: 100 });
      validated.memoryTypeFilter = validateOptionalEnum(args.memory_type, 'memory_type', MEMORY_TYPES);
      validated.mode = validateOptionalEnum(args.mode, 'mode', SEARCH_MODES);
      break;

    case 'memory_timeline':
      validated.anchorId = validateString(args.anchor_id, 'anchor_id');
      validated.depthBefore = validateOptionalNumber(args.depth_before, 'depth_before', { min: 0, max: MAX_DEPTH });
      validated.depthAfter = validateOptionalNumber(args.depth_after, 'depth_after', { min: 0, max: MAX_DEPTH });
      break;

    case 'memory_add':
      validated.content = validateString(args.content, 'content', { maxLength: MAX_CONTENT_LENGTH });
      validated.sector = validateOptionalEnum(args.sector, 'sector', MEMORY_SECTORS);
      validated.memoryType = validateOptionalEnum(args.type, 'type', MEMORY_TYPES);
      validated.tags = args.tags
        ? validateArray(args.tags, 'tags', (item, i) => validateString(item, `tags[${i}]`, { maxLength: 100 }))
        : undefined;
      validated.importance = validateOptionalNumber(args.importance, 'importance', { min: 0, max: 1 });
      validated.context = validateOptionalString(args.context, 'context', { maxLength: MAX_CONTENT_LENGTH });
      validated.scopePath = validateOptionalString(args.scope_path, 'scope_path', { maxLength: 500 });
      validated.scopeModule = validateOptionalString(args.scope_module, 'scope_module', { maxLength: 100 });
      break;

    case 'memory_reinforce':
      validated.memoryId = validateString(args.memory_id, 'memory_id');
      validated.amount = validateOptionalNumber(args.amount, 'amount', { min: 0, max: 1 });
      break;

    case 'memory_deemphasize':
      validated.memoryId = validateString(args.memory_id, 'memory_id');
      validated.amount = validateOptionalNumber(args.amount, 'amount', { min: 0, max: 1 });
      break;

    case 'memory_delete':
      validated.memoryId = validateString(args.memory_id, 'memory_id');
      validated.hard = args.hard ?? false;
      break;

    case 'memory_supersede':
      validated.oldMemoryId = validateString(args.old_memory_id, 'old_memory_id');
      validated.newMemoryId = validateString(args.new_memory_id, 'new_memory_id');
      break;

    case 'docs_search':
      validated.query = validateString(args.query, 'query', { maxLength: MAX_QUERY_LENGTH });
      validated.limit = validateOptionalNumber(args.limit, 'limit', { min: 1, max: MAX_LIMIT });
      break;

    case 'docs_ingest':
      validated.path = validateOptionalString(args.path, 'path', { maxLength: 4096 });
      validated.url = validateOptionalString(args.url, 'url', { maxLength: 4096 });
      validated.content = validateOptionalString(args.content, 'content', { maxLength: MAX_CONTENT_LENGTH });
      validated.title = validateOptionalString(args.title, 'title', { maxLength: 500 });
      break;

    case 'code_search':
      validated.query = validateString(args.query, 'query', { maxLength: MAX_QUERY_LENGTH });
      validated.language = validateOptionalString(args.language, 'language', { maxLength: 20 });
      validated.limit = validateOptionalNumber(args.limit, 'limit', { min: 1, max: MAX_LIMIT });
      break;

    case 'code_index':
      validated.force = args.force ?? false;
      validated.dryRun = args.dry_run ?? false;
      break;

    default:
      throw new Error(`Unknown tool: ${name}`);
  }

  return validated;
}

async function handleToolCall(name: string, args: ToolArgs, cwd: string): Promise<string> {
  const start = Date.now();
  log.debug('mcp', 'Tool call received', { name, cwd });

  const validated = validateToolArgs(name, args);

  const project = await getOrCreateProject(cwd);
  const embeddingService = await createEmbeddingService();
  const search = createSearchService(embeddingService);
  const store = createMemoryStore();
  const docs = createDocumentService(embeddingService);

  switch (name) {
    case 'memory_search': {
      const results = await search.search({
        query: validated.query!,
        projectId: project.id,
        sector: validated.sector,
        memoryType: validated.memoryTypeFilter,
        limit: validated.limit ?? 10,
        mode: validated.mode ?? 'hybrid',
        includeSuperseded: validated.includeSuperseded ?? false,
        scopePath: validated.scopePath,
        scopeModule: validated.scopeModule,
      });
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return formatSearchResults(results);
    }

    case 'memory_timeline': {
      const timeline = await search.timeline(
        validated.anchorId!,
        validated.depthBefore ?? 5,
        validated.depthAfter ?? 5,
      );
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return formatTimeline(timeline);
    }

    case 'memory_add': {
      const memory = await store.create(
        {
          content: validated.content!,
          sector: validated.sector,
          memoryType: validated.memoryType,
          tags: validated.tags,
          importance: validated.importance,
          context: validated.context,
          scopePath: validated.scopePath,
          scopeModule: validated.scopeModule,
          tier: 'project',
        },
        project.id,
      );
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return `Memory created: ${memory.id} (sector: ${memory.sector}, salience: ${memory.salience})`;
    }

    case 'memory_reinforce': {
      const memory = await store.reinforce(validated.memoryId!, validated.amount ?? 0.1);
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return `Memory reinforced: ${memory.id} (new salience: ${memory.salience.toFixed(2)})`;
    }

    case 'memory_deemphasize': {
      const memory = await store.deemphasize(validated.memoryId!, validated.amount ?? 0.2);
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return `Memory de-emphasized: ${memory.id} (new salience: ${memory.salience.toFixed(2)})`;
    }

    case 'memory_delete': {
      await store.delete(validated.memoryId!, validated.hard ?? false);
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return validated.hard
        ? `Memory permanently deleted: ${validated.memoryId}`
        : `Memory soft-deleted: ${validated.memoryId} (can be restored)`;
    }

    case 'memory_supersede': {
      await supersede(validated.oldMemoryId!, validated.newMemoryId!);
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return `Memory ${validated.oldMemoryId} marked as superseded by ${validated.newMemoryId}`;
    }

    case 'docs_search': {
      const results = await docs.search(validated.query!, project.id, validated.limit ?? 5);
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return formatDocResults(results);
    }

    case 'docs_ingest': {
      const doc = await docs.ingest({
        projectId: project.id,
        path: validated.path,
        url: validated.url,
        content: validated.content,
        title: validated.title,
      });
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return `Document ingested: ${doc.title ?? doc.id}`;
    }

    case 'code_search': {
      const codeIndex = createCodeIndexService(embeddingService);
      const state = await codeIndex.getState(project.id);

      if (!state) {
        return `IMPORTANT: Tell the user that project code has not been indexed yet.
They should run \`ccmemory watch .\` in the project directory to enable
semantic code search, or you can run it for them via Bash.

No indexed code to search.`;
      }

      const timeSinceIndex = Date.now() - state.lastIndexedAt;
      const hoursAgo = Math.floor(timeSinceIndex / (1000 * 60 * 60));
      let staleWarning = '';
      if (hoursAgo > 24) {
        staleWarning = `\n\nNote: Index is ${hoursAgo} hours old. Consider re-indexing with \`ccmemory code-index\`.`;
      }

      const results = await codeIndex.search({
        query: validated.query!,
        projectId: project.id,
        language: validated.language as CodeLanguage | undefined,
        limit: validated.limit ?? 10,
      });
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
      return formatCodeSearchResults(results) + staleWarning;
    }

    case 'code_index': {
      const codeIndex = createCodeIndexService(embeddingService);
      const progress = await codeIndex.index(cwd, project.id, {
        force: validated.force ?? false,
        dryRun: validated.dryRun ?? false,
      });
      log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });

      if (validated.dryRun) {
        return `Dry run complete: Found ${progress.totalFiles} code files to index.`;
      }

      let result = `Code indexing complete:
- Files scanned: ${progress.scannedFiles}
- Files indexed: ${progress.indexedFiles}`;

      if (progress.errors.length > 0) {
        result += `\n- Errors: ${progress.errors.length}`;
      }

      return result;
    }

    default:
      log.warn('mcp', 'Unknown tool requested', { name });
      throw new Error(`Unknown tool: ${name}`);
  }
}

function formatSearchResults(results: SearchResult[]): string {
  if (results.length === 0) return 'No memories found.';

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
        const sessionDate = new Date(r.sourceSession.startedAt).toISOString().slice(0, 16);
        lines.push(
          `Session: ${sessionDate}${r.sourceSession.summary ? ` - ${r.sourceSession.summary.slice(0, 50)}...` : ''}`,
        );
      }

      if (r.relatedMemoryCount > 0) {
        lines.push(`Related: ${r.relatedMemoryCount} memories`);
      }

      lines.push(`Content: ${mem.content.slice(0, 300)}${mem.content.length > 300 ? '...' : ''}`);

      return lines.join('\n');
    })
    .join('\n\n---\n\n');
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
    for (const [, session] of sessions) {
      const sessionDate = new Date(session.startedAt).toISOString().slice(0, 16);
      lines.push(`  - ${sessionDate}: ${session.summary ?? 'No summary'}`);
    }
  }

  return lines.join('\n');
}

function formatDocResults(results: DocumentSearchResult[]): string {
  if (results.length === 0) return 'No documents found.';

  return results
    .map((r, i) => {
      return `[${i + 1}] ${r.document.title ?? 'Untitled'} (score: ${r.score.toFixed(2)})
Source: ${r.document.sourcePath ?? r.document.sourceUrl ?? 'inline'}
Match: ${r.chunk.content.slice(0, 200)}...`;
    })
    .join('\n\n');
}

function formatCodeSearchResults(results: CodeSearchResult[]): string {
  if (results.length === 0) return 'No code found matching your query.';

  return results
    .map((r, i) => {
      const lines = [
        `[${i + 1}] ${r.path}:${r.startLine}-${r.endLine}`,
        `Language: ${r.language} | Type: ${r.chunkType} | Score: ${r.score.toFixed(3)}`,
      ];

      if (r.symbols.length > 0) {
        lines.push(`Symbols: ${r.symbols.join(', ')}`);
      }

      const preview = r.content.split('\n').slice(0, 10).join('\n');
      lines.push('');
      lines.push('```' + r.language);
      lines.push(preview);
      if (r.content.split('\n').length > 10) {
        lines.push('...');
      }
      lines.push('```');

      return lines.join('\n');
    })
    .join('\n\n---\n\n');
}

const server = new Server({ name: 'ccmemory', version: '1.0.0' }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS.map(t => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async request => {
  const { name, arguments: args } = request.params;
  const cwd = process.env['CLAUDE_PROJECT_DIR'] ?? process.cwd();
  const start = Date.now();

  try {
    const result = await handleToolCall(name, (args ?? {}) as ToolArgs, cwd);
    log.info('mcp', 'Tool call completed', { name, ms: Date.now() - start });
    return {
      content: [{ type: 'text' as const, text: result }],
    };
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    log.error('mcp', 'Tool call failed', {
      name,
      error: errMsg,
      ms: Date.now() - start,
    });
    return {
      content: [{ type: 'text' as const, text: `Error: ${errMsg}` }],
      isError: true,
    };
  }
});

async function main(): Promise<void> {
  log.info('mcp', 'Starting MCP server', { pid: process.pid });
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log.info('mcp', 'MCP server connected');
}

main().catch((err: Error) => {
  log.error('mcp', 'MCP server error', { error: err.message });
  process.exit(1);
});
