# CCMemory

Self-contained memory plugin for Claude Code. Provides persistent, searchable memory across sessions using vector embeddings (Ollama/OpenRouter) and full-text search.

## Quick Start

```bash
bun install                    # Install dependencies
bun run build                  # Build single executable (dist/ccmemory)
bun run bin:install            # Build and copy to ~/.local/bin/ccmemory
bun run test                   # Run all tests
```

## Architecture

CCMemory uses a 5-sector memory model:

- **Episodic**: Tool observations, session events
- **Semantic**: Facts, documentation, code knowledge
- **Procedural**: How-to instructions, workflows
- **Emotional**: Frustrations, preferences, successes
- **Reflective**: Session summaries, insights

Memories have tiers: `session` (temporary) → `project` (persistent).

## Project Structure

```
src/
├── main.ts        → Unified entry point (CLI, MCP, hooks)
├── db/            → libSQL database, schema, migrations
├── services/
│   ├── embedding/ → Ollama/OpenRouter vector providers
│   ├── memory/    → Memory store, dedup, decay, sessions
│   ├── search/    → FTS5, vector search, hybrid ranking
│   └── documents/ → Chunking, ingestion
├── mcp/           → MCP server for Claude Code integration
├── cli/           → CLI commands (search, show, serve, etc.)
├── hooks/         → Hook handlers (capture, summarize, cleanup)
├── webui/         → React SSR + WebSocket browser UI
└── utils/         → Paths, logging utilities

scripts/           → Build scripts (prebuild-webui.ts)
plugin/            → Claude Code plugin (bin/ccmemory + configs)
tests/integration/ → Integration tests
```

## Development

### Type Safety (CRITICAL)

**NO `any`. NO `@ts-ignore`. NO `as any`.**

```typescript
// Use type for data shapes
type Memory = { id: string; content: string; sector: MemorySector };

// Use unknown with type guards
function isMemory(obj: unknown): obj is Memory {
  return typeof obj === "object" && obj !== null && "id" in obj;
}
```

### Code Style

- Modern ESM with `.js` extensions in imports
- `@libsql/client` for DB, `Bun.serve()` for HTTP
- Async/await, no callbacks
- No barrel files, no comments (self-documenting code)

### Testing

```bash
bun run test                        # All tests
```

- Unit tests: `src/**/__test__/*.test.ts` (colocated)
- Integration tests: `tests/integration/*.test.ts`

### Logging

```typescript
import { log } from "../utils/log.js";

log.debug("module", "Message", { context: "value" });
log.info("module", "Message");
log.warn("module", "Message");
log.error("module", "Message", { error: err.message });
```

Control via `LOG_LEVEL` env var: `debug`, `info`, `warn`, `error` (default: `info`).

## CLI Commands

```bash
ccmemory search "query"          # Search memories
ccmemory show <id>               # Show memory details
ccmemory delete <id>             # Soft delete memory
ccmemory serve                   # Start WebUI
ccmemory shutdown                # Shutdown WebUI
ccmemory stats                   # Show statistics
ccmemory health                  # Check system health
ccmemory config get <key>        # View configuration
ccmemory import <file>           # Import memories
ccmemory export <file>           # Export memories
```

## MCP Tools

When used as a Claude Code plugin, these tools are available:

- `memory_search` - Search memories by query
- `memory_timeline` - View memories around a point in time
- `memory_add` - Create new memory
- `memory_reinforce` - Increase memory salience
- `memory_deemphasize` - Decrease memory salience
- `memory_delete` - Soft delete memory
- `memory_supersede` - Mark memory as superseded
- `docs_search` - Search ingested documents
- `docs_ingest` - Ingest document for search

## Hooks

The plugin uses Claude Code hooks:

- **PostToolUse**: Captures tool observations as episodic memories
- **Stop**: Generates session summary as reflective memory
- **SessionEnd**: Promotes high-salience session memories to project tier

## Configuration

Environment variables:

- `CCMEMORY_DATA_DIR` - Data directory (default: XDG data dir)
- `CCMEMORY_CONFIG_DIR` - Config directory (default: XDG config dir)
- `CCMEMORY_CACHE_DIR` - Cache directory (default: XDG cache dir)
- `LOG_LEVEL` - Logging level (debug/info/warn/error)
- `OPENROUTER_API_KEY` - API key for OpenRouter embedding fallback
