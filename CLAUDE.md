# CCMemory

Self-contained memory plugin for Claude Code. Vectorizes via Ollama, searchable on-demand.

## On Context Clear: Do This First

1. Read `spec/99-tasks.md` — find `[~]` (in progress) or first `[ ]`
2. **Mark `[~]` IMMEDIATELY** before coding
3. Read relevant spec file for details
4. Implement with tests (`bun run test`)
5. **Mark `[x]` when tests pass**
6. Commit changes with descriptive message

## Task Tracking (CRITICAL)

**Update `spec/99-tasks.md` or work gets lost.**

| Before coding | After tests pass |
|---------------|------------------|
| `[ ]` → `[~]` | `[~]` → `[x]` |

## Type Safety (CRITICAL)

**NO `any`. NO `@ts-ignore`. NO `as any`. NO `as unknown as ...`.**

**Prefer `type` over `interface`:**
```typescript
// Good: type for data shapes
type Memory = {
  id: string;
  content: string;
  sector: MemorySector;
};

// Good: type for unions
type MemorySector = "episodic" | "semantic" | "procedural" | "emotional" | "reflective";

// Only use interface for declaration merging or class implementation
interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
}
```

Use `unknown` with type guards:
```typescript
function isMemory(obj: unknown): obj is Memory {
  return typeof obj === "object" && obj !== null && "id" in obj;
}
```

Handle `null`/`undefined` explicitly.

## Code Style

- Modern ESM with `.js` extensions in imports
- Modern TypeScript const assertions, template literals, `satisfies`
- NO barrel files (`index.ts` re-exports)
- NO comments — write self-documenting code
- Async/await, no callbacks
- `@libsql/client` for DB (not bun:sqlite)
- `Bun.serve()` for HTTP (not express)

## Specs

| Spec | Topic |
|------|-------|
| `spec/99-tasks.md` | **Task list — read first** |
| `spec/00-overview.md` | project overview |
| `spec/01-database.md` | libSQL, schema, migrations |
| `spec/02-embedding.md` | Ollama/OpenRouter |
| `spec/03-memory.md` | Types, dedup, decay |
| `spec/04-search.md` | FTS5, vectors, ranking |
| `spec/05-documents.md` | Chunking, ingestion |
| `spec/06-plugin.md` | Hooks, MCP server |
| `spec/07-cli.md` | CLI commands |
| `spec/08-webui.md` | Browser UI |

## Scripts (you may need to update these as you work)

```bash
bun run typecheck     # Type check
bun run test          # All tests
bun run build:all     # Build everything
bun run ollama:check  # Verify Ollama if not working
bun run db:counts     # Check DB stats
```

## Structure

```
src/db/           → libSQL connection, schema
src/services/     → embedding, memory, search, documents
src/mcp/          → MCP server (stdio)
src/cli/          → CLI commands
src/webui/        → Browser UI (Bun + React SSR + WebSocket)
src/utils/        → Shared utilities (paths, logging)
scripts/          → Hook scripts (capture, summarize, cleanup)
plugin/           → Plugin config (hooks.json, .mcp.json)
spec/             → Specifications
tests/            → Integration/e2e tests only
```

## Test Structure

- **Unit tests**: Colocated in `__test__` dirs next to source (`src/**/__test__/*.test.ts`)
- **Integration/e2e tests**: In `tests/`

## Logging

Use the unified logger from `src/utils/log.ts`:
```typescript
import { log } from "../utils/log.js";

log.debug("embedding", "Computing vector", { model: "qwen3" });
log.info("memory", "Created memory", { id: "abc123" });
log.warn("search", "Slow query", { ms: 500 });
log.error("db", "Connection failed", { error: err.message });
```

Log levels controlled via `LOG_LEVEL` env var: `debug`, `info`, `warn`, `error` (default: `info`).
