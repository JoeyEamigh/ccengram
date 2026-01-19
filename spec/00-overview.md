# CCMemory: Architecture Overview

## Project Summary

CCMemory is a Claude Code memory plugin that captures agent activity, vectorizes it via Ollama (with OpenRouter fallback), and makes it searchable on-demand without context pollution.

## Design Principles

1. **Silent Capture**: No context injection on SessionStart - only on-demand search
2. **Self-Contained**: Only requires Ollama (or OpenRouter API key)
3. **Portable**: Single libSQL database file, XDG-compliant paths
4. **Well-Behaved**: No zombie processes, proper lifecycle management
5. **Multi-Instance Safe**: WAL mode, no port conflicts, session isolation
6. **Context-Efficient**: Minimal tokens for maximum information retrieval

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code Session                       │
├─────────────────────────────────────────────────────────────┤
│  Hooks (PostToolUse, Stop, SessionEnd)                      │
│  ├── capture.ts      → Silent tool observation capture      │
│  ├── summarize.ts    → Session summary via SDK agent        │
│  └── cleanup.ts      → Process cleanup on session end       │
├─────────────────────────────────────────────────────────────┤
│  MCP Server (stdio)                                          │
│  ├── memory_search   → Hybrid FTS + vector search           │
│  ├── memory_timeline → Chronological context navigation     │
│  ├── memory_add      → Manual memory creation               │
│  ├── docs_search     → Search ingested documents            │
│  └── docs_ingest     → Ingest txt/md documents              │
├─────────────────────────────────────────────────────────────┤
│  Core Services                                               │
│  ├── Database        → libSQL with WAL mode                 │
│  ├── Embedding       → Ollama primary, OpenRouter fallback  │
│  ├── Memory          → CRUD, deduplication, decay           │
│  └── Search          → Hybrid ranking with salience         │
├─────────────────────────────────────────────────────────────┤
│  Storage                                                     │
│  └── $XDG_DATA_HOME/ccmemory/memories.db                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Differentiators

### vs claude-mem
| Aspect | claude-mem | CCMemory |
|--------|-----------|----------|
| Context injection | SessionStart auto-inject | On-demand only |
| Architecture | HTTP worker daemon | Direct DB access |
| Process management | Zombies possible | Explicit cleanup |
| Concurrent instances | Port conflicts | WAL mode, no ports |

### vs OpenMemory
| Aspect | OpenMemory | CCMemory |
|--------|------------|----------|
| Integration | Generic API | Tight Claude Code plugin |
| Dependencies | Postgres + Redis | Self-contained libSQL |
| Setup | Complex infrastructure | Single file DB |

## Memory Model

### Tiers
| Tier | Scope | Decay | Purpose |
|------|-------|-------|---------|
| Session | Current session | None | Working context |
| Project | Per-project | Slow (salience) | Architecture, decisions |
| Global | Cross-project | Aggressive | User preferences, patterns |

### Types
| Type | Description | Decay Rate |
|------|-------------|------------|
| decision | Architecture/design choices | 0.005 |
| procedure | Workflows, patterns | 0.01 |
| discovery | Codebase facts | 0.02 |
| preference | User style, conventions | 0.008 |

## Technology Stack

- **Runtime**: Bun
- **Database**: libSQL (Turso's SQLite fork) with native vectors
- **Embeddings**: Ollama (qwen3-embedding) / OpenRouter fallback
- **Plugin SDK**: @anthropic-ai/claude-agent-sdk (for summarization)
- **MCP**: @modelcontextprotocol/sdk

## File Structure

```
ccmemory/
├── src/
│   ├── db/
│   │   ├── database.ts          # libSQL connection, WAL mode
│   │   ├── schema.ts            # Table definitions
│   │   └── migrations.ts        # Schema migrations
│   ├── services/
│   │   ├── embedding/
│   │   │   ├── index.ts         # Provider selection
│   │   │   ├── ollama.ts        # Ollama provider
│   │   │   └── openrouter.ts    # OpenRouter fallback
│   │   ├── memory/
│   │   │   ├── store.ts         # Memory CRUD
│   │   │   ├── types.ts         # Memory type classification
│   │   │   ├── decay.ts         # Salience decay
│   │   │   └── dedup.ts         # Simhash deduplication
│   │   ├── search/
│   │   │   ├── hybrid.ts        # FTS + vector hybrid
│   │   │   └── ranking.ts       # Score computation
│   │   └── documents/
│   │       ├── ingest.ts        # Document ingestion
│   │       └── chunk.ts         # Text chunking
│   ├── mcp/
│   │   └── server.ts            # stdio MCP server
│   ├── cli/
│   │   └── index.ts             # CLI entry point
│   └── webui/
│       └── server.ts            # Bun.serve WebUI
├── scripts/
│   ├── capture.ts               # PostToolUse hook
│   ├── summarize.ts             # Stop hook
│   └── cleanup.ts               # SessionEnd hook
├── plugin/
│   ├── plugin.json              # Plugin manifest
│   ├── hooks/
│   │   └── hooks.json           # Hook configuration
│   └── .mcp.json                # MCP server configuration
├── spec/                        # This directory
├── tests/                       # E2E/integration test files
└── CLAUDE.md                    # Agent instructions
```

## XDG Paths

```
$XDG_CONFIG_HOME/ccmemory/config.json     # User settings
$XDG_DATA_HOME/ccmemory/memories.db       # Database
$XDG_CACHE_HOME/ccmemory/embeddings/      # Embedding cache (optional)
```

Default fallbacks:
- Linux: `~/.config`, `~/.local/share`, `~/.cache`
- macOS: `~/Library/Application Support`, same, `~/Library/Caches`
- Windows: `%APPDATA%`, `%LOCALAPPDATA%`, `%LOCALAPPDATA%\cache`

## Spec Files

1. **01-database.md** - libSQL setup, schema, migrations
2. **02-embedding.md** - Embedding providers
3. **03-memory.md** - Memory storage, types, decay
4. **04-search.md** - Hybrid search and ranking
5. **05-documents.md** - Document ingestion
6. **06-plugin.md** - Claude Code plugin integration
7. **07-cli.md** - CLI commands
8. **08-webui.md** - WebUI server
9. **99-tasks.md** - Master task list

## Implementation Phases

### Phase 1: Core Infrastructure
Database, XDG paths, embedding service foundation

### Phase 2: Memory System
CRUD operations, deduplication, type classification, decay

### Phase 3: Plugin Integration
Hooks (capture, summarize, cleanup), MCP server

### Phase 4: Search
FTS5, vector similarity, hybrid ranking

### Phase 5: Documents
Ingestion, chunking, separate search

### Phase 6: CLI
Commands, config, import/export, diagnostics

### Phase 7: WebUI
On-demand server, search UI, settings

### Phase 8: Polish
Tests, optimization, error handling
