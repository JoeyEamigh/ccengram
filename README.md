# CCEngram

Intelligent memory and code search for Claude Code.

### Views

- **Dashboard**: Overview of memories, code, and entities
- **Memories**: Browse and search memories
- **Code**: Browse indexed code chunks
- **Docs**: Browse ingested documents
- **Sessions**: Browse session history
- **Search**: Unified search across all data types

### Keybindings

| Key     | Action           |
| ------- | ---------------- |
| `1-6`   | Switch views     |
| `/`     | Search           |
| `j/k`   | Navigate up/down |
| `Enter` | Select/expand    |
| `q`     | Quit             |
| `?`     | Help             |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Claude Code Plugin                     │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ UserPrompt  │ PostToolUse │ PreCompact  │   MCP Server     │
│   Hook      │   Capture   │  Summarize  │     Tools        │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                       CCEngram Daemon                      │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Memory    │    Code     │    Docs     │   Embedding      │
│   Service   │   Indexer   │   Ingester  │   Service        │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       LanceDB                               │
└─────────────────────────────────────────────────────────────┘
```
