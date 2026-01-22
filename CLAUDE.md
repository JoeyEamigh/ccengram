# CCEngram Build Agent

You are building **CCEngram** - intelligent memory and code search for Claude Code.

## Architecture Summary

**Single daemon + thin clients** via Unix socket IPC:

- `ccengram daemon` - Long-running process (embeddings, file watcher, all logic)
- `ccengram mcp` - Thin stdio proxy for Claude Code MCP
- `ccengram hook <name>` - Thin hook client, sends event to daemon

**Storage: SQLite** (per-project isolation)

- Each project gets its own database at `~/.local/share/ccengram/projects/<hash>/`
- Tables: `memories`, `code_chunks`, `documents`, `sessions`, `entities`

**Search: Pure vector** - Semantic similarity via embeddings

## Crate Structure

```
crates/
├── core/       # Domain types (Memory, Sector, CodeChunk, Config)
├── db/         # SQLite wrapper, per-project connections
├── embedding/  # Ollama embedding provider
├── index/      # File scanner, tree-sitter parser, chunker
├── extract/    # Dedup (SimHash), decay, sector classification
├── llm/        # LLM integration for summarization
├── daemon/     # Unix socket server, request router, tools/hooks
├── tui/        # Terminal user interface (ratatui)
└── cli/        # Binary: main.rs, MCP proxy, hook client, commands
```

## Type Safety Rules

- **NO `any`**. NO `unwrap()` in library code. Use `?` and proper error types.
- Use `thiserror` for error enums
- Validate all inputs at boundaries

## Testing

- Unit tests: Colocated in `src/` as `#[cfg(test)]` modules
- Integration tests: `tests/integration/`
- Run with `cargo test`

## Sample Commands

```bash
cargo build                     # Build all
cargo test                      # Run tests
cargo clippy --all              # Lint all
cargo fmt --all                 # Format all
cargo run -p cli -- daemon      # Run daemon
cargo run -p cli -- search "q"  # Search
cargo run -p cli -- tui         # Launch TUI
```

## Tool Presets

**Recommendation**: Start with `minimal` preset. Hooks handle memory capture automatically, so Claude only needs search tools.

| Preset | Tools | Description |
|--------|-------|-------------|
| **minimal** | `memory_search`, `code_search`, `docs_search` | Read-only (3 tools) |
| **standard** | Above + `memory_add`, `memory_reinforce`, `memory_deemphasize`, `memory_timeline`, `entity_top`, `project_stats` | Standard (9 tools) |
| **full** | All 36 tools | Everything |

Initialize project config:

```bash
ccengram config --init --preset minimal
```

## MemExplore Subagent

Generate a MemExplore subagent for codebase exploration:

```bash
ccengram agent
# Creates .claude/agents/MemExplore.md
```

The MemExplore agent has access to:
- `memory_search` - Search memories
- `code_search` - Search indexed code
- `docs_search` - Search ingested documents
- `memory_timeline` - Get chronological context
- `entity_top` - Get top mentioned entities

Use for:
- Finding past decisions and their rationale
- Locating relevant code by semantic meaning
- Recalling user preferences and patterns
- Exploring codebase structure

## CLI Commands

```bash
ccengram daemon                     # Start daemon (required)
ccengram search "query"             # Search memories
ccengram search-code "query"        # Search code
ccengram search-docs "query"        # Search documents

# Index subcommands
ccengram index                      # Index code (default)
ccengram index code                 # Index code files
ccengram index code --force         # Re-index all code
ccengram index code --stats         # Show code index stats
ccengram index docs                 # Index documents from configured directory
ccengram index docs --directory ./  # Index docs from specific directory
ccengram index file <path>          # Index a single file (auto-detects type)

ccengram watch                      # Start file watcher
ccengram tui                        # Launch TUI
ccengram mcp                        # MCP server (for plugin)
ccengram hook <name>                # Handle hook event
ccengram migrate                    # Migrate embeddings
ccengram agent                      # Generate MemExplore agent
```

## Document Indexing

Configure document auto-indexing in `.claude/ccengram.toml`:

```toml
[docs]
# Directory to watch for documents (relative to project root)
directory = "docs"

# File extensions to treat as documents
extensions = ["md", "txt", "rst", "adoc", "org"]

# Maximum document file size (bytes)
max_file_size = 5242880  # 5MB
```

When `docs.directory` is set, the file watcher will automatically index documents when they change. The config file is also watched and reloaded on changes.
