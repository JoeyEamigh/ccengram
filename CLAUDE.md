# CCEngram Build Agent

You are building **CCEngram** - intelligent memory and code search for Claude Code.

## Architecture Summary

**Single daemon + thin clients** via Unix socket IPC:

- `ccengram daemon` - Long-running process (embeddings, file watcher, all logic)
- `ccengram mcp` - Thin stdio proxy for Claude Code MCP
- `ccengram hook <name>` - Thin hook client, sends event to daemon

**Storage: LanceDB** (per-project isolation)

- Each project gets its own database at `~/.local/share/ccengram/projects/<hash>/`
- Tables: `memories`, `code_chunks`, `documents`, `sessions`, `entities`

**Search: Pure vector** - Semantic similarity via embeddings

## Crate Structure

```
crates/
├── core/       # Domain types (Memory, Sector, CodeChunk, Config)
├── db/         # LanceDB wrapper, per-project connections
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
cargo build                               # Build all
cargo test                                # Run tests
cargo clippy --all-targets --all-features # Lint all
cargo fmt --all                           # Format all
```

## Tool Presets

**Recommendation**: Start with `minimal` preset. Hooks handle memory capture automatically, so Claude only needs search tools.

| Preset | Tools | Description |
|--------|-------|-------------|
| **minimal** | `memory_search`, `code_search`, `docs_search` | Read-only (3 tools) |
| **standard** | Above + `memory_add`, `memory_reinforce`, `memory_deemphasize`, `code_context`, `doc_context`, `memory_timeline`, `entity_top`, `project_stats` | Standard (11 tools) |
| **full** | All 38 tools | Everything |

Initialize project config:

```bash
ccengram config init --preset minimal
```

## CLI Commands

```bash
# Core
ccengram daemon                     # Start daemon (required)
ccengram watch                      # Start file watcher
ccengram tui                        # Launch TUI
ccengram mcp                        # MCP server (for plugin)
ccengram hook <name>                # Handle hook event

# Search (memories, code, docs)
ccengram search memories "query"    # Search memories
ccengram search code "query"        # Search code
ccengram search docs "query"        # Search documents

# Context retrieval
ccengram context <chunk_id>         # Get context (auto-detects code vs doc)
ccengram context <id> --before 30   # More lines/chunks before
ccengram context <id> --after 30    # More lines/chunks after
ccengram context <id> --json        # JSON output

# Memory management
ccengram memory show <id>           # Show memory details
ccengram memory delete <id>         # Soft-delete a memory
ccengram memory restore <id>        # Restore a soft-deleted memory
ccengram memory deleted             # List soft-deleted memories
ccengram memory export              # Export memories to file
ccengram memory archive             # Archive old low-salience memories

# Index management
ccengram index                      # Index code (default)
ccengram index code                 # Index code files
ccengram index code --force         # Re-index all code
ccengram index code --stats         # Show code index stats
ccengram index docs                 # Index documents from configured directory
ccengram index docs --directory ./  # Index docs from specific directory
ccengram index file <path>          # Index a single file (auto-detects type)

# Configuration
ccengram config help                # Show presets and config locations
ccengram config show                # Show effective config
ccengram config init                # Create project config (standard preset)
ccengram config init --preset minimal
ccengram config reset               # Reset user config to defaults

# Project management
ccengram projects list              # List all indexed projects
ccengram projects show <path>       # Show project details
ccengram projects clean <path>      # Remove a project's data
ccengram projects clean-all         # Remove all project data

# Logs
ccengram logs                       # Show last 50 lines of logs
ccengram logs -f                    # Follow logs in real-time
ccengram logs -n 100                # Show last 100 lines
ccengram logs --level error         # Filter by log level
ccengram logs --open                # Open log directory

# Utilities
ccengram stats                      # Show statistics
ccengram health                     # Health check
ccengram migrate                    # Migrate embeddings
ccengram update                     # Check for updates
ccengram agent                      # Generate MemExplore agent

# Shell completions
ccengram completions bash           # Generate bash completions
ccengram completions zsh            # Generate zsh completions
ccengram completions fish           # Generate fish completions
ccengram completions powershell     # Generate PowerShell completions
```
