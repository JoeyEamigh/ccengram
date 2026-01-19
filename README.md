# CCMemory

Self-contained memory plugin for Claude Code. Provides persistent, searchable memory across sessions with automatic vectorization via Ollama.

## Features

- **5-Sector Memory Model**: Episodic, semantic, procedural, emotional, and reflective memory types
- **Hybrid Search**: Combined FTS5 full-text and vector similarity search
- **Automatic Capture**: PostToolUse hooks capture tool observations
- **Session Summaries**: Stop hooks generate reflective summaries
- **Salience Decay**: Time-based memory decay with reinforcement on access
- **Deduplication**: SimHash-based duplicate detection with automatic boosting
- **Document Ingestion**: Chunk and vectorize files/URLs for RAG
- **WebUI**: Real-time browser interface with WebSocket updates
- **Multi-Agent Support**: Track memories across concurrent Claude Code instances
- **Single Binary**: No runtime dependencies - just download and run

## Quick Start

### Prerequisites

CCMemory requires [Ollama](https://ollama.ai/) for local embeddings:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Install Ollama (macOS)
brew install ollama

# Start Ollama and pull the embedding model
ollama serve &
ollama pull qwen3-embedding
```

Alternatively, set `OPENROUTER_API_KEY` for cloud-based embeddings.

### Option 1: Quick Install (Recommended)

Install with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/joeyguerra/ccmemory/main/scripts/install.sh | bash
```

This installs to `~/.local/bin/ccmemory` (user-scoped, no sudo required). Make sure `~/.local/bin` is in your PATH.

Verify installation:

```bash
ccmemory --version
ccmemory health
```

### Option 2: Build from Source

Requires [Bun](https://bun.sh/) v1.0+ for building:

```bash
# Clone the repository
git clone https://github.com/your-username/ccmemory.git
cd ccmemory

# Install dependencies and build
bun install
bun run build

# The executable is at dist/ccmemory
./dist/ccmemory --version
```

## Claude Code Plugin Installation

### Option 1: Marketplace Install (Recommended)

The easiest way to install CCMemory as a Claude Code plugin:

```bash
# In Claude Code, add the marketplace
/plugin marketplace add your-username/ccmemory

# Install the plugin
/plugin install ccmemory@ccmemory-marketplace
```

The plugin will automatically download the correct binary for your platform from GitHub releases on first use.

### Option 2: Manual Installation

Clone and copy the plugin directory:

```bash
# Clone the repository
git clone https://github.com/your-username/ccmemory.git

# Copy the plugin to Claude Code
cp -r ccmemory/plugin ~/.claude/plugins/ccmemory

# Restart Claude Code to activate
```

The binary will be downloaded automatically when the plugin is first used.

### Option 3: Build from Source

If you want to use a locally-built binary:

```bash
# Clone and build
git clone https://github.com/your-username/ccmemory.git
cd ccmemory
bun install
bun run build

# Copy plugin and binary
cp -r plugin ~/.claude/plugins/ccmemory
cp dist/ccmemory ~/.claude/plugins/ccmemory/bin/

# Restart Claude Code to activate
```

## CLI Usage

```bash
# Search memories
ccmemory search "authentication flow"

# Search with filters
ccmemory search "API design" --sector semantic --limit 5

# Show memory details
ccmemory show <memory-id>

# Show related memories
ccmemory show <memory-id> --related

# View statistics
ccmemory stats

# Check system health
ccmemory health

# View/set configuration
ccmemory config
ccmemory config embedding.provider ollama

# Start WebUI
ccmemory serve --port 37778 --open

# Shutdown running server
ccmemory shutdown

# Check for updates
ccmemory update --check

# Update to latest version
ccmemory update

# Force update (even if up-to-date)
ccmemory update --force

# Import a document
ccmemory import document.md --title "Project Docs"

# Export memories
ccmemory export --output memories.json
```

## MCP Tools

When used as a Claude Code plugin, these tools are available:

| Tool                 | Description                                |
| -------------------- | ------------------------------------------ |
| `memory_search`      | Hybrid search with sector filtering        |
| `memory_timeline`    | Chronological view with session grouping   |
| `memory_add`         | Create new memory with auto-classification |
| `memory_reinforce`   | Increase salience (mark as useful)         |
| `memory_deemphasize` | Decrease salience (mark as less relevant)  |
| `memory_delete`      | Soft delete a memory                       |
| `memory_supersede`   | Replace old memory with new version        |
| `docs_search`        | Search ingested documents                  |
| `docs_ingest`        | Ingest file, URL, or raw content           |

### Tool Modes

CCMemory supports three tool exposure modes to control which MCP tools are available to Claude Code:

| Mode     | Tools Available                                   | Use Case                                                     |
| -------- | ------------------------------------------------- | ------------------------------------------------------------ |
| `full`   | All 9 tools                                       | Full control over memory management                          |
| `recall` | `memory_search`, `memory_timeline`, `docs_search` | Read-only access (memories captured automatically via hooks) |
| `custom` | User-specified list                               | Fine-grained control                                         |

**Configuration (in order of priority):**

1. **Environment variable**: `CCMEMORY_TOOLS_MODE=recall`
2. **Per-project config**: Create `.claude/ccmemory.local.md` in your project:
   ```yaml
   ---
   tools:
     mode: recall
   ---
   ```
3. **Global config**: `ccmemory config tools.mode recall`
4. **Database setting**: Via WebUI Settings

**Example: Per-project recall mode**

For projects where you only want Claude to recall memories (not manually add them), create `.claude/ccmemory.local.md`:

```yaml
---
tools:
  mode: recall
---
```

**Example: Custom tool selection**

```yaml
---
tools:
  mode: custom
  enabledTools:
    - memory_search
    - memory_add
    - docs_search
---
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code Plugin                       │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ PostToolUse │    Stop     │ SessionEnd  │   MCP Server     │
│   Capture   │  Summarize  │   Cleanup   │     Tools        │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       ▼             ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory Service                          │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Sectors   │    Decay    │   Dedup     │  Relationships   │
│  (5 types)  │  (salience) │  (simhash)  │   (4 types)      │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       ▼             ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Search Service                          │
├─────────────────────────────┬───────────────────────────────┤
│         FTS5 Search         │       Vector Search           │
│    (full-text + snippets)   │   (cosine similarity)         │
└──────────────┬──────────────┴───────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Embedding Service                        │
├─────────────────────────────┬───────────────────────────────┤
│     Ollama Provider         │    OpenRouter Provider        │
│   (qwen3-embedding)         │   (API fallback)              │
└──────────────┬──────────────┴───────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       libSQL Database                        │
│              (WAL mode, F32_BLOB vectors)                   │
└─────────────────────────────────────────────────────────────┘
```

## Memory Sectors

| Sector         | Description                              | Decay Rate           |
| -------------- | ---------------------------------------- | -------------------- |
| **Episodic**   | What happened (tool calls, observations) | Fast (0.1/day)       |
| **Semantic**   | Facts and knowledge                      | Slow (0.02/day)      |
| **Procedural** | How to do things (commands, workflows)   | Very slow (0.01/day) |
| **Emotional**  | Preferences and reactions                | Medium (0.05/day)    |
| **Reflective** | Session summaries and insights           | Very slow (0.01/day) |

## Configuration

### Environment Variables

| Variable              | Description                                    | Default                   |
| --------------------- | ---------------------------------------------- | ------------------------- |
| `CCMEMORY_DATA_DIR`   | Database location                              | `~/.local/share/ccmemory` |
| `CCMEMORY_CONFIG_DIR` | Config files                                   | `~/.config/ccmemory`      |
| `CCMEMORY_CACHE_DIR`  | Cache files                                    | `~/.cache/ccmemory`       |
| `CCMEMORY_TOOLS_MODE` | Tool exposure mode: `full`, `recall`, `custom` | `full`                    |
| `OPENROUTER_API_KEY`  | OpenRouter API key (fallback)                  | (none)                    |
| `LOG_LEVEL`           | Log verbosity: debug, info, warn, error        | `info`                    |

### Data Directories by Platform

| Platform | Data                                     | Config                           | Cache                           |
| -------- | ---------------------------------------- | -------------------------------- | ------------------------------- |
| Linux    | `~/.local/share/ccmemory`                | `~/.config/ccmemory`             | `~/.cache/ccmemory`             |
| macOS    | `~/Library/Application Support/ccmemory` | `~/Library/Preferences/ccmemory` | `~/Library/Caches/ccmemory`     |
| Windows  | `%LOCALAPPDATA%\ccmemory`                | `%APPDATA%\ccmemory`             | `%LOCALAPPDATA%\ccmemory\cache` |

### Embedding Configuration

Configure via CLI, WebUI, or config file:

**CLI:**

```bash
# View current config
ccmemory config

# Use Ollama (default)
ccmemory config embedding.provider ollama

# Use OpenRouter
ccmemory config embedding.provider openrouter
```

**WebUI:**
Navigate to Settings (gear icon) in the WebUI to configure:

- Embedding provider and model
- Capture settings (enable/disable, max result size)
- Other runtime options

## WebUI

Start the WebUI server:

```bash
ccmemory serve --port 37778 --open
```

Features:

- Real-time memory updates via WebSocket
- Search with sector filters
- Timeline view with session grouping
- Memory reinforcement/de-emphasis
- Multi-agent activity tracking
- **Settings page** for configuration (embedding provider, capture settings, etc.)

## Development

Requires [Bun](https://bun.sh/) v1.0+ for development.

```bash
# Install dependencies
bun install

# Type check
bun run typecheck

# Run all tests (396 tests)
bun run test

# Run specific test file
bun test src/services/memory/__test__/store.test.ts

# Development mode (with hot reload)
bun run dev:cli           # CLI
bun run dev:mcp           # MCP server
bun run dev:serve         # WebUI

# Check Ollama status
bun run ollama:check

# View database stats
bun run db:counts
```

### Building

```bash
# Build for current platform
bun run build

# Build for all platforms
bun run build:all-platforms

# Build and copy to plugin directory
bun run build:plugin

# Individual platform builds
bun run build:linux       # Linux x64
bun run build:macos       # macOS x64
bun run build:macos-arm   # macOS ARM64
bun run build:windows     # Windows x64
```

### Project Structure

```
src/
├── cli/              # CLI commands
├── db/               # libSQL connection, schema, migrations
├── hooks/            # Hook handlers (capture, summarize, cleanup)
├── mcp/              # MCP server (stdio transport)
├── services/
│   ├── documents/    # Chunking, ingestion
│   ├── embedding/    # Ollama/OpenRouter providers
│   ├── memory/       # Store, sectors, decay, dedup
│   └── search/       # FTS5, vector, hybrid ranking
├── utils/            # Paths, logging
└── webui/            # Bun server, React SSR, WebSocket

scripts/              # Build scripts
plugin/               # Claude Code plugin (ready to copy)
spec/                 # Design specifications
tests/                # Integration tests
```

## Troubleshooting

### Ollama Connection Failed

```
Error: Failed to connect to Ollama at http://localhost:11434
```

**Solutions:**

1. Ensure Ollama is running: `ollama serve`
2. Check if the port is blocked by firewall
3. Verify Ollama URL: `curl http://localhost:11434/api/tags`

### Model Not Found

```
Error: Model qwen3-embedding not found
```

**Solutions:**

1. Pull the model: `ollama pull qwen3-embedding`
2. Check available models: `ollama list`

### Permission Denied

```
Error: EACCES: permission denied
```

**Solutions:**

1. Create directory manually: `mkdir -p ~/.local/share/ccmemory`
2. Fix permissions: `chmod 755 ~/.local/share/ccmemory`

### Database Locked

```
Error: database is locked
```

**Solutions:**

1. Only one write process should access the database at a time
2. Check for zombie processes: `ps aux | grep ccmemory`
3. Remove stale lock files if necessary

## Updating

### Auto-Update (Recommended)

CCMemory includes a built-in update command:

```bash
# Check for updates
ccmemory update --check

# Update to latest version
ccmemory update

# Force update (even if already up-to-date)
ccmemory update --force
```

### Plugin (Automatic)

The plugin wrapper automatically checks for updates once every 24 hours in the background. No manual action required.

### Reinstall via Script

Re-run the install script to get the latest version:

```bash
curl -fsSL https://raw.githubusercontent.com/joeyguerra/ccmemory/main/scripts/install.sh | bash
```

### From Source

```bash
cd ccmemory
git pull origin main
bun install
bun run build
```

## Releasing (For Maintainers)

To create a new release:

```bash
# Build binaries for all platforms
bun run build:all-platforms

# Tag the release
git tag v1.0.0
git push origin v1.0.0

# Create GitHub release and upload binaries:
# - dist/ccmemory-linux-x64
# - dist/ccmemory-darwin-x64
# - dist/ccmemory-darwin-arm64
# - dist/ccmemory-windows-x64.exe
```

Users with the plugin installed will automatically receive the update within an hour.

## Uninstalling

```bash
# Remove plugin
rm -rf ~/.claude/plugins/ccmemory

# Remove CLI (default location)
rm ~/.local/bin/ccmemory

# Remove data (optional - this deletes all memories!)
rm -rf ~/.local/share/ccmemory
rm -rf ~/.config/ccmemory
rm -rf ~/.cache/ccmemory
```

## License

MIT
