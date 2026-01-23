# CCEngram

Intelligent memory and code search for Claude Code. Provides persistent, searchable memory across sessions with vector embeddings via Ollama.

## Features

- **5-Sector Memory Model**: Episodic, semantic, procedural, emotional, and reflective memory types
- **Semantic Search**: Vector similarity search for memories, code, and documents
- **Code Indexing**: Semantic search over your entire codebase
- **File Watcher**: Background daemon keeps code index updated on file changes
- **Automatic Capture**: Hooks capture context automatically during Claude Code sessions
- **Salience Decay**: Time-based memory decay with reinforcement on access
- **TUI**: Terminal user interface for browsing and searching
- **Multi-Agent Support**: Track memories across concurrent Claude Code instances
- **Single Binary**: Written in Rust - fast, safe, no runtime dependencies

## Quick Start

### Prerequisites

CCEngram requires [Ollama](https://ollama.ai/) for local embeddings:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Install Ollama (macOS)
brew install ollama

# Start Ollama and pull the embedding model
ollama serve
ollama pull qwen3-embedding
```

### Recommended Ollama Configuration

For optimal embedding performance during bulk indexing, configure Ollama's systemd service:

```bash
sudo systemctl edit ollama
```

Add the following configuration:

```ini
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=32768" # see below for VRAM-based adjustments
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_GPU_OVERHEAD=2147483648"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=16"
Environment="OLLAMA_MAX_QUEUE=2048"
```

Then restart Ollama:

```bash
sudo systemctl restart ollama
```

**Why these values:**
- `MAX_LOADED_MODELS=1` - Prevents OOM by limiting to one model
- `KEEP_ALIVE=30m` - Keeps model loaded, avoids reload latency during indexing
- `NUM_PARALLEL=16` - Queue depth for burst requests (throughput is GPU-bound regardless)
- `MAX_QUEUE=2048` - Large queue for bulk indexing operations
- `GPU_OVERHEAD=2GB` - Reserves VRAM headroom for your desktop environment

> **Note**: Embedding throughput is GPU-bound. Increasing `NUM_PARALLEL` only increases queue depth, not actual throughput. The GPU processes requests as fast as it can regardless of parallelism settings.

### VRAM Requirements

The `qwen3-embedding` model (Q4_K_M quantization, 7.6B parameters) has the following VRAM requirements:

| Component | Size |
|-----------|------|
| Base (model weights) | ~4.9 GB |
| Per 1K context tokens | ~231 MB |

**Total VRAM by context length:**

| Context Length | Total VRAM |
|----------------|------------|
| 2K | 5.3 GB |
| 4K | 5.8 GB |
| 8K | 6.8 GB |
| 16K | 8.6 GB |
| 32K | 12.4 GB |

**GPU compatibility:**

| GPU | VRAM | Max Context | Configuration |
|-----|------|-------------|---------------|
| RTX 3090 / 4090 | 24 GB | 32K+ | Full context, use default config |
| RTX 3060 / 4060 Ti / Intel B580 | 12 GB | ~24K | `OLLAMA_CONTEXT_LENGTH=16384` recommended |
| RTX 4060 / 3050 | 8 GB | ~12K | `OLLAMA_CONTEXT_LENGTH=8192` required |

**Configuration for lower VRAM GPUs (8-12GB):**

```bash
sudo systemctl edit ollama
```

```ini
[Service]
# For 12GB VRAM (RTX 3060, Intel B580)
Environment="OLLAMA_CONTEXT_LENGTH=16384"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_GPU_OVERHEAD=2147483648"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_MAX_QUEUE=1024"

# For 8GB VRAM (RTX 4060), use instead:
# Environment="OLLAMA_CONTEXT_LENGTH=8192"
# Environment="OLLAMA_NUM_PARALLEL=4"
```

> **Note**: Lower context lengths reduce maximum batch sizes for embedding operations. CCEngram automatically adapts batch sizes based on the configured context length. With 8K context, expect ~2x longer indexing times compared to 32K context.

### Option 1: Quick Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/JoeyEamigh/ccengram/main/scripts/install.sh | bash
```

This installs to `~/.local/bin/ccengram`. Make sure `~/.local/bin` is in your PATH.

Verify installation:

```bash
ccengram --version
ccengram health
```

### Option 2: Build from Source

Requires Rust 1.75+:

```bash
# Clone the repository
git clone https://github.com/JoeyEamigh/ccengram.git
cd ccengram

# Build and install
cargo install --path crates/cli

# Or build without installing
cargo build --release -p cli
./target/release/ccengram --version
```

## Claude Code Plugin Installation

### Marketplace Install (Recommended)

```bash
# In Claude Code, add the marketplace
/plugin marketplace add JoeyEamigh/ccengram

# Install the plugin
/plugin install ccengram@ccengram-marketplace
```

The plugin will automatically download the correct binary for your platform from GitHub releases on first use.

### Manual Installation

Clone and copy the plugin directory:

```bash
git clone https://github.com/JoeyEamigh/ccengram.git
cp -r ccengram/plugin ~/.claude/plugins/ccengram
```

## Tool Presets

CCEngram supports three tool presets to control which MCP tools are available to Claude Code:

| Preset       | Tools                                                                                                            | Description                             |
| ------------ | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **minimal**  | `memory_search`, `code_search`, `docs_search`                                                                    | Read-only search (recommended to start) |
| **standard** | Above + `memory_add`, `memory_reinforce`, `memory_deemphasize`, `memory_timeline`, `entity_top`, `project_stats` | Standard operations (9 tools)           |
| **full**     | All 36 tools                                                                                                     | Full control over all memory operations |

**Recommendation**: Start with `minimal`. Hooks handle memory capture automatically.

Configure in `.claude/ccengram.toml`:

```toml
[tools]
preset = "minimal"  # or "standard", "full"

# Or specify individual tools:
# preset = "custom"
# enabled = ["memory_search", "code_search", "docs_search"]
```

Initialize project config:

```bash
ccengram config --init --preset minimal
```

## Configuration

Configuration file: `.claude/ccengram.toml` (per-project) or `~/.config/ccengram/config.toml` (global)

```toml
[tools]
preset = "minimal"

[embedding]
provider = "ollama"
model = "qwen3-embedding"
dimensions = 4096

[decay]
episodic_rate = 0.1
semantic_rate = 0.02
procedural_rate = 0.01
emotional_rate = 0.05
reflective_rate = 0.01

[search]
default_limit = 10
min_similarity = 0.3

[docs]
# Directory to watch for documents (file watcher auto-indexes on change)
directory = "docs"
# File extensions to treat as documents
extensions = ["md", "txt", "rst", "adoc", "org"]
# Maximum document file size (bytes)
max_file_size = 5242880  # 5MB
```

The config file is watched for changes. Docs settings are reloaded automatically; embedding changes require a daemon restart.

## Memory Sectors

| Sector         | Description                              | Decay Rate           |
| -------------- | ---------------------------------------- | -------------------- |
| **Episodic**   | What happened (tool calls, observations) | Fast (0.1/day)       |
| **Semantic**   | Facts and knowledge                      | Slow (0.02/day)      |
| **Procedural** | How to do things (commands, workflows)   | Very slow (0.01/day) |
| **Emotional**  | Preferences and reactions                | Medium (0.05/day)    |
| **Reflective** | Session summaries and insights           | Very slow (0.01/day) |

## CLI Usage

```bash
# Start the daemon (required for all operations)
ccengram daemon

# Search memories
ccengram search "authentication flow"
ccengram search "API design" --sector semantic --limit 5

# Search code semantically
ccengram search-code "error handling patterns"
ccengram search-code "database connection" --language rust

# Search documents
ccengram search-docs "architecture decisions"

# Index management (subcommands)
ccengram index                       # Index code (default)
ccengram index code                  # Index code files
ccengram index code --force          # Re-index all code
ccengram index code --stats          # Show code index stats
ccengram index code --export f.json  # Export code index
ccengram index code --load f.json    # Load code index
ccengram index docs                  # Index docs from configured dir
ccengram index docs -d ./docs        # Index docs from specific dir
ccengram index file README.md        # Index a single file

# File watcher
ccengram watch                    # Start file watcher
ccengram watch --status           # Check watcher status
ccengram watch --stop             # Stop watcher

# Other commands
ccengram stats                    # View statistics
ccengram health                   # Check system health
ccengram config --show            # Show current config
ccengram config --init            # Create project config
ccengram migrate                  # Migrate embeddings
ccengram agent                    # Generate MemExplore subagent
ccengram tui                      # Launch TUI
ccengram mcp                      # Start MCP server (for plugin)
```

## TUI

Launch the terminal user interface:

```bash
ccengram tui
```

### Views

- **Dashboard**: Overview of memories, code, and entities
- **Memories**: Browse and search memories
- **Code**: Browse indexed code chunks
- **Docs**: Browse ingested documents
- **Entities**: View extracted entities and relationships
- **Sessions**: Browse session history
- **Search**: Unified search across all data types

### Keybindings

| Key     | Action           |
| ------- | ---------------- |
| `Tab`   | Switch views     |
| `/`     | Search           |
| `j/k`   | Navigate up/down |
| `Enter` | Select/expand    |
| `q`     | Quit             |
| `?`     | Help             |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code Plugin                       │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ UserPrompt  │ PostToolUse │ PreCompact  │   MCP Server     │
│   Hook      │   Capture   │  Summarize  │     Tools        │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       CCEngram Daemon                        │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Memory    │    Code     │    Docs     │   Embedding      │
│   Service   │   Indexer   │   Ingester  │   Service        │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     SQLite Database                          │
│            (WAL mode, vector extensions)                     │
└─────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Build all crates
cargo build

# Build release binary
cargo build --release -p cli

# Run tests
cargo test

# Run specific crate tests
cargo test -p core
cargo test -p daemon

# Format code
cargo fmt

# Lint
cargo clippy
```

### Crate Structure

```
crates/
├── cli/        # CLI binary and commands
├── core/       # Core types, config, tool definitions
├── daemon/     # Background daemon, MCP server, hooks
├── db/         # Database layer, migrations
├── embedding/  # Ollama embedding service
├── extract/    # Entity extraction, classification
├── index/      # Code indexing, file watching
├── llm/        # LLM integration
└── tui/        # Terminal user interface
```

## Troubleshooting

### Daemon Not Running

```
Error: Daemon is not running. Start it with: ccengram daemon
```

Start the daemon in a separate terminal:

```bash
ccengram daemon
```

Or run it in the background:

```bash
ccengram daemon &
```

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

### Embedding Dimension Mismatch

If you change embedding models, run migration:

```bash
ccengram migrate --force
```

## License

MIT
