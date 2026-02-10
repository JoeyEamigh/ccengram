# CCEngram User Guide

Intelligent memory and code search for Claude Code.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Projects](#projects)
4. [Configuration](#configuration)
5. [File Watching](#file-watching)
6. [CLI Reference](#cli-reference)
7. [TUI Guide](#tui-guide)
8. [How Memories Work](#how-memories-work)
9. [Troubleshooting](#troubleshooting)
10. [CLAUDE.md Recommendations](#claudemd-recommendations)
11. [Supported Languages](#supported-languages)

---

## Installation

### Prerequisites

- **Embedding Provider**: Uses **llama.cpp** by default (no API keys needed, models auto-download). For better speed and performance, cloud providers are recommended:
  - [OpenRouter](https://openrouter.ai/) (cloud, recommended)
  - [DeepInfra](https://deepinfra.com/) (cloud, supports embedding + reranking)
  - [Ollama](https://ollama.ai/) (local alternative)

### Method 1: One-Line Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/JoeyEamigh/ccengram/main/scripts/install.sh | bash
```

This installs the `ccengram` binary to `~/.local/bin/`.

### Method 2: Claude Code Plugin

Install from the Claude Code plugin marketplace:

```
/plugin marketplace add JoeyEamigh/ccengram
/plugin install ccengram
```

The binary auto-downloads on first use.

### Method 3: From Source

```bash
cargo install --git https://github.com/JoeyEamigh/ccengram --bin ccengram
```

To build with llama.cpp support (local embedding + reranking):

```bash
# Vulkan GPU (most portable, default feature)
cargo install --git https://github.com/JoeyEamigh/ccengram --bin ccengram --features vulkan

# CUDA GPU
cargo install --git https://github.com/JoeyEamigh/ccengram --bin ccengram --features cuda

# Apple Metal
cargo install --git https://github.com/JoeyEamigh/ccengram --bin ccengram --features metal
```

Build requirements for llama.cpp: CMake, C/C++ compiler, and the relevant GPU SDK (Vulkan SDK, CUDA toolkit, or Xcode).

### Verify Installation

```bash
ccengram --version
ccengram health
```

### Supported Platforms

- Linux x64, ARM64
- macOS x64, ARM64 (Apple Silicon)
- Windows x64

---

## Quick Start

### Step 1: Initialize Your Project

CCEngram uses **llama.cpp** for embedding and reranking by default - no API keys or configuration needed. Models are downloaded automatically from HuggingFace on first use.

> [!TIP]
> **For better speed and performance**, use a cloud embedding provider. See [Embedding Providers](#embedding-providers) below.

### Step 2: Initialize Your Project

Navigate to your project and create a project-specific config:

```bash
cd /your/project
ccengram config init
```

This creates `.claude/ccengram.toml`. The default `minimal` preset is recommended for most users:

- `minimal` - 2 tools: `explore`, `context` (recommended, default)
- `standard` - 11 tools: search + memory management + code maintenance
- `full` - 34 tools: everything

```bash
ccengram config init --preset standard  # If you want more tools
```

### Step 3: Index Your Codebase

```bash
ccengram index code
```

### Step 4: File Watching (Automatic)

The file watcher **automatically starts** after indexing and whenever an indexed project is accessed. Your index stays up-to-date as you edit files.

See [File Watching](#file-watching) for details on manual control and startup scan modes.

### Step 5: Start Using

**With Claude Code:** MCP tools are automatically available. Claude uses `explore` and `context` to search your codebase and memories.

**CLI Search:**

```bash
ccengram search memories "authentication flow"
ccengram search code "error handling"
```

**Interactive TUI:**

```bash
ccengram tui
```

---

## Projects

CCEngram maintains completely separate data for each project.

### Project Isolation

Each project gets:

- Its own LanceDB database at `~/.local/share/ccengram/projects/{id}/`
- Isolated memories - what you learn in one project stays there
- Separate code index and documents
- Independent configuration via `.claude/ccengram.toml`

### Project Identification

Projects are identified by:

1. **Git root** - If in a git repository, the repo root is the project
2. **Directory path** - Otherwise, the directory you're in becomes the project

**Git worktrees** that share the same repository will share memories by default. You can override this with the `[workspace]` config section.

### Managing Projects

```bash
# List all indexed projects
ccengram projects list

# Show details for a project (memory count, code stats, etc.)
ccengram projects show /path/to/project

# Remove all data for a project
ccengram projects clean /path/to/project

# Remove ALL project data (dangerous!)
ccengram projects clean-all
```

### Workspace Aliasing

To share memories between related projects (e.g., multiple clones of the same repo):

```toml
# In .claude/ccengram.toml
[workspace]
alias = "/path/to/main-repo"  # Share memories with this project
```

---

## Configuration

### Two-Level Config System

CCEngram uses a two-level configuration system:

| Level       | Path                             | Purpose                                       |
| ----------- | -------------------------------- | --------------------------------------------- |
| **Global**  | `~/.config/ccengram/config.toml` | Embedding provider, daemon settings, defaults |
| **Project** | `.claude/ccengram.toml`          | Project-specific overrides                    |

**Important:** Project config overrides global config.

### Config Commands

```bash
ccengram config show                     # Show effective configuration
ccengram config init                     # Generate project config (minimal preset)
ccengram config init --preset standard   # Generate with specific preset
ccengram config reset                    # Reset global config to defaults
```

Global config (`~/.config/ccengram/config.toml`) is created automatically on first use.

### Global Config (Daemon-Level Settings)

These sections **must** be in `~/.config/ccengram/config.toml`:

```toml
[embedding]
provider = "llamacpp"             # "llamacpp" (default), "openrouter", "deepinfra", or "ollama"
dimensions = 1024                 # 1024 for llamacpp 0.6B, 4096 for cloud 8B models
context_length = 32768
# For cloud providers (recommended for speed and performance):
# provider = "openrouter"         # Set OPENROUTER_API_KEY env var
# model = "qwen/qwen3-embedding-8b"
# dimensions = 4096
# For LlamaCpp customization:
# llamacpp_model_repo = "Qwen/Qwen3-Embedding-0.6B-GGUF"  # default
# llamacpp_gpu_layers = -1        # GPU layers to offload (-1 = all)

[reranker]
enabled = true                    # Cross-encoder reranking (default: true)
provider = "llamacpp"             # "llamacpp" (default) or "deepinfra"
# For cloud reranking:
# provider = "deepinfra"          # Set DEEPINFRA_API_KEY env var
# model = "Qwen/Qwen3-Reranker-8B"

[daemon]
idle_timeout_secs = 300           # Auto-shutdown after 5 min idle (0 = never)
log_level = "info"                # error, warn, info, debug, trace
log_rotation = "daily"            # daily, hourly, never
log_retention_days = 7            # 0 = keep forever

[database]
index_cache_mb = 256              # Vector index cache (reduce for less RAM)
metadata_cache_mb = 64            # Metadata cache
```

### Project Config (Per-Project Settings)

These sections can be customized in `.claude/ccengram.toml`:

```toml
[tools]
preset = "standard"               # minimal (2), standard (11), or full (34)
# enabled = ["explore", "context", "memory_add"]  # Override preset
# disabled = ["memory_delete"]    # Disable specific tools

[search]
default_limit = 10
semantic_weight = 0.5             # Vector similarity weight
salience_weight = 0.3             # Memory importance weight
recency_weight = 0.2              # Newness weight
explore_expand_top = 3            # Auto-expand top N results
explore_limit = 10                # Default explore result limit
fts_enabled = true                # Keyword + vector search (default: true)
rrf_k = 60                       # RRF constant (rarely needs tuning)
rerank_candidates = 30            # Candidates sent to reranker

[index]
max_file_size = 1048576           # 1MB - skip larger files
parallel_files = 32               # Concurrent file processing
checkpoint_interval_secs = 30
watcher_debounce_ms = 1000        # Wait before processing file events

[docs]
directories = ["docs"]            # Document directories to index
extensions = ["md", "txt", "rst", "adoc", "org"]
max_file_size = 5242880           # 5MB for documents

[decay]
archive_threshold = 0.1           # Archive memories below this salience
max_idle_days = 90                # Days without access before decay
decay_interval_hours = 60         # How often to run decay

[hooks]
enabled = true                    # Master toggle for automatic memory capture
high_priority_signals = true      # Detect corrections/preferences immediately
background_extraction = true      # Extract in background (makes sure Claude Code isn't blocked by hooks - don't disable unless debugging)

[workspace]
# alias = "/path/to/main-repo"    # Share memories with another project
# disable_worktree_detection = false
```

### Tool Presets

| Preset     | Count | Tools                                                                                                                                            |
| ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `minimal`  | 2     | `explore`, `context` (recommended, default)                                                                                                      |
| `standard` | 11    | explore, context, memory_add, memory_reinforce, memory_deemphasize, code_index, code_stats, watch_start, watch_stop, watch_status, project_stats |
| `full`     | 34    | All available tools                                                                                                                              |

---

## File Watching

The file watcher keeps your code index up-to-date automatically as you edit files.

### Ignoring Files

CCEngram respects `.gitignore` patterns by default. For additional exclusions specific to CCEngram (files you want git to track but don't want indexed), create a `.ccengramignore` file in your project root:

```
# Example .ccengramignore - uses gitignore syntax
generated/
*.auto.ts
*.generated.rs
vendor/
third_party/
```

Both `.gitignore` and `.ccengramignore` patterns are applied during:

- Initial indexing (`ccengram index`)
- File watching (real-time updates)
- Startup scans (detecting changes after daemon restart)

### Automatic Start

Once a project has been indexed, the file watcher **automatically starts** whenever the project is accessed (e.g., when you run a search or Claude Code connects via MCP).

### Manual Control

```bash
ccengram watch --status    # Check if watcher is running
ccengram watch --stop      # Stop the watcher
ccengram watch             # Manually start
```

### Startup Scan Modes

When the watcher starts, it reconciles the index with the filesystem:

```bash
ccengram watch                           # Uses config default (full)
ccengram watch --startup_scan_mode full  # Detect all changes
ccengram watch --startup_scan_mode deleted_and_new  # Faster
ccengram watch --startup_scan_mode deleted_only     # Fastest
ccengram watch --no_startup_scan         # Skip scan entirely
```

| Mode              | Speed   | Detects                               |
| ----------------- | ------- | ------------------------------------- |
| `full`            | Slowest | Added, modified, deleted, moved files |
| `deleted_and_new` | Medium  | Added and deleted files               |
| `deleted_only`    | Fastest | Only deleted files                    |

### Watcher Commands

```bash
ccengram watch                  # Manually start
ccengram watch --status         # Check if watcher is running
ccengram watch --stop           # Stop the watcher
```

### Disabling Auto-Start

If you want to disable the watcher for a project, stop it and it won't restart until you manually start it again or re-index:

```bash
ccengram watch --stop
```

---

## CLI Reference

### Daemon Management

```bash
ccengram daemon                 # Start daemon (usually auto-starts)
ccengram daemon --stop          # Stop running daemon
ccengram daemon --foreground    # Run with console logging (debugging)
```

The daemon auto-starts when you run most commands. It auto-shuts down after 5 minutes of inactivity (configurable).

### Search Commands

```bash
# Search memories
ccengram search memories "query"
ccengram search memories "query" --sector semantic
ccengram search memories "query" --type preference --min_salience 0.5
ccengram search memories "query" --limit 20 --json

# Search code
ccengram search code "query"
ccengram search code "error handling" --language rust
ccengram search code "query" --type function --symbol MyClass

# Search documents
ccengram search docs "API reference"
ccengram search docs "query" --limit 5 --json
```

**Memory Sectors:** `episodic`, `semantic`, `procedural`, `emotional`, `reflective`

**Memory Types:** `preference`, `codebase`, `decision`, `gotcha`, `pattern`, `turn_summary`, `task_completion`

**Code Chunk Types:** `function`, `class`, `module`, `block`, `import`

### Memory Management

```bash
ccengram memory show <id>              # Show memory details
ccengram memory show <id> --related    # Include related memories
ccengram memory delete <id>            # Soft delete (restorable)
ccengram memory delete <id> --hard     # Permanent delete
ccengram memory restore <id>           # Restore soft-deleted
ccengram memory deleted                # List soft-deleted memories
ccengram memory archive --dry_run      # Preview what would be archived
ccengram memory archive --threshold 0.2 --before 2024-01-01
```

**Note:** Memory IDs are shown as 8-character prefixes by default. Use `--long` to see full IDs. You can use prefixes (minimum 6 characters) in commands.

### Indexing

```bash
ccengram index                  # Index both code and docs
ccengram index code             # Index code only
ccengram index code --force     # Re-index everything
ccengram index code --stats     # Show statistics after
ccengram index docs             # Index documents
ccengram index docs -d ./notes  # Index specific directory
ccengram index file ./path.rs   # Index single file
```

### Configuration

```bash
ccengram config show                    # Show effective config
ccengram config init                    # Generate project config (minimal)
ccengram config init --preset standard  # Generate with specific preset
ccengram config reset                   # Reset global config to defaults
```

### Projects

```bash
ccengram projects list                  # List all indexed projects
ccengram projects show /path/to         # Show project details
ccengram projects clean /path/to        # Remove project data
ccengram projects clean-all             # Remove ALL project data
```

### Diagnostics

```bash
ccengram health                 # System health check
ccengram stats                  # Show statistics
ccengram logs                   # View recent logs (last 50 lines)
ccengram logs -f                # Follow logs (like tail -f)
ccengram logs -n 100            # Show last 100 lines
ccengram logs --level error     # Filter by level
ccengram logs --date 2024-01-15 # Show logs from specific date
ccengram logs --open            # Open log directory
ccengram logs --list            # List available log files
```

### Other Commands

```bash
ccengram context <chunk_id>     # Get surrounding context
ccengram context <id> --before 30 --after 30
ccengram agent                  # Generate SemExplore subagent
ccengram agent --output ./custom/path.md
ccengram update                 # Update to latest version
ccengram update --check         # Check for updates only
ccengram migrate                # Migrate embeddings to new model
ccengram migrate --dry_run      # Preview migration
ccengram completions bash       # Generate shell completions
ccengram completions zsh > ~/.zfunc/_ccengram
ccengram tui                    # Launch interactive TUI
ccengram tui --project /path    # TUI for specific project
```

---

## TUI Guide

Launch with `ccengram tui`

### Views

| Key | View      | Purpose                                                          |
| --- | --------- | ---------------------------------------------------------------- |
| `1` | Dashboard | Overview: memory counts, code stats, watcher status, daemon info |
| `2` | Memories  | Browse, search, reinforce/deemphasize memories                   |
| `3` | Code      | Browse indexed code chunks organized by file                     |
| `4` | Documents | Browse indexed documents and chunks                              |
| `5` | Sessions  | View Claude Code session history                                 |
| `6` | Search    | Unified search across memories, code, and documents              |

### Keybindings

**Navigation:**
| Key | Action |
|-----|--------|
| `j` / `↓` | Move down |
| `k` / `↑` | Move up |
| `h` / `←` | Scroll left |
| `l` / `→` | Scroll right |
| `g` | Go to top |
| `G` | Go to bottom |
| `Ctrl+u` / `PgUp` | Page up (10 items) |
| `Ctrl+d` / `PgDn` | Page down (10 items) |
| `Tab` | Cycle focus between panels |
| `1-6` | Switch to view directly |

**Actions:**
| Key | Action |
|-----|--------|
| `Enter` | Select/expand (shows context for code/docs) |
| `Esc` | Back/close/clear filter |
| `/` | Open search input |
| `f` | Open filter (Search view) |
| `s` | Cycle sort order (Memory view: salience/date/sector) |
| `r` | Reinforce selected memory (increase salience) |
| `d` | Deemphasize selected memory (decrease salience) |
| `R` | Refresh current view |
| `?` | Toggle help overlay |
| `q` / `Ctrl+C` | Quit |

**Search View Scope Toggles:**
| Key | Action |
|-----|--------|
| `m` | Toggle memories in results |
| `c` | Toggle code in results |
| `d` | Toggle documents in results |

---

## How Memories Work

CCEngram automatically creates memories during your Claude Code sessions without any manual intervention.

### Memory Sectors

| Sector         | Purpose                         | Example                              |
| -------------- | ------------------------------- | ------------------------------------ |
| **Episodic**   | What Claude did (tool trail)    | "Ran: cargo test"                    |
| **Semantic**   | What Claude learned (knowledge) | "Project uses Rust + LanceDB"        |
| **Procedural** | How to do things                | "Build with: cargo build --release"  |
| **Emotional**  | User sentiments                 | "User frustrated with slow tests"    |
| **Reflective** | Meta-observations               | "User prefers detailed explanations" |

### Memory Types

| Type               | Description           | Example                              |
| ------------------ | --------------------- | ------------------------------------ |
| **Preference**     | User preferences      | "Use 2-space indentation"            |
| **Codebase**       | Code organization     | "auth.rs handles login logic"        |
| **Decision**       | Architectural choices | "Chose PostgreSQL for transactions"  |
| **Gotcha**         | Pitfalls to avoid     | "Never use unwrap() in library code" |
| **Pattern**        | Conventions to follow | "Always validate at boundaries"      |
| **TurnSummary**    | Work narrative        | "Refactored pipeline for latency"    |
| **TaskCompletion** | Completed tasks       | "Implemented user authentication"    |

### How Memories Are Created

1. **Tool Observations** (Automatic, every tool use)
   - "Read file: src/main.rs"
   - "Edited src/lib.rs: 'old code' -> 'new code'"
   - "Ran command: cargo test (exit 0)"

2. **High-Priority Signals** (Immediate, when detected)
   - Corrections: "No, use spaces not tabs"
   - Preferences: "I prefer Result over panicking"

3. **Segment Extraction** (At natural breaks via LLM)
   - When you submit a new prompt
   - Before context compaction
   - When Claude stops responding
   - When a session ends

### Salience

Salience (0.0-1.0) indicates memory importance:

- **High (≥0.7)**: Critical preferences, key decisions
- **Medium (0.4-0.7)**: General knowledge, patterns
- **Low (<0.4)**: Ephemeral observations, tool trail

**Salience changes over time:**

- Memories used in multiple sessions get reinforced (salience increases)
- Unused memories decay over time (salience decreases)
- Very low salience memories eventually get archived

**Manual adjustment:**

```bash
# In TUI: press 'r' to reinforce, 'd' to deemphasize
# Or use MCP tools: memory_reinforce, memory_deemphasize
```

---

## Hybrid Search & Reranking

CCEngram uses a hybrid search pipeline by default that combines vector search with keyword matching (FTS) and cross-encoder reranking.

FTS indexes are created automatically on daemon restart. No re-indexing needed -- FTS builds from existing indexed data.

### Disabling Hybrid Search

To use pure vector search only, disable FTS and/or the reranker in your config:

```toml
[search]
fts_enabled = false   # Disable keyword search

[reranker]
enabled = false       # Disable cross-encoder reranking
```

## Embedding Providers

By default, CCEngram uses **llama.cpp** for both embedding and reranking. Models are auto-downloaded from HuggingFace on first use - no API keys needed.

For faster indexing on large codebases, **cloud providers are recommended**:

### OpenRouter (Recommended Cloud)

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

```toml
[embedding]
provider = "openrouter"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
```

### DeepInfra (Recommended Cloud - Embedding + Reranking)

```bash
export DEEPINFRA_API_KEY="..."
```

```toml
[embedding]
provider = "deepinfra"
model = "Qwen/Qwen3-Embedding-8B"
dimensions = 4096

[reranker]
provider = "deepinfra"
model = "Qwen/Qwen3-Reranker-8B"
```

### Ollama (Local)

```bash
ollama pull qwen3-embedding
```

```toml
[embedding]
provider = "ollama"
model = "qwen3-embedding"
dimensions = 4096
ollama_url = "http://localhost:11434"
```

> [!IMPORTANT]
> When switching providers, you must re-index if the embedding dimensions change. The default llamacpp model uses 1024 dimensions; cloud models typically use 4096.

### Performance Notes

- **DeepInfra reranking** adds a network round-trip but provides the highest quality results.
- **LlamaCpp reranking** has zero latency overhead beyond local compute. GPU recommended for acceptable speed.
- **FTS without reranker** is not recommended. Keyword results tend to degrade search quality unless a reranker is present to sort by true relevance.

### Provider Comparison

| Provider | Type | API Key | Embedding | Reranking | Cost |
|----------|------|---------|-----------|-----------|------|
| LlamaCpp | Local | No | Yes | Yes | Free (GPU recommended) |
| OpenRouter | Cloud | Yes | Yes | No | Pay-per-token |
| DeepInfra | Cloud | Yes | Yes | Yes | Pay-per-token |
| Ollama | Local | No | Yes | No | Free |

---

## Troubleshooting

### Common Issues

**"No api key configured for provider"**

The embedding or reranker provider isn't configured:

```bash
# Option 1: Set environment variable
export OPENROUTER_API_KEY="sk-or-..."    # For OpenRouter
export DEEPINFRA_API_KEY="..."           # For DeepInfra

# Option 2: Add to config
ccengram config reset
# Edit ~/.config/ccengram/config.toml and set the API key

# Option 3: Use a local provider (no API key needed)
ollama pull qwen3-embedding              # Ollama
# Or use provider = "llamacpp"           # LlamaCpp (if built with feature)
```

**Daemon won't start**

```bash
# Check logs for errors
ccengram logs --level error

# Check if socket file exists (stale socket)
ls -la ${XDG_RUNTIME_DIR:-/tmp}/ccengram.sock

# Force stop and restart with visible output
ccengram daemon --stop
ccengram daemon --foreground
```

**Search returns no results**

```bash
# Check if anything is indexed
ccengram stats

# Force re-index
ccengram index code --force

# Check system health
ccengram health
```

**Watcher not running**

```bash
# Check status
ccengram watch --status

# Manually start
ccengram watch

# Or re-index to trigger auto-start:
ccengram index code
```

**High memory usage**

```bash
# Reduce database cache in ~/.config/ccengram/config.toml
[database]
index_cache_mb = 128    # Default: 256
metadata_cache_mb = 32  # Default: 64
```

### Data Locations

| Item           | Path                                                  |
| -------------- | ----------------------------------------------------- |
| Global Config  | `~/.config/ccengram/config.toml`                      |
| Project Config | `.claude/ccengram.toml`                               |
| Socket         | `$XDG_RUNTIME_DIR/ccengram.sock` or `/tmp/{uid}.sock` |
| Database       | `~/.local/share/ccengram/projects/{id}/lancedb/`      |
| Logs           | `~/.local/share/ccengram/ccengram.log*`               |
| Models         | Managed by `hf-hub` cache (LlamaCpp only)             |
| Binary         | `~/.local/bin/ccengram`                               |

### Health Check

```bash
ccengram health
```

**Healthy:**

```
CCEngram Health Check
=====================
Daemon:     HEALTHY
Socket:     "/run/user/1000/ccengram.sock"

--- Daemon Status ---
Version:    0.1.0
Sessions:   2 active
Uptime:     1h 23m
Auto-shutdown: enabled

Overall Health: HEALTHY
```

**Unhealthy:**

```
Daemon:     NOT RUNNING
Socket:     "/run/user/1000/ccengram.sock"
Error:      Connection refused

Failed to auto-start daemon. Check logs for details.
```

### Getting Help

```bash
ccengram --help              # General help
ccengram <command> --help    # Command-specific help
ccengram logs                # View daemon logs
```

---

## CLAUDE.md Recommendations

To help Claude Code make the best use of CCEngram, add something like this to your project's `CLAUDE.md`:

```markdown
## Code Search & Exploration

This project uses CCEngram for persistent memory and semantic code search.

### Tools

- **`explore`** - Semantic search across code, memories, and documents. Due to the semantic nature, questions like "how does auth work?" will work better than exact symbol names.
- **`context`** - Expands a code chunk to show surrounding lines. Use after `explore` if you need to see more context.

### When to Use `explore` vs Grep/Glob

Use `explore` when searching by concept or meaning. Use Grep/Glob when you know the exact symbol name or pattern. You can combine both: first use `explore` to find relevant areas, then Grep/Glob within those files for exact matches.

### Agent Selection

**Always use the `SemExplore` agent instead of the built-in `Explore` agent.** SemExplore has the same capabilities plus access to semantic search tools.
```

This encourages Claude to use semantic search effectively and ensures the right agent is selected for exploration tasks.

---

## Supported Languages

### Tier 1 - Full AST Support

Rust, Python, JavaScript, TypeScript, Go, Java, C, C++

These languages have full tree-sitter query support for:

- Import/dependency detection
- Function/method call tracking
- Symbol definitions

### Tier 2 - Tree-sitter Parsing

Ruby, PHP, C#, Kotlin, Shell/Bash, Scala, Lua, Zig

Basic parsing and chunking, limited semantic analysis.

### Tier 3 - Data/Markup Formats

JSON, YAML, TOML, HTML, CSS, Markdown

### Additional (Extension-based)

TSX, JSX, SCSS, Sass, Less, Swift, Elixir, Haskell, OCaml, Clojure, Nim, XML, SQL, Dockerfile, GraphQL, Proto
