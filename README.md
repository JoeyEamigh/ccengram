# CCEngram

Semantic code search and persistent memory for Claude Code.

CCEngram lets Claude Code search your codebase by meaning, not just keywords. Ask questions like "how does authentication work?" or "where are errors handled?" and get relevant results even when the exact terms don't appear in the code. Optionally, enable automatic memory extraction (uses Haiku on your Claude Code subscription) to have Claude remember your preferences, decisions, and project-specific patterns across sessions.

> [!WARNING]
> This project was primarily vibecoded. It is not a reflection of my skills and definitely not Rust best practices. I made it to solve a specific problem (working in large monorepos with Claude), so likely won't accept a bunch of tangential features.

The intention is that this plugin is as set-and-forget as possible. Once you run an initial indexing for a project, CCEngram's file watcher keeps the index up-to-date automatically as you code. The daemon will shut itself down after 5 minutes of inactivity, and automatically restart and reconcile index changes as requests come in.

## Features

- **Hybrid Code Search** - Find code by meaning (vector) and keywords (FTS), fused via Reciprocal Rank Fusion
- **Cross-Encoder Reranking** - Reranker refines results after initial retrieval (enabled by default via llama.cpp)
- **Persistent Memory** - Preferences, decisions, and patterns remembered across sessions (opt-in per project)
- **File Watching** - Index updates automatically as you code
- **Per-Project Isolation** - Each project has its own index and memories
- **Interactive TUI** - Browse and manage code, documents, and memories

## Installation

### Prerequisites

CCEngram uses **llama.cpp** for embedding and reranking by default - no API keys required. Models are downloaded automatically from HuggingFace on first use.

For better speed and performance, cloud providers are recommended:

- **[OpenRouter](https://openrouter.ai/)** (cloud, recommended) - Set `OPENROUTER_API_KEY`
- **[DeepInfra](https://deepinfra.com/)** (cloud, recommended) - Set `DEEPINFRA_API_KEY`
- **[Ollama](https://ollama.ai/)** (local) - Run embeddings locally

### Install

```bash
curl -fsSL https://raw.githubusercontent.com/JoeyEamigh/ccengram/main/scripts/install.sh | bash
```

Or install from source:

```bash
cargo install --git https://github.com/JoeyEamigh/ccengram --bin ccengram
```

For in-process llama.cpp with GPU acceleration (recommended for local inference):

```bash
cargo install --git https://github.com/JoeyEamigh/ccengram --bin ccengram --features llama-cpp,vulkan
# GPU options: vulkan (recommended), cuda, metal
```

### Claude Code Plugin

Install from the Claude Code plugin marketplace:

```
/plugin marketplace add JoeyEamigh/ccengram
/plugin install ccengram
```

## Quick Start

### 1. Initialize Your Project

Navigate to your project and create a project-specific config:

```bash
cd /your/project
ccengram config init # creates the project config
ccengram agent # installs the SemExplore agent
```

This creates `.claude/ccengram.toml`. The default `minimal` preset is recommended for most users. You can choose a different preset:

- `minimal` - 2 tools: `explore`, `context` (recommended)
- `standard` - 11 tools: search + memory management + code maintenance
- `full` - 34 tools: everything

```bash
ccengram config init --preset standard  # If you want the agent to be able to modify the database
```

### 2. Index Your Codebase

```bash
ccengram index
```

This scans your project and creates semantic embeddings. By default, CCEngram uses llama.cpp for embedding and reranking - models are downloaded automatically on first use, no configuration needed.

> [!TIP]
> **For better speed and performance**, use a cloud embedding provider. OpenRouter and DeepInfra are recommended. See [Embedding Providers](#embedding-providers) below.

### 3. File Watching (Automatic)

The file watcher **automatically starts** after indexing and whenever an indexed project is accessed. Your index stays up-to-date as you edit files.

#### Ignoring Files

CCEngram respects `.gitignore` patterns. For additional exclusions specific to CCEngram, create a `.ccengramignore` file in your project root using the same syntax:

```
# Example .ccengramignore
generated/
*.auto.ts
vendor/
```

To manually control the watcher:

```bash
ccengram watch --status    # Check watcher status
ccengram watch --stop      # Stop the watcher
ccengram watch             # Manually start (if stopped)
```

The watcher performs a **startup scan** when launched to detect any files that changed while it wasn't running.

### 4. Start Using

**With Claude Code:** MCP tools are automatically available. Claude uses `explore` and `context` to search your codebase and memories.

#### CLAUDE.md Recommendations

To help Claude Code make the best use of CCEngram, add something like this to your project's `CLAUDE.md`:

```markdown
## Semantic Code Search

This project uses CCEngram for semantic code search.

### Tools

- **`explore`** - Semantic search across code and documents. Questions like "how does auth work?" find relevant code even without exact keyword matches.
- **`context`** - Expands a code chunk to show surrounding lines. Use after `explore` if you need to see more context.

### When to Use `explore` vs Grep/Glob

Use `explore` when searching by concept or meaning. Use Grep/Glob when you know the exact symbol name or pattern. You can combine both: first use `explore` to find relevant areas, then Grep/Glob within those files for exact matches.

### Agent Selection

**Always use the `SemExplore` agent instead of the built-in `Explore` agent.** SemExplore has the same capabilities plus access to semantic search tools.
```

## Projects

CCEngram maintains separate data for each project:

```bash
ccengram projects list              # See all indexed projects
ccengram projects show /path/to    # Show project details
ccengram projects clean /path/to   # Remove project data
```

Each project gets:

- Its own LanceDB database at `~/.local/share/ccengram/projects/{id}/`
- Isolated memories, code index, and documents
- Independent configuration via `.claude/ccengram.toml`

**Project identification:** Based on git root (if in a repo) or directory path. Git worktrees sharing the same repo share memories by default.

## Configuration

### Two-Level Config System

| Level       | Path                             | Purpose                                       |
| ----------- | -------------------------------- | --------------------------------------------- |
| **Global**  | `~/.config/ccengram/config.toml` | Embedding provider, daemon settings, defaults |
| **Project** | `.claude/ccengram.toml`          | Project-specific overrides                    |

Project config overrides global config.

### Config Commands

```bash
ccengram config show                     # Show effective configuration
ccengram config init                     # Generate project config (minimal preset, recommended)
ccengram config init --preset standard   # Generate with specific preset
ccengram config reset                    # Reset global config to defaults
```

Global config (`~/.config/ccengram/config.toml`) is created automatically on first use.

### Global Config Sections

These settings are daemon-level and **must** be in global config:

```toml
[embedding]
provider = "llamacpp"             # "llamacpp" (default), "openrouter", "deepinfra", or "ollama"
dimensions = 1024                 # 1024 for llamacpp 0.6B, 4096 for cloud 8B models

[reranker]
enabled = true                    # Cross-encoder reranking (default: true)
provider = "llamacpp"             # "llamacpp" (default) or "deepinfra"

[daemon]
idle_timeout_secs = 300           # Auto-shutdown after 5 min idle (0 = never)
log_level = "info"                # error, warn, info, debug, trace
log_rotation = "daily"
log_retention_days = 7

[database]
index_cache_mb = 256
metadata_cache_mb = 64
```

### Project Config Sections

These can be customized per-project:

```toml
[tools]
preset = "minimal"                # minimal (recommended), standard, or full

[search]
default_limit = 10
semantic_weight = 0.5
salience_weight = 0.3
recency_weight = 0.2
fts_enabled = true                # Keyword search + vector search (default: true)
rrf_k = 60                        # Reciprocal Rank Fusion constant
rerank_candidates = 30            # Max candidates sent to reranker

[index]
max_file_size = 1048576           # 1MB
parallel_files = 32

[docs]
directories = ["docs"]
extensions = ["md", "txt", "rst", "adoc", "org"]

[decay]
archive_threshold = 0.1
max_idle_days = 90

[hooks]
enabled = false                   # Set to true to enable automatic memory creation

[workspace]
# alias = "/path/to/main-repo"    # Share memories with another project (useful for worktrees)
```

## Embedding Providers

By default, CCEngram uses **llama.cpp** for both embedding and reranking. This works out of the box with no API keys - models are auto-downloaded from HuggingFace on first use. For better speed and performance, cloud providers are recommended.

### OpenRouter (Recommended Cloud)

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

```toml
# ~/.config/ccengram/config.toml
[embedding]
provider = "openrouter"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
```

### DeepInfra (Recommended Cloud)

Supports both embedding and reranking.

```bash
export DEEPINFRA_API_KEY="..."
```

```toml
# ~/.config/ccengram/config.toml
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
# ~/.config/ccengram/config.toml
[embedding]
provider = "ollama"
model = "qwen3-embedding"
dimensions = 4096
ollama_url = "http://localhost:11434"
```

> [!IMPORTANT]
> When switching providers, you must re-index if the embedding dimensions change. The default llamacpp model uses 1024 dimensions; cloud models typically use 4096.

## How It Works

CCEngram runs as a daemon that Claude Code connects to via MCP.

### Hybrid Search

Your codebase is parsed into semantic chunks (functions, classes, etc.) and embedded into a vector database. When you search, queries are matched by meaning rather than exact text. "Where is authentication handled?" finds auth-related code even if it doesn't contain the word "authentication".

By default, CCEngram also performs full-text keyword search in parallel and merges results using Reciprocal Rank Fusion (RRF). This catches exact identifier matches that vector search might miss. A cross-encoder reranker then re-scores the top candidates for higher precision.

### Optional: Persistent Memory

When enabled in config (`hooks.enabled = true`), CCEngram can also create persistent memories from your sessions:

1. **Tool observations** are recorded (files read, commands run, edits made)
2. **High-priority signals** like corrections and preferences are captured immediately
3. **At natural breaks**, an LLM extracts learnings (decisions, gotchas, patterns)

Memories have **salience** (0.0-1.0) that determines search ranking. Frequently-used memories get reinforced; unused ones eventually archive.

| Type           | Example                                     |
| -------------- | ------------------------------------------- |
| **Preference** | "User prefers 2-space indentation"          |
| **Decision**   | "Chose PostgreSQL for transactional safety" |
| **Gotcha**     | "Never use unwrap() in library code"        |
| **Pattern**    | "Always validate inputs at boundaries"      |

To enable memory for a project, add to `.claude/ccengram.toml`:

```toml
[hooks]
enabled = true
```

## TUI

Launch with `ccengram tui`

## Supported Languages

**Full AST support:** Rust, Python, JavaScript, TypeScript, Go, Java, C, C++

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
├─────────────┴──────┬──────┴─────────────┴──────────────────┤
│   Hybrid Search    │   Reranker                            │
│   Vector + FTS     │   llama.cpp / DeepInfra               │
└──────┬─────────────┴───────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                 LanceDB (per-project)                       │
│             Vector Index + FTS Index                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Locations

| Item           | Path                                     |
| -------------- | ---------------------------------------- |
| Global Config  | `~/.config/ccengram/config.toml`         |
| Project Config | `.claude/ccengram.toml`                  |
| Database       | `~/.local/share/ccengram/projects/{id}/` |
| Models         | `~/.cache/huggingface/hub/` (llama.cpp)  |
| Logs           | `~/.local/share/ccengram/ccengram.log*`  |
| Socket         | `$XDG_RUNTIME_DIR/ccengram.sock`         |

## Other Notes

> [!NOTE]
> **Windows is not currently supported.** Should work on WSL though.

> Why jemalloc?

CCEngram uses LanceDB which relies on Arrow, which for some reason on Linux wasn't releasing memory back to the OS with the default allocator. The program now idles at around 40MiB vs about 12GiB.

## Documentation

See [`docs/user-guide.md`](docs/user-guide.md) for comprehensive documentation.
