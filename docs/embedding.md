# Embedding System

This document explains the embedding system used in CCEngram, including the provider architecture and configuration options.

## Overview

CCEngram uses embedding models (default: **Qwen3-Embedding-0.6B** via llama.cpp) producing high-dimensional embeddings for code, memory, and document search. The embedding system supports multiple providers through a unified architecture:

1. **Multi-provider**: Four embedding backends -- LlamaCpp (default, in-process via llama-cpp-2), OpenRouter, DeepInfra, and Ollama. Cloud providers (OpenRouter, DeepInfra) are recommended for better speed and performance.
2. **Trust the model**: Modern embedding models understand semantic relationships between programming concepts without explicit synonym dictionaries.
3. **Hybrid search**: Vector embeddings power semantic search, complemented by keyword search (FTS) and cross-encoder reranking by default (see `embedding_search.md`).

## EmbeddingMode

The embedding system distinguishes between two modes:

### Document Mode

Used when **indexing** content (code chunks, memories, documents). Text is embedded as-is without any instruction prefix.

```rust
// Indexing a code chunk
let embedding = provider.embed(chunk.content, EmbeddingMode::Document).await?;
```

### Query Mode

Used when **searching** for content. For instruction-aware models like qwen3-embedding, queries are formatted with an instruction prefix that tells the model the retrieval task.

```rust
// Searching for code
let embedding = provider.embed(query, EmbeddingMode::Query).await?;
```

The formatted query looks like:

```
Instruct: Given a code search query, retrieve relevant code snippets and documentation that match the query
Query:authentication jwt
```

This instruction format is specific to qwen3-embedding. The model uses the instruction to understand the retrieval task and produce better query embeddings.

## Configuration

The query instruction is configurable in `ccengram.toml`:

```toml
[embedding]
# Optional instruction for query mode (qwen3-embedding style)
# Set to empty string "" to disable instruction formatting
query_instruction = "Given a code search query, retrieve relevant code snippets and documentation that match the query"
```

### When to disable

If using an embedding model that doesn't support instruction-based retrieval (e.g., older sentence-transformers models), set `query_instruction = ""` to embed queries as raw text.

## Why Pure Semantic Search?

### The Problem with Hardcoded Expansion

Previously, code search used a 300+ line synonym dictionary:

```rust
("auth", &["authentication", "authorization", "login", "session", "token", "jwt", "oauth"...])
```

This approach has several problems:

1. **Limited coverage**: Misses domain-specific terms in the user's codebase. "login flow" won't expand to "OAuth callback handler" even if they're semantically related.

2. **No confidence weighting**: Treats all expansions equally. Expanding "auth" to 10 terms adds noise when only 2 are relevant.

3. **Maintenance burden**: Requires manual updates as terminology evolves.

4. **Performance**: Expanded queries can be longer, slower to embed.

### The Solution

The 4096-dimensional embedding space naturally encodes semantic relationships:

- The vector for "auth" is close to "authentication", "jwt", "oauth"
- The vector for "mutex" is close to "lock", "concurrent", "sync"
- The vector for "LTV" is close to "lifetime value" (domain knowledge)

By trusting the embedding model, we get:

- **Better recall**: Finds domain-specific relationships the synonym map missed
- **Weighted similarity**: Closer vectors = stronger relationship
- **Zero maintenance**: The model already knows these relationships
- **Domain adaptability**: Works with any codebase terminology

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EmbeddingProvider trait                   │
├─────────────────────────────────────────────────────────────┤
│  embed(text, mode) -> Vec<f32>                              │
│  embed_batch(texts, mode) -> Vec<Vec<f32>>                  │
└─────────────────────────────────────────────────────────────┘
              │
    ┌─────────┼──────────────┬──────────────────┐
    ▼         ▼              ▼                   ▼
┌────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Ollama │ │ OpenAiCompat │ │ OpenAiCompat │ │ LlamaCpp     │
│Provider│ │ (OpenRouter) │ │ (DeepInfra)  │ │ Provider     │
│ /api/  │ │ /v1/embed    │ │ /v1/embed    │ │ (in-process) │
│ embed  │ └──────┬───────┘ └──────┬───────┘ │ feature-gate │
└────────┘        └──────┬─────────┘         └──────────────┘
                         ▼
                ┌─────────────────┐
                │ ResilientProv.  │
                │ (Retries/Split) │
                └─────────────────┘
```

### Providers

| Provider | Type | API Format | Auth | Notes |
|----------|------|------------|------|-------|
| **Ollama** | Local | `/api/embed` (Ollama-native) | None | Separate implementation |
| **OpenRouter** | Cloud | `/v1/embeddings` (OpenAI-compat) | `OPENROUTER_API_KEY` | Via `OpenAiCompatibleProvider` |
| **DeepInfra** | Cloud | `/v1/embeddings` (OpenAI-compat) | `DEEPINFRA_API_KEY` | Via `OpenAiCompatibleProvider` |
| **LlamaCpp** | In-process | Direct FFI via `llama-cpp-2` | None | Feature-gated (`llama-cpp` feature) |

### `OpenAiCompatibleProvider`

OpenRouter, DeepInfra, and external llama-server all use the same OpenAI `/v1/embeddings` API format. Rather than duplicating code, a single `OpenAiCompatibleProvider` handles all three. It is configured with a base URL, optional API key, and model name. The old `OpenRouterProvider` was replaced by this generic provider.

The `LlamaCpp` config variant can also use this provider when pointing at an external `llama-server` HTTP endpoint.

### LlamaCpp In-Process Embedding

When the `llama-cpp` feature is enabled and `provider = "llamacpp"`, CCEngram loads a GGUF embedding model in-process via `llama-cpp-2` FFI bindings. No subprocess or HTTP involved.

- Models auto-download from HuggingFace on first use via `hf-hub`
- Default model: `Qwen/Qwen3-Embedding-0.6B-GGUF` (~639MB)
- GPU offloading via Vulkan (default), CUDA, or Metal feature flags
- Uses `spawn_blocking` for CPU/GPU-bound inference calls

### Provider Layers

1. **Base providers** (Ollama, OpenAiCompatible, LlamaCpp): Handle API communication and instruction formatting
2. **ResilientProvider**: Wraps cloud providers with retry logic, exponential backoff, and batch splitting on failure

## Search Flow

The search pipeline now supports hybrid retrieval. See `embedding_search.md` for the full pipeline diagram. At the embedding level:

```
Query: "how does auth work"
         │
         ▼
┌─────────────────────────┐
│   format_for_embedding  │
│   (Query mode)          │
└─────────────────────────┘
         │
         ▼ "Instruct: Given a code search...\nQuery:how does auth work"
┌─────────────────────────┐
│   Embedding Provider    │
│   (any of the 4 above)  │
└─────────────────────────┘
         │
         ▼ Vec<f32>
    ┌─────────┐
    │  Hybrid  │  Vector search + optional FTS + optional reranking
    │ Pipeline │  (see embedding_search.md)
    └─────────┘
```

## Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENROUTER_API_KEY` | OpenRouter | Yes (or set in config) |
| `DEEPINFRA_API_KEY` | DeepInfra | Yes (or set in config) |
| `OLLAMA_URL` | Ollama | No (defaults to `http://localhost:11434`) |

## References

- [qwen3-embedding documentation](https://huggingface.co/Alibaba-NLP/qwen3-embedding-8b)
- `crates/backend/src/embedding/mod.rs` - EmbeddingMode and provider trait
- `crates/backend/src/embedding/openai_compat.rs` - OpenAiCompatibleProvider (OpenRouter, DeepInfra, llama-server)
- `crates/backend/src/embedding/llamacpp.rs` - LlamaCpp in-process provider (feature-gated)
- `crates/backend/src/embedding/ollama.rs` - Ollama implementation
- `crates/backend/src/service/code/search.rs` - Code search with ranking
