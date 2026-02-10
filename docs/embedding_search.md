# Hybrid Search Design

CCEngram supports a multi-stage search pipeline: parallel vector + keyword retrieval, RRF fusion, and cross-encoder reranking. This document explains the design philosophy and pipeline architecture.

## Search Modes

Users effectively get three search modes:

1. **Hybrid + Rerank** (`fts_enabled = true`, `reranker.enabled = true`): Full pipeline. **The default.** Keyword search finds exact matches, vector search finds semantic matches, reranker sorts by true relevance. Best quality.
2. **Vector-only** (`fts_enabled = false`, `reranker.enabled = false`): Pure semantic similarity. Good for conceptual queries.
3. **Hybrid** (`fts_enabled = true`, `reranker.enabled = false`): Vector + keyword with RRF fusion. Not recommended -- keyword search without reranking tends to degrade results.

Recommendation: use mode 1 (default, best quality) or mode 2 (simple).

## Pipeline Architecture

```
Query
  |
  +---> Vector Search (LanceDB ANN, top 50) -------+
  |                                                  |
  +---> Keyword Search (LanceDB FTS, top 50) -------+   [only if fts_enabled]
  |                                                  |
  v                                                  v
          Fusion (RRF if both, passthrough if vector-only)
                       |
                       v
              Top 30 candidates
                       |
            [if reranker enabled]
                       |
                       v
              Cross-encoder rerank
                       |
                       v
         Domain-specific post-ranking
              (existing logic)
                       |
                       v
                Final results
```

When `fts_enabled = false`, the pipeline degrades to vector-only retrieval. When the reranker is disabled, fusion results pass directly to domain-specific ranking.

## Why Vector Search?

High-dimensional embeddings naturally encode semantic relationships. The model understands that "auth" relates to "authentication", "JWT", and "OAuth" without explicit synonym dictionaries. By trusting the embedding model:

- **Better recall**: Finds domain-specific relationships no synonym map could cover
- **Weighted similarity**: Closer vectors = stronger relationship
- **Zero maintenance**: The model already knows these relationships

## Keyword Search (FTS)

### LanceDB Full-Text Search

LanceDB provides native FTS indexes alongside vector indexes on the same tables. No additional infrastructure needed.

FTS indexes are created on:
- `code_chunks.embedding_text` -- enriched text containing definitions, signatures, symbols
- `memories.content` -- natural language memory content
- `documents.content` -- document text

### Code Tokenizer

Code identifiers need pre-processing for effective FTS. The tokenizer in `context/files/code/tokenize.rs` splits identifiers for indexing:

```
"camelCase"          -> "camelcase camel case"
"snake_case"         -> "snake_case snake case"
"HTTPServer"         -> "httpserver http server"
"src/auth/handler.rs" -> "src auth handler rs"
```

Rules:
- Split on `_` (snake_case), case transitions (camelCase), path separators
- Preserve original token alongside splits (exact matches still work)
- Lowercase all tokens
- Filter common code stop words (`fn`, `pub`, `struct`, `impl`, `def`, `class`, etc.)

For memories and documents, no special pre-processing is needed -- the default tokenizer handles natural language well.

### FTS Index Lifecycle

- Created during `ProjectDb::connect()` alongside scalar indexes
- Rebuilt on daemon startup and after full re-index operations
- No column changes needed -- reuses existing `embedding_text` for code

## Reciprocal Rank Fusion (RRF)

When both vector and FTS results are available, they're merged using RRF:

```
score(d) = sum(1 / (k + rank_i(d))) for each ranker i
```

`k=60` is the standard constant from the original RRF paper. Items appearing in both result sets get boosted scores. The `rrf_k` parameter is configurable but rarely needs tuning.

Implementation: `crates/backend/src/service/util/fusion.rs`

## Reranking

### Overview

After fusion produces top-N candidates, an optional cross-encoder reranker scores each (query, document) pair for relevance. Cross-encoders are more accurate than embedding similarity because they see query and document together, but too expensive for first-stage retrieval.

### Providers

| Provider | Type | API | Notes |
|----------|------|-----|-------|
| **DeepInfra** | Cloud | DeepInfra native inference API | Default model: `Qwen/Qwen3-Reranker-8B` |
| **LlamaCpp** | In-process | Direct FFI via `llama-cpp-2` | Default model: `jina-reranker-v2-base-multilingual` |

Embedding and reranking providers are fully independent -- mix and match freely:

| Embedding | Reranker | Use Case |
|-----------|----------|----------|
| Ollama | DeepInfra | Local embedding, cloud reranking |
| DeepInfra | DeepInfra | All-cloud, best quality |
| LlamaCpp | LlamaCpp | Zero API keys, fully local |
| OpenRouter | None | Vector-only (current behavior) |

### Position-Aware Blending

When a reranker is present, RRF scores are blended with reranker scores using position-aware weights:

| RRF Position | RRF Weight | Reranker Weight | Rationale |
|--------------|------------|-----------------|-----------|
| 0-2 | 0.75 | 0.25 | Trust retrieval for top hits |
| 3-9 | 0.60 | 0.40 | Balanced |
| 10+ | 0.40 | 0.60 | Trust reranker more for lower-ranked |

This ensures top retrieval results maintain stability even when the reranker slightly disagrees, while allowing the reranker to promote genuinely relevant items from deeper in the list.

## Domain-Specific Integration

### Code Search

With hybrid search, the existing ranking signals are redistributed:
- **RRF score** (vector + keyword fusion) replaces the separate semantic and symbol signals
- **Importance** (caller_count, visibility) remains as a post-ranking signal
- When FTS is enabled, the in-memory `calculate_symbol_boost` is skipped (FTS subsumes it)

### Memory Search

Memory ranking combines RRF scores with salience, recency, and sector boost as post-ranking signals. Reranking is useful for memories since natural language content is where cross-encoders excel.

### Document Search

Simplest integration: RRF fusion + reranking, no additional domain-specific signals.

## Config Reference

```toml
[search]
fts_enabled = true        # Keyword search alongside vector search (default: true)
rrf_k = 60               # RRF constant (standard value, rarely needs tuning)
rerank_candidates = 30    # Candidates sent to reranker after fusion
```

See `embedding.md` for embedding provider configuration and `user-guide.md` for full config reference.

## Key Files

| Path | Purpose |
|------|---------|
| `service/code/search.rs` | Code search with hybrid pipeline |
| `service/memory/search.rs` | Memory search with hybrid pipeline |
| `service/docs/search.rs` | Document search with hybrid pipeline |
| `service/util/fusion.rs` | RRF implementation and position-aware blending |
| `context/files/code/tokenize.rs` | Code identifier tokenizer for FTS |
| `rerank/mod.rs` | RerankerProvider trait and types |
| `rerank/deepinfra.rs` | DeepInfra reranker implementation |
| `rerank/llamacpp.rs` | LlamaCpp reranker (feature-gated) |
| `db/code/codes.rs` | Code chunk DB operations including FTS search |
| `db/memory/memories.rs` | Memory DB operations including FTS search |
| `db/document/documents.rs` | Document DB operations including FTS search |
| `db/connection.rs` | FTS index creation |
