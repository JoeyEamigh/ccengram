# LanceDB Memory Analysis & Recommendations

## Executive Summary

Memory usage during indexing grows to ~4.6 GB and doesn't release after pipeline completion. The root cause is **LanceDB's internal caching behavior combined with how we open tables**.

**Key finding**: 91.7% of memory growth (+4,218 MB of +4,601 MB total) comes directly from `upsert_code_chunks_batch` → LanceDB's `merge_insert` operation. Our data preparation code (`prep_delta`) consistently shows +0.0 MB - the clones are not the issue.

## Root Cause Analysis

### 1. LanceDB Default Cache Sizes Are Enormous

From Lance documentation:

| Cache | Default Size | Purpose |
|-------|-------------|---------|
| Index Cache | **6 GiB** | Vector/scalar index data for fast lookups |
| Metadata Cache | **1 GiB** | Dataset manifests, schemas, statistics |

**Total default per-table: 7 GiB**

### 2. Caches Are NOT Shared Between Tables by Default

From Lance documentation:
> "The metadata cache is not shared between tables by default. For best performance you should create a single table and share it across your application. Alternatively, you can create a single session and specify it when you open tables."

This means **each table gets its own caches** unless you explicitly share a Session:

| Without Shared Session | With Shared Session |
|------------------------|---------------------|
| 8 tables × 6 GiB index cache = **48 GiB** | All tables share **one** index cache |
| 8 tables × 1 GiB metadata cache = **8 GiB** | All tables share **one** metadata cache |
| **56 GiB theoretical maximum** | Bounded by your Session config |

Current code pattern:
```rust
pub async fn code_chunks_table(&self) -> Result<lancedb::Table> {
    Ok(self.connection.open_table("code_chunks").execute().await?)
}
```

Every call to `*_table()` opens a fresh table handle with its own default caches. This explains the 4.6 GB growth - we're accumulating multiple per-table caches during indexing.

With a shared Session:
- All tables use the same bounded caches
- Dropping Table handles doesn't free cached data (it stays in the Session)
- The ~500 MB fluctuations we observed are likely temporary Arrow/merge buffers

### 3. `merge_insert` Requires Reading Existing Data

The `when_not_matched_by_source_delete` clause forces LanceDB to:
1. Scan existing table data to find rows to delete
2. Load index data to perform the merge
3. Cache this data for potential reuse

With 97 flushes during a full index, each flush populates caches with table data.

### 4. We Have 8 Tables

```
memories, code_chunks, sessions, documents,
session_memories, memory_relationships, document_metadata, indexed_files
```

Without a shared Session, each table maintains independent caches:
- **Theoretical maximum**: 8 tables × 7 GiB = **56 GiB**
- **Observed**: 4.6 GB (not all tables heavily used during indexing)

The `code_chunks` table dominates indexing writes, explaining why memory stabilizes around 4.6 GB rather than reaching the full theoretical maximum.

## Observed Memory Behavior

From the instrumented pipeline run:

```
Baseline:           47.1 MB
First flush pre:    960.9 MB  (+913 MB before any DB write - embedder buffering)
Final:              4,648.3 MB
Total growth:       +4,601 MB

Breakdown by operation:
- code_upsert_delta:    +4,218 MB (91.7%)
- indexed_files_delta:  ~100 MB (2%)
- metadata/doc:         negligible
- prep_delta:           ALWAYS +0.0 MB (our code is clean)
```

## Proposed Solutions

### Solution 1: Shared Session with Controlled Cache Sizes (Primary Fix)

Create a single `Session` with reasonable cache limits and share it across all operations.

```rust
use lance::session::Session;
use object_store::ObjectStoreRegistry;

// At connection creation time
let registry = Arc::new(ObjectStoreRegistry::default());
let session = Arc::new(Session::new(
    256 * 1024 * 1024,   // 256 MB index cache (not 6 GB!)
    64 * 1024 * 1024,    // 64 MB metadata cache (not 1 GB!)
    registry,
));

let connection = lancedb::connect(&db_path)
    .session(session.clone())
    .execute()
    .await?;
```

**Why these sizes?**
- 256 MB index cache: Sufficient for typical vector search patterns
- 64 MB metadata cache: Enough for table metadata without bloat
- Total: 320 MB vs 7 GB default per table

### Solution 2: Configure Per-Table Cache via `lance_read_params`

For finer control, configure caches when opening tables:

```rust
use lance::dataset::ReadParams;

pub async fn code_chunks_table(&self) -> Result<lancedb::Table> {
    let read_params = ReadParams {
        index_cache_size_bytes: 128 * 1024 * 1024,  // 128 MB
        metadata_cache_size_bytes: 32 * 1024 * 1024, // 32 MB
        session: Some(self.shared_session.clone()),  // Share session!
        ..Default::default()
    };

    Ok(self.connection
        .open_table("code_chunks")
        .lance_read_params(read_params)
        .execute()
        .await?)
}
```

### Solution 3: Hold Table Handles Permanently (Performance Optimization)

**Important clarification**: With a shared Session, dropping Table handles does **not** free cached memory. The Table struct is lightweight:
- An Arc reference to the Connection/Session
- Table name and metadata
- Minimal per-table state

The actual cached data (indices, metadata, file manifests) lives in the **Session**. Dropping a Table doesn't evict that data - it stays cached until LRU eviction or Session drop.

The ~500MB fluctuations observed during indexing are likely from:
- Arrow RecordBatch allocations during merge_insert (temporary)
- Intermediate merge operation buffers
- OS memory reporting variance

**This means caching Table handles is a pure performance optimization** (avoiding re-open overhead), not a memory optimization. With a shared Session controlling cache sizes, the simplest approach is to just hold Table handles for the lifetime of ProjectDb:

```rust
pub struct ProjectDb {
    pub project_id: ProjectId,
    pub connection: Connection,
    pub vector_dim: usize,
    session: Arc<Session>,

    // Hold tables permanently - no Option, no locks needed
    // Table is Send + Sync, so concurrent access is safe
    code_chunks: Table,
    documents: Table,
    indexed_files: Table,
    document_metadata: Table,
    memories: Table,
    sessions: Table,
    session_memories: Table,
    memory_relationships: Table,
}

impl ProjectDb {
    pub fn code_chunks_table(&self) -> &Table {
        &self.code_chunks
    }

    pub fn documents_table(&self) -> &Table {
        &self.documents
    }
    // ... etc - simple field access, no async, no Result
}
```

**Benefits:**
- Zero per-operation overhead (no `open_table()` calls)
- No locking required - Table is Send + Sync
- Simpler code - synchronous field access instead of async open
- Tables initialized once in `open_at_path()` after `ensure_tables()`

**Why not Option/OnceCell/lazy init?**
- Dropping tables doesn't save memory (Session holds the caches)
- All tables will be used anyway during normal operation
- Lazy init adds complexity for no benefit
- We control all writes, so staleness isn't a concern

### Solution 4: Environment Variables for Quick Testing

Lance respects environment variables for thread pools. While not directly controlling cache, reducing I/O threads may reduce memory pressure:

```bash
export LANCE_IO_THREADS=4    # Default: 8 (local), 64 (cloud)
export LANCE_CPU_THREADS=4   # Default: num_cpus
```

### Solution 5: Periodic Cache Stats Monitoring

Use Session's introspection methods to monitor cache behavior:

```rust
impl ProjectDb {
    pub async fn log_cache_stats(&self) {
        if let Some(session) = &self.session {
            let index_stats = session.index_cache_stats().await;
            let meta_stats = session.metadata_cache_stats().await;
            let size = session.size_bytes();

            info!(
                index_hits = index_stats.hits,
                index_misses = index_stats.misses,
                meta_hits = meta_stats.hits,
                meta_misses = meta_stats.misses,
                total_size_mb = size / (1024 * 1024),
                "LanceDB cache stats"
            );
        }
    }
}
```

## Recommended Implementation

### Phase 1: Add DatabaseConfig Section

```rust
// In domain/config.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    /// Index cache size in MB (default: 256)
    pub index_cache_mb: usize,

    /// Metadata cache size in MB (default: 64)
    pub metadata_cache_mb: usize,

    /// Log cache stats periodically during indexing (default: false)
    pub log_cache_stats: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            index_cache_mb: 256,
            metadata_cache_mb: 64,
            log_cache_stats: false,
        }
    }
}
```

### Phase 2: Update ProjectDb

```rust
use lancedb::{Session, ObjectStoreRegistry, Table};
use std::sync::Arc;

pub struct ProjectDb {
    pub project_id: ProjectId,
    pub connection: Connection,
    pub vector_dim: usize,
    session: Arc<Session>,

    // Hold all table handles permanently (see Solution 3)
    code_chunks: Table,
    documents: Table,
    indexed_files: Table,
    document_metadata: Table,
    memories: Table,
    sessions_table: Table,  // renamed to avoid confusion with Session
    session_memories: Table,
    memory_relationships: Table,
}

impl ProjectDb {
    pub async fn open_at_path(
        project_id: ProjectId,
        db_path: PathBuf,
        config: Arc<Config>
    ) -> Result<Self> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let session = Arc::new(Session::new(
            config.database.index_cache_mb * 1024 * 1024,
            config.database.metadata_cache_mb * 1024 * 1024,
            registry,
        ));

        let connection = connect(db_path.to_string_lossy().as_ref())
            .session(session.clone())
            .execute()
            .await?;

        // Ensure tables exist (creates if missing)
        // ... ensure_tables logic ...

        // Open all tables once, hold forever
        let code_chunks = connection.open_table("code_chunks").execute().await?;
        let documents = connection.open_table("documents").execute().await?;
        let indexed_files = connection.open_table("indexed_files").execute().await?;
        let document_metadata = connection.open_table("document_metadata").execute().await?;
        let memories = connection.open_table("memories").execute().await?;
        let sessions_table = connection.open_table("sessions").execute().await?;
        let session_memories = connection.open_table("session_memories").execute().await?;
        let memory_relationships = connection.open_table("memory_relationships").execute().await?;

        Ok(Self {
            project_id,
            connection,
            vector_dim: config.embedding.dimensions,
            session,
            code_chunks,
            documents,
            indexed_files,
            document_metadata,
            memories,
            sessions_table,
            session_memories,
            memory_relationships,
        })
    }

    // Simple accessors - no async, no Result, no overhead
    pub fn code_chunks(&self) -> &Table { &self.code_chunks }
    pub fn documents(&self) -> &Table { &self.documents }
    pub fn indexed_files(&self) -> &Table { &self.indexed_files }
    // ... etc
}
```

### Phase 3: Consider Write-Optimized Mode for Indexing

During bulk indexing, caches provide little benefit (we're writing, not reading). Consider:

```rust
// For bulk indexing operations, use minimal caches
let bulk_session = Arc::new(Session::new(
    16 * 1024 * 1024,   // 16 MB - just enough for merge operations
    8 * 1024 * 1024,    // 8 MB
    registry.clone(),
));

// For query operations, use larger caches
let query_session = Arc::new(Session::new(
    512 * 1024 * 1024,  // 512 MB - good for repeated searches
    64 * 1024 * 1024,
    registry.clone(),
));
```

## Expected Memory Behavior After Fix

| Scenario | Current | With Fix |
|----------|---------|----------|
| Idle baseline | ~47 MB | ~47 MB |
| During indexing | ~4.6 GB | ~500 MB - 1 GB |
| After indexing | ~4.6 GB (stuck) | ~300-400 MB |
| During queries | ~4.6 GB | ~400-600 MB |
| After queries | ~4.6 GB | drops to ~300 MB |

## Alternative Considerations

### Why Not Just Disable Caching?

Setting cache sizes to 0 disables caching entirely. This would:
- ✅ Eliminate memory growth
- ❌ Significantly slow down `merge_insert` (needs to re-read data each time)
- ❌ Devastate query performance (every search re-reads index from disk)

**Recommendation**: Don't disable caching; tune it.

### What About Table Compaction?

LanceDB supports `table.optimize()` and `compact_files()`. These:
- Reduce on-disk fragmentation
- May reduce metadata cache size (fewer fragments = smaller manifest)
- Don't directly address in-memory cache accumulation

Worth doing periodically, but won't fix the core issue.

### Memory Allocator Considerations

Rust's default allocator (system malloc) may not return memory to OS promptly. Consider:
- `jemalloc` with `background_thread` for periodic purging
- `mimalloc` for better memory reuse patterns

This is a secondary optimization after fixing cache configuration.

## Summary

1. **Root cause**: Each table gets its own caches by default (6 GiB index + 1 GiB metadata per table)
2. **Our code is clean**: `prep_delta` always shows +0.0 MB - clones are not the issue
3. **Primary fix (memory)**: Create shared `Session` with 256 MB index + 64 MB metadata caches - all tables share one bounded cache
4. **Secondary fix (performance)**: Hold Table handles as struct fields - no repeated `open_table()` calls
5. **Key insight**: With a shared Session, Table handles are lightweight; dropping them doesn't free cached data
6. **Monitoring**: Add cache stats logging for visibility

### Architecture After Fix

```
ProjectDb
├── session: Arc<Session>        ← Controls cache sizes (256 MB + 64 MB)
├── connection: Connection       ← Uses our session
├── code_chunks: Table          ← Held permanently, uses shared session cache
├── documents: Table            ← Held permanently, uses shared session cache
├── ... (6 more tables)         ← All share the same bounded cache
```

The fix is straightforward:
1. Create a `Session` with controlled cache sizes
2. Pass it to the Connection at creation time
3. Open all tables once, hold them as struct fields
4. Simple synchronous accessors instead of async `open_table()` calls

This should reduce peak memory from 4.6 GB to under 1 GB during indexing, eliminate per-operation table open overhead, and properly bound memory regardless of dataset size.
