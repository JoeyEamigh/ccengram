# CCEngram Logging Guidelines

This document establishes logging standards for the CCEngram codebase.

## Log Levels

Use the correct level for each situation:

### `error!` - Failures requiring attention
- Operations that failed and cannot recover
- Database corruption or connection failures
- External service unavailability (after retries exhausted)
- Configuration errors that prevent functionality

```rust
error!("Failed to connect to LanceDB: {}", err);
error!(session_id = %id, "Session lookup failed: {}", err);
```

### `warn!` - Degraded but functional
- Fallback behavior activated
- Retryable failures (before exhausting retries)
- Deprecated usage or configuration
- Missing optional dependencies

```rust
warn!("OpenRouter unavailable, falling back to Ollama");
warn!(file = %path, "File not found, skipping");
```

### `info!` - Significant operational events
- Daemon/service lifecycle (start, stop, ready)
- Major operations completed (indexing, migrations)
- Configuration loaded with non-default values
- Session/project lifecycle events
- Background job results (decay, cleanup)

```rust
info!("Daemon started on {}", socket_path);
info!(project = %name, files = count, "Indexing complete");
info!(session_id = %id, "Session started");
```

### `debug!` - Diagnostic information
- Request/response flow (method names, IDs)
- Decision points and branches taken
- Cache hits/misses
- Individual file operations
- Batch processing progress

```rust
debug!(method = %name, id = ?req_id, "Handling request");
debug!(file = %path, cached = hit, "File lookup");
debug!(batch = n, total = t, "Processing batch");
```

### `trace!` - Detailed execution flow
- Function entry/exit for complex operations
- Full request/response payloads (truncated)
- Individual item processing in loops
- Timing measurements
- State transitions

```rust
trace!(query = %q, "Embedding request");
trace!(chunk_id = %id, score = s, "Search result");
trace!(elapsed_ms = ms, "Operation completed");
```

## Structured Logging

Use structured fields for machine-parseable logs:

```rust
// Good - structured fields
info!(
    session_id = %session_id,
    project = %project_name,
    memories_created = count,
    "Session ended"
);

// Avoid - embedded in message
info!("Session {} for project {} ended with {} memories", id, name, count);
```

### Common Field Names

Use consistent field names across the codebase:

| Field | Description | Example |
|-------|-------------|---------|
| `session_id` | Session identifier | `%session_id` |
| `project` | Project name | `%project_name` |
| `project_id` | Project hash ID | `%project_id` |
| `file` | File path | `%path.display()` |
| `method` | RPC method name | `%method` |
| `elapsed_ms` | Duration in milliseconds | `elapsed.as_millis()` |
| `count` | Item count | `items.len()` |
| `batch` | Batch number | `batch_num` |
| `total` | Total items | `total_count` |
| `err` | Error details | `%err` |

## What to Log

### Daemon Lifecycle
- Startup with configuration summary
- Socket binding success/failure
- Embedding provider initialization
- Scheduler startup
- Shutdown initiation and completion

### Request Handling
- `debug!`: Every incoming request (method, id)
- `debug!`: Request completion with timing
- `warn!`: Malformed requests
- `error!`: Request handling failures

### Hook Processing
- `info!`: Hook event received (type, session)
- `debug!`: Processing steps
- `info!`: Memories created/updated
- `debug!`: Extraction decisions (skip reasons)

### Embedding Operations
- `debug!`: Batch size and text count
- `trace!`: Individual embedding requests
- `debug!`: API response times
- `warn!`: Rate limiting or throttling
- `error!`: Provider failures

### Database Operations
- `debug!`: Connection creation
- `trace!`: Individual queries (table, operation)
- `debug!`: Migration execution
- `info!`: Schema migrations completed
- `error!`: Query failures

### Indexing
- `info!`: Scan started with config
- `debug!`: File discovered/processed
- `debug!`: Chunk creation
- `info!`: Scan completed with stats
- `warn!`: Parse failures for individual files

### Background Jobs
- `info!`: Job started
- `debug!`: Progress updates
- `info!`: Job completed with results
- `warn!`: Job skipped (reason)

## What NOT to Log

Avoid trivial or noisy logs:

```rust
// Bad - too trivial
debug!("Entering function");
debug!("Checking condition");
trace!("Loop iteration {}", i);  // For every item

// Bad - sensitive data
debug!("Processing content: {}", full_content);  // May contain user code
info!("API key: {}", api_key);

// Bad - redundant with error propagation
debug!("About to call database");
// ... call database
debug!("Database call returned");
```

### Specific Anti-patterns

1. **Don't log every loop iteration** - Log batch progress instead
2. **Don't log function entry/exit for simple functions** - Only for complex operations
3. **Don't log the same information at multiple levels** - Pick one
4. **Don't log full file contents** - Log paths and sizes instead
5. **Don't log API keys, tokens, or credentials** - Never
6. **Don't log every cache hit** - Log cache stats periodically instead

## Spans for Complex Operations

Use tracing spans for operations that span multiple functions:

```rust
use tracing::{instrument, Span};

#[instrument(skip(self, embeddings), fields(batch_size = texts.len()))]
async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    // Operation is automatically tracked
}

// Or manually:
let span = tracing::info_span!("index_project", project = %name);
let _guard = span.enter();
```

## Performance Considerations

1. **Use lazy evaluation** for expensive formatting:
   ```rust
   trace!(data = ?expensive_debug_impl, "Details");  // Only formats if trace enabled
   ```

2. **Avoid allocation in hot paths**:
   ```rust
   // Bad - allocates even if debug disabled
   debug!("Items: {}", items.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "));

   // Better - use Debug trait
   debug!(items = ?items, "Processing");
   ```

3. **Use `%` for Display, `?` for Debug**:
   ```rust
   debug!(path = %path.display(), error = ?err, "Failed");
   ```

## Example: Well-Logged Function

```rust
#[instrument(skip(self, db, embeddings), fields(project = %project_name))]
async fn index_files(
    &self,
    db: &ProjectDb,
    embeddings: &dyn EmbeddingProvider,
    files: Vec<PathBuf>,
    project_name: &str,
) -> Result<IndexResult> {
    info!(file_count = files.len(), "Starting file indexing");

    let mut indexed = 0;
    let mut failed = 0;

    for (batch_num, batch) in files.chunks(100).enumerate() {
        debug!(batch = batch_num, size = batch.len(), "Processing batch");

        for file in batch {
            match self.index_file(db, embeddings, file).await {
                Ok(chunks) => {
                    trace!(file = %file.display(), chunks = chunks, "File indexed");
                    indexed += 1;
                }
                Err(e) => {
                    warn!(file = %file.display(), err = %e, "Failed to index file");
                    failed += 1;
                }
            }
        }
    }

    info!(indexed = indexed, failed = failed, "Indexing complete");
    Ok(IndexResult { indexed, failed })
}
```

## Testing Logs

When testing, you can capture logs:

```rust
#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    #[traced_test]
    #[tokio::test]
    async fn test_something() {
        // logs are captured and can be asserted
        assert!(logs_contain("expected message"));
    }
}
```
