# Database Specification

## Overview

CCMemory uses libSQL (Turso's SQLite fork) for storage with native vector support. The database is a single portable file with WAL mode for concurrent access.

**Note:** We evaluated `@tursodatabase/database` but it's beta with vector indexing on roadmap only. We evaluated Drizzle ORM but it has a vector insertion bug (#3899). Raw `@libsql/client` is the most reliable choice for vector operations.

## Dependencies

```json
{
  "@libsql/client": "^0.14.0"
}
```

## Files to Create

- `src/db/database.ts` - Connection management
- `src/db/schema.ts` - Table definitions
- `src/db/migrations.ts` - Schema migrations
- `src/utils/paths.ts` - XDG path utilities

## XDG Path Utilities

### Interface

```typescript
// src/utils/paths.ts
export interface Paths {
  config: string;   // $XDG_CONFIG_HOME/ccmemory
  data: string;     // $XDG_DATA_HOME/ccmemory
  cache: string;    // $XDG_CACHE_HOME/ccmemory
  db: string;       // $XDG_DATA_HOME/ccmemory/memories.db
}

export function getPaths(): Paths;
export function ensureDirectories(): Promise<void>;
```

### Implementation Notes

- Use `process.env.XDG_*` with fallbacks
- Platform-specific defaults (Linux, macOS, Windows)
- Create directories with `mkdir -p` equivalent

### Test Specification

```typescript
// src/utils/__test__/paths.test.ts (colocated)
import { describe, test, expect, beforeEach, afterEach } from "bun:test";

describe("XDG Paths", () => {
  test("uses XDG_DATA_HOME when set", () => {
    process.env.XDG_DATA_HOME = "/tmp/test-data";
    const paths = getPaths();
    expect(paths.data).toBe("/tmp/test-data/ccmemory");
  });

  test("falls back to ~/.local/share on Linux", () => {
    delete process.env.XDG_DATA_HOME;
    // Mock platform as linux
    const paths = getPaths();
    expect(paths.data).toContain(".local/share/ccmemory");
  });

  test("database path is under data directory", () => {
    const paths = getPaths();
    expect(paths.db).toBe(`${paths.data}/memories.db`);
  });
});
```

## Logging Utilities

Unified file-based logging for all ccmemory components. Handles multiple concurrent instances safely.

### Interface

```typescript
// src/utils/log.ts
type LogLevel = "debug" | "info" | "warn" | "error";

type LogContext = Record<string, unknown>;

type Logger = {
  debug(module: string, message: string, context?: LogContext): void;
  info(module: string, message: string, context?: LogContext): void;
  warn(module: string, message: string, context?: LogContext): void;
  error(module: string, message: string, context?: LogContext): void;
  setLevel(level: LogLevel): void;
  getLevel(): LogLevel;
  flush(): Promise<void>;
};

export const log: Logger;
export function getLogPath(): string;
```

### Implementation Notes

```typescript
// src/utils/log.ts
import { getPaths, ensureDirectories } from "./paths.js";

const LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3
};

let currentLevel: LogLevel = (process.env.LOG_LEVEL as LogLevel) || "info";
let logFile: Bun.FileSink | null = null;
let logPath: string | null = null;

// Get or create the log file path
export function getLogPath(): string {
  if (!logPath) {
    const paths = getPaths();
    logPath = `${paths.data}/ccmemory.log`;
  }
  return logPath;
}

// Initialize log file (lazy, on first write)
async function ensureLogFile(): Promise<Bun.FileSink> {
  if (!logFile) {
    await ensureDirectories();
    const path = getLogPath();
    // Open in append mode - safe for multiple instances
    logFile = Bun.file(path).writer({ highWaterMark: 1024 });
  }
  return logFile;
}

// Format: [timestamp] [LEVEL] [pid:module] message {context}
function formatLine(level: LogLevel, module: string, message: string, context?: LogContext): string {
  const timestamp = new Date().toISOString();
  const pid = process.pid;
  const contextStr = context ? ` ${JSON.stringify(context)}` : "";
  return `[${timestamp}] [${level.toUpperCase().padEnd(5)}] [${pid}:${module}] ${message}${contextStr}\n`;
}

function shouldLog(level: LogLevel): boolean {
  return LEVELS[level] >= LEVELS[currentLevel];
}

async function writeLog(level: LogLevel, module: string, message: string, context?: LogContext): Promise<void> {
  if (!shouldLog(level)) return;

  const line = formatLine(level, module, message, context);

  try {
    const writer = await ensureLogFile();
    writer.write(line);
    // Flush immediately for error level to ensure it's written
    if (level === "error") {
      await writer.flush();
    }
  } catch (err) {
    // Fallback to stderr if file write fails
    console.error(`[LOG WRITE FAILED] ${line}`);
  }
}

export const log: Logger = {
  debug(module, message, context) {
    writeLog("debug", module, message, context);
  },
  info(module, message, context) {
    writeLog("info", module, message, context);
  },
  warn(module, message, context) {
    writeLog("warn", module, message, context);
  },
  error(module, message, context) {
    writeLog("error", module, message, context);
  },
  setLevel(level) {
    currentLevel = level;
  },
  getLevel() {
    return currentLevel;
  },
  async flush() {
    if (logFile) {
      await logFile.flush();
    }
  }
};

// Flush on process exit
process.on("beforeExit", async () => {
  await log.flush();
});
```

### Log File Location

- **Path**: `$XDG_DATA_HOME/ccmemory/ccmemory.log`
- **Format**: Append-only, newline-delimited
- **Multi-instance**: Each line includes PID to distinguish instances
- **Rotation**: Manual via CLI (`ccmemory log rotate`) or external tool

### Log Line Format

```
[2024-01-15T10:30:45.123Z] [INFO ] [12345:embedding] Embedded batch {"count":10,"ms":234}
[2024-01-15T10:30:45.456Z] [DEBUG] [12345:memory] Created memory {"id":"abc123","sector":"semantic"}
[2024-01-15T10:30:46.789Z] [ERROR] [67890:mcp] Tool failed {"tool":"memory_search","error":"timeout"}
```

### Usage in Other Modules

**IMPORTANT**: All specs should import and use this logger:

```typescript
import { log } from "../utils/log.js";

// Always use: log.level(module, message, context?)
log.info("embedding", "Provider initialized", { provider: "ollama", model: "qwen3" });
log.debug("memory", "Checking for duplicates", { simhash: hash });
log.warn("search", "Slow query detected", { ms: 500, query: q });
log.error("db", "Connection failed", { error: err.message });
```

### Test Specification

```typescript
// src/utils/__test__/log.test.ts (colocated)
import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { log, getLogPath } from "./log.js";
import { unlink, readFile } from "node:fs/promises";

describe("Logger", () => {
  const testLogPath = "/tmp/ccmemory-test.log";

  beforeEach(async () => {
    // Clean up test log
    try { await unlink(testLogPath); } catch {}
    log.setLevel("debug");
  });

  afterEach(async () => {
    await log.flush();
  });

  test("writes to log file", async () => {
    log.info("test", "hello world");
    await log.flush();

    const content = await readFile(getLogPath(), "utf-8");
    expect(content).toContain("[INFO ]");
    expect(content).toContain("[test]");
    expect(content).toContain("hello world");
  });

  test("respects log level", async () => {
    log.setLevel("warn");
    log.debug("test", "should not appear");
    log.info("test", "should not appear");
    log.warn("test", "should appear");
    await log.flush();

    const content = await readFile(getLogPath(), "utf-8");
    expect(content).not.toContain("should not appear");
    expect(content).toContain("should appear");
  });

  test("includes PID for multi-instance support", async () => {
    log.info("test", "pid test");
    await log.flush();

    const content = await readFile(getLogPath(), "utf-8");
    expect(content).toContain(`[${process.pid}:test]`);
  });

  test("serializes context as JSON", async () => {
    log.info("test", "with context", { count: 5, name: "foo" });
    await log.flush();

    const content = await readFile(getLogPath(), "utf-8");
    expect(content).toContain('{"count":5,"name":"foo"}');
  });

  test("handles concurrent writes", async () => {
    const promises = [];
    for (let i = 0; i < 100; i++) {
      promises.push(log.info("test", `message ${i}`));
    }
    await Promise.all(promises);
    await log.flush();

    const content = await readFile(getLogPath(), "utf-8");
    const lines = content.trim().split("\n");
    // All messages should be written (may be interleaved with other tests)
    expect(lines.length).toBeGreaterThanOrEqual(100);
  });

  test("flushes immediately on error level", async () => {
    log.error("test", "critical error");
    // Should be flushed without explicit flush()
    const content = await readFile(getLogPath(), "utf-8");
    expect(content).toContain("critical error");
  });
});
```

## Database Connection

### Interface

```typescript
// src/db/database.ts
import { Client } from "@libsql/client";

export interface Database {
  client: Client;
  execute(sql: string, args?: any[]): Promise<any>;
  batch(statements: Array<{sql: string, args?: any[]}>): Promise<any[]>;
  transaction<T>(fn: (tx: Transaction) => Promise<T>): Promise<T>;
  close(): void;
}

export function createDatabase(dbPath?: string): Promise<Database>;
export function getDatabase(): Database; // Singleton access
```

### Implementation Notes

```typescript
import { createClient } from "@libsql/client";

export async function createDatabase(dbPath?: string): Promise<Database> {
  const paths = getPaths();
  await ensureDirectories();

  const client = createClient({
    url: `file:${dbPath || paths.db}`
  });

  // Enable WAL mode for concurrent access
  await client.execute("PRAGMA journal_mode=WAL");
  await client.execute("PRAGMA busy_timeout=5000");
  await client.execute("PRAGMA synchronous=NORMAL");

  // Run migrations
  await runMigrations(client);

  return wrapClient(client);
}
```

### WAL Mode Benefits

- Multiple readers + single writer
- Better performance for concurrent access
- Multiple Claude Code sessions without conflicts
- Safe for process crashes (auto-recovery)

### Test Specification

```typescript
// src/db/__test__/database.test.ts (colocated)
describe("Database", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
  });

  afterEach(() => {
    db.close();
  });

  test("creates database with WAL mode", async () => {
    const result = await db.execute("PRAGMA journal_mode");
    expect(result.rows[0][0]).toBe("wal");
  });

  test("executes parameterized queries", async () => {
    await db.execute("CREATE TABLE test (id INTEGER, name TEXT)");
    await db.execute("INSERT INTO test VALUES (?, ?)", [1, "foo"]);
    const result = await db.execute("SELECT * FROM test WHERE id = ?", [1]);
    expect(result.rows[0]).toEqual([1, "foo"]);
  });

  test("supports batch operations", async () => {
    await db.execute("CREATE TABLE test (id INTEGER)");
    const results = await db.batch([
      { sql: "INSERT INTO test VALUES (?)", args: [1] },
      { sql: "INSERT INTO test VALUES (?)", args: [2] },
      { sql: "SELECT COUNT(*) FROM test", args: [] }
    ]);
    expect(results[2].rows[0][0]).toBe(2);
  });

  test("transactions rollback on error", async () => {
    await db.execute("CREATE TABLE test (id INTEGER UNIQUE)");
    await db.execute("INSERT INTO test VALUES (1)");

    await expect(
      db.transaction(async (tx) => {
        await tx.execute("INSERT INTO test VALUES (2)");
        await tx.execute("INSERT INTO test VALUES (1)"); // Duplicate
      })
    ).rejects.toThrow();

    const result = await db.execute("SELECT COUNT(*) FROM test");
    expect(result.rows[0][0]).toBe(1); // Rollback happened
  });
});
```

## Schema

### Memory Sectors (5-Sector Model)

Based on OpenMemory's research, memories are classified into 5 sectors with different decay rates:

| Sector       | Description                                  | Decay Rate | Examples                           |
| ------------ | -------------------------------------------- | ---------- | ---------------------------------- |
| `episodic`   | Events, conversations, specific interactions | 0.02/day   | "User asked about auth flow"       |
| `semantic`   | Facts, knowledge, preferences                | 0.005/day  | "Prefers tabs over spaces"         |
| `procedural` | Skills, workflows, how-to knowledge          | 0.01/day   | "Deploy via `bun run deploy`"      |
| `emotional`  | Sentiments, frustrations, satisfactions      | 0.003/day  | "Frustrated by slow tests"         |
| `reflective` | Insights, patterns, lessons learned          | 0.008/day  | "This codebase favors composition" |

### Core Tables

```sql
-- src/db/schema.ts (exported as SQL strings)

-- Embedding models (track dimensions for model switching)
CREATE TABLE IF NOT EXISTS embedding_models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,  -- 'ollama' | 'openrouter'
    dimensions INTEGER NOT NULL,
    is_active INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
);

-- Projects (isolation by directory)
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT,
    settings_json TEXT,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
);

-- Sessions (Claude Code sessions with rich context)
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    summary TEXT,
    user_prompt TEXT,
    -- Context for the session
    context_json TEXT,          -- {"working_dir": "...", "git_branch": "...", "task": "..."}
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- Memories (core knowledge store with 5-sector model)
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    content_hash TEXT,          -- MD5 for deduplication

    -- Classification (5-sector model)
    sector TEXT NOT NULL,       -- 'episodic' | 'semantic' | 'procedural' | 'emotional' | 'reflective'
    tier TEXT DEFAULT 'project', -- 'session' | 'project'
    importance REAL DEFAULT 0.5, -- Base importance for decay calculation
    categories_json TEXT,       -- Auto-labeled categories

    -- Deduplication
    simhash TEXT,

    -- Salience/Reinforcement
    salience REAL DEFAULT 1.0,  -- Current strength (decays over time, boosted on access)
    access_count INTEGER DEFAULT 0,

    -- Timestamps (bi-temporal model)
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_accessed INTEGER NOT NULL,
    valid_from INTEGER,         -- When this fact became true (optional)
    valid_until INTEGER,        -- When this fact ceased being true (null = current)

    -- Soft delete
    is_deleted INTEGER DEFAULT 0,
    deleted_at INTEGER,

    -- Embedding
    embedding_model_id TEXT,

    -- Extracted metadata
    tags_json TEXT,
    concepts_json TEXT,
    files_json TEXT,

    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_models(id)
);

-- Session-Memory links (track which memories were used in which sessions)
CREATE TABLE IF NOT EXISTS session_memories (
    session_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    usage_type TEXT NOT NULL,   -- 'created' | 'recalled' | 'updated' | 'reinforced'
    PRIMARY KEY (session_id, memory_id, created_at),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Entities (graph nodes for relationship tracking)
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- 'person' | 'project' | 'technology' | 'concept' | 'file' | 'error'
    summary TEXT,
    embedding_model_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_models(id)
);

-- Memory-Entity links (waypoints connecting memories to entities)
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT NOT NULL,         -- 'subject' | 'object' | 'context'
    created_at INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

-- Memory relationships (explicit graph edges)
CREATE TABLE IF NOT EXISTS memory_relationships (
    id TEXT PRIMARY KEY,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,  -- 'SUPERSEDES' | 'CONTRADICTS' | 'RELATED_TO' | 'BUILDS_ON' | etc.
    created_at INTEGER NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_until INTEGER,        -- null = currently valid
    confidence REAL DEFAULT 1.0,
    extracted_by TEXT NOT NULL, -- 'user' | 'llm' | 'system'
    FOREIGN KEY (source_memory_id) REFERENCES memories(id),
    FOREIGN KEY (target_memory_id) REFERENCES memories(id)
);

-- Memory vectors (separate for model flexibility)
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    vector F32_BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_models(id)
);

-- Vector similarity index
CREATE INDEX IF NOT EXISTS memory_vectors_idx ON memory_vectors (
    libsql_vector_idx(vector)
);

-- Documents (ingested txt/md files)
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_path TEXT,           -- Original file path
    source_url TEXT,            -- Or URL if fetched
    source_type TEXT NOT NULL,  -- 'txt' | 'md' | 'url'
    title TEXT,
    full_content TEXT NOT NULL,
    checksum TEXT,              -- For change detection
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- Document chunks (for vector search)
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    tokens_estimate INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Document vectors
CREATE TABLE IF NOT EXISTS document_vectors (
    chunk_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    vector F32_BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_models(id)
);

-- Document vector index
CREATE INDEX IF NOT EXISTS document_vectors_idx ON document_vectors (
    libsql_vector_idx(vector)
);

-- FTS5 for keyword search on memories
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    summary,
    concepts_json,
    tags_json,
    content='memories',
    content_rowid='rowid'
);

-- FTS5 triggers to keep in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary, concepts_json, tags_json)
    VALUES (NEW.rowid, NEW.content, NEW.summary, NEW.concepts_json, NEW.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, concepts_json, tags_json)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.summary, OLD.concepts_json, OLD.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, concepts_json, tags_json)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.summary, OLD.concepts_json, OLD.tags_json);
    INSERT INTO memories_fts(rowid, content, summary, concepts_json, tags_json)
    VALUES (NEW.rowid, NEW.content, NEW.summary, NEW.concepts_json, NEW.tags_json);
END;

-- FTS5 for document chunks
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    content='document_chunks',
    content_rowid='rowid'
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories(sector);
CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_simhash ON memories(simhash);
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_until);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_memories_session ON session_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_session_memories_memory ON session_memories(memory_id);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON memory_relationships(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON memory_relationships(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relationship_type);

CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc ON document_chunks(document_id);
```

### Relationship Types

```typescript
// src/db/types.ts
type RelationshipType =
  | 'SUPERSEDES'      // New info replaces old
  | 'CONTRADICTS'     // Conflicting information
  | 'RELATED_TO'      // General semantic connection
  | 'BUILDS_ON'       // Extends previous knowledge
  | 'CONFIRMS'        // Reinforces existing info
  | 'APPLIES_TO'      // Memory applies to specific context
  | 'DEPENDS_ON'      // Prerequisite relationship
  | 'ALTERNATIVE_TO'; // Different approach to same problem
```

### Schema Test Specification

```typescript
// src/db/__test__/schema.test.ts (colocated with schema.ts)
describe("Schema", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
  });

  test("creates all required tables", async () => {
    const tables = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    );
    const names = tables.rows.map(r => r[0]);

    expect(names).toContain("memories");
    expect(names).toContain("memory_vectors");
    expect(names).toContain("documents");
    expect(names).toContain("document_chunks");
    expect(names).toContain("projects");
    expect(names).toContain("sessions");
    expect(names).toContain("session_memories");
    expect(names).toContain("entities");
    expect(names).toContain("memory_entities");
    expect(names).toContain("memory_relationships");
    expect(names).toContain("embedding_models");
  });

  test("creates FTS5 virtual tables", async () => {
    const tables = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts%'"
    );
    expect(tables.rows.length).toBeGreaterThanOrEqual(2);
  });

  test("creates vector indexes", async () => {
    const indexes = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '%vectors_idx%'"
    );
    expect(indexes.rows.length).toBeGreaterThanOrEqual(2);
  });

  test("FTS triggers are created", async () => {
    const triggers = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='trigger'"
    );
    const names = triggers.rows.map(r => r[0]);
    expect(names).toContain("memories_ai");
    expect(names).toContain("memories_ad");
    expect(names).toContain("memories_au");
  });

  test("session_memories tracks memory usage", async () => {
    // Setup project and session
    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute("INSERT INTO sessions (id, project_id, started_at) VALUES ('s1', 'p1', 0)");
    await db.execute(`INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
                      VALUES ('m1', 'p1', 'test', 'semantic', 0, 0, 0)`);

    // Track usage
    await db.execute(`INSERT INTO session_memories (session_id, memory_id, created_at, usage_type)
                      VALUES ('s1', 'm1', 0, 'recalled')`);

    const result = await db.execute("SELECT usage_type FROM session_memories WHERE memory_id = 'm1'");
    expect(result.rows[0][0]).toBe("recalled");
  });

  test("memory_relationships tracks SUPERSEDES", async () => {
    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(`INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
                      VALUES ('m1', 'p1', 'old fact', 'semantic', 0, 0, 0)`);
    await db.execute(`INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
                      VALUES ('m2', 'p1', 'new fact', 'semantic', 1, 1, 1)`);

    await db.execute(`INSERT INTO memory_relationships (id, source_memory_id, target_memory_id, relationship_type, created_at, valid_from, extracted_by)
                      VALUES ('r1', 'm2', 'm1', 'SUPERSEDES', 1, 1, 'system')`);

    const result = await db.execute("SELECT relationship_type FROM memory_relationships WHERE source_memory_id = 'm2'");
    expect(result.rows[0][0]).toBe("SUPERSEDES");
  });
});
```

## Migrations

### Interface

```typescript
// src/db/migrations.ts
export interface Migration {
  version: number;
  name: string;
  up: string;   // SQL to apply
  down: string; // SQL to rollback
}

export const migrations: Migration[];
export async function runMigrations(client: Client): Promise<void>;
export async function getCurrentVersion(client: Client): Promise<number>;
```

### Implementation Notes

```typescript
// Schema versioning table
const MIGRATIONS_TABLE = `
CREATE TABLE IF NOT EXISTS _migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
)
`;

export async function runMigrations(client: Client): Promise<void> {
  await client.execute(MIGRATIONS_TABLE);

  const currentVersion = await getCurrentVersion(client);
  const pending = migrations.filter(m => m.version > currentVersion);

  for (const migration of pending) {
    await client.batch([
      { sql: migration.up },
      {
        sql: "INSERT INTO _migrations (version, name) VALUES (?, ?)",
        args: [migration.version, migration.name]
      }
    ], "write");
  }
}
```

### Test Specification

```typescript
// src/db/__test__/migrations.test.ts (colocated)
describe("Migrations", () => {
  test("migrations are ordered by version", () => {
    for (let i = 1; i < migrations.length; i++) {
      expect(migrations[i].version).toBeGreaterThan(migrations[i-1].version);
    }
  });

  test("all migrations have unique versions", () => {
    const versions = migrations.map(m => m.version);
    const unique = new Set(versions);
    expect(unique.size).toBe(versions.length);
  });

  test("migrations run idempotently", async () => {
    const db1 = await createDatabase(":memory:");
    const db2 = await createDatabase(":memory:");

    // Running twice should produce same schema
    await runMigrations(db1.client);
    await runMigrations(db1.client);

    const v1 = await getCurrentVersion(db1.client);
    const v2 = await getCurrentVersion(db2.client);
    expect(v1).toBe(v2);
  });
});
```

## Vector Operations

### libSQL Vector Functions

```sql
-- Store vector (use F32_BLOB for native storage)
INSERT INTO memory_vectors (memory_id, model_id, vector, dim)
VALUES (?, ?, vector(?), ?);

-- Vector similarity search using index
SELECT memory_id, vector_distance_cos(vector, vector(?)) as distance
FROM memory_vectors
WHERE rowid IN (
    SELECT rowid FROM vector_top_k('memory_vectors_idx', vector(?), 20)
)
ORDER BY distance ASC;

-- Note: vector() function converts JSON array to F32_BLOB
-- vector_distance_cos() computes cosine distance (0 = identical, 2 = opposite)
```

### Vector Test Specification

```typescript
// src/db/__test__/vectors.test.ts (colocated)
describe("Vector Operations", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
  });

  test("stores and retrieves vectors", async () => {
    await db.execute(
      "INSERT INTO embedding_models (id, name, provider, dimensions) VALUES (?, ?, ?, ?)",
      ["test-model", "test", "test", 4]
    );

    // Insert a memory first
    await db.execute(
      `INSERT INTO projects (id, path) VALUES ('proj1', '/test')`,
    );
    await db.execute(
      `INSERT INTO memories (id, project_id, content, memory_type, created_at, updated_at, last_accessed)
       VALUES ('mem1', 'proj1', 'test', 'discovery', 0, 0, 0)`
    );

    // Store vector
    const vec = [0.1, 0.2, 0.3, 0.4];
    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["mem1", "test-model", JSON.stringify(vec), 4]
    );

    // Retrieve and verify dimension
    const result = await db.execute(
      "SELECT dim FROM memory_vectors WHERE memory_id = ?",
      ["mem1"]
    );
    expect(result.rows[0][0]).toBe(4);
  });

  test("finds similar vectors with vector_top_k", async () => {
    // Setup: create 10 vectors
    // Query: find top 3 similar
    // Assert: results ordered by similarity
  });

  test("handles different vector dimensions", async () => {
    // Store vectors with different dimensions (from different models)
    // Verify they don't interfere with each other
  });
});
```

## Acceptance Criteria

- [ ] Database creates successfully with WAL mode
- [ ] All tables and indexes created on first run (including new session_memories, entities, memory_entities, memory_relationships)
- [ ] Migrations run idempotently
- [ ] FTS5 stays in sync via triggers
- [ ] Vector storage and retrieval works
- [ ] vector_top_k finds similar vectors
- [ ] Multiple concurrent connections work
- [ ] XDG paths respected on all platforms
- [ ] Database survives process crashes (WAL recovery)
- [ ] Session-memory links track usage types correctly
- [ ] Memory relationships can be created and queried
- [ ] Soft delete works (is_deleted flag)
- [ ] Bi-temporal queries work (valid_from/valid_until)
