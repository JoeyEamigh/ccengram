import type { Client } from '@libsql/client';
import { EXTRACTION_SCHEMA_STATEMENTS, FTS_STATEMENTS, INDEX_STATEMENTS, SCHEMA_STATEMENTS } from './schema.js';

export type Migration = {
  version: number;
  name: string;
  statements: string[];
};

const MIGRATIONS_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS _migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
)
`;

export const migrations: Migration[] = [
  {
    version: 1,
    name: 'initial_schema',
    statements: SCHEMA_STATEMENTS,
  },
  {
    version: 2,
    name: 'fts_tables',
    statements: FTS_STATEMENTS,
  },
  {
    version: 3,
    name: 'indexes',
    statements: INDEX_STATEMENTS,
  },
  {
    version: 4,
    name: 'config_and_compound_indexes',
    statements: [
      `CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
      )`,
      `CREATE INDEX IF NOT EXISTS idx_memories_project_created
        ON memories(project_id, created_at DESC) WHERE is_deleted = 0`,
      `CREATE INDEX IF NOT EXISTS idx_memories_project_sector_created
        ON memories(project_id, sector, created_at DESC) WHERE is_deleted = 0`,
    ],
  },
  {
    version: 5,
    name: 'conceptual_extraction',
    statements: [
      ...EXTRACTION_SCHEMA_STATEMENTS,
      `ALTER TABLE memories ADD COLUMN memory_type TEXT`,
      `ALTER TABLE memories ADD COLUMN context TEXT`,
      `ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.5`,
      `ALTER TABLE memories ADD COLUMN segment_id TEXT`,
      `CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type) WHERE memory_type IS NOT NULL`,
      `CREATE INDEX IF NOT EXISTS idx_extraction_segments_session ON extraction_segments(session_id)`,
      `CREATE INDEX IF NOT EXISTS idx_extraction_segments_project ON extraction_segments(project_id)`,
    ],
  },
  {
    version: 6,
    name: 'task_completion_tracking',
    statements: [
      `ALTER TABLE segment_accumulators ADD COLUMN completed_tasks_json TEXT DEFAULT '[]'`,
    ],
  },
  {
    version: 7,
    name: 'code_indexing',
    statements: [
      `ALTER TABLE documents ADD COLUMN language TEXT`,
      `ALTER TABLE documents ADD COLUMN line_count INTEGER`,
      `ALTER TABLE documents ADD COLUMN is_code INTEGER DEFAULT 0`,
      `ALTER TABLE document_chunks ADD COLUMN start_line INTEGER`,
      `ALTER TABLE document_chunks ADD COLUMN end_line INTEGER`,
      `ALTER TABLE document_chunks ADD COLUMN chunk_type TEXT`,
      `ALTER TABLE document_chunks ADD COLUMN symbols_json TEXT`,
      `CREATE TABLE IF NOT EXISTS code_index_state (
        project_id TEXT PRIMARY KEY,
        last_indexed_at INTEGER,
        indexed_files INTEGER DEFAULT 0,
        gitignore_hash TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id)
      )`,
      `CREATE TABLE IF NOT EXISTS indexed_files (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        path TEXT NOT NULL,
        checksum TEXT NOT NULL,
        mtime INTEGER NOT NULL,
        indexed_at INTEGER NOT NULL,
        UNIQUE(project_id, path)
      )`,
      `CREATE INDEX IF NOT EXISTS idx_indexed_files_project ON indexed_files(project_id)`,
      `CREATE INDEX IF NOT EXISTS idx_indexed_files_path ON indexed_files(project_id, path)`,
      `CREATE INDEX IF NOT EXISTS idx_documents_code ON documents(project_id, is_code) WHERE is_code = 1`,
      `CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language) WHERE language IS NOT NULL`,
    ],
  },
  {
    version: 8,
    name: 'dedup_and_decay_optimization',
    statements: [
      `UPDATE memories SET simhash_prefix = substr(simhash, 1, 4) WHERE simhash IS NOT NULL AND simhash_prefix IS NULL`,
      `CREATE INDEX IF NOT EXISTS idx_memories_simhash_prefix
        ON memories(project_id, simhash_prefix) WHERE simhash_prefix IS NOT NULL AND is_deleted = 0`,
      `CREATE INDEX IF NOT EXISTS idx_memories_next_decay
        ON memories(next_decay_at) WHERE next_decay_at IS NOT NULL AND is_deleted = 0`,
      `CREATE INDEX IF NOT EXISTS idx_memories_content_hash
        ON memories(project_id, content_hash) WHERE content_hash IS NOT NULL AND is_deleted = 0`,
    ],
  },
  {
    version: 9,
    name: 'index_checkpoints',
    statements: [
      `CREATE TABLE IF NOT EXISTS index_checkpoints (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        phase TEXT NOT NULL,
        processed_files_json TEXT NOT NULL DEFAULT '[]',
        pending_files_json TEXT NOT NULL DEFAULT '[]',
        progress_json TEXT NOT NULL,
        created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
        updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
        FOREIGN KEY (project_id) REFERENCES projects(id)
      )`,
      `CREATE INDEX IF NOT EXISTS idx_checkpoints_project ON index_checkpoints(project_id)`,
      `ALTER TABLE code_index_state ADD COLUMN total_bytes INTEGER DEFAULT 0`,
      `ALTER TABLE code_index_state ADD COLUMN total_tokens INTEGER DEFAULT 0`,
      `ALTER TABLE code_index_state ADD COLUMN total_chunks INTEGER DEFAULT 0`,
    ],
  },
  {
    version: 10,
    name: 'scoped_memories',
    statements: [
      `ALTER TABLE memories ADD COLUMN scope_path TEXT`,
      `ALTER TABLE memories ADD COLUMN scope_module TEXT`,
      `CREATE INDEX IF NOT EXISTS idx_memories_scope_path ON memories(project_id, scope_path) WHERE scope_path IS NOT NULL AND is_deleted = 0`,
      `CREATE INDEX IF NOT EXISTS idx_memories_scope_module ON memories(project_id, scope_module) WHERE scope_module IS NOT NULL AND is_deleted = 0`,
    ],
  },
];

export async function getCurrentVersion(client: Client): Promise<number> {
  try {
    const result = await client.execute('SELECT MAX(version) as version FROM _migrations');
    const row = result.rows[0];
    if (row && row['version'] !== null) {
      return Number(row['version']);
    }
    return 0;
  } catch {
    return 0;
  }
}

export async function runMigrations(client: Client): Promise<void> {
  await client.execute(MIGRATIONS_TABLE_SQL);

  const currentVersion = await getCurrentVersion(client);
  const pending = migrations.filter(m => m.version > currentVersion);

  for (const migration of pending) {
    for (const sql of migration.statements) {
      await client.execute(sql);
    }

    await client.execute({
      sql: 'INSERT INTO _migrations (version, name) VALUES (?, ?)',
      args: [migration.version, migration.name],
    });
  }
}
