export const SCHEMA_STATEMENTS = [
  `CREATE TABLE IF NOT EXISTS embedding_models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    is_active INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
)`,
  `CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT,
    settings_json TEXT,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
)`,
  `CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    summary TEXT,
    user_prompt TEXT,
    context_json TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id)
)`,
  `CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    content_hash TEXT,
    sector TEXT NOT NULL,
    tier TEXT DEFAULT 'project', -- 'session' | 'project'
    importance REAL DEFAULT 0.5,
    categories_json TEXT,
    simhash TEXT,
    simhash_prefix TEXT,
    salience REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_accessed INTEGER NOT NULL,
    valid_from INTEGER,
    valid_until INTEGER,
    is_deleted INTEGER DEFAULT 0,
    deleted_at INTEGER,
    embedding_model_id TEXT,
    tags_json TEXT,
    concepts_json TEXT,
    files_json TEXT,
    next_decay_at INTEGER,
    decay_rate REAL,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_models(id)
)`,
  `CREATE TABLE IF NOT EXISTS session_memories (
    session_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    usage_type TEXT NOT NULL,
    PRIMARY KEY (session_id, memory_id, created_at),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
)`,
  `CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    summary TEXT,
    embedding_model_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_models(id)
)`,
  `CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
)`,
  `CREATE TABLE IF NOT EXISTS memory_relationships (
    id TEXT PRIMARY KEY,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_until INTEGER,
    confidence REAL DEFAULT 1.0,
    extracted_by TEXT NOT NULL,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id),
    FOREIGN KEY (target_memory_id) REFERENCES memories(id)
)`,
  `CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    vector F32_BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_models(id)
)`,
  `CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_path TEXT,
    source_url TEXT,
    source_type TEXT NOT NULL,
    title TEXT,
    full_content TEXT NOT NULL,
    checksum TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
)`,
  `CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    tokens_estimate INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
)`,
  `CREATE TABLE IF NOT EXISTS document_vectors (
    chunk_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    vector F32_BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_models(id)
)`,
];

export const FTS_STATEMENTS = [
  `CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    summary,
    concepts_json,
    tags_json,
    content='memories',
    content_rowid='rowid'
)`,
  `CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary, concepts_json, tags_json)
    VALUES (NEW.rowid, NEW.content, NEW.summary, NEW.concepts_json, NEW.tags_json);
END`,
  `CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, concepts_json, tags_json)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.summary, OLD.concepts_json, OLD.tags_json);
END`,
  `CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, concepts_json, tags_json)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.summary, OLD.concepts_json, OLD.tags_json);
    INSERT INTO memories_fts(rowid, content, summary, concepts_json, tags_json)
    VALUES (NEW.rowid, NEW.content, NEW.summary, NEW.concepts_json, NEW.tags_json);
END`,
  `CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    content='document_chunks',
    content_rowid='rowid'
)`,
  `CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON document_chunks BEGIN
    INSERT INTO documents_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END`,
  `CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON document_chunks BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content)
    VALUES ('delete', OLD.rowid, OLD.content);
END`,
  `CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content)
    VALUES ('delete', OLD.rowid, OLD.content);
    INSERT INTO documents_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END`,
];

export const EXTRACTION_SCHEMA_STATEMENTS = [
  `CREATE TABLE IF NOT EXISTS extraction_segments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    user_prompts_json TEXT,
    files_read_json TEXT,
    files_modified_json TEXT,
    tool_call_count INTEGER DEFAULT 0,
    memories_extracted INTEGER DEFAULT 0,
    extraction_tokens INTEGER,
    segment_start INTEGER,
    segment_end INTEGER,
    extraction_duration_ms INTEGER,
    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (project_id) REFERENCES projects(id)
  )`,
  `CREATE TABLE IF NOT EXISTS segment_accumulators (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    segment_start INTEGER NOT NULL,
    user_prompts_json TEXT DEFAULT '[]',
    files_read_json TEXT DEFAULT '[]',
    files_modified_json TEXT DEFAULT '[]',
    commands_run_json TEXT DEFAULT '[]',
    errors_encountered_json TEXT DEFAULT '[]',
    searches_performed_json TEXT DEFAULT '[]',
    last_assistant_message TEXT,
    tool_call_count INTEGER DEFAULT 0,
    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (project_id) REFERENCES projects(id)
  )`,
];

export const INDEX_STATEMENTS = [
  `CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id) WHERE is_deleted = 0`,
  `CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories(sector)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_simhash ON memories(simhash)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_until)`,
  `CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)`,
  `CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_session_memories_session ON session_memories(session_id)`,
  `CREATE INDEX IF NOT EXISTS idx_session_memories_memory ON session_memories(memory_id)`,
  `CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)`,
  `CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)`,
  `CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id)`,
  `CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id)`,
  `CREATE INDEX IF NOT EXISTS idx_relationships_source ON memory_relationships(source_memory_id)`,
  `CREATE INDEX IF NOT EXISTS idx_relationships_target ON memory_relationships(target_memory_id)`,
  `CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relationship_type)`,
  `CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id)`,
  `CREATE INDEX IF NOT EXISTS idx_document_chunks_doc ON document_chunks(document_id)`,
];
