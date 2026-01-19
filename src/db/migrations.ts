import type { Client } from "@libsql/client";
import { SCHEMA_STATEMENTS, FTS_STATEMENTS, INDEX_STATEMENTS } from "./schema.js";

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
    name: "initial_schema",
    statements: SCHEMA_STATEMENTS,
  },
  {
    version: 2,
    name: "fts_tables",
    statements: FTS_STATEMENTS,
  },
  {
    version: 3,
    name: "indexes",
    statements: INDEX_STATEMENTS,
  },
  {
    version: 4,
    name: "config_and_compound_indexes",
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
];

export async function getCurrentVersion(client: Client): Promise<number> {
  try {
    const result = await client.execute(
      "SELECT MAX(version) as version FROM _migrations"
    );
    const row = result.rows[0];
    if (row && row["version"] !== null) {
      return Number(row["version"]);
    }
    return 0;
  } catch {
    return 0;
  }
}

export async function runMigrations(client: Client): Promise<void> {
  await client.execute(MIGRATIONS_TABLE_SQL);

  const currentVersion = await getCurrentVersion(client);
  const pending = migrations.filter((m) => m.version > currentVersion);

  for (const migration of pending) {
    for (const sql of migration.statements) {
      await client.execute(sql);
    }

    await client.execute({
      sql: "INSERT INTO _migrations (version, name) VALUES (?, ?)",
      args: [migration.version, migration.name],
    });
  }
}
