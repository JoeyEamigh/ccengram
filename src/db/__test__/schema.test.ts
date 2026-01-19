import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createDatabase, type Database } from "../database.js";

describe("Schema", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
  });

  afterEach(() => {
    db.close();
  });

  test("creates all required tables", async () => {
    const result = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    );
    const names = result.rows.map((r) => r["name"]);

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
    const result = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts%'"
    );
    const names = result.rows.map((r) => r["name"]);
    expect(names).toContain("memories_fts");
    expect(names).toContain("documents_fts");
  });

  test("FTS triggers are created", async () => {
    const result = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='trigger'"
    );
    const names = result.rows.map((r) => r["name"]);
    expect(names).toContain("memories_ai");
    expect(names).toContain("memories_ad");
    expect(names).toContain("memories_au");
    expect(names).toContain("documents_ai");
    expect(names).toContain("documents_ad");
    expect(names).toContain("documents_au");
  });

  test("session_memories tracks memory usage", async () => {
    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      "INSERT INTO sessions (id, project_id, started_at) VALUES ('s1', 'p1', 0)"
    );
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m1', 'p1', 'test', 'semantic', 0, 0, 0)`
    );

    await db.execute(
      `INSERT INTO session_memories (session_id, memory_id, created_at, usage_type)
       VALUES ('s1', 'm1', 0, 'recalled')`
    );

    const result = await db.execute(
      "SELECT usage_type FROM session_memories WHERE memory_id = 'm1'"
    );
    expect(result.rows[0]?.["usage_type"]).toBe("recalled");
  });

  test("memory_relationships tracks SUPERSEDES", async () => {
    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m1', 'p1', 'old fact', 'semantic', 0, 0, 0)`
    );
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m2', 'p1', 'new fact', 'semantic', 1, 1, 1)`
    );

    await db.execute(
      `INSERT INTO memory_relationships
       (id, source_memory_id, target_memory_id, relationship_type, created_at, valid_from, extracted_by)
       VALUES ('r1', 'm2', 'm1', 'SUPERSEDES', 1, 1, 'system')`
    );

    const result = await db.execute(
      "SELECT relationship_type FROM memory_relationships WHERE source_memory_id = 'm2'"
    );
    expect(result.rows[0]?.["relationship_type"]).toBe("SUPERSEDES");
  });

  test("soft delete flag exists on memories", async () => {
    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed, is_deleted)
       VALUES ('m1', 'p1', 'test', 'semantic', 0, 0, 0, 1)`
    );

    const result = await db.execute(
      "SELECT is_deleted FROM memories WHERE id = 'm1'"
    );
    expect(result.rows[0]?.["is_deleted"]).toBe(1);
  });

  test("memories have bi-temporal columns", async () => {
    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    const now = Date.now();
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed, valid_from, valid_until)
       VALUES ('m1', 'p1', 'test', 'semantic', ?, ?, ?, ?, ?)`,
      [now, now, now, now - 1000, now + 1000]
    );

    const result = await db.execute(
      "SELECT valid_from, valid_until FROM memories WHERE id = 'm1'"
    );
    expect(result.rows[0]?.["valid_from"]).toBe(now - 1000);
    expect(result.rows[0]?.["valid_until"]).toBe(now + 1000);
  });
});
