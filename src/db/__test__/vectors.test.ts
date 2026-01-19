import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createDatabase, type Database } from "../database.js";

describe("Vector Operations", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
  });

  afterEach(() => {
    db.close();
  });

  test("stores vectors using vector() function", async () => {
    await db.execute(
      "INSERT INTO embedding_models (id, name, provider, dimensions) VALUES (?, ?, ?, ?)",
      ["test-model", "test", "test", 4]
    );

    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m1', 'p1', 'test content', 'semantic', 0, 0, 0)`
    );

    const vec = [0.1, 0.2, 0.3, 0.4];
    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["m1", "test-model", JSON.stringify(vec), 4]
    );

    const result = await db.execute(
      "SELECT dim FROM memory_vectors WHERE memory_id = ?",
      ["m1"]
    );
    expect(result.rows[0]?.["dim"]).toBe(4);
  });

  test("calculates vector cosine distance", async () => {
    await db.execute(
      "INSERT INTO embedding_models (id, name, provider, dimensions) VALUES (?, ?, ?, ?)",
      ["test-model", "test", "test", 4]
    );

    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m1', 'p1', 'test 1', 'semantic', 0, 0, 0)`
    );
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m2', 'p1', 'test 2', 'semantic', 0, 0, 0)`
    );

    const vec1 = [1.0, 0.0, 0.0, 0.0];
    const vec2 = [0.0, 1.0, 0.0, 0.0];

    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["m1", "test-model", JSON.stringify(vec1), 4]
    );
    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["m2", "test-model", JSON.stringify(vec2), 4]
    );

    const result = await db.execute(
      `SELECT memory_id, vector_distance_cos(vector, vector(?)) as distance
       FROM memory_vectors
       ORDER BY distance ASC`,
      [JSON.stringify(vec1)]
    );

    expect(result.rows[0]?.["memory_id"]).toBe("m1");
    expect(result.rows[0]?.["distance"]).toBe(0);
    expect(Number(result.rows[1]?.["distance"])).toBeGreaterThan(0);
  });

  test("handles different vector dimensions per model", async () => {
    await db.execute(
      "INSERT INTO embedding_models (id, name, provider, dimensions) VALUES (?, ?, ?, ?)",
      ["model-4d", "test4", "test", 4]
    );
    await db.execute(
      "INSERT INTO embedding_models (id, name, provider, dimensions) VALUES (?, ?, ?, ?)",
      ["model-8d", "test8", "test", 8]
    );

    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m1', 'p1', 'test 1', 'semantic', 0, 0, 0)`
    );
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m2', 'p1', 'test 2', 'semantic', 0, 0, 0)`
    );

    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["m1", "model-4d", JSON.stringify([0.1, 0.2, 0.3, 0.4]), 4]
    );
    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["m2", "model-8d", JSON.stringify([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), 8]
    );

    const result = await db.execute(
      "SELECT memory_id, dim FROM memory_vectors ORDER BY dim"
    );
    expect(result.rows[0]?.["dim"]).toBe(4);
    expect(result.rows[1]?.["dim"]).toBe(8);
  });

  test("cascades delete from memories to vectors", async () => {
    await db.execute(
      "INSERT INTO embedding_models (id, name, provider, dimensions) VALUES (?, ?, ?, ?)",
      ["test-model", "test", "test", 4]
    );

    await db.execute("INSERT INTO projects (id, path) VALUES ('p1', '/test')");
    await db.execute(
      `INSERT INTO memories (id, project_id, content, sector, created_at, updated_at, last_accessed)
       VALUES ('m1', 'p1', 'test', 'semantic', 0, 0, 0)`
    );
    await db.execute(
      "INSERT INTO memory_vectors (memory_id, model_id, vector, dim) VALUES (?, ?, vector(?), ?)",
      ["m1", "test-model", JSON.stringify([0.1, 0.2, 0.3, 0.4]), 4]
    );

    let vectorCount = await db.execute("SELECT COUNT(*) as cnt FROM memory_vectors");
    expect(vectorCount.rows[0]?.["cnt"]).toBe(1);

    await db.execute("DELETE FROM memories WHERE id = 'm1'");

    vectorCount = await db.execute("SELECT COUNT(*) as cnt FROM memory_vectors");
    expect(vectorCount.rows[0]?.["cnt"]).toBe(0);
  });
});
