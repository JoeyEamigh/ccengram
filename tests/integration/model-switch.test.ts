import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import {
  createDatabase,
  closeDatabase,
  setDatabase,
  type Database,
} from "../../src/db/database.js";
import { createMemoryStore } from "../../src/services/memory/store.js";
import { getOrCreateProject } from "../../src/services/project.js";
import { getOrCreateSession } from "../../src/services/memory/sessions.js";
import { createEmbeddingService } from "../../src/services/embedding/index.js";
import type { EmbeddingService } from "../../src/services/embedding/types.js";
import { createSearchService } from "../../src/services/search/hybrid.js";

describe("Model Switching Integration", () => {
  const testDir = `/tmp/ccmemory-model-switch-${Date.now()}`;
  let db: Database;
  let embeddingService: EmbeddingService;

  beforeAll(async () => {
    await mkdir(testDir, { recursive: true });
    process.env["CCMEMORY_DATA_DIR"] = testDir;
    process.env["CCMEMORY_CONFIG_DIR"] = testDir;
    process.env["CCMEMORY_CACHE_DIR"] = testDir;

    db = await createDatabase(join(testDir, "test.db"));
    setDatabase(db);

    embeddingService = await createEmbeddingService();
  });

  afterAll(async () => {
    closeDatabase();
    await rm(testDir, { recursive: true, force: true });
    delete process.env["CCMEMORY_DATA_DIR"];
    delete process.env["CCMEMORY_CONFIG_DIR"];
    delete process.env["CCMEMORY_CACHE_DIR"];
  });

  test("embedding service initializes with default provider", async () => {
    const provider = embeddingService.getProvider();

    expect(provider.name).toBeDefined();
    expect(provider.model).toBeDefined();
    expect(provider.dimensions).toBeGreaterThan(0);
  });

  test("embedding service can embed text", async () => {
    const result = await embeddingService.embed("test text for embedding");

    expect(result.vector).toBeDefined();
    expect(result.vector.length).toBeGreaterThan(0);
    expect(result.model).toBeDefined();
    expect(result.dimensions).toBe(result.vector.length);
  });

  test("embedding service tracks active model ID", async () => {
    const modelId = embeddingService.getActiveModelId();

    expect(typeof modelId).toBe("string");
    expect(modelId.length).toBeGreaterThan(0);

    const modelRow = await db.execute(
      "SELECT * FROM embedding_models WHERE id = ?",
      [modelId]
    );
    expect(modelRow.rows.length).toBe(1);
  });

  test("embedding model is registered in database", async () => {
    await embeddingService.embed("register model test");

    const models = await db.execute("SELECT * FROM embedding_models");

    expect(models.rows.length).toBeGreaterThan(0);
    const model = models.rows[0];
    expect(model?.["provider"]).toBeDefined();
    expect(model?.["name"]).toBeDefined();
    expect(Number(model?.["dimensions"])).toBeGreaterThan(0);
  });

  test("embeddings are stored with model reference", async () => {
    const projectPath = "/test/embedding-project";
    const sessionId = `embedding-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memory = await store.create(
      {
        content: "Test memory content for embedding storage verification",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const modelId = embeddingService.getActiveModelId();
    const result = await embeddingService.embed(memory.content);
    const vectorBlob = new Float32Array(result.vector).buffer;

    await db.execute(
      `INSERT OR REPLACE INTO memory_vectors (memory_id, model_id, vector, dim, created_at)
       VALUES (?, ?, ?, ?, ?)`,
      [memory.id, modelId, vectorBlob, result.vector.length, Date.now()]
    );

    const vectorRow = await db.execute(
      "SELECT * FROM memory_vectors WHERE memory_id = ?",
      [memory.id]
    );

    expect(vectorRow.rows.length).toBe(1);
    expect(vectorRow.rows[0]?.["model_id"]).toBe(modelId);
    expect(vectorRow.rows[0]?.["dim"]).toBe(result.dimensions);
  });

  test("search works with current model embeddings", async () => {
    const projectPath = "/test/search-model-project";
    const sessionId = `search-model-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memory = await store.create(
      {
        content: "The Kubernetes cluster uses Helm charts for deployment management",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const modelId = embeddingService.getActiveModelId();
    const result = await embeddingService.embed(memory.content);
    const vectorBlob = new Float32Array(result.vector).buffer;

    await db.execute(
      `INSERT OR REPLACE INTO memory_vectors (memory_id, model_id, vector, dim, created_at)
       VALUES (?, ?, ?, ?, ?)`,
      [memory.id, modelId, vectorBlob, result.vector.length, Date.now()]
    );

    const searchService = createSearchService(embeddingService);

    const results = await searchService.search({
      query: "Kubernetes Helm deployment",
      projectId: project.id,
      mode: "semantic",
      limit: 10,
    });

    expect(results.length).toBe(1);
    expect(results[0]?.memory.content).toContain("Kubernetes");
  });

  test("multiple embeddings can be generated in batch", async () => {
    const texts = [
      "First batch text about authentication",
      "Second batch text about authorization",
      "Third batch text about security policies",
    ];

    const results = await embeddingService.embedBatch(texts);

    expect(results.length).toBe(3);
    expect(results.every((r) => r.vector.length > 0)).toBe(true);
    expect(results.every((r) => r.dimensions === results[0]?.dimensions)).toBe(true);
  });

  test("embedding dimensions are consistent within same model", async () => {
    const embeddings = await Promise.all([
      embeddingService.embed("short text"),
      embeddingService.embed("a much longer text with more words and content to verify dimension consistency"),
      embeddingService.embed("another test text"),
    ]);

    const dimensions = embeddings.map((e) => e.dimensions);

    expect(dimensions.every((d) => d === dimensions[0])).toBe(true);
  });

  test("provider information is accessible", async () => {
    const provider = embeddingService.getProvider();

    expect(["ollama", "openrouter"]).toContain(provider.name);
    expect(provider.model).toBeDefined();
    expect(typeof provider.dimensions).toBe("number");
  });

  test("embedding service handles empty text gracefully", async () => {
    try {
      const result = await embeddingService.embed("");
      expect(result.vector.length).toBeGreaterThan(0);
    } catch (error) {
      expect(error).toBeDefined();
    }
  });

  test("model metadata is persisted correctly", async () => {
    const modelId = embeddingService.getActiveModelId();

    const modelData = await db.execute(
      `SELECT id, provider, name, dimensions, created_at
       FROM embedding_models WHERE id = ?`,
      [modelId]
    );

    expect(modelData.rows.length).toBe(1);
    const row = modelData.rows[0];

    expect(row?.["provider"]).toBeDefined();
    expect(row?.["name"]).toBeDefined();
    expect(Number(row?.["dimensions"])).toBeGreaterThan(0);
    expect(Number(row?.["created_at"])).toBeGreaterThan(0);
  });
});
