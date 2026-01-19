import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  createDatabase,
  closeDatabase,
  setDatabase,
  getDatabase,
  type Database,
} from "../../db/database.js";
import { createMemoryStore } from "../../services/memory/store.js";
import { createSearchService } from "../../services/search/hybrid.js";
import { createDocumentService } from "../../services/documents/ingest.js";
import { supersede } from "../../services/memory/relationships.js";
import type { EmbeddingService, EmbeddingResult } from "../../services/embedding/types.js";

function createMockEmbeddingService(): EmbeddingService {
  const mockVector = Array(128).fill(0.1);

  return {
    getProvider: () => ({
      name: "mock",
      model: "test-model",
      dimensions: 128,
      embed: async () => mockVector,
      embedBatch: async () => [],
      isAvailable: async () => true,
    }),
    embed: async (): Promise<EmbeddingResult> => ({
      vector: mockVector,
      model: "test-model",
      dimensions: 128,
      cached: false,
    }),
    embedBatch: async (texts: string[]): Promise<EmbeddingResult[]> =>
      texts.map(() => ({
        vector: mockVector,
        model: "test-model",
        dimensions: 128,
        cached: false,
      })),
    getActiveModelId: () => "mock:test-model",
    switchProvider: async () => {},
  };
}

describe("MCP Server Tools", () => {
  let db: Database;
  let embeddingService: EmbeddingService;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
    setDatabase(db);
    embeddingService = createMockEmbeddingService();

    const now = Date.now();
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ["proj1", "/test/path", "Test Project", now, now]
    );

    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ["mock:test-model", "test-model", "mock", 128, 1, now]
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  describe("memory_search", () => {
    test("returns results with session context", async () => {
      const store = createMemoryStore();
      await store.create(
        { content: "Test memory about React components" },
        "proj1"
      );

      const search = createSearchService(embeddingService);
      const results = await search.search({
        query: "React",
        projectId: "proj1",
        limit: 5,
      });

      expect(results.length).toBeGreaterThan(0);
      expect(results[0]?.memory.content).toContain("React");
    });

    test("excludes superseded memories by default", async () => {
      const store = createMemoryStore();
      const oldMem = await store.create(
        { content: "Old API endpoint documentation" },
        "proj1"
      );
      const newMem = await store.create(
        { content: "New API endpoint documentation" },
        "proj1"
      );
      await supersede(oldMem.id, newMem.id);

      const search = createSearchService(embeddingService);
      const results = await search.search({
        query: "API endpoint",
        projectId: "proj1",
        includeSuperseded: false,
      });

      const oldFound = results.find((r) => r.memory.id === oldMem.id);
      expect(oldFound).toBeUndefined();
    });

    test("includes superseded when requested", async () => {
      const store = createMemoryStore();
      const oldMem = await store.create(
        { content: "Outdated configuration settings" },
        "proj1"
      );
      const newMem = await store.create(
        { content: "Updated configuration settings" },
        "proj1"
      );
      await supersede(oldMem.id, newMem.id);

      const search = createSearchService(embeddingService);
      const results = await search.search({
        query: "configuration",
        projectId: "proj1",
        includeSuperseded: true,
      });

      const oldFound = results.find((r) => r.memory.id === oldMem.id);
      expect(oldFound).toBeDefined();
      expect(oldFound?.isSuperseded).toBe(true);
    });
  });

  describe("memory_add", () => {
    test("creates memory with sector", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        {
          content: "User prefers TypeScript for new projects",
          sector: "emotional",
        },
        "proj1"
      );

      expect(memory.id).toBeDefined();
      expect(memory.sector).toBe("emotional");
    });

    test("auto-classifies sector when not provided", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        {
          content: "To deploy, run npm build then upload to S3",
        },
        "proj1"
      );

      expect(memory.sector).toBe("procedural");
    });
  });

  describe("memory_reinforce", () => {
    test("increases salience from lower value", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        { content: "Important architectural decision" },
        "proj1"
      );
      await store.deemphasize(memory.id, 0.5);
      const loweredMem = await store.get(memory.id);
      const initialSalience = loweredMem?.salience ?? 0;

      const reinforced = await store.reinforce(memory.id, 0.2);

      expect(reinforced.salience).toBeGreaterThan(initialSalience);
    });

    test("has diminishing returns", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        { content: "High salience fact" },
        "proj1"
      );

      const gains: number[] = [];
      let current = memory;
      for (let i = 0; i < 5; i++) {
        const before = current.salience;
        current = await store.reinforce(current.id, 0.2);
        gains.push(current.salience - before);
      }

      expect(current.salience).toBeLessThanOrEqual(1.0);
      expect(current.salience).toBeGreaterThan(0.8);

      for (let i = 1; i < gains.length; i++) {
        const prevGain = gains[i - 1] ?? 0;
        const currGain = gains[i] ?? 0;
        expect(currGain).toBeLessThanOrEqual(prevGain + 0.001);
      }
    });
  });

  describe("memory_deemphasize", () => {
    test("reduces salience", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        { content: "Less important information" },
        "proj1"
      );
      const initialSalience = memory.salience;

      const deemphasized = await store.deemphasize(memory.id, 0.3);

      expect(deemphasized.salience).toBeLessThan(initialSalience);
    });
  });

  describe("memory_delete", () => {
    test("soft deletes by default", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        { content: "To be soft deleted" },
        "proj1"
      );

      await store.delete(memory.id, false);

      const database = await getDatabase();
      const result = await database.execute(
        "SELECT is_deleted FROM memories WHERE id = ?",
        [memory.id]
      );
      expect(result.rows[0]?.["is_deleted"]).toBe(1);
    });

    test("hard deletes when requested", async () => {
      const store = createMemoryStore();
      const memory = await store.create(
        { content: "To be permanently deleted" },
        "proj1"
      );

      await store.delete(memory.id, true);

      const database = await getDatabase();
      const result = await database.execute(
        "SELECT * FROM memories WHERE id = ?",
        [memory.id]
      );
      expect(result.rows.length).toBe(0);
    });
  });

  describe("memory_supersede", () => {
    test("creates relationship", async () => {
      const store = createMemoryStore();
      const oldMem = await store.create(
        { content: "Old approach to authentication" },
        "proj1"
      );
      const newMem = await store.create(
        { content: "New approach to authentication" },
        "proj1"
      );

      await supersede(oldMem.id, newMem.id);

      const database = await getDatabase();
      const rels = await database.execute(
        `SELECT * FROM memory_relationships
         WHERE source_memory_id = ? AND target_memory_id = ?
         AND relationship_type = 'SUPERSEDES'`,
        [newMem.id, oldMem.id]
      );
      expect(rels.rows.length).toBe(1);
    });

    test("sets valid_until on old memory", async () => {
      const store = createMemoryStore();
      const oldMem = await store.create(
        { content: "Outdated API design" },
        "proj1"
      );
      const newMem = await store.create(
        { content: "Updated API design" },
        "proj1"
      );

      await supersede(oldMem.id, newMem.id);

      const database = await getDatabase();
      const result = await database.execute(
        "SELECT valid_until FROM memories WHERE id = ?",
        [oldMem.id]
      );
      expect(result.rows[0]?.["valid_until"]).not.toBeNull();
    });
  });

  describe("memory_timeline", () => {
    test("includes session info", async () => {
      const database = await getDatabase();
      await database.execute(
        `INSERT INTO sessions (id, project_id, started_at, summary) VALUES (?, ?, ?, ?)`,
        ["sess1", "proj1", Date.now() - 3600000, "Test session"]
      );

      const store = createMemoryStore();
      await store.create({ content: "First action" }, "proj1", "sess1");
      await new Promise((r) => setTimeout(r, 10));
      const m2 = await store.create({ content: "Second action" }, "proj1", "sess1");

      const search = createSearchService(embeddingService);
      const timeline = await search.timeline(m2.id);

      expect(timeline.anchor.id).toBe(m2.id);
      expect(timeline.before.length).toBeGreaterThan(0);
    });
  });

  describe("docs_ingest and docs_search", () => {
    test("ingests and searches document", async () => {
      const docs = createDocumentService(embeddingService);

      const doc = await docs.ingest({
        projectId: "proj1",
        content: "React is a JavaScript library for building user interfaces.",
        title: "React Overview",
      });

      expect(doc.title).toBe("React Overview");

      const results = await docs.search("JavaScript library", "proj1", 5);
      expect(results.length).toBeGreaterThan(0);
    });
  });
});

describe("Tool Response Formatting", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
    setDatabase(db);

    const now = Date.now();
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ["proj1", "/test/path", "Test Project", now, now]
    );

    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ["mock:test-model", "test-model", "mock", 128, 1, now]
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  test("search results include memory ID", async () => {
    const store = createMemoryStore();
    const memory = await store.create(
      { content: "Searchable memory content" },
      "proj1"
    );

    const search = createSearchService(createMockEmbeddingService());
    const results = await search.search({
      query: "searchable",
      projectId: "proj1",
    });

    expect(results.length).toBeGreaterThan(0);
    expect(results[0]?.memory.id).toBe(memory.id);
  });

  test("empty search returns appropriate message", async () => {
    const search = createSearchService(createMockEmbeddingService());
    const results = await search.search({
      query: "nonexistent",
      projectId: "proj1",
    });

    expect(results.length).toBe(0);
  });
});
