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
import {
  getOrCreateSession,
  createSessionService,
} from "../../src/services/memory/sessions.js";
import { createSearchService } from "../../src/services/search/hybrid.js";
import { createEmbeddingService } from "../../src/services/embedding/index.js";

describe("Full Capture Flow Integration", () => {
  const testDir = `/tmp/ccmemory-capture-integration-${Date.now()}`;
  let db: Database;

  beforeAll(async () => {
    await mkdir(testDir, { recursive: true });
    process.env["CCMEMORY_DATA_DIR"] = testDir;
    process.env["CCMEMORY_CONFIG_DIR"] = testDir;
    process.env["CCMEMORY_CACHE_DIR"] = testDir;

    db = await createDatabase(join(testDir, "test.db"));
    setDatabase(db);
  });

  afterAll(async () => {
    closeDatabase();
    await rm(testDir, { recursive: true, force: true });
    delete process.env["CCMEMORY_DATA_DIR"];
    delete process.env["CCMEMORY_CONFIG_DIR"];
    delete process.env["CCMEMORY_CACHE_DIR"];
  });

  test("simulates full capture workflow: session creation -> memory capture -> session end", async () => {
    const projectPath = "/test/project";
    const sessionId = `test-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    expect(project.id).toBeDefined();
    expect(project.path).toBe(projectPath);

    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const toolObservations = [
      {
        content: "Tool: Read\nRead file: src/components/Button.tsx",
        sector: "episodic" as const,
        files: ["src/components/Button.tsx"],
      },
      {
        content: "Tool: Edit\nEdited file: src/components/Button.tsx\nAdded onClick handler",
        sector: "episodic" as const,
        files: ["src/components/Button.tsx"],
      },
      {
        content: "Tool: Bash\nCommand: npm test\nOutput: All tests passed",
        sector: "episodic" as const,
      },
    ];

    const createdMemories = [];
    for (const obs of toolObservations) {
      const memory = await store.create(
        {
          content: obs.content,
          sector: obs.sector,
          tier: "session",
          files: obs.files,
        },
        project.id,
        sessionId
      );
      createdMemories.push(memory);
    }

    expect(createdMemories).toHaveLength(3);
    expect(createdMemories.every((m) => m.sector === "episodic")).toBe(true);
    expect(createdMemories.every((m) => m.tier === "session")).toBe(true);

    const sessionMemories = await store.getBySession(sessionId);
    expect(sessionMemories).toHaveLength(3);

    const sessionService = createSessionService();
    const summary = "Session completed with 3 tool observations.\nFiles accessed: Button.tsx\nCommands run: npm test";
    await sessionService.end(sessionId, summary);

    const sessionRow = await db.execute(
      "SELECT * FROM sessions WHERE id = ?",
      [sessionId]
    );
    expect(sessionRow.rows).toHaveLength(1);
    expect(sessionRow.rows[0]?.["ended_at"]).toBeDefined();
    expect(sessionRow.rows[0]?.["summary"]).toBe(summary);
  });

  test("memories are searchable after capture", async () => {
    const projectPath = "/test/searchable-project";
    const sessionId = `search-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: "The authentication module uses JWT tokens for user validation",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    await store.create(
      {
        content: "Database connections are pooled with a maximum of 10 connections",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const embeddingService = await createEmbeddingService();
    const searchService = createSearchService(embeddingService);

    const ftsResults = await searchService.search({
      query: "authentication JWT",
      projectId: project.id,
      limit: 10,
    });

    expect(ftsResults.length).toBeGreaterThan(0);
    expect(ftsResults.some((r) => r.memory.content.includes("JWT"))).toBe(true);
  });

  test("deduplication prevents duplicate memories", async () => {
    const projectPath = "/test/dedup-project";
    const sessionId = `dedup-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const content = "The API endpoint is located at /api/v1/users for user management operations";

    const mem1 = await store.create(
      { content, sector: "semantic", tier: "project" },
      project.id,
      sessionId
    );

    const sessionId2 = `dedup-session-2-${Date.now()}`;
    await getOrCreateSession(sessionId2, project.id);

    const mem2 = await store.create(
      { content, sector: "semantic", tier: "project" },
      project.id,
      sessionId2
    );

    expect(mem2.id).toBe(mem1.id);
    expect(mem2.salience).toBeGreaterThanOrEqual(mem1.salience);
  });

  test("memory reinforcement increases salience", async () => {
    const projectPath = "/test/reinforce-project";
    const sessionId = `reinforce-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memory = await store.create(
      {
        content: "Important configuration setting for production deployment",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    await store.deemphasize(memory.id, 0.5);
    const lowSalience = await store.get(memory.id);
    expect(lowSalience?.salience).toBe(0.5);

    const reinforced = await store.reinforce(memory.id, 0.3);
    expect(reinforced.salience).toBeGreaterThan(0.5);
    expect(reinforced.accessCount).toBeGreaterThan(memory.accessCount);
  });

  test("session tier promotion for high-salience memories", async () => {
    const projectPath = "/test/promotion-project";
    const sessionId = `promotion-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memory = await store.create(
      {
        content: "Critical bug fix for memory leak in event handler",
        sector: "episodic",
        tier: "session",
      },
      project.id,
      sessionId
    );

    expect(memory.tier).toBe("session");
    expect(memory.salience).toBe(1.0);

    await db.execute(
      `UPDATE memories
       SET tier = 'project', updated_at = ?
       WHERE id = ? AND tier = 'session' AND salience > 0.7`,
      [Date.now(), memory.id]
    );

    const promoted = await store.get(memory.id);
    expect(promoted?.tier).toBe("project");
  });

  test("memory relationships are tracked", async () => {
    const projectPath = "/test/relationship-project";
    const sessionId = `relationship-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const original = await store.create(
      {
        content: "Original implementation uses synchronous file operations",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const updated = await store.create(
      {
        content: "Updated implementation uses asynchronous file operations for better performance",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const now = Date.now();
    await db.execute(
      `INSERT INTO memory_relationships (id, source_memory_id, target_memory_id, relationship_type, extracted_by, created_at, valid_from)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        crypto.randomUUID(),
        updated.id,
        original.id,
        "SUPERSEDES",
        "user",
        now,
        now,
      ]
    );

    await db.execute(
      "UPDATE memories SET valid_until = ? WHERE id = ?",
      [Date.now(), original.id]
    );

    const relationships = await db.execute(
      "SELECT * FROM memory_relationships WHERE source_memory_id = ?",
      [updated.id]
    );

    expect(relationships.rows).toHaveLength(1);
    expect(relationships.rows[0]?.["relationship_type"]).toBe("SUPERSEDES");
    expect(relationships.rows[0]?.["target_memory_id"]).toBe(original.id);

    const invalidated = await store.get(original.id);
    expect(invalidated?.validUntil).toBeDefined();
  });
});
