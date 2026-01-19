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
import { getOrCreateSession, createSessionService } from "../../src/services/memory/sessions.js";

describe("Concurrent Instance Integration", () => {
  const testDir = `/tmp/ccmemory-concurrent-integration-${Date.now()}`;
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

  test("multiple sessions can write to the same project concurrently", async () => {
    const projectPath = "/test/concurrent-project";
    const project = await getOrCreateProject(projectPath);

    const sessionIds = [
      `session-a-${Date.now()}`,
      `session-b-${Date.now()}`,
      `session-c-${Date.now()}`,
    ];

    await Promise.all(
      sessionIds.map((sid) => getOrCreateSession(sid, project.id))
    );

    const store = createMemoryStore();

    const writePromises = sessionIds.flatMap((sessionId, idx) =>
      Array.from({ length: 5 }, (_, i) =>
        store.create(
          {
            content: `Memory from session ${sessionId} number ${idx * 5 + i}, unique content: ${crypto.randomUUID()}`,
            sector: "episodic",
            tier: "session",
          },
          project.id,
          sessionId
        )
      )
    );

    const results = await Promise.all(writePromises);

    expect(results).toHaveLength(15);
    expect(results.every((m) => m.id !== undefined)).toBe(true);

    const uniqueIds = new Set(results.map((m) => m.id));
    expect(uniqueIds.size).toBe(15);

    for (const sessionId of sessionIds) {
      const sessionMemories = await store.getBySession(sessionId);
      expect(sessionMemories.length).toBeGreaterThanOrEqual(5);
    }
  });

  test("concurrent reinforcement operations are handled correctly", async () => {
    const projectPath = "/test/reinforce-concurrent";
    const project = await getOrCreateProject(projectPath);
    const sessionId = `reinforce-session-${Date.now()}`;
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memory = await store.create(
      {
        content: "Shared memory for concurrent reinforcement testing",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    await store.deemphasize(memory.id, 0.5);
    const lowSalience = await store.get(memory.id);
    expect(lowSalience?.salience).toBe(0.5);

    const reinforcePromises = Array.from({ length: 5 }, () =>
      store.reinforce(memory.id, 0.1)
    );

    const reinforced = await Promise.all(reinforcePromises);

    expect(reinforced.every((m) => m.salience >= 0.5)).toBe(true);

    const finalMemory = await store.get(memory.id);
    expect(finalMemory).not.toBeNull();
    expect(finalMemory?.accessCount).toBeGreaterThanOrEqual(5);
  });

  test("session end operations are isolated", async () => {
    const projectPath = "/test/session-end";
    const project = await getOrCreateProject(projectPath);

    const sessionIds = [`end-a-${Date.now()}`, `end-b-${Date.now()}`];
    await Promise.all(sessionIds.map((sid) => getOrCreateSession(sid, project.id)));

    const store = createMemoryStore();
    const sessionService = createSessionService();

    for (const sessionId of sessionIds) {
      await store.create(
        {
          content: `Memory for ${sessionId}`,
          sector: "episodic",
          tier: "session",
        },
        project.id,
        sessionId
      );
    }

    await sessionService.end(sessionIds[0]!, `Summary for session A`);

    const session0 = await db.execute(
      "SELECT * FROM sessions WHERE id = ?",
      [sessionIds[0]!]
    );
    const session1 = await db.execute(
      "SELECT * FROM sessions WHERE id = ?",
      [sessionIds[1]!]
    );

    expect(session0.rows[0]?.["ended_at"]).toBeDefined();
    expect(session0.rows[0]?.["summary"]).toContain("session A");

    expect(session1.rows[0]?.["ended_at"]).toBeNull();
    expect(session1.rows[0]?.["summary"]).toBeNull();
  });

  test("project isolation is maintained under concurrent access", async () => {
    const project1 = await getOrCreateProject("/test/project-1");
    const project2 = await getOrCreateProject("/test/project-2");

    const session1 = `proj1-session-${Date.now()}`;
    const session2 = `proj2-session-${Date.now()}`;

    await getOrCreateSession(session1, project1.id);
    await getOrCreateSession(session2, project2.id);

    const store = createMemoryStore();

    const writePromises = [
      ...Array.from({ length: 10 }, (_, i) =>
        store.create(
          {
            content: `Project 1 memory number ${i}: unique identifier proj1 with uuid ${crypto.randomUUID()}`,
            sector: "semantic",
            tier: "project",
          },
          project1.id,
          session1
        )
      ),
      ...Array.from({ length: 10 }, (_, i) =>
        store.create(
          {
            content: `Project 2 memory number ${i}: unique identifier proj2 with uuid ${crypto.randomUUID()}`,
            sector: "semantic",
            tier: "project",
          },
          project2.id,
          session2
        )
      ),
    ];

    await Promise.all(writePromises);

    const proj1Memories = await store.list({ projectId: project1.id, limit: 100 });
    const proj2Memories = await store.list({ projectId: project2.id, limit: 100 });

    expect(proj1Memories.every((m) => m.content.includes("proj1"))).toBe(true);
    expect(proj2Memories.every((m) => m.content.includes("proj2"))).toBe(true);

    expect(proj1Memories.length).toBe(10);
    expect(proj2Memories.length).toBe(10);
  });

  test("database WAL mode handles concurrent writes efficiently", async () => {
    const projectPath = "/test/wal-test";
    const project = await getOrCreateProject(projectPath);
    const sessionId = `wal-session-${Date.now()}`;
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const startTime = Date.now();

    const contents = Array.from({ length: 50 }, (_, i) =>
      `Heavy write ${crypto.randomUUID()} test entry ${i} unique ${crypto.randomUUID()} more content to differentiate ${Math.random().toString(36).substring(2)}`
    );

    const heavyWritePromises = contents.map((content) =>
      store.create(
        {
          content,
          sector: "episodic",
          tier: "session",
        },
        project.id,
        sessionId
      )
    );

    await Promise.all(heavyWritePromises);

    const elapsed = Date.now() - startTime;

    expect(elapsed).toBeLessThan(5000);

    const memories = await store.getBySession(sessionId);
    expect(memories.length).toBeGreaterThanOrEqual(45);
  });

  test("concurrent get and update operations are consistent", async () => {
    const projectPath = "/test/get-update";
    const project = await getOrCreateProject(projectPath);
    const sessionId = `get-update-session-${Date.now()}`;
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memory = await store.create(
      {
        content: "Original content for get-update test",
        sector: "semantic",
        tier: "project",
        tags: ["original"],
      },
      project.id,
      sessionId
    );

    const concurrentOps = [
      store.get(memory.id),
      store.touch(memory.id),
      store.update(memory.id, { tags: ["updated", "concurrent"] }),
      store.get(memory.id),
      store.touch(memory.id),
    ];

    const results = await Promise.all(concurrentOps);

    const finalMemory = await store.get(memory.id);
    expect(finalMemory).not.toBeNull();
    expect(finalMemory?.accessCount).toBeGreaterThanOrEqual(2);
    expect(finalMemory?.tags).toContain("updated");
  });

  test("simhash deduplication works across concurrent creates", async () => {
    const projectPath = "/test/dedup-concurrent";
    const project = await getOrCreateProject(projectPath);

    const sessionIds = [`dedup-a-${Date.now()}`, `dedup-b-${Date.now()}`];
    await Promise.all(sessionIds.map((sid) => getOrCreateSession(sid, project.id)));

    const store = createMemoryStore();

    const duplicateContent = "This is the exact same content that should be deduplicated across sessions";

    const createPromises = sessionIds.map((sessionId) =>
      store.create(
        {
          content: duplicateContent,
          sector: "semantic",
          tier: "project",
        },
        project.id,
        sessionId
      )
    );

    const createdMemories = await Promise.all(createPromises);

    expect(createdMemories[0]?.id).toBe(createdMemories[1]?.id);
  });
});
