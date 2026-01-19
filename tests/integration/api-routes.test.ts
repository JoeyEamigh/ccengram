import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import {
  createDatabase,
  closeDatabase,
  setDatabase,
  getDatabase,
  type Database,
} from "../../src/db/database.js";
import { createMemoryStore } from "../../src/services/memory/store.js";
import { getOrCreateProject } from "../../src/services/project.js";
import { getOrCreateSession } from "../../src/services/memory/sessions.js";
import { handleAPI } from "../../src/webui/api/routes.js";

describe("API Routes Integration", () => {
  const testDir = `/tmp/ccmemory-api-routes-${Date.now()}`;
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

  describe("GET /api/config", () => {
    test("returns default config values when config table is empty", async () => {
      const req = new Request("http://localhost/api/config", { method: "GET" });
      const response = await handleAPI(req, "/api/config");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.config).toBeDefined();
      expect(data.config.embeddingProvider).toBe("ollama");
      expect(data.config.captureEnabled).toBe("true");
      expect(data.config.captureThreshold).toBe("0.3");
    });
  });

  describe("PUT /api/config", () => {
    test("updates config value and persists it", async () => {
      const putReq = new Request("http://localhost/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: "embeddingProvider", value: "openrouter" }),
      });
      const putResponse = await handleAPI(putReq, "/api/config");

      expect(putResponse.status).toBe(200);
      const putData = await putResponse.json();
      expect(putData.ok).toBe(true);

      const getReq = new Request("http://localhost/api/config", { method: "GET" });
      const getResponse = await handleAPI(getReq, "/api/config");

      const getData = await getResponse.json();
      expect(getData.config.embeddingProvider).toBe("openrouter");
    });

    test("updates captureEnabled config", async () => {
      const putReq = new Request("http://localhost/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: "captureEnabled", value: "false" }),
      });
      const response = await handleAPI(putReq, "/api/config");

      expect(response.status).toBe(200);

      const getReq = new Request("http://localhost/api/config", { method: "GET" });
      const getResponse = await handleAPI(getReq, "/api/config");
      const getData = await getResponse.json();
      expect(getData.config.captureEnabled).toBe("false");
    });
  });

  describe("POST /api/memories/clear", () => {
    test("clears all memories when no projectId provided", async () => {
      const project = await getOrCreateProject("/test/clear-all-project");
      const sessionId = `clear-session-${Date.now()}`;
      await getOrCreateSession(sessionId, project.id);

      const db = await getDatabase();

      const beforeCount = await db.execute(
        "SELECT COUNT(*) as count FROM memories WHERE is_deleted = 0"
      );
      const initialCount = Number(beforeCount.rows[0]?.["count"]);

      const store = createMemoryStore();
      const ts = Date.now();
      await store.create(
        { content: `Unique memory content about database configuration ${ts}-1`, sector: "semantic", tier: "project" },
        project.id,
        sessionId
      );
      await store.create(
        { content: `Completely different topic about React components ${ts}-2`, sector: "procedural", tier: "project" },
        project.id,
        sessionId
      );

      const afterCreation = await db.execute(
        "SELECT COUNT(*) as count FROM memories WHERE is_deleted = 0"
      );
      const countAfterCreation = Number(afterCreation.rows[0]?.["count"]);
      expect(countAfterCreation).toBe(initialCount + 2);

      const req = new Request("http://localhost/api/memories/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const response = await handleAPI(req, "/api/memories/clear");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.ok).toBe(true);
      expect(data.deleted).toBeGreaterThanOrEqual(2);

      const afterCount = await db.execute(
        "SELECT COUNT(*) as count FROM memories WHERE is_deleted = 0"
      );
      expect(Number(afterCount.rows[0]?.["count"])).toBe(0);
    });

    test("clears only memories for specified project", async () => {
      const ts = Date.now();
      const project1 = await getOrCreateProject(`/test/clear-project-1-${ts}`);
      const project2 = await getOrCreateProject(`/test/clear-project-2-${ts}`);
      const sessionId1 = `clear-session-1-${ts}`;
      const sessionId2 = `clear-session-2-${ts}`;
      await getOrCreateSession(sessionId1, project1.id);
      await getOrCreateSession(sessionId2, project2.id);

      const store = createMemoryStore();
      await store.create(
        { content: `Project 1 database architecture documentation ${ts}`, sector: "semantic", tier: "project" },
        project1.id,
        sessionId1
      );
      await store.create(
        { content: `Project 2 frontend testing strategies ${ts}`, sector: "semantic", tier: "project" },
        project2.id,
        sessionId2
      );

      const req = new Request("http://localhost/api/memories/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId: project1.id }),
      });
      const response = await handleAPI(req, "/api/memories/clear");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.ok).toBe(true);
      expect(data.deleted).toBeGreaterThanOrEqual(1);

      const db = await getDatabase();
      const project2Count = await db.execute(
        "SELECT COUNT(*) as count FROM memories WHERE project_id = ? AND is_deleted = 0",
        [project2.id]
      );
      expect(Number(project2Count.rows[0]?.["count"])).toBeGreaterThanOrEqual(1);
    });
  });

  describe("GET /api/search with project filter", () => {
    test("filters search results by project", async () => {
      const ts = Date.now();
      const project1 = await getOrCreateProject(`/test/search-project-1-${ts}`);
      const project2 = await getOrCreateProject(`/test/search-project-2-${ts}`);
      const sessionId1 = `search-session-1-${ts}`;
      const sessionId2 = `search-session-2-${ts}`;
      await getOrCreateSession(sessionId1, project1.id);
      await getOrCreateSession(sessionId2, project2.id);

      const store = createMemoryStore();
      await store.create(
        { content: `xyzzy_unique_searchable_token_${ts} in project 1 database config`, sector: "semantic", tier: "project" },
        project1.id,
        sessionId1
      );
      await store.create(
        { content: `xyzzy_unique_searchable_token_${ts} in project 2 react components`, sector: "semantic", tier: "project" },
        project2.id,
        sessionId2
      );

      const req = new Request(
        `http://localhost/api/search?q=xyzzy_unique_searchable_token&project=${project1.id}`,
        { method: "GET" }
      );
      const response = await handleAPI(req, "/api/search");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.results.length).toBeGreaterThanOrEqual(1);
      expect(data.results.every((r: { memory: { projectId: string } }) => r.memory.projectId === project1.id)).toBe(true);
    });
  });

  describe("GET /api/health", () => {
    test("returns ok status", async () => {
      const req = new Request("http://localhost/api/health", { method: "GET" });
      const response = await handleAPI(req, "/api/health");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.ok).toBe(true);
    });
  });

  describe("GET /api/stats", () => {
    test("returns memory statistics", async () => {
      const req = new Request("http://localhost/api/stats", { method: "GET" });
      const response = await handleAPI(req, "/api/stats");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.totals).toBeDefined();
      expect(typeof data.totals.memories).toBe("number");
      expect(typeof data.totals.projectMemories).toBe("number");
      expect(typeof data.totals.documents).toBe("number");
      expect(typeof data.totals.sessions).toBe("number");
      expect(data.bySector).toBeDefined();
    });
  });

  describe("GET /api/projects", () => {
    test("returns list of projects with memory counts", async () => {
      const req = new Request("http://localhost/api/projects", { method: "GET" });
      const response = await handleAPI(req, "/api/projects");

      expect(response.status).toBe(200);
      const data = await response.json();
      expect(Array.isArray(data.projects)).toBe(true);
    });
  });

  describe("404 handling", () => {
    test("returns 404 for unknown routes", async () => {
      const req = new Request("http://localhost/api/unknown", { method: "GET" });
      const response = await handleAPI(req, "/api/unknown");

      expect(response.status).toBe(404);
      const data = await response.json();
      expect(data.error).toBe("Not found");
    });
  });

  describe("GET /api/page-data with project filter", () => {
    test("returns recent memories for project when no query provided", async () => {
      const ts = Date.now();
      const project = await getOrCreateProject(`/test/pagedata-project-${ts}`);
      const sessionId = `pagedata-session-${ts}`;
      await getOrCreateSession(sessionId, project.id);

      const store = createMemoryStore();
      await store.create(
        { content: `First memory for pagedata test ${ts}`, sector: "semantic", tier: "project" },
        project.id,
        sessionId
      );
      await store.create(
        { content: `Second memory for pagedata test ${ts}`, sector: "procedural", tier: "project" },
        project.id,
        sessionId
      );

      const req = new Request(
        `http://localhost/api/page-data?path=${encodeURIComponent(`/search?project=${project.id}`)}`,
        { method: "GET" }
      );
      const response = await handleAPI(req, "/api/page-data");

      expect(response.status).toBe(200);
      const data = (await response.json()) as {
        type: string;
        results: { memory: { id: string; projectId: string } }[];
        projectId: string;
      };
      expect(data.type).toBe("search");
      expect(data.projectId).toBe(project.id);
      expect(data.results.length).toBeGreaterThanOrEqual(2);
      expect(data.results.every((r) => r.memory.projectId === project.id)).toBe(true);
    });

    test("returns empty results when no project filter and no query", async () => {
      const req = new Request(
        `http://localhost/api/page-data?path=${encodeURIComponent("/search")}`,
        { method: "GET" }
      );
      const response = await handleAPI(req, "/api/page-data");

      expect(response.status).toBe(200);
      const data = (await response.json()) as {
        type: string;
        results: unknown[];
      };
      expect(data.type).toBe("search");
      expect(data.results.length).toBe(0);
    });
  });
});
