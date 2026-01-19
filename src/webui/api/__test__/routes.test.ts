import { describe, expect, test, beforeEach, afterEach, mock } from "bun:test";
import {
  createDatabase,
  closeDatabase,
  setDatabase,
  type Database,
} from "../../../db/database.js";
import { handleAPI } from "../routes.js";
import { setServer, broadcastToRoom } from "../../ws/handler.js";

describe("API Routes", () => {
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
      `INSERT INTO sessions (id, project_id, started_at) VALUES (?, ?, ?)`,
      ["sess1", "proj1", now]
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  describe("/api/hooks/memory-created webhook", () => {
    test("broadcasts memory:created event with sessionId", async () => {
      const now = Date.now();
      await db.execute(
        `INSERT INTO memories (id, content, sector, tier, salience, project_id, created_at, updated_at, last_accessed, is_deleted, access_count)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ["mem1", "Test content", "episodic", "session", 1.0, "proj1", now, now, now, 0, 0]
      );
      await db.execute(
        `INSERT INTO session_memories (session_id, memory_id, usage_type, created_at) VALUES (?, ?, ?, ?)`,
        ["sess1", "mem1", "created", now]
      );

      const broadcasts: { room: string; message: unknown }[] = [];
      const mockServer = {
        publish: (room: string, data: string) => {
          broadcasts.push({ room, message: JSON.parse(data) });
        },
      };
      setServer(mockServer);

      const req = new Request("http://localhost/api/hooks/memory-created", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          memoryId: "mem1",
          projectId: "proj1",
          sessionId: "sess1",
        }),
      });

      const res = await handleAPI(req, "/api/hooks/memory-created");
      const data = await res.json();

      expect(res.status).toBe(200);
      expect(data).toEqual({ ok: true });

      expect(broadcasts).toHaveLength(2);

      const projectBroadcast = broadcasts.find((b) => b.room === "proj1");
      expect(projectBroadcast).toBeDefined();
      const projectMsg = projectBroadcast!.message as Record<string, unknown>;
      expect(projectMsg.type).toBe("memory:created");
      expect(projectMsg.sessionId).toBe("sess1");
      expect((projectMsg.memory as Record<string, unknown>).id).toBe("mem1");

      const globalBroadcast = broadcasts.find((b) => b.room === "global");
      expect(globalBroadcast).toBeDefined();
      const globalMsg = globalBroadcast!.message as Record<string, unknown>;
      expect(globalMsg.type).toBe("memory:created");
      expect(globalMsg.sessionId).toBe("sess1");
      expect(globalMsg.projectId).toBe("proj1");
    });

    test("returns 400 if memoryId missing", async () => {
      const req = new Request("http://localhost/api/hooks/memory-created", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId: "proj1",
        }),
      });

      const res = await handleAPI(req, "/api/hooks/memory-created");
      expect(res.status).toBe(400);
    });

    test("returns 404 if memory not found", async () => {
      const req = new Request("http://localhost/api/hooks/memory-created", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          memoryId: "non-existent",
          projectId: "proj1",
        }),
      });

      const res = await handleAPI(req, "/api/hooks/memory-created");
      expect(res.status).toBe(404);
    });

    test("works without sessionId for MCP tool calls", async () => {
      const now = Date.now();
      await db.execute(
        `INSERT INTO memories (id, content, sector, tier, salience, project_id, created_at, updated_at, last_accessed, is_deleted, access_count)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ["mem2", "Via MCP", "semantic", "project", 1.0, "proj1", now, now, now, 0, 0]
      );

      const broadcasts: { room: string; message: unknown }[] = [];
      const mockServer = {
        publish: (room: string, data: string) => {
          broadcasts.push({ room, message: JSON.parse(data) });
        },
      };
      setServer(mockServer);

      const req = new Request("http://localhost/api/hooks/memory-created", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          memoryId: "mem2",
          projectId: "proj1",
        }),
      });

      const res = await handleAPI(req, "/api/hooks/memory-created");
      expect(res.status).toBe(200);

      expect(broadcasts).toHaveLength(2);
      const globalMsg = broadcasts.find((b) => b.room === "global")?.message as Record<string, unknown>;
      expect(globalMsg.sessionId).toBeUndefined();
    });
  });
});
