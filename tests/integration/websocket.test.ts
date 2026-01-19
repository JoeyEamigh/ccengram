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
import { startServer, type ServerResult } from "../../src/webui/server.js";

describe("Multi-Agent WebSocket Integration", () => {
  const testDir = `/tmp/ccmemory-websocket-integration-${Date.now()}`;
  let db: Database;
  let serverResult: ServerResult;
  const port = 37780 + Math.floor(Math.random() * 1000);

  beforeAll(async () => {
    await mkdir(testDir, { recursive: true });
    process.env["CCMEMORY_DATA_DIR"] = testDir;
    process.env["CCMEMORY_CONFIG_DIR"] = testDir;
    process.env["CCMEMORY_CACHE_DIR"] = testDir;

    db = await createDatabase(join(testDir, "test.db"));
    setDatabase(db);

    serverResult = await startServer({
      port,
      sessionId: `test-session-${Date.now()}`,
    });
  });

  afterAll(async () => {
    if (serverResult.server) {
      serverResult.server.stop();
    }
    if (serverResult.checkInterval) {
      clearInterval(serverResult.checkInterval);
    }
    closeDatabase();
    await rm(testDir, { recursive: true, force: true });
    delete process.env["CCMEMORY_DATA_DIR"];
    delete process.env["CCMEMORY_CONFIG_DIR"];
    delete process.env["CCMEMORY_CACHE_DIR"];
  });

  test("server starts and serves health endpoint", async () => {
    const response = await fetch(`http://localhost:${port}/api/health`);
    expect(response.ok).toBe(true);

    const data = (await response.json()) as { ok: boolean };
    expect(data.ok).toBe(true);
  });

  test("WebSocket connection can be established", async () => {
    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    const connected = await new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), 5000);
      ws.onopen = () => {
        clearTimeout(timeout);
        resolve(true);
      };
      ws.onerror = () => {
        clearTimeout(timeout);
        resolve(false);
      };
    });

    expect(connected).toBe(true);
    ws.close();
  });

  test("WebSocket responds to ping with pong", async () => {
    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const pong = await new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string };
        if (data.type === "pong") {
          clearTimeout(timeout);
          resolve(true);
        }
      };
      ws.send(JSON.stringify({ type: "ping" }));
    });

    expect(pong).toBe(true);
    ws.close();
  });

  test("WebSocket can subscribe to project updates", async () => {
    const projectPath = "/test/ws-project";
    const project = await getOrCreateProject(projectPath);

    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const subscribed = await new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string; room?: string };
        if (data.type === "subscribed" && data.room === project.id) {
          clearTimeout(timeout);
          resolve(true);
        }
      };
      ws.send(JSON.stringify({ type: "subscribe:project", projectId: project.id }));
    });

    expect(subscribed).toBe(true);
    ws.close();
  });

  test("WebSocket can reinforce memory", async () => {
    const projectPath = "/test/ws-reinforce";
    const sessionId = `ws-session-${Date.now()}`;
    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();
    const memory = await store.create(
      {
        content: "Test memory for WebSocket reinforcement",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    await store.deemphasize(memory.id, 0.5);
    const lowSalience = await store.get(memory.id);
    expect(lowSalience?.salience).toBe(0.5);

    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const reinforced = await new Promise<{ memory: { salience: number } } | null>((resolve) => {
      const timeout = setTimeout(() => resolve(null), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string; memory?: { salience: number } };
        if (data.type === "memory:updated" && data.memory) {
          clearTimeout(timeout);
          resolve(data as { memory: { salience: number } });
        }
      };
      ws.send(JSON.stringify({ type: "memory:reinforce", memoryId: memory.id, amount: 0.3 }));
    });

    expect(reinforced).not.toBeNull();
    expect(reinforced?.memory.salience).toBeGreaterThan(0.5);
    ws.close();
  });

  test("WebSocket can deemphasize memory", async () => {
    const projectPath = "/test/ws-deemphasize";
    const sessionId = `ws-deemph-session-${Date.now()}`;
    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();
    const memory = await store.create(
      {
        content: "Test memory for WebSocket de-emphasis",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    expect(memory.salience).toBe(1.0);

    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const deemphasized = await new Promise<{ memory: { salience: number } } | null>((resolve) => {
      const timeout = setTimeout(() => resolve(null), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string; memory?: { salience: number } };
        if (data.type === "memory:updated" && data.memory) {
          clearTimeout(timeout);
          resolve(data as { memory: { salience: number } });
        }
      };
      ws.send(JSON.stringify({ type: "memory:deemphasize", memoryId: memory.id, amount: 0.5 }));
    });

    expect(deemphasized).not.toBeNull();
    expect(deemphasized?.memory.salience).toBeLessThan(1.0);
    ws.close();
  });

  test("WebSocket can delete memory", async () => {
    const projectPath = "/test/ws-delete";
    const sessionId = `ws-delete-session-${Date.now()}`;
    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();
    const memory = await store.create(
      {
        content: "Test memory for WebSocket deletion",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const deleted = await new Promise<{ memoryId: string } | null>((resolve) => {
      const timeout = setTimeout(() => resolve(null), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string; memoryId?: string };
        if (data.type === "memory:deleted") {
          clearTimeout(timeout);
          resolve(data as { memoryId: string });
        }
      };
      ws.send(JSON.stringify({ type: "memory:delete", memoryId: memory.id }));
    });

    expect(deleted).not.toBeNull();
    expect(deleted?.memoryId).toBe(memory.id);

    const deletedMemory = await store.get(memory.id);
    expect(deletedMemory?.isDeleted).toBe(true);

    ws.close();
  });

  test("multiple WebSocket clients can connect simultaneously", async () => {
    const clients: WebSocket[] = [];
    const connected: boolean[] = [];

    for (let i = 0; i < 3; i++) {
      const ws = new WebSocket(`ws://localhost:${port}/ws`);
      clients.push(ws);

      const isConnected = await new Promise<boolean>((resolve) => {
        const timeout = setTimeout(() => resolve(false), 5000);
        ws.onopen = () => {
          clearTimeout(timeout);
          resolve(true);
        };
        ws.onerror = () => {
          clearTimeout(timeout);
          resolve(false);
        };
      });

      connected.push(isConnected);
    }

    expect(connected.every((c) => c)).toBe(true);
    expect(clients.every((ws) => ws.readyState === WebSocket.OPEN)).toBe(true);

    clients.forEach((ws) => ws.close());
  });

  test("WebSocket handles invalid messages gracefully", async () => {
    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const errorReceived = await new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string };
        if (data.type === "error") {
          clearTimeout(timeout);
          resolve(true);
        }
      };
      ws.send("not valid json {{{");
    });

    expect(errorReceived).toBe(true);
    ws.close();
  });

  test("WebSocket handles missing memoryId for operations", async () => {
    const ws = new WebSocket(`ws://localhost:${port}/ws`);

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const errorReceived = await new Promise<{ type: string; message: string } | null>((resolve) => {
      const timeout = setTimeout(() => resolve(null), 5000);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as { type: string; message?: string };
        if (data.type === "error") {
          clearTimeout(timeout);
          resolve(data as { type: string; message: string });
        }
      };
      ws.send(JSON.stringify({ type: "memory:reinforce" }));
    });

    expect(errorReceived).not.toBeNull();
    expect(errorReceived?.message).toContain("memoryId");
    ws.close();
  });

  test("API endpoints are accessible", async () => {
    const projectPath = "/test/api-project";
    const sessionId = `api-session-${Date.now()}`;
    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();
    await store.create(
      {
        content: "Test memory for API endpoint testing",
        sector: "semantic",
        tier: "project",
      },
      project.id,
      sessionId
    );

    const searchResponse = await fetch(
      `http://localhost:${port}/api/search?query=test&projectId=${project.id}`
    );
    expect(searchResponse.ok).toBe(true);

    const statsResponse = await fetch(`http://localhost:${port}/api/stats`);
    expect(statsResponse.ok).toBe(true);

    const sessionsResponse = await fetch(`http://localhost:${port}/api/sessions`);
    expect(sessionsResponse.ok).toBe(true);
  });
});
