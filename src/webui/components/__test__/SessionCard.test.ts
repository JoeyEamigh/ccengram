import { describe, expect, test } from "bun:test";
import type { Memory } from "../../../services/memory/types.js";

type SessionCardMemory = {
  id: string;
  content: string;
  summary?: string;
  sector: string;
  salience: number;
  createdAt: number;
};

type WebSocketMessage = {
  type: string;
  memory?: Memory;
  sessionId?: string;
  projectId?: string;
};

function filterMessagesForSession(
  messages: WebSocketMessage[],
  sessionId: string
): WebSocketMessage[] {
  return messages.filter(
    (m) => m.type === "memory:created" && m.sessionId === sessionId
  );
}

function convertToSessionCardMemory(memory: Memory): SessionCardMemory {
  return {
    id: memory.id,
    content: memory.content,
    summary: memory.summary,
    sector: memory.sector,
    salience: memory.salience,
    createdAt: memory.createdAt,
  };
}

describe("SessionCard real-time updates", () => {
  test("filterMessagesForSession returns only messages for the session", () => {
    const messages: WebSocketMessage[] = [
      {
        type: "memory:created",
        sessionId: "sess1",
        memory: {
          id: "mem1",
          content: "Test 1",
          sector: "episodic",
          salience: 1.0,
          tier: "session",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          projectId: "proj1",
          isDeleted: false,
          accessCount: 0,
          lastAccessed: Date.now(),
          importance: 0.5,
          tags: [],
          files: [],
          concepts: [],
          contentHash: "abc",
          simhash: "def",
        } as Memory,
      },
      {
        type: "memory:created",
        sessionId: "sess2",
        memory: {
          id: "mem2",
          content: "Test 2",
          sector: "episodic",
          salience: 1.0,
          tier: "session",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          projectId: "proj1",
          isDeleted: false,
          accessCount: 0,
          lastAccessed: Date.now(),
          importance: 0.5,
          tags: [],
          files: [],
          concepts: [],
          contentHash: "ghi",
          simhash: "jkl",
        } as Memory,
      },
      {
        type: "memory:updated",
        sessionId: "sess1",
        memory: {
          id: "mem3",
          content: "Updated",
          sector: "episodic",
          salience: 0.8,
          tier: "session",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          projectId: "proj1",
          isDeleted: false,
          accessCount: 0,
          lastAccessed: Date.now(),
          importance: 0.5,
          tags: [],
          files: [],
          concepts: [],
          contentHash: "mno",
          simhash: "pqr",
        } as Memory,
      },
    ];

    const sess1Messages = filterMessagesForSession(messages, "sess1");

    expect(sess1Messages).toHaveLength(1);
    expect(sess1Messages[0]?.memory?.id).toBe("mem1");
  });

  test("convertToSessionCardMemory extracts correct fields", () => {
    const memory: Memory = {
      id: "mem1",
      content: "Full content here",
      summary: "Short summary",
      sector: "procedural",
      salience: 0.75,
      tier: "project",
      createdAt: 1700000000000,
      updatedAt: 1700000001000,
      projectId: "proj1",
      isDeleted: false,
      accessCount: 5,
      lastAccessed: 1700000002000,
      importance: 0.8,
      tags: ["tag1"],
      files: ["file.ts"],
      concepts: ["concept1"],
      contentHash: "abc",
      simhash: "def",
    };

    const cardMemory = convertToSessionCardMemory(memory);

    expect(cardMemory).toEqual({
      id: "mem1",
      content: "Full content here",
      summary: "Short summary",
      sector: "procedural",
      salience: 0.75,
      createdAt: 1700000000000,
    });
  });

  test("new memories are prepended to existing list", () => {
    const existing: SessionCardMemory[] = [
      { id: "old1", content: "Old 1", sector: "episodic", salience: 1.0, createdAt: 1000 },
      { id: "old2", content: "Old 2", sector: "semantic", salience: 0.9, createdAt: 900 },
    ];

    const newMemory: SessionCardMemory = {
      id: "new1",
      content: "New memory",
      sector: "episodic",
      salience: 1.0,
      createdAt: 2000,
    };

    const updated = [newMemory, ...existing].slice(0, 5);

    expect(updated).toHaveLength(3);
    expect(updated[0]?.id).toBe("new1");
    expect(updated[1]?.id).toBe("old1");
    expect(updated[2]?.id).toBe("old2");
  });

  test("duplicate memories are not added", () => {
    const existing: SessionCardMemory[] = [
      { id: "mem1", content: "Original", sector: "episodic", salience: 1.0, createdAt: 1000 },
    ];

    const newMemory: SessionCardMemory = {
      id: "mem1",
      content: "Duplicate",
      sector: "episodic",
      salience: 1.0,
      createdAt: 1000,
    };

    const existingIds = new Set(existing.map((m) => m.id));
    const updated = existingIds.has(newMemory.id)
      ? existing
      : [newMemory, ...existing].slice(0, 5);

    expect(updated).toHaveLength(1);
    expect(updated[0]?.content).toBe("Original");
  });

  test("memory count increments when new memory received", () => {
    let memoryCount = 5;

    const messageForThisSession: WebSocketMessage = {
      type: "memory:created",
      sessionId: "sess1",
      memory: {
        id: "new1",
        content: "New",
        sector: "episodic",
        salience: 1.0,
        tier: "session",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        projectId: "proj1",
        isDeleted: false,
        accessCount: 0,
        lastAccessed: Date.now(),
        importance: 0.5,
        tags: [],
        files: [],
        concepts: [],
        contentHash: "abc",
        simhash: "def",
      } as Memory,
    };

    if (messageForThisSession.type === "memory:created" && messageForThisSession.sessionId === "sess1") {
      memoryCount += 1;
    }

    expect(memoryCount).toBe(6);
  });
});
