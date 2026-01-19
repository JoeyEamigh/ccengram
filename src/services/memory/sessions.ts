import { getDatabase } from "../../db/database.js";
import { log } from "../../utils/log.js";
import type { Memory, UsageType } from "./types.js";
import { rowToMemory } from "./utils.js";

export type Session = {
  id: string;
  projectId: string;
  startedAt: number;
  endedAt?: number;
  summary?: string;
  userPrompt?: string;
  context: Record<string, unknown>;
};

export type SessionInput = {
  projectId: string;
  userPrompt?: string;
  context?: Record<string, unknown>;
};

export type SessionStats = {
  memoriesCreated: number;
  memoriesRecalled: number;
  memoriesUpdated: number;
  memoriesReinforced: number;
  totalMemories: number;
};

export type SessionService = {
  create(input: SessionInput): Promise<Session>;
  get(id: string): Promise<Session | null>;
  end(id: string, summary?: string): Promise<Session>;
  getStats(id: string): Promise<SessionStats>;
  getSessionMemories(id: string): Promise<Memory[]>;
  promoteSessionMemories(id: string, minUsageCount?: number): Promise<number>;
  getActiveSession(projectId: string): Promise<Session | null>;
};

function rowToSession(row: Record<string, unknown>): Session {
  return {
    id: String(row["id"]),
    projectId: String(row["project_id"]),
    startedAt: Number(row["started_at"]),
    endedAt: row["ended_at"] ? Number(row["ended_at"]) : undefined,
    summary: row["summary"] ? String(row["summary"]) : undefined,
    userPrompt: row["user_prompt"] ? String(row["user_prompt"]) : undefined,
    context: parseJsonObject(row["context_json"]),
  };
}

function parseJsonObject(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (typeof parsed === "object" && parsed !== null) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
  }
  return {};
}

export async function getOrCreateSession(
  sessionId: string,
  projectId: string
): Promise<Session> {
  const db = await getDatabase();

  const existing = await db.execute("SELECT * FROM sessions WHERE id = ?", [
    sessionId,
  ]);

  if (existing.rows.length > 0 && existing.rows[0]) {
    return rowToSession(existing.rows[0]);
  }

  const now = Date.now();
  await db.execute(
    `INSERT INTO sessions (id, project_id, started_at, context_json)
     VALUES (?, ?, ?, ?)`,
    [sessionId, projectId, now, "{}"]
  );

  log.info("session", "Created session", { id: sessionId, projectId });

  return {
    id: sessionId,
    projectId,
    startedAt: now,
    context: {},
  };
}

export function createSessionService(): SessionService {
  const service: SessionService = {
    async create(input: SessionInput): Promise<Session> {
      const db = await getDatabase();
      const id = crypto.randomUUID();
      const now = Date.now();

      log.debug("session", "Creating session", {
        projectId: input.projectId,
      });

      await db.execute(
        `INSERT INTO sessions (id, project_id, started_at, user_prompt, context_json)
         VALUES (?, ?, ?, ?, ?)`,
        [
          id,
          input.projectId,
          now,
          input.userPrompt ?? null,
          JSON.stringify(input.context ?? {}),
        ]
      );

      log.info("session", "Session created", { id, projectId: input.projectId });

      const created = await service.get(id);
      if (!created) {
        throw new Error("Failed to get created session");
      }
      return created;
    },

    async get(id: string): Promise<Session | null> {
      const db = await getDatabase();

      const result = await db.execute("SELECT * FROM sessions WHERE id = ?", [id]);

      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;
      return rowToSession(row);
    },

    async end(id: string, summary?: string): Promise<Session> {
      const db = await getDatabase();
      const now = Date.now();

      log.info("session", "Ending session", { id, hasSummary: !!summary });

      if (summary) {
        await db.execute(
          `UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?`,
          [now, summary, id]
        );
      } else {
        await db.execute(
          `UPDATE sessions SET ended_at = ? WHERE id = ?`,
          [now, id]
        );
      }

      const ended = await service.get(id);
      if (!ended) {
        throw new Error("Failed to get ended session");
      }
      return ended;
    },

    async getStats(id: string): Promise<SessionStats> {
      const db = await getDatabase();

      const result = await db.execute(
        `SELECT usage_type, COUNT(*) as count
         FROM session_memories
         WHERE session_id = ?
         GROUP BY usage_type`,
        [id]
      );

      const stats: SessionStats = {
        memoriesCreated: 0,
        memoriesRecalled: 0,
        memoriesUpdated: 0,
        memoriesReinforced: 0,
        totalMemories: 0,
      };

      for (const row of result.rows) {
        const usageType = String(row["usage_type"]) as UsageType;
        const count = Number(row["count"]);

        switch (usageType) {
          case "created":
            stats.memoriesCreated = count;
            break;
          case "recalled":
            stats.memoriesRecalled = count;
            break;
          case "updated":
            stats.memoriesUpdated = count;
            break;
          case "reinforced":
            stats.memoriesReinforced = count;
            break;
        }
      }

      stats.totalMemories =
        stats.memoriesCreated +
        stats.memoriesRecalled +
        stats.memoriesUpdated +
        stats.memoriesReinforced;

      return stats;
    },

    async getSessionMemories(id: string): Promise<Memory[]> {
      const db = await getDatabase();

      const result = await db.execute(
        `SELECT DISTINCT m.* FROM memories m
         JOIN session_memories sm ON sm.memory_id = m.id
         WHERE sm.session_id = ?
         ORDER BY sm.created_at DESC`,
        [id]
      );

      return result.rows.map(rowToMemory);
    },

    async promoteSessionMemories(id: string, minUsageCount = 2): Promise<number> {
      const db = await getDatabase();
      const now = Date.now();

      log.info("session", "Promoting session memories", { id, minUsageCount });

      const result = await db.execute(
        `SELECT memory_id, COUNT(*) as usage_count
         FROM session_memories
         WHERE session_id = ?
         GROUP BY memory_id
         HAVING COUNT(*) >= ?`,
        [id, minUsageCount]
      );

      const memoryIds = result.rows.map((row) => String(row["memory_id"]));

      if (memoryIds.length === 0) {
        return 0;
      }

      const placeholders = memoryIds.map(() => "?").join(", ");
      await db.execute(
        `UPDATE memories
         SET tier = 'project', updated_at = ?
         WHERE id IN (${placeholders}) AND tier = 'session'`,
        [now, ...memoryIds]
      );

      log.info("session", "Promoted memories", { count: memoryIds.length });

      return memoryIds.length;
    },

    async getActiveSession(projectId: string): Promise<Session | null> {
      const db = await getDatabase();

      const result = await db.execute(
        `SELECT * FROM sessions
         WHERE project_id = ? AND ended_at IS NULL
         ORDER BY started_at DESC
         LIMIT 1`,
        [projectId]
      );

      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;
      return rowToSession(row);
    },
  };

  return service;
}
