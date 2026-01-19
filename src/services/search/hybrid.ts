import { getDatabase } from "../../db/database.js";
import { log } from "../../utils/log.js";
import type { Memory, MemorySector, MemoryTier, UsageType } from "../memory/types.js";
import { rowToMemory } from "../memory/utils.js";
import { getSupersedingMemory } from "../memory/relationships.js";
import type { EmbeddingService } from "../embedding/types.js";
import { searchFTS } from "./fts.js";
import { searchVector } from "./vector.js";
import { computeScore, type RankingWeights, DEFAULT_WEIGHTS } from "./ranking.js";

export type SearchMode = "hybrid" | "semantic" | "keyword";

export type SearchOptions = {
  query: string;
  projectId?: string;
  sector?: MemorySector;
  tier?: MemoryTier;
  limit?: number;
  minSalience?: number;
  includeDocuments?: boolean;
  includeSuperseded?: boolean;
  sessionId?: string;
  mode?: SearchMode;
  weights?: RankingWeights;
};

export type SessionSummary = {
  id: string;
  startedAt: number;
  summary?: string;
  projectId: string;
};

export type SearchResult = {
  memory: Memory;
  score: number;
  matchType: "semantic" | "keyword" | "both";
  highlights?: string[];
  sourceSession?: SessionSummary;
  isSuperseded: boolean;
  supersededBy?: {
    id: string;
    content: string;
    createdAt: number;
  };
  relatedMemoryCount: number;
};

export type SessionContext = {
  session: {
    id: string;
    startedAt: number;
    endedAt?: number;
    summary?: string;
    projectId: string;
  };
  memoriesInSession: number;
  usageType: UsageType;
};

export type TimelineResult = {
  anchor: Memory;
  before: Memory[];
  after: Memory[];
  sessions: Map<string, SessionSummary>;
};

export type SearchService = {
  search(options: SearchOptions): Promise<SearchResult[]>;
  timeline(anchorId: string, depthBefore?: number, depthAfter?: number): Promise<TimelineResult>;
  getSessionContext(memoryId: string): Promise<SessionContext | null>;
};

async function getMemoryById(id: string): Promise<Memory | null> {
  const db = await getDatabase();
  const result = await db.execute("SELECT * FROM memories WHERE id = ?", [id]);
  if (result.rows.length === 0) return null;
  const row = result.rows[0];
  if (!row) return null;
  return rowToMemory(row);
}

async function getSourceSession(memoryId: string): Promise<SessionSummary | undefined> {
  const db = await getDatabase();
  const result = await db.execute(
    `SELECT s.id, s.started_at, s.summary, s.project_id
     FROM session_memories sm
     JOIN sessions s ON sm.session_id = s.id
     WHERE sm.memory_id = ? AND sm.usage_type = 'created'
     LIMIT 1`,
    [memoryId]
  );

  if (result.rows.length === 0) return undefined;

  const row = result.rows[0];
  if (!row) return undefined;

  return {
    id: String(row["id"]),
    startedAt: Number(row["started_at"]),
    summary: row["summary"] ? String(row["summary"]) : undefined,
    projectId: String(row["project_id"]),
  };
}

async function getRelatedMemoryCount(memoryId: string): Promise<number> {
  const db = await getDatabase();
  const result = await db.execute(
    `SELECT COUNT(*) as count FROM memory_relationships
     WHERE (source_memory_id = ? OR target_memory_id = ?)
       AND valid_until IS NULL`,
    [memoryId, memoryId]
  );

  const row = result.rows[0];
  return row ? Number(row["count"]) : 0;
}

async function batchGetSourceSessions(
  memoryIds: string[]
): Promise<Map<string, SessionSummary>> {
  if (memoryIds.length === 0) return new Map();

  const db = await getDatabase();
  const placeholders = memoryIds.map(() => "?").join(",");
  const result = await db.execute(
    `SELECT sm.memory_id, s.id, s.started_at, s.summary, s.project_id
     FROM session_memories sm
     JOIN sessions s ON sm.session_id = s.id
     WHERE sm.memory_id IN (${placeholders}) AND sm.usage_type = 'created'`,
    memoryIds
  );

  const map = new Map<string, SessionSummary>();
  for (const row of result.rows) {
    const memoryId = String(row["memory_id"]);
    map.set(memoryId, {
      id: String(row["id"]),
      startedAt: Number(row["started_at"]),
      summary: row["summary"] ? String(row["summary"]) : undefined,
      projectId: String(row["project_id"]),
    });
  }
  return map;
}

async function batchGetSupersedingMemories(
  memoryIds: string[]
): Promise<Map<string, { id: string; content: string; createdAt: number }>> {
  if (memoryIds.length === 0) return new Map();

  const db = await getDatabase();
  const placeholders = memoryIds.map(() => "?").join(",");
  const result = await db.execute(
    `SELECT mr.target_memory_id, m.id, m.content, m.created_at
     FROM memory_relationships mr
     JOIN memories m ON mr.source_memory_id = m.id
     WHERE mr.target_memory_id IN (${placeholders})
       AND mr.relationship_type = 'SUPERSEDES'
       AND mr.valid_until IS NULL
       AND m.is_deleted = 0`,
    memoryIds
  );

  const map = new Map<string, { id: string; content: string; createdAt: number }>();
  for (const row of result.rows) {
    const targetId = String(row["target_memory_id"]);
    map.set(targetId, {
      id: String(row["id"]),
      content: String(row["content"]).slice(0, 200),
      createdAt: Number(row["created_at"]),
    });
  }
  return map;
}

async function batchGetRelatedCounts(
  memoryIds: string[]
): Promise<Map<string, number>> {
  if (memoryIds.length === 0) return new Map();

  const db = await getDatabase();
  const placeholders = memoryIds.map(() => "?").join(",");
  const allIds = [...memoryIds, ...memoryIds];

  const result = await db.execute(
    `SELECT memory_id, COUNT(*) as count FROM (
       SELECT source_memory_id as memory_id FROM memory_relationships
       WHERE source_memory_id IN (${placeholders}) AND valid_until IS NULL
       UNION ALL
       SELECT target_memory_id as memory_id FROM memory_relationships
       WHERE target_memory_id IN (${placeholders}) AND valid_until IS NULL
     ) GROUP BY memory_id`,
    allIds
  );

  const map = new Map<string, number>();
  for (const row of result.rows) {
    map.set(String(row["memory_id"]), Number(row["count"]));
  }
  return map;
}

async function checkSessionLink(memoryId: string, sessionId: string): Promise<boolean> {
  const db = await getDatabase();
  const result = await db.execute(
    `SELECT 1 FROM session_memories
     WHERE memory_id = ? AND session_id = ?
     LIMIT 1`,
    [memoryId, sessionId]
  );
  return result.rows.length > 0;
}

async function reinforceMemory(id: string, amount: number): Promise<void> {
  const db = await getDatabase();
  const now = Date.now();
  await db.execute(
    `UPDATE memories
     SET salience = MIN(1.0, salience + ? * (1.0 - salience)),
         last_accessed = ?,
         access_count = access_count + 1,
         updated_at = ?
     WHERE id = ? AND is_deleted = 0`,
    [amount, now, now, id]
  );
}

async function linkToSession(
  memoryId: string,
  sessionId: string,
  usageType: UsageType
): Promise<void> {
  const db = await getDatabase();
  const now = Date.now();
  try {
    await db.execute(
      `INSERT INTO session_memories (session_id, memory_id, created_at, usage_type)
       VALUES (?, ?, ?, ?)`,
      [sessionId, memoryId, now, usageType]
    );
  } catch {
    // Ignore duplicate key errors
  }
}

export function createSearchService(
  embeddingService: EmbeddingService | null
): SearchService {
  const service: SearchService = {
    async search(options: SearchOptions): Promise<SearchResult[]> {
      const {
        query,
        projectId,
        sector,
        tier,
        limit = 10,
        minSalience = 0,
        includeSuperseded = false,
        sessionId,
        weights = DEFAULT_WEIGHTS,
      } = options;

      let { mode = "hybrid" } = options;

      if (!embeddingService && mode !== "keyword") {
        log.warn("search", "No embedding service available, falling back to keyword search", {
          requestedMode: mode,
        });
        mode = "keyword";
      }

      const start = Date.now();
      log.info("search", "Hybrid search starting", {
        query: query.slice(0, 50),
        mode,
        projectId,
        degraded: !embeddingService,
      });

      const [ftsResults, vectorResults] = await Promise.all([
        mode !== "semantic" ? searchFTS(query, projectId, limit * 2) : [],
        mode !== "keyword" && embeddingService
          ? searchVector(query, embeddingService, projectId, limit * 2)
          : [],
      ]);

      const resultMap = new Map<
        string,
        {
          ftsRank: number;
          similarity: number;
          snippet?: string;
        }
      >();

      for (const r of ftsResults) {
        resultMap.set(r.memoryId, {
          ftsRank: r.rank,
          similarity: 0,
          snippet: r.snippet,
        });
      }

      for (const r of vectorResults) {
        const existing = resultMap.get(r.memoryId);
        if (existing) {
          existing.similarity = r.similarity;
        } else {
          resultMap.set(r.memoryId, {
            ftsRank: 0,
            similarity: r.similarity,
          });
        }
      }

      const memoryIds = Array.from(resultMap.keys());
      const memories = await Promise.all(memoryIds.map(getMemoryById));

      type CandidateResult = {
        memory: Memory;
        data: { ftsRank: number; similarity: number; snippet?: string };
        score: number;
        matchType: "semantic" | "keyword" | "both";
      };
      const candidates: CandidateResult[] = [];

      for (let i = 0; i < memories.length; i++) {
        const memory = memories[i];
        const memoryId = memoryIds[i];
        if (!memory || memory.isDeleted || !memoryId) continue;

        if (sector && memory.sector !== sector) continue;
        if (tier && memory.tier !== tier) continue;
        if (memory.salience < minSalience) continue;
        if (!includeSuperseded && memory.validUntil) continue;

        if (sessionId) {
          const hasLink = await checkSessionLink(memory.id, sessionId);
          if (!hasLink) continue;
        }

        const data = resultMap.get(memory.id);
        if (!data) continue;

        const score = computeScore(
          memory,
          data.similarity,
          data.ftsRank,
          weights
        );

        const matchType: "semantic" | "keyword" | "both" =
          data.similarity > 0 && data.ftsRank !== 0
            ? "both"
            : data.similarity > 0
              ? "semantic"
              : "keyword";

        candidates.push({ memory, data, score, matchType });
      }

      const candidateIds = candidates.map((c) => c.memory.id);
      const [sessionMap, supersededMap, relatedCountMap] = await Promise.all([
        batchGetSourceSessions(candidateIds),
        batchGetSupersedingMemories(candidateIds),
        batchGetRelatedCounts(candidateIds),
      ]);

      const results: SearchResult[] = candidates.map((c) => {
        const supersedingMemory = supersededMap.get(c.memory.id);
        return {
          memory: c.memory,
          score: c.score,
          matchType: c.matchType,
          highlights: c.data.snippet ? [c.data.snippet] : undefined,
          sourceSession: sessionMap.get(c.memory.id),
          isSuperseded: !!c.memory.validUntil,
          supersededBy: supersedingMemory,
          relatedMemoryCount: relatedCountMap.get(c.memory.id) ?? 0,
        };
      });

      results.sort((a, b) => b.score - a.score);

      const topResults = results.slice(0, limit);

      for (const result of topResults) {
        await reinforceMemory(result.memory.id, 0.02);
        if (sessionId) {
          await linkToSession(result.memory.id, sessionId, "recalled");
        }
      }

      log.info("search", "Hybrid search complete", {
        total: results.length,
        returned: topResults.length,
        mode,
        ms: Date.now() - start,
      });

      return topResults;
    },

    async timeline(
      anchorId: string,
      depthBefore = 5,
      depthAfter = 5
    ): Promise<TimelineResult> {
      log.debug("search", "Timeline query", { anchorId, depthBefore, depthAfter });

      const db = await getDatabase();
      const anchor = await getMemoryById(anchorId);

      if (!anchor) {
        log.warn("search", "Timeline anchor not found", { anchorId });
        throw new Error("Anchor memory not found");
      }

      const [beforeResult, afterResult] = await Promise.all([
        db.execute(
          `SELECT * FROM memories
           WHERE project_id = ? AND created_at < ? AND is_deleted = 0
           ORDER BY created_at DESC
           LIMIT ?`,
          [anchor.projectId, anchor.createdAt, depthBefore]
        ),
        db.execute(
          `SELECT * FROM memories
           WHERE project_id = ? AND created_at > ? AND is_deleted = 0
           ORDER BY created_at ASC
           LIMIT ?`,
          [anchor.projectId, anchor.createdAt, depthAfter]
        ),
      ]);

      const before = beforeResult.rows.map(rowToMemory).reverse();
      const after = afterResult.rows.map(rowToMemory);

      const allMemories = [...before, anchor, ...after];
      const sessions = new Map<string, SessionSummary>();

      for (const memory of allMemories) {
        const sessionContext = await service.getSessionContext(memory.id);
        if (sessionContext && !sessions.has(sessionContext.session.id)) {
          sessions.set(sessionContext.session.id, {
            id: sessionContext.session.id,
            startedAt: sessionContext.session.startedAt,
            summary: sessionContext.session.summary,
            projectId: sessionContext.session.projectId,
          });
        }
      }

      return { anchor, before, after, sessions };
    },

    async getSessionContext(memoryId: string): Promise<SessionContext | null> {
      const db = await getDatabase();
      const result = await db.execute(
        `SELECT s.*, sm.usage_type,
                (SELECT COUNT(*) FROM session_memories WHERE session_id = s.id) as memory_count
         FROM session_memories sm
         JOIN sessions s ON sm.session_id = s.id
         WHERE sm.memory_id = ?
         ORDER BY sm.created_at DESC
         LIMIT 1`,
        [memoryId]
      );

      if (result.rows.length === 0) return null;

      const row = result.rows[0];
      if (!row) return null;

      return {
        session: {
          id: String(row["id"]),
          startedAt: Number(row["started_at"]),
          endedAt: row["ended_at"] ? Number(row["ended_at"]) : undefined,
          summary: row["summary"] ? String(row["summary"]) : undefined,
          projectId: String(row["project_id"]),
        },
        memoriesInSession: Number(row["memory_count"]),
        usageType: String(row["usage_type"]) as UsageType,
      };
    },
  };

  return service;
}
