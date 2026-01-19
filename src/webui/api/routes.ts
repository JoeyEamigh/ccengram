import { createSearchService } from "../../services/search/hybrid.js";
import { createMemoryStore } from "../../services/memory/store.js";
import { createEmbeddingService } from "../../services/embedding/index.js";
import { getDatabase } from "../../db/database.js";
import { broadcastToRoom } from "../ws/handler.js";
import { log } from "../../utils/log.js";
import type { MemorySector } from "../../services/memory/types.js";
import type { EmbeddingService } from "../../services/embedding/types.js";
import type { InValue } from "@libsql/client";

type JsonResponse = { [key: string]: unknown };

function json(data: JsonResponse, status = 200): Response {
  return Response.json(data, {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

let cachedEmbeddingService: EmbeddingService | null = null;

async function getEmbeddingService(): Promise<EmbeddingService> {
  if (!cachedEmbeddingService) {
    cachedEmbeddingService = await createEmbeddingService();
  }
  return cachedEmbeddingService;
}

export async function handleAPI(req: Request, path: string): Promise<Response> {
  const start = Date.now();
  log.debug("webui", "API request", { method: req.method, path });
  const url = new URL(req.url);

  try {
    if (path === "/api/health") {
      return json({ ok: true });
    }

    if (path === "/api/search" && req.method === "GET") {
      const query = url.searchParams.get("q") ?? "";
      const sector = url.searchParams.get("sector") as MemorySector | null;
      const sessionId = url.searchParams.get("session");
      const projectId = url.searchParams.get("project");
      const includeSuperseded =
        url.searchParams.get("include_superseded") === "true";
      const limit = parseInt(url.searchParams.get("limit") ?? "20");

      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const results = await search.search({
        query,
        sector: sector ?? undefined,
        sessionId: sessionId ?? undefined,
        projectId: projectId ?? undefined,
        includeSuperseded,
        limit,
        mode: "hybrid",
      });

      log.debug("webui", "API search complete", {
        query: query.slice(0, 30),
        results: results.length,
        ms: Date.now() - start,
      });

      return json({ results });
    }

    if (path.startsWith("/api/memory/") && req.method === "GET") {
      const id = path.replace("/api/memory/", "");
      const store = createMemoryStore();
      const memory = await store.get(id);
      if (!memory) {
        return json({ error: "Memory not found" }, 404);
      }
      return json({ memory });
    }

    if (path === "/api/timeline" && req.method === "GET") {
      const anchorId = url.searchParams.get("anchor");
      if (!anchorId) {
        return json({ error: "Missing anchor parameter" }, 400);
      }
      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const data = await search.timeline(anchorId, 10, 10);
      return json({ data });
    }

    if (path === "/api/timeline/browse" && req.method === "GET") {
      const projectId = url.searchParams.get("project");
      const dateStr = url.searchParams.get("date");
      const sector = url.searchParams.get("sector") as MemorySector | null;
      const limit = parseInt(url.searchParams.get("limit") ?? "50");
      const offset = parseInt(url.searchParams.get("offset") ?? "0");

      const data = await browseTimeline({
        projectId,
        dateStr,
        sector,
        limit,
        offset,
      });
      return json(data);
    }

    if (path === "/api/sessions" && req.method === "GET") {
      const projectId = url.searchParams.get("project");
      const sessions = await getRecentSessions(projectId);
      return json({ sessions });
    }

    if (path.startsWith("/api/sessions/") && path.endsWith("/memories") && req.method === "GET") {
      const sessionId = path.replace("/api/sessions/", "").replace("/memories", "");
      const limit = parseInt(url.searchParams.get("limit") ?? "10");
      const memories = await getSessionMemories(sessionId, limit);
      return json({ memories });
    }

    if (path === "/api/stats" && req.method === "GET") {
      const stats = await getStats();
      return json(stats);
    }

    if (path === "/api/projects" && req.method === "GET") {
      const projects = await getProjects();
      return json({ projects });
    }

    if (path === "/api/page-data" && req.method === "GET") {
      const pagePath = url.searchParams.get("path") ?? "/";
      const data = await fetchPageData(new URL(pagePath, req.url));
      return json(data);
    }

    if (path === "/api/config" && req.method === "GET") {
      const config = await getConfig();
      return json({ config });
    }

    if (path === "/api/config" && req.method === "PUT") {
      const body = (await req.json()) as { key: string; value: string };
      await setConfig(body.key, body.value);
      return json({ ok: true });
    }

    if (path === "/api/memories/clear" && req.method === "POST") {
      const body = (await req.json()) as { projectId?: string };
      const deleted = await clearMemories(body.projectId);
      return json({ ok: true, deleted });
    }

    if (path === "/api/hooks/memory-created" && req.method === "POST") {
      const body = (await req.json()) as {
        memoryId?: string;
        projectId?: string;
        sessionId?: string;
      };

      if (!body.memoryId) {
        return json({ error: "memoryId is required" }, 400);
      }

      const store = createMemoryStore();
      const memory = await store.get(body.memoryId);

      if (!memory) {
        return json({ error: "Memory not found" }, 404);
      }

      const projectId = body.projectId ?? memory.projectId;

      broadcastToRoom(projectId, {
        type: "memory:created",
        memory,
        sessionId: body.sessionId,
      });
      broadcastToRoom("global", {
        type: "memory:created",
        memory,
        projectId,
        sessionId: body.sessionId,
      });

      log.debug("webui", "Memory creation broadcast", {
        memoryId: body.memoryId,
        projectId,
        sessionId: body.sessionId,
      });

      return json({ ok: true });
    }

    log.warn("webui", "API route not found", { path });
    return json({ error: "Not found" }, 404);
  } catch (err) {
    log.error("webui", "API error", {
      path,
      error: err instanceof Error ? err.message : String(err),
      ms: Date.now() - start,
    });
    return json(
      { error: err instanceof Error ? err.message : String(err) },
      500
    );
  }
}

async function getRecentSessions(
  projectId?: string | null
): Promise<unknown[]> {
  const db = await getDatabase();
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  const args: InValue[] = [cutoff];
  if (projectId) args.push(projectId);

  const result = await db.execute(
    `
    SELECT
      s.*,
      COUNT(DISTINCT sm.memory_id) as memory_count,
      MAX(m.created_at) as last_activity
    FROM sessions s
    LEFT JOIN session_memories sm ON s.id = sm.session_id
    LEFT JOIN memories m ON sm.memory_id = m.id
    WHERE s.started_at > ? ${projectId ? "AND s.project_id = ?" : ""}
    GROUP BY s.id
    ORDER BY s.started_at DESC
    LIMIT 50
    `,
    args
  );

  return result.rows.map((row) => ({
    id: row.id,
    projectId: row.project_id,
    startedAt: row.started_at,
    endedAt: row.ended_at,
    summary: row.summary,
    memoryCount: row.memory_count ?? 0,
    lastActivity: row.last_activity,
  }));
}

async function getStats(): Promise<JsonResponse> {
  const db = await getDatabase();

  const counts = await db.execute(`
    SELECT
      (SELECT COUNT(*) FROM memories WHERE is_deleted = 0) as total_memories,
      (SELECT COUNT(*) FROM memories WHERE tier = 'project' AND is_deleted = 0) as project_memories,
      (SELECT COUNT(*) FROM documents) as total_documents,
      (SELECT COUNT(*) FROM sessions) as total_sessions
  `);

  const bySector = await db.execute(`
    SELECT sector, COUNT(*) as count
    FROM memories
    WHERE is_deleted = 0
    GROUP BY sector
  `);

  const totalsRow = counts.rows[0];
  return {
    totals: {
      memories: totalsRow?.["total_memories"] ?? 0,
      projectMemories: totalsRow?.["project_memories"] ?? 0,
      documents: totalsRow?.["total_documents"] ?? 0,
      sessions: totalsRow?.["total_sessions"] ?? 0,
    },
    bySector: Object.fromEntries(
      bySector.rows.map((r) => [String(r["sector"]), Number(r["count"])])
    ),
  };
}

async function getProjects(): Promise<unknown[]> {
  const db = await getDatabase();
  const result = await db.execute(`
    SELECT
      p.id,
      p.path,
      p.name,
      p.created_at,
      p.updated_at,
      COUNT(DISTINCT m.id) as memory_count,
      COUNT(DISTINCT s.id) as session_count,
      MAX(COALESCE(m.created_at, s.started_at)) as last_activity
    FROM projects p
    LEFT JOIN memories m ON m.project_id = p.id AND m.is_deleted = 0
    LEFT JOIN sessions s ON s.project_id = p.id
    GROUP BY p.id
    ORDER BY last_activity DESC NULLS LAST
  `);
  return result.rows;
}

type BrowseOptions = {
  projectId?: string | null;
  dateStr?: string | null;
  sector?: MemorySector | null;
  limit: number;
  offset: number;
};

async function browseTimeline(options: BrowseOptions): Promise<JsonResponse> {
  const db = await getDatabase();

  let startTime: number;
  let endTime: number;

  if (options.dateStr) {
    const d = new Date(options.dateStr);
    startTime = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    endTime = startTime + 24 * 60 * 60 * 1000 - 1;
  } else {
    endTime = Date.now();
    startTime = endTime - 7 * 24 * 60 * 60 * 1000;
  }

  const args: InValue[] = [startTime, endTime];
  let whereClause = "m.created_at >= ? AND m.created_at <= ? AND m.is_deleted = 0";

  if (options.projectId) {
    whereClause += " AND m.project_id = ?";
    args.push(options.projectId);
  }
  if (options.sector) {
    whereClause += " AND m.sector = ?";
    args.push(options.sector);
  }

  const memoriesResult = await db.execute(
    `
    SELECT m.*,
           s.id as source_session_id,
           s.summary as source_session_summary
    FROM memories m
    LEFT JOIN session_memories sm ON sm.memory_id = m.id AND sm.usage_type = 'created'
    LEFT JOIN sessions s ON sm.session_id = s.id
    WHERE ${whereClause}
    ORDER BY m.created_at DESC
    LIMIT ? OFFSET ?
    `,
    [...args, options.limit, options.offset]
  );

  const dateArgs: InValue[] = [startTime, endTime];
  let dateWhere = "created_at >= ? AND created_at <= ? AND is_deleted = 0";
  if (options.projectId) {
    dateWhere += " AND project_id = ?";
    dateArgs.push(options.projectId);
  }
  if (options.sector) {
    dateWhere += " AND sector = ?";
    dateArgs.push(options.sector);
  }

  const dateAggregates = await db.execute(
    `
    SELECT
      DATE(created_at / 1000, 'unixepoch', 'localtime') as date,
      COUNT(*) as count
    FROM memories
    WHERE ${dateWhere}
    GROUP BY DATE(created_at / 1000, 'unixepoch', 'localtime')
    ORDER BY date DESC
    LIMIT 30
    `,
    dateArgs
  );

  return {
    memories: memoriesResult.rows,
    dateAggregates: dateAggregates.rows,
    hasMore: memoriesResult.rows.length === options.limit,
  };
}

async function fetchPageData(url: URL): Promise<JsonResponse> {
  const path = url.pathname;
  const searchParams = url.searchParams;

  if (path === "/projects") {
    const projects = await getProjects();
    return { type: "projects", projects };
  }

  if (path === "/" || path === "/search") {
    const query = searchParams.get("q");
    const projectId = searchParams.get("project");
    const sessionId = searchParams.get("session");
    if (query) {
      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const results = await search.search({
        query,
        projectId: projectId ?? undefined,
        sessionId: sessionId ?? undefined,
        limit: 20,
      });
      return { type: "search", results, projectId, sessionId };
    }
    if (projectId) {
      const results = await getRecentProjectMemories(projectId, 20);
      return { type: "search", results, projectId, sessionId };
    }
    if (sessionId) {
      const results = await getRecentSessionMemories(sessionId, 20);
      return { type: "search", results, projectId, sessionId };
    }
    return { type: "search", results: [], projectId, sessionId };
  }

  if (path === "/agents") {
    const sessions = await getRecentSessions(searchParams.get("project"));
    const recentActivity = await getRecentActivity(15);
    return { type: "agents", sessions, recentActivity };
  }

  if (path === "/timeline") {
    const anchorId = searchParams.get("anchor");
    if (anchorId) {
      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const data = await search.timeline(anchorId, 10, 10);
      return { type: "timeline", data, browseMode: false };
    }
    const projectId = searchParams.get("project");
    const dateStr = searchParams.get("date");
    const sector = searchParams.get("sector") as MemorySector | null;
    const browseData = await browseTimeline({
      projectId,
      dateStr,
      sector,
      limit: 50,
      offset: 0,
    });
    const projects = await getProjects();
    return { type: "timeline", browseMode: true, ...browseData, projects };
  }

  return { type: "home" };
}

type ConfigMap = {
  embeddingProvider: string;
  captureEnabled: string;
  captureThreshold: string;
};

async function getConfig(): Promise<ConfigMap> {
  const db = await getDatabase();
  const result = await db.execute(
    "SELECT key, value FROM config WHERE key IN (?, ?, ?)",
    ["embeddingProvider", "captureEnabled", "captureThreshold"]
  );

  const config: ConfigMap = {
    embeddingProvider: "ollama",
    captureEnabled: "true",
    captureThreshold: "0.3",
  };

  for (const row of result.rows) {
    const key = String(row["key"]) as keyof ConfigMap;
    if (key in config) {
      config[key] = String(row["value"]);
    }
  }

  return config;
}

async function setConfig(key: string, value: string): Promise<void> {
  const db = await getDatabase();
  await db.execute(
    `INSERT INTO config (key, value, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
    [key, value, Date.now()]
  );
}

async function clearMemories(projectId?: string): Promise<number> {
  const db = await getDatabase();
  const now = Date.now();

  let result;
  if (projectId) {
    result = await db.execute(
      "UPDATE memories SET is_deleted = 1, deleted_at = ? WHERE project_id = ? AND is_deleted = 0",
      [now, projectId]
    );
  } else {
    result = await db.execute(
      "UPDATE memories SET is_deleted = 1, deleted_at = ? WHERE is_deleted = 0",
      [now]
    );
  }

  return result.rowsAffected;
}

async function getSessionMemories(
  sessionId: string,
  limit: number
): Promise<unknown[]> {
  const db = await getDatabase();
  const result = await db.execute(
    `SELECT m.id, m.content, m.summary, m.sector, m.salience, m.created_at
     FROM memories m
     JOIN session_memories sm ON sm.memory_id = m.id
     WHERE sm.session_id = ? AND sm.usage_type = 'created' AND m.is_deleted = 0
     ORDER BY m.created_at DESC
     LIMIT ?`,
    [sessionId, limit]
  );

  return result.rows.map((row) => ({
    id: row["id"],
    content: row["content"],
    summary: row["summary"],
    sector: row["sector"],
    salience: row["salience"],
    createdAt: row["created_at"],
  }));
}

async function getRecentActivity(limit: number = 20): Promise<unknown[]> {
  const db = await getDatabase();
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;

  const result = await db.execute(
    `SELECT m.id, m.content, m.summary, m.sector, m.salience, m.project_id, m.created_at
     FROM memories m
     WHERE m.is_deleted = 0 AND m.created_at > ?
     ORDER BY m.created_at DESC
     LIMIT ?`,
    [cutoff, limit]
  );

  return result.rows.map((row) => ({
    id: `${row["id"]}-${row["created_at"]}`,
    type: "created",
    memory: {
      id: row["id"],
      content: row["content"],
      sector: row["sector"],
      salience: row["salience"],
      summary: row["summary"],
    },
    projectId: row["project_id"],
    timestamp: Number(row["created_at"]),
  }));
}

async function getRecentProjectMemories(
  projectId: string,
  limit: number
): Promise<unknown[]> {
  const db = await getDatabase();
  const result = await db.execute(
    `SELECT m.*,
            s.id as source_session_id,
            s.summary as source_session_summary
     FROM memories m
     LEFT JOIN session_memories sm ON sm.memory_id = m.id AND sm.usage_type = 'created'
     LEFT JOIN sessions s ON sm.session_id = s.id
     WHERE m.project_id = ? AND m.is_deleted = 0
     ORDER BY m.created_at DESC
     LIMIT ?`,
    [projectId, limit]
  );

  return result.rows.map((row) => ({
    memory: {
      id: row["id"],
      content: row["content"],
      summary: row["summary"],
      sector: row["sector"],
      tier: row["tier"],
      salience: row["salience"],
      projectId: row["project_id"],
      createdAt: row["created_at"],
      updatedAt: row["updated_at"],
      validUntil: row["valid_until"],
      isDeleted: row["is_deleted"],
    },
    score: Number(row["salience"]),
    matchType: "keyword" as const,
    isSuperseded: !!row["valid_until"],
    relatedMemoryCount: 0,
    sourceSession: row["source_session_id"]
      ? {
          id: row["source_session_id"],
          summary: row["source_session_summary"],
        }
      : undefined,
  }));
}

async function getRecentSessionMemories(
  sessionId: string,
  limit: number
): Promise<unknown[]> {
  const db = await getDatabase();
  const result = await db.execute(
    `SELECT m.*,
            s.id as source_session_id,
            s.summary as source_session_summary
     FROM memories m
     JOIN session_memories sm ON sm.memory_id = m.id
     LEFT JOIN sessions s ON sm.session_id = s.id
     WHERE sm.session_id = ? AND sm.usage_type = 'created' AND m.is_deleted = 0
     ORDER BY m.created_at DESC
     LIMIT ?`,
    [sessionId, limit]
  );

  return result.rows.map((row) => ({
    memory: {
      id: row["id"],
      content: row["content"],
      summary: row["summary"],
      sector: row["sector"],
      tier: row["tier"],
      salience: row["salience"],
      projectId: row["project_id"],
      createdAt: row["created_at"],
      updatedAt: row["updated_at"],
      validUntil: row["valid_until"],
      isDeleted: row["is_deleted"],
    },
    score: Number(row["salience"]),
    matchType: "keyword" as const,
    isSuperseded: !!row["valid_until"],
    relatedMemoryCount: 0,
    sourceSession: row["source_session_id"]
      ? {
          id: row["source_session_id"],
          summary: row["source_session_summary"],
        }
      : undefined,
  }));
}
