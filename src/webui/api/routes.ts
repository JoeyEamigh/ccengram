import { createSearchService } from "../../services/search/hybrid.js";
import { createMemoryStore } from "../../services/memory/store.js";
import { createEmbeddingService } from "../../services/embedding/index.js";
import { getDatabase } from "../../db/database.js";
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
      const includeSuperseded =
        url.searchParams.get("include_superseded") === "true";
      const limit = parseInt(url.searchParams.get("limit") ?? "20");

      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const results = await search.search({
        query,
        sector: sector ?? undefined,
        sessionId: sessionId ?? undefined,
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

    if (path === "/api/sessions" && req.method === "GET") {
      const projectId = url.searchParams.get("project");
      const sessions = await getRecentSessions(projectId);
      return json({ sessions });
    }

    if (path === "/api/stats" && req.method === "GET") {
      const stats = await getStats();
      return json(stats);
    }

    if (path === "/api/page-data" && req.method === "GET") {
      const pagePath = url.searchParams.get("path") ?? "/";
      const data = await fetchPageData(new URL(pagePath, req.url));
      return json(data);
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
  return result.rows;
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

async function fetchPageData(url: URL): Promise<JsonResponse> {
  const path = url.pathname;
  const searchParams = url.searchParams;

  if (path === "/" || path === "/search") {
    const query = searchParams.get("q");
    if (query) {
      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const results = await search.search({ query, limit: 20 });
      return { type: "search", results };
    }
    return { type: "search", results: [] };
  }

  if (path === "/agents") {
    const sessions = await getRecentSessions(searchParams.get("project"));
    return { type: "agents", sessions };
  }

  if (path === "/timeline") {
    const anchorId = searchParams.get("anchor");
    if (anchorId) {
      const embedding = await getEmbeddingService();
      const search = createSearchService(embedding);
      const data = await search.timeline(anchorId, 10, 10);
      return { type: "timeline", data };
    }
    return { type: "timeline", data: null };
  }

  return { type: "home" };
}
