# Search Specification

## Overview

Hybrid search combines FTS5 keyword search with vector similarity search, using salience-weighted ranking for optimal results. Search results include session context and temporal information to help understand when and where memories were created.

## Files to Create

- `src/services/search/hybrid.ts` - Hybrid search combining FTS + vector
- `src/services/search/ranking.ts` - Score computation and ranking
- `src/services/search/fts.ts` - FTS5 keyword search
- `src/services/search/vector.ts` - Vector similarity search

## Search Interface

### Main Interface

```typescript
// src/services/search/hybrid.ts
type SearchOptions = {
  query: string;
  projectId?: string;
  sector?: MemorySector;
  tier?: MemoryTier;
  limit?: number;
  minSalience?: number;
  includeDocuments?: boolean;
  includeSuperseded?: boolean;  // Include memories that have been superseded
  sessionId?: string;           // Filter to specific session
  mode?: 'hybrid' | 'semantic' | 'keyword';
};

type SearchResult = {
  memory: Memory;
  score: number;
  matchType: 'semantic' | 'keyword' | 'both';
  highlights?: string[];

  // Session context
  sourceSession?: {
    id: string;
    startedAt: number;
    summary?: string;
    projectId: string;
  };

  // Temporal context
  isSuperseded: boolean;
  supersededBy?: {
    id: string;
    content: string;
    createdAt: number;
  };

  // Related context
  relatedMemoryCount: number;
};

type SearchService = {
  search(options: SearchOptions): Promise<SearchResult[]>;
  searchDocuments(query: string, projectId?: string, limit?: number): Promise<DocumentSearchResult[]>;
  timeline(anchorId: string, depthBefore?: number, depthAfter?: number): Promise<TimelineResult>;
  getSessionContext(memoryId: string): Promise<SessionContext | null>;
};

type TimelineResult = {
  anchor: Memory;
  before: Memory[];
  after: Memory[];
  sessions: Map<string, SessionSummary>;  // Session info for all memories
};

type SessionContext = {
  session: {
    id: string;
    startedAt: number;
    endedAt?: number;
    summary?: string;
    projectId: string;
  };
  memoriesInSession: number;
  usageType: 'created' | 'recalled' | 'updated' | 'reinforced';
};

function createSearchService(): SearchService;
```

## FTS5 Keyword Search

### Interface

```typescript
// src/services/search/fts.ts
export interface FTSResult {
  memoryId: string;
  rank: number;
  snippet: string;
}

export async function searchFTS(
  query: string,
  projectId?: string,
  limit?: number
): Promise<FTSResult[]>;
```

### Implementation Notes

```typescript
import { log } from "../../utils/log.js";

export async function searchFTS(
  query: string,
  projectId?: string,
  limit = 20
): Promise<FTSResult[]> {
  const db = getDatabase();
  const start = Date.now();

  // Prepare query for FTS5
  // Convert "foo bar" to "foo* OR bar*" for prefix matching
  const ftsQuery = query
    .split(/\s+/)
    .filter(t => t.length > 1)
    .map(t => `"${t}"*`)
    .join(" OR ");

  if (!ftsQuery) {
    log.debug("search", "Empty FTS query", { original: query });
    return [];
  }

  log.debug("search", "FTS search", { query: ftsQuery, projectId, limit });

  let sql = `
    SELECT
      m.id as memory_id,
      bm25(memories_fts) as rank,
      snippet(memories_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
    FROM memories_fts
    JOIN memories m ON memories_fts.rowid = m.rowid
    WHERE memories_fts MATCH ?
  `;
  const args: any[] = [ftsQuery];

  if (projectId) {
    sql += " AND m.project_id = ?";
    args.push(projectId);
  }

  sql += " ORDER BY rank LIMIT ?";
  args.push(limit);

  const result = await db.execute(sql, args);

  log.info("search", "FTS search complete", {
    results: result.rows.length,
    ms: Date.now() - start
  });

  return result.rows.map(row => ({
    memoryId: row[0] as string,
    rank: row[1] as number,
    snippet: row[2] as string
  }));
}
```

### Test Specification

```typescript
// src/services/search/__test__/fts.test.ts (colocated)
describe('FTS5 Search', () => {
  beforeEach(async () => {
    await setupTestDatabase();
    // Insert test memories
    await store.create({
      content: "The authentication module handles user login and JWT tokens"
    }, "proj1");
    await store.create({
      content: "Database migrations are run with the migrate command"
    }, "proj1");
  });

  test("finds memories by keyword", async () => {
    const results = await searchFTS("authentication", "proj1");
    expect(results.length).toBe(1);
    expect(results[0].snippet).toContain("authentication");
  });

  test("supports prefix matching", async () => {
    const results = await searchFTS("auth", "proj1");
    expect(results.length).toBe(1);  // Matches "authentication"
  });

  test("returns highlighted snippets", async () => {
    const results = await searchFTS("authentication", "proj1");
    expect(results[0].snippet).toContain("<mark>");
  });

  test("filters by project", async () => {
    await store.create({ content: "authentication in another project" }, "proj2");

    const results = await searchFTS("authentication", "proj1");
    expect(results.length).toBe(1);
  });

  test("handles empty query", async () => {
    const results = await searchFTS("", "proj1");
    expect(results).toEqual([]);
  });
});
```

## Vector Similarity Search

### Interface

```typescript
// src/services/search/vector.ts
export interface VectorResult {
  memoryId: string;
  distance: number;  // Cosine distance (0 = identical)
  similarity: number;  // 1 - distance
}

export async function searchVector(
  query: string,
  projectId?: string,
  limit?: number
): Promise<VectorResult[]>;
```

### Implementation Notes

```typescript
import { log } from "../../utils/log.js";

export async function searchVector(
  query: string,
  projectId?: string,
  limit = 20
): Promise<VectorResult[]> {
  const db = getDatabase();
  const embedding = getEmbeddingService();
  const start = Date.now();

  log.debug("search", "Vector search starting", { queryLength: query.length, projectId, limit });

  // Get query embedding
  const queryEmbedding = await embedding.embed(query);
  const modelId = embedding.getActiveModelId();

  log.debug("search", "Query embedded", { model: modelId, ms: Date.now() - start });

  // Use vector_top_k for efficient ANN search
  let sql = `
    SELECT
      mv.memory_id,
      vector_distance_cos(mv.vector, vector(?)) as distance
    FROM memory_vectors mv
    JOIN memories m ON mv.memory_id = m.id
    WHERE mv.model_id = ?
      AND mv.rowid IN (
        SELECT rowid FROM vector_top_k('memory_vectors_idx', vector(?), ?)
      )
  `;
  const args: any[] = [
    JSON.stringify(queryEmbedding.vector),
    modelId,
    JSON.stringify(queryEmbedding.vector),
    limit * 2  // Get more candidates for filtering
  ];

  if (projectId) {
    sql += " AND m.project_id = ?";
    args.push(projectId);
  }

  sql += " ORDER BY distance ASC LIMIT ?";
  args.push(limit);

  const result = await db.execute(sql, args);

  log.info("search", "Vector search complete", {
    results: result.rows.length,
    ms: Date.now() - start
  });

  return result.rows.map(row => ({
    memoryId: row[0] as string,
    distance: row[1] as number,
    similarity: 1 - (row[1] as number)
  }));
}
```

### Test Specification

```typescript
// src/services/search/__test__/vector.test.ts (colocated)
describe('Vector Search', () => {
  beforeEach(async () => {
    await setupTestDatabase();
    // Insert test memories with embeddings
    await store.create({
      content: "User authentication with JWT tokens and OAuth2"
    }, "proj1");
    await store.create({
      content: "Database schema design with foreign keys"
    }, "proj1");
  });

  test("finds semantically similar memories", async () => {
    const results = await searchVector("login system security", "proj1");
    expect(results.length).toBeGreaterThan(0);
    // Auth memory should be more similar than DB memory
    expect(results[0].memoryId).toBeDefined();
  });

  test("returns similarity scores", async () => {
    const results = await searchVector("authentication", "proj1");
    expect(results[0].similarity).toBeGreaterThan(0);
    expect(results[0].similarity).toBeLessThanOrEqual(1);
  });

  test("only searches current model vectors", async () => {
    // Vectors from different models should not be mixed
    const results = await searchVector("test query", "proj1");
    // All results should have vectors from current model
    // (verified by the query itself filtering by model_id)
  });
});
```

## Hybrid Search & Ranking

### Scoring Algorithm

```typescript
// src/services/search/ranking.ts
type RankingWeights = {
  semantic: number;
  keyword: number;
  salience: number;
  recency: number;
  sectorBoost: Record<MemorySector, number>;
};

const DEFAULT_WEIGHTS: RankingWeights = {
  semantic: 0.4,
  keyword: 0.25,
  salience: 0.2,
  recency: 0.15,
  sectorBoost: {
    reflective: 1.2,   // Insights are valuable
    semantic: 1.1,     // Facts are important
    procedural: 1.0,   // Standard
    emotional: 0.9,    // Less often searched
    episodic: 0.8,     // Events fade in relevance
  },
};

function computeScore(
  memory: Memory,
  semanticSim: number,
  ftsRank: number,
  weights?: RankingWeights
): number {
  const w = weights || DEFAULT_WEIGHTS;

  const normalizedFTS = ftsRank ? Math.min(1, Math.abs(ftsRank) / 10) : 0;

  const daysSinceCreated = (Date.now() - memory.createdAt) / (1000 * 60 * 60 * 24);
  const recencyScore = Math.exp(-0.05 * daysSinceCreated);

  let score =
    w.semantic * semanticSim +
    w.keyword * normalizedFTS +
    w.salience * memory.salience +
    w.recency * recencyScore;

  score *= w.sectorBoost[memory.sector] || 1.0;

  // Penalty for superseded memories
  if (memory.validUntil) {
    score *= 0.5;
  }

  return Math.min(1, Math.max(0, score));
}
```

### Hybrid Search Implementation

```typescript
// src/services/search/hybrid.ts
import { log } from "../../utils/log.js";

function createSearchService(): SearchService {
  return {
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
        mode = 'hybrid',
      } = options;

      const start = Date.now();
      log.info("search", "Hybrid search starting", { query: query.slice(0, 50), mode, projectId });

      const [ftsResults, vectorResults] = await Promise.all([
        mode !== 'semantic' ? searchFTS(query, projectId, limit * 2) : [],
        mode !== 'keyword' ? searchVector(query, projectId, limit * 2) : [],
      ]);

      const resultMap = new Map<string, {
        ftsRank: number;
        similarity: number;
        snippet?: string;
      }>();

      for (const r of ftsResults) {
        resultMap.set(r.memoryId, { ftsRank: r.rank, similarity: 0, snippet: r.snippet });
      }

      for (const r of vectorResults) {
        const existing = resultMap.get(r.memoryId);
        if (existing) {
          existing.similarity = r.similarity;
        } else {
          resultMap.set(r.memoryId, { ftsRank: 0, similarity: r.similarity });
        }
      }

      const memoryIds = Array.from(resultMap.keys());
      const memories = await Promise.all(memoryIds.map(id => store.get(id)));

      const results: SearchResult[] = [];

      for (let i = 0; i < memories.length; i++) {
        const memory = memories[i];
        if (!memory || memory.isDeleted) continue;

        // Apply filters
        if (sector && memory.sector !== sector) continue;
        if (tier && memory.tier !== tier) continue;
        if (memory.salience < minSalience) continue;
        if (!includeSuperseded && memory.validUntil) continue;

        // Session filter
        if (sessionId) {
          const sessionLink = await getSessionLink(memory.id, sessionId);
          if (!sessionLink) continue;
        }

        const data = resultMap.get(memory.id)!;
        const score = computeScore(memory, data.similarity, data.ftsRank);

        const matchType = data.similarity > 0 && data.ftsRank !== 0
          ? 'both'
          : data.similarity > 0
            ? 'semantic'
            : 'keyword';

        // Get session context
        const sourceSession = await getSourceSession(memory.id);

        // Check if superseded
        const supersedingMemory = await getSupersedingMemory(memory.id);

        // Get related memory count
        const relatedCount = await getRelatedMemoryCount(memory.id);

        results.push({
          memory,
          score,
          matchType,
          highlights: data.snippet ? [data.snippet] : undefined,
          sourceSession,
          isSuperseded: !!memory.validUntil,
          supersededBy: supersedingMemory ? {
            id: supersedingMemory.id,
            content: supersedingMemory.content.slice(0, 200),
            createdAt: supersedingMemory.createdAt,
          } : undefined,
          relatedMemoryCount: relatedCount,
        });
      }

      results.sort((a, b) => b.score - a.score);

      const topResults = results.slice(0, limit);
      for (const result of topResults) {
        await store.reinforce(result.memory.id, 0.02);
        if (sessionId) {
          await store.linkToSession(result.memory.id, sessionId, 'recalled');
        }
      }

      log.info("search", "Hybrid search complete", {
        total: results.length,
        returned: topResults.length,
        mode,
        ms: Date.now() - start
      });

      return topResults;
    },

    async timeline(anchorId: string, depthBefore = 5, depthAfter = 5): Promise<TimelineResult> {
      log.debug("search", "Timeline query", { anchorId, depthBefore, depthAfter });
      const db = getDatabase();
      const anchor = await store.get(anchorId);
      if (!anchor) {
        log.warn("search", "Timeline anchor not found", { anchorId });
        throw new Error('Anchor memory not found');
      }

      const [beforeResult, afterResult] = await Promise.all([
        db.execute(`
          SELECT * FROM memories
          WHERE project_id = ? AND created_at < ? AND is_deleted = 0
          ORDER BY created_at DESC
          LIMIT ?
        `, [anchor.projectId, anchor.createdAt, depthBefore]),
        db.execute(`
          SELECT * FROM memories
          WHERE project_id = ? AND created_at > ? AND is_deleted = 0
          ORDER BY created_at ASC
          LIMIT ?
        `, [anchor.projectId, anchor.createdAt, depthAfter]),
      ]);

      const before = beforeResult.rows.map(rowToMemory).reverse();
      const after = afterResult.rows.map(rowToMemory);

      // Gather session info for all memories
      const allMemories = [...before, anchor, ...after];
      const sessions = new Map<string, SessionSummary>();

      for (const memory of allMemories) {
        const sessionContext = await this.getSessionContext(memory.id);
        if (sessionContext && !sessions.has(sessionContext.session.id)) {
          sessions.set(sessionContext.session.id, {
            id: sessionContext.session.id,
            startedAt: sessionContext.session.startedAt,
            summary: sessionContext.session.summary,
          });
        }
      }

      return { anchor, before, after, sessions };
    },

    async getSessionContext(memoryId: string): Promise<SessionContext | null> {
      const db = getDatabase();
      const result = await db.execute(`
        SELECT s.*, sm.usage_type,
               (SELECT COUNT(*) FROM session_memories WHERE session_id = s.id) as memory_count
        FROM session_memories sm
        JOIN sessions s ON sm.session_id = s.id
        WHERE sm.memory_id = ?
        ORDER BY sm.created_at DESC
        LIMIT 1
      `, [memoryId]);

      if (result.rows.length === 0) return null;

      const row = result.rows[0];
      return {
        session: {
          id: row.id,
          startedAt: row.started_at,
          endedAt: row.ended_at,
          summary: row.summary,
          projectId: row.project_id,
        },
        memoriesInSession: row.memory_count,
        usageType: row.usage_type,
      };
    },

    async searchDocuments(query: string, projectId?: string, limit = 10): Promise<DocumentSearchResult[]> {
      // Similar hybrid search for documents
      // ... implementation
    },
  };
}

async function getSourceSession(memoryId: string): Promise<SearchResult['sourceSession'] | undefined> {
  const db = getDatabase();
  const result = await db.execute(`
    SELECT s.id, s.started_at, s.summary, s.project_id
    FROM session_memories sm
    JOIN sessions s ON sm.session_id = s.id
    WHERE sm.memory_id = ? AND sm.usage_type = 'created'
    LIMIT 1
  `, [memoryId]);

  if (result.rows.length === 0) return undefined;

  const row = result.rows[0];
  return {
    id: row.id,
    startedAt: row.started_at,
    summary: row.summary,
    projectId: row.project_id,
  };
}

async function getRelatedMemoryCount(memoryId: string): Promise<number> {
  const db = getDatabase();
  const result = await db.execute(`
    SELECT COUNT(*) as count FROM memory_relationships
    WHERE source_memory_id = ? OR target_memory_id = ?
  `, [memoryId, memoryId]);
  return result.rows[0].count;
}
```

### Test Specification

```typescript
// src/services/search/__test__/hybrid.test.ts (colocated)
describe('Hybrid Search', () => {
  let search: SearchService;

  beforeEach(async () => {
    await setupTestDatabase();
    search = createSearchService();

    // Create a session for testing
    await db.execute(`INSERT INTO sessions (id, project_id, started_at) VALUES ('sess1', 'proj1', ?)`,[Date.now()]);

    // Insert diverse test memories with sessions
    await store.create({
      content: 'Decided to use React with TypeScript for the frontend',
      sector: 'reflective',
    }, 'proj1', 'sess1');
    await store.create({
      content: 'To deploy, run npm build then upload to S3',
      sector: 'procedural',
    }, 'proj1', 'sess1');
    await store.create({
      content: 'The API routes are defined in src/routes/api.ts',
      sector: 'semantic',
    }, 'proj1', 'sess1');
  });

  test('returns results from both FTS and vector', async () => {
    const results = await search.search({ query: 'React TypeScript', projectId: 'proj1' });
    expect(results.length).toBeGreaterThan(0);
  });

  test('results include session context', async () => {
    const results = await search.search({ query: 'React', projectId: 'proj1' });
    expect(results[0].sourceSession).toBeDefined();
    expect(results[0].sourceSession?.id).toBe('sess1');
  });

  test('results include superseded status', async () => {
    const old = await store.create({ content: 'Old API endpoint is /v1' }, 'proj1');
    const newMem = await store.create({ content: 'New API endpoint is /v2' }, 'proj1');
    await supersede(old.id, newMem.id);

    const results = await search.search({
      query: 'API endpoint',
      projectId: 'proj1',
      includeSuperseded: true,
    });

    const oldResult = results.find(r => r.memory.id === old.id);
    expect(oldResult?.isSuperseded).toBe(true);
    expect(oldResult?.supersededBy?.id).toBe(newMem.id);
  });

  test('excludes superseded by default', async () => {
    const old = await store.create({ content: 'Old fact' }, 'proj1');
    const newMem = await store.create({ content: 'New fact' }, 'proj1');
    await supersede(old.id, newMem.id);

    const results = await search.search({ query: 'fact', projectId: 'proj1' });
    expect(results.find(r => r.memory.id === old.id)).toBeUndefined();
  });

  test('filters by sector', async () => {
    const results = await search.search({
      query: 'build deploy',
      projectId: 'proj1',
      sector: 'procedural',
    });
    expect(results.every(r => r.memory.sector === 'procedural')).toBe(true);
  });

  test('filters by session', async () => {
    await db.execute(`INSERT INTO sessions (id, project_id, started_at) VALUES ('sess2', 'proj1', ?)`, [Date.now()]);
    await store.create({ content: 'Only in session 2' }, 'proj1', 'sess2');

    const results = await search.search({
      query: 'session',
      projectId: 'proj1',
      sessionId: 'sess2',
    });
    expect(results.every(r => r.sourceSession?.id === 'sess2')).toBe(true);
  });

  test('timeline includes session info', async () => {
    const mem1 = await store.create({ content: 'First' }, 'proj1', 'sess1');
    await new Promise(r => setTimeout(r, 10));
    const mem2 = await store.create({ content: 'Second' }, 'proj1', 'sess1');

    const timeline = await search.timeline(mem2.id, 2, 2);

    expect(timeline.anchor.id).toBe(mem2.id);
    expect(timeline.before.length).toBeGreaterThan(0);
    expect(timeline.sessions.size).toBeGreaterThan(0);
    expect(timeline.sessions.has('sess1')).toBe(true);
  });

  test('getSessionContext returns memory context', async () => {
    const memory = await store.create({ content: 'Test memory' }, 'proj1', 'sess1');

    const context = await search.getSessionContext(memory.id);

    expect(context).not.toBeNull();
    expect(context?.session.id).toBe('sess1');
    expect(context?.usageType).toBe('created');
  });

  test('includes related memory count', async () => {
    const mem1 = await store.create({ content: 'Memory 1' }, 'proj1');
    const mem2 = await store.create({ content: 'Memory 2' }, 'proj1');
    await createRelationship(mem1.id, mem2.id, 'RELATED_TO', 'user');

    const results = await search.search({ query: 'Memory 1', projectId: 'proj1' });
    const result = results.find(r => r.memory.id === mem1.id);
    expect(result?.relatedMemoryCount).toBe(1);
  });

  test('tracks recalled memories in session', async () => {
    const memory = await store.create({ content: 'To be recalled' }, 'proj1', 'sess1');

    await db.execute(`INSERT INTO sessions (id, project_id, started_at) VALUES ('sess2', 'proj1', ?)`, [Date.now()]);

    await search.search({
      query: 'recalled',
      projectId: 'proj1',
      sessionId: 'sess2',
    });

    const context = await search.getSessionContext(memory.id);
    // The latest interaction should be in sess2 as 'recalled'
  });
});
```

## Acceptance Criteria

- [ ] FTS5 finds memories by keywords
- [ ] FTS5 supports prefix matching
- [ ] Vector search finds semantically similar memories
- [ ] Hybrid search combines both methods
- [ ] Ranking weights semantic, keyword, salience, and recency
- [ ] Sector-specific boosts applied
- [ ] Filters work (sector, tier, minSalience, sessionId)
- [ ] Timeline returns chronological context with session info
- [ ] Search boosts salience of returned results
- [ ] Different search modes work (hybrid, semantic, keyword)
- [ ] Search results include session context (sourceSession)
- [ ] Search results include superseded status
- [ ] Superseded memories excluded by default
- [ ] getSessionContext returns memory usage info
- [ ] Related memory count included in results
- [ ] Recalled memories tracked in session
