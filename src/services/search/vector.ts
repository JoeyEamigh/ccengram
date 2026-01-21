import { getDatabase } from '../../db/database.js';
import { log } from '../../utils/log.js';
import type { EmbeddingService } from '../embedding/types.js';

export type VectorResult = {
  memoryId: string;
  distance: number;
  similarity: number;
};

export type BatchedSearchOptions = {
  batchSize?: number;
  earlyTerminationThreshold?: number;
  maxBatches?: number;
  candidateIds?: string[];
};

const DEFAULT_BATCH_SIZE = 1000;
const DEFAULT_EARLY_TERMINATION_THRESHOLD = 0.95;
const DEFAULT_MAX_BATCHES = 100;

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}

function parseVector(blob: unknown): number[] {
  if (blob instanceof Uint8Array || blob instanceof ArrayBuffer) {
    const buffer = blob instanceof ArrayBuffer ? blob : blob.buffer;
    return Array.from(new Float32Array(buffer));
  }

  if (typeof blob === 'string') {
    try {
      return JSON.parse(blob) as number[];
    } catch {
      return [];
    }
  }

  if (Array.isArray(blob)) {
    return blob as number[];
  }

  return [];
}

export async function searchVector(
  query: string,
  embeddingService: EmbeddingService,
  projectId?: string,
  limit = 20,
): Promise<VectorResult[]> {
  const db = await getDatabase();
  const start = Date.now();

  log.debug('search', 'Vector search starting', {
    queryLength: query.length,
    projectId,
    limit,
  });

  const instructedQuery = `Instruct: Given a natural language query, find semantically related user preferences, decisions, and knowledge\nQuery:${query}`;
  const queryEmbedding = await embeddingService.embed(instructedQuery);
  const modelId = embeddingService.getActiveModelId();

  log.debug('search', 'Query embedded', {
    model: modelId,
    ms: Date.now() - start,
  });

  let sql = `
    SELECT
      mv.memory_id,
      mv.vector
    FROM memory_vectors mv
    JOIN memories m ON mv.memory_id = m.id
    WHERE mv.model_id = ?
      AND m.is_deleted = 0
  `;
  const args: (string | number)[] = [modelId];

  if (projectId) {
    sql += ' AND m.project_id = ?';
    args.push(projectId);
  }

  const result = await db.execute(sql, args);

  const scored: VectorResult[] = [];

  for (const row of result.rows) {
    const memoryId = String(row['memory_id']);
    const vectorData = row['vector'];
    const vector = parseVector(vectorData);

    if (vector.length !== queryEmbedding.dimensions) {
      continue;
    }

    const similarity = cosineSimilarity(queryEmbedding.vector, vector);
    const distance = 1 - similarity;

    scored.push({
      memoryId,
      distance,
      similarity,
    });
  }

  scored.sort((a, b) => b.similarity - a.similarity);

  const topResults = scored.slice(0, limit);

  log.info('search', 'Vector search complete', {
    candidates: result.rows.length,
    results: topResults.length,
    ms: Date.now() - start,
  });

  return topResults;
}

export async function searchVectorBatched(
  query: string,
  embeddingService: EmbeddingService,
  projectId?: string,
  limit = 20,
  options: BatchedSearchOptions = {},
): Promise<VectorResult[]> {
  const db = await getDatabase();
  const start = Date.now();

  const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
  const earlyTerminationThreshold = options.earlyTerminationThreshold ?? DEFAULT_EARLY_TERMINATION_THRESHOLD;
  const maxBatches = options.maxBatches ?? DEFAULT_MAX_BATCHES;

  log.debug('search', 'Batched vector search starting', {
    queryLength: query.length,
    projectId,
    limit,
    batchSize,
  });

  const instructedQuery = `Instruct: Given a natural language query, find semantically related user preferences, decisions, and knowledge\nQuery:${query}`;
  const queryEmbedding = await embeddingService.embed(instructedQuery);
  const modelId = embeddingService.getActiveModelId();

  log.debug('search', 'Query embedded', {
    model: modelId,
    ms: Date.now() - start,
  });

  let totalCandidates = 0;
  let batchCount = 0;
  const results: VectorResult[] = [];
  let offset = 0;
  let foundHighQualityResults = false;

  while (batchCount < maxBatches) {
    let sql: string;
    const args: (string | number)[] = [];

    if (options.candidateIds && options.candidateIds.length > 0) {
      const placeholders = options.candidateIds.map(() => '?').join(',');
      sql = `
        SELECT
          mv.memory_id,
          mv.vector
        FROM memory_vectors mv
        JOIN memories m ON mv.memory_id = m.id
        WHERE mv.model_id = ?
          AND m.is_deleted = 0
          AND mv.memory_id IN (${placeholders})
        LIMIT ? OFFSET ?
      `;
      args.push(modelId, ...options.candidateIds, batchSize, offset);
    } else {
      sql = `
        SELECT
          mv.memory_id,
          mv.vector
        FROM memory_vectors mv
        JOIN memories m ON mv.memory_id = m.id
        WHERE mv.model_id = ?
          AND m.is_deleted = 0
      `;
      args.push(modelId);

      if (projectId) {
        sql += ' AND m.project_id = ?';
        args.push(projectId);
      }

      sql += ' LIMIT ? OFFSET ?';
      args.push(batchSize, offset);
    }

    const batchResult = await db.execute(sql, args);
    const rowCount = batchResult.rows.length;
    totalCandidates += rowCount;
    batchCount++;

    if (rowCount === 0) {
      break;
    }

    for (const row of batchResult.rows) {
      const memoryId = String(row['memory_id']);
      const vectorData = row['vector'];
      const vector = parseVector(vectorData);

      if (vector.length !== queryEmbedding.dimensions) {
        continue;
      }

      const similarity = cosineSimilarity(queryEmbedding.vector, vector);
      const distance = 1 - similarity;

      results.push({
        memoryId,
        distance,
        similarity,
      });
    }

    results.sort((a, b) => b.similarity - a.similarity);

    if (results.length >= limit) {
      const topResult = results[0];
      if (topResult && topResult.similarity >= earlyTerminationThreshold) {
        foundHighQualityResults = true;
        log.debug('search', 'Early termination triggered', {
          similarity: topResult.similarity,
          threshold: earlyTerminationThreshold,
          batchesProcessed: batchCount,
        });
        break;
      }
    }

    if (rowCount < batchSize) {
      break;
    }

    offset += batchSize;
  }

  const topResults = results.slice(0, limit);

  log.info('search', 'Batched vector search complete', {
    totalCandidates,
    batchesProcessed: batchCount,
    results: topResults.length,
    earlyTermination: foundHighQualityResults,
    ms: Date.now() - start,
  });

  return topResults;
}

export async function getCandidateIdsFromFTS(
  query: string,
  projectId?: string,
  maxCandidates = 5000,
): Promise<string[]> {
  const db = await getDatabase();

  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter(t => t.length > 2)
    .map(t => `"${t}"*`)
    .join(' OR ');

  if (!tokens) {
    return [];
  }

  let sql = `
    SELECT DISTINCT m.id
    FROM memories m
    JOIN memories_fts fts ON m.rowid = fts.rowid
    WHERE memories_fts MATCH ?
      AND m.is_deleted = 0
  `;
  const args: (string | number)[] = [tokens];

  if (projectId) {
    sql += ' AND m.project_id = ?';
    args.push(projectId);
  }

  sql += ' LIMIT ?';
  args.push(maxCandidates);

  const result = await db.execute(sql, args);
  return result.rows.map(row => String(row['id']));
}

export async function searchVectorOptimized(
  query: string,
  embeddingService: EmbeddingService,
  projectId?: string,
  limit = 20,
  options: BatchedSearchOptions = {},
): Promise<VectorResult[]> {
  const candidateIds = await getCandidateIdsFromFTS(query, projectId);

  log.debug('search', 'FTS pre-filter results', {
    candidates: candidateIds.length,
    query: query.slice(0, 50),
  });

  if (candidateIds.length > 0 && candidateIds.length < 5000) {
    return searchVectorBatched(query, embeddingService, projectId, limit, {
      ...options,
      candidateIds,
    });
  }

  return searchVectorBatched(query, embeddingService, projectId, limit, options);
}
