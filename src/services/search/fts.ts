import { getDatabase } from '../../db/database.js';
import { log } from '../../utils/log.js';

export type FTSResult = {
  memoryId: string;
  rank: number;
  snippet: string;
};

const MAX_QUERY_LENGTH = 10000;
const FTS5_SPECIAL_CHARS = /[*^:(){}[\]\-+'"\\]/g;

function sanitizeToken(token: string): string {
  return token.replace(FTS5_SPECIAL_CHARS, '');
}

function prepareQuery(query: string): string {
  const truncated = query.length > MAX_QUERY_LENGTH ? query.slice(0, MAX_QUERY_LENGTH) : query;

  const tokens = truncated
    .split(/\s+/)
    .map(t => sanitizeToken(t))
    .filter(t => t.length > 1)
    .map(t => `"${t}"*`)
    .join(' OR ');

  return tokens;
}

export async function searchFTS(query: string, projectId?: string, limit = 20): Promise<FTSResult[]> {
  const db = await getDatabase();
  const start = Date.now();

  const ftsQuery = prepareQuery(query);

  if (!ftsQuery) {
    log.debug('search', 'Empty FTS query', { original: query });
    return [];
  }

  log.debug('search', 'FTS search', { query: ftsQuery, projectId, limit });

  let sql: string;
  const args: (string | number)[] = [ftsQuery];

  if (projectId) {
    sql = `
      SELECT
        m.id as memory_id,
        bm25(memories_fts) as rank,
        snippet(memories_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
      FROM memories_fts
      JOIN memories m ON memories_fts.rowid = m.rowid
      WHERE memories_fts MATCH ?
        AND m.project_id = ?
        AND m.is_deleted = 0
      ORDER BY rank
      LIMIT ?
    `;
    args.push(projectId);
    args.push(limit);
  } else {
    sql = `
      SELECT
        m.id as memory_id,
        bm25(memories_fts) as rank,
        snippet(memories_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
      FROM memories_fts
      JOIN memories m ON memories_fts.rowid = m.rowid
      WHERE memories_fts MATCH ?
        AND m.is_deleted = 0
      ORDER BY rank
      LIMIT ?
    `;
    args.push(limit);
  }

  try {
    const result = await db.execute(sql, args);

    log.info('search', 'FTS search complete', {
      results: result.rows.length,
      ms: Date.now() - start,
    });

    return result.rows.map(row => ({
      memoryId: String(row['memory_id']),
      rank: Number(row['rank']),
      snippet: String(row['snippet']),
    }));
  } catch (error) {
    log.error('search', 'FTS search failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    return [];
  }
}
