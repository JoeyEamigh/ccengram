import { getDatabase } from '../../db/database.js';
import { log } from '../../utils/log.js';
import type { Memory } from './types.js';
import { rowToMemory } from './utils.js';

const JACCARD_THRESHOLD = 0.8;

function fnv1a64(str: string): bigint {
  let hash = 14695981039346656037n;
  const mask = (1n << 64n) - 1n;
  for (let i = 0; i < str.length; i++) {
    hash ^= BigInt(str.charCodeAt(i));
    hash = (hash * 1099511628211n) & mask;
  }
  return hash;
}

export function computeSimhash(text: string): string {
  const tokens = text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 2);

  if (tokens.length === 0) {
    return '0000000000000000';
  }

  const vector: number[] = [];
  for (let i = 0; i < 64; i++) {
    vector.push(0);
  }

  for (const token of tokens) {
    const hash = fnv1a64(token);
    for (let i = 0; i < 64; i++) {
      const currentValue = vector[i] ?? 0;
      if ((hash >> BigInt(i)) & 1n) {
        vector[i] = currentValue + 1;
      } else {
        vector[i] = currentValue - 1;
      }
    }
  }

  let result = 0n;
  for (let i = 0; i < 64; i++) {
    const value = vector[i] ?? 0;
    if (value > 0) {
      result |= 1n << BigInt(i);
    }
  }

  return result.toString(16).padStart(16, '0');
}

export function hammingDistance(hash1: string, hash2: string): number {
  const h1 = BigInt('0x' + hash1);
  const h2 = BigInt('0x' + hash2);
  const xor = h1 ^ h2;

  let count = 0;
  let n = xor;
  while (n > 0n) {
    count += Number(n & 1n);
    n >>= 1n;
  }
  return count;
}

export function isDuplicate(hash1: string, hash2: string, threshold = 3): boolean {
  return hammingDistance(hash1, hash2) <= threshold;
}

export function getSimhashPrefix(simhash: string): string {
  return simhash.slice(0, 4);
}

function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(t => t.length > 2),
  );
}

export function computeJaccardSimilarity(text1: string, text2: string): number {
  const set1 = tokenize(text1);
  const set2 = tokenize(text2);

  if (set1.size === 0 && set2.size === 0) return 1.0;
  if (set1.size === 0 || set2.size === 0) return 0.0;

  let intersection = 0;
  for (const token of set1) {
    if (set2.has(token)) intersection++;
  }

  const union = set1.size + set2.size - intersection;
  return intersection / union;
}

function getAdaptiveThreshold(contentLength: number): number {
  if (contentLength < 50) return 2;
  if (contentLength < 200) return 3;
  if (contentLength < 500) return 4;
  return 5;
}

export async function findSimilarMemory(simhash: string, projectId: string, threshold = 3): Promise<Memory | null> {
  const db = await getDatabase();
  const prefix = getSimhashPrefix(simhash);

  const result = await db.execute(
    `SELECT * FROM memories
     WHERE project_id = ?
       AND is_deleted = 0
       AND simhash IS NOT NULL
       AND (simhash_prefix = ? OR simhash_prefix IS NULL)
     ORDER BY created_at DESC`,
    [projectId, prefix],
  );

  log.debug('dedup', 'Simhash lookup', { projectId, prefix, candidates: result.rows.length });

  for (const row of result.rows) {
    const rowSimhash = row['simhash'];
    if (typeof rowSimhash === 'string' && isDuplicate(simhash, rowSimhash, threshold)) {
      return rowToMemory(row);
    }
  }

  return null;
}

export type DuplicateCheckResult = {
  isDuplicate: boolean;
  existingMemory?: Memory;
  matchType?: 'exact' | 'simhash' | 'jaccard';
};

export async function checkDuplicate(
  content: string,
  contentHash: string,
  simhash: string,
  projectId: string,
): Promise<DuplicateCheckResult> {
  const db = await getDatabase();

  const exactMatch = await db.execute(
    `SELECT * FROM memories
     WHERE project_id = ?
       AND is_deleted = 0
       AND content_hash = ?
     LIMIT 1`,
    [projectId, contentHash],
  );

  if (exactMatch.rows.length > 0) {
    log.debug('dedup', 'Exact content hash match found', { projectId, contentHash });
    return {
      isDuplicate: true,
      existingMemory: rowToMemory(exactMatch.rows[0]!),
      matchType: 'exact',
    };
  }

  const prefix = getSimhashPrefix(simhash);
  const threshold = getAdaptiveThreshold(content.length);

  const bucketResult = await db.execute(
    `SELECT * FROM memories
     WHERE project_id = ?
       AND is_deleted = 0
       AND simhash_prefix = ?
     ORDER BY created_at DESC`,
    [projectId, prefix],
  );

  log.debug('dedup', 'Simhash bucket lookup', { projectId, prefix, threshold, candidates: bucketResult.rows.length });

  for (const row of bucketResult.rows) {
    const rowSimhash = row['simhash'];
    const rowContent = row['content'];

    if (typeof rowSimhash !== 'string') continue;

    const distance = hammingDistance(simhash, rowSimhash);

    if (distance <= threshold) {
      if (distance <= 2) {
        return {
          isDuplicate: true,
          existingMemory: rowToMemory(row),
          matchType: 'simhash',
        };
      }

      if (typeof rowContent === 'string') {
        const jaccard = computeJaccardSimilarity(content, rowContent);
        if (jaccard >= JACCARD_THRESHOLD) {
          log.debug('dedup', 'Jaccard verification passed', { jaccard, threshold: JACCARD_THRESHOLD });
          return {
            isDuplicate: true,
            existingMemory: rowToMemory(row),
            matchType: 'jaccard',
          };
        }
      }
    }
  }

  return { isDuplicate: false };
}

export async function computeMD5(text: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}
