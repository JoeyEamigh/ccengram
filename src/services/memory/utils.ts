import type { Row } from '@libsql/client';
import type { Memory, MemorySector, MemoryTier, MemoryType } from './types.js';

function parseJsonArray(value: unknown): string[] {
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        return parsed.filter((item): item is string => typeof item === 'string');
      }
    } catch {
      return [];
    }
  }
  return [];
}

function asNumber(value: unknown, defaultValue: number): number {
  if (typeof value === 'number') return value;
  if (typeof value === 'bigint') return Number(value);
  return defaultValue;
}

function asString(value: unknown, defaultValue: string): string {
  if (typeof value === 'string') return value;
  return defaultValue;
}

function asOptionalString(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  return undefined;
}

function asOptionalNumber(value: unknown): number | undefined {
  if (typeof value === 'number') return value;
  if (typeof value === 'bigint') return Number(value);
  return undefined;
}

export function rowToMemory(row: Row): Memory {
  return {
    id: asString(row['id'], ''),
    projectId: asString(row['project_id'], ''),
    content: asString(row['content'], ''),
    summary: asOptionalString(row['summary']),
    contentHash: asOptionalString(row['content_hash']),
    sector: asString(row['sector'], 'semantic') as MemorySector,
    tier: asString(row['tier'], 'project') as MemoryTier,
    importance: asNumber(row['importance'], 0.5),
    categories: parseJsonArray(row['categories_json']),
    memoryType: asOptionalString(row['memory_type']) as MemoryType | undefined,
    context: asOptionalString(row['context']),
    confidence: asNumber(row['confidence'], 0.5),
    segmentId: asOptionalString(row['segment_id']),
    simhash: asOptionalString(row['simhash']),
    salience: asNumber(row['salience'], 1.0),
    accessCount: asNumber(row['access_count'], 0),
    createdAt: asNumber(row['created_at'], Date.now()),
    updatedAt: asNumber(row['updated_at'], Date.now()),
    lastAccessed: asNumber(row['last_accessed'], Date.now()),
    validFrom: asOptionalNumber(row['valid_from']),
    validUntil: asOptionalNumber(row['valid_until']),
    isDeleted: Boolean(row['is_deleted']),
    deletedAt: asOptionalNumber(row['deleted_at']),
    embeddingModelId: asOptionalString(row['embedding_model_id']),
    tags: parseJsonArray(row['tags_json']),
    concepts: parseJsonArray(row['concepts_json']),
    files: parseJsonArray(row['files_json']),
    scopePath: asOptionalString(row['scope_path']),
    scopeModule: asOptionalString(row['scope_module']),
  };
}
