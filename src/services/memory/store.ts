import { getDatabase } from '../../db/database.js';
import { log } from '../../utils/log.js';
import { publishEvent } from '../events/pubsub.js';
import { computeMD5, computeSimhash, findSimilarMemory, getSimhashPrefix } from './dedup.js';
import type { ListOptions, Memory, MemoryInput, UsageType } from './types.js';
import { classifyMemorySector, MEMORY_TYPE_TO_SECTOR } from './types.js';
import { rowToMemory } from './utils.js';

export type MemoryStore = {
  create(input: MemoryInput, projectId: string, sessionId?: string): Promise<Memory>;
  get(id: string): Promise<Memory | null>;
  update(id: string, updates: Partial<MemoryInput>): Promise<Memory>;
  delete(id: string, hard?: boolean): Promise<void>;
  restore(id: string): Promise<Memory>;
  list(options: ListOptions): Promise<Memory[]>;
  touch(id: string): Promise<void>;
  reinforce(id: string, amount?: number): Promise<Memory>;
  deemphasize(id: string, amount?: number): Promise<Memory>;
  linkToSession(memoryId: string, sessionId: string, usageType: UsageType): Promise<void>;
  getBySession(sessionId: string): Promise<Memory[]>;
};

function extractConcepts(content: string): string[] {
  const concepts: string[] = [];

  const codePatterns = [
    /`([^`]+)`/g,
    /\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b/g,
    /\b([a-z]+_[a-z_]+)\b/g,
    /\/([\w\-./]+\.\w+)/g,
  ];

  for (const pattern of codePatterns) {
    const matches = content.matchAll(pattern);
    for (const match of matches) {
      const concept = match[1];
      if (concept && concept.length > 2 && concept.length < 50) {
        concepts.push(concept);
      }
    }
  }

  const uniqueConcepts = [...new Set(concepts)];
  return uniqueConcepts.slice(0, 20);
}

export function createMemoryStore(): MemoryStore {
  const store: MemoryStore = {
    async create(input: MemoryInput, projectId: string, sessionId?: string): Promise<Memory> {
      const db = await getDatabase();
      const id = crypto.randomUUID();
      const now = Date.now();

      const sector = input.memoryType
        ? MEMORY_TYPE_TO_SECTOR[input.memoryType]
        : input.sector || classifyMemorySector(input.content);
      const tier = input.tier || 'project';
      const importance = input.importance ?? 0.5;

      const simhash = computeSimhash(input.content);
      const contentHash = await computeMD5(input.content);

      log.debug('memory', 'Creating memory', {
        sector,
        tier,
        projectId,
        simhash: simhash.slice(0, 8),
      });

      const existing = await findSimilarMemory(simhash, projectId);
      if (existing && !existing.isDeleted) {
        log.info('memory', 'Duplicate detected, reinforcing existing', {
          existingId: existing.id,
        });
        await store.reinforce(existing.id, 0.1);
        if (sessionId) {
          await store.linkToSession(existing.id, sessionId, 'reinforced');
        }
        const reinforced = await store.get(existing.id);
        if (!reinforced) {
          throw new Error('Failed to get reinforced memory');
        }
        return reinforced;
      }

      const concepts = input.concepts && input.concepts.length > 0
        ? input.concepts
        : extractConcepts(input.content);
      const confidence = input.confidence ?? 0.5;
      const summary = input.summary ?? null;

      const simhashPrefix = getSimhashPrefix(simhash);

      await db.execute(
        `INSERT INTO memories (
          id, project_id, content, summary, content_hash, sector, tier, importance,
          simhash, simhash_prefix, salience, access_count, created_at, updated_at,
          last_accessed, valid_from, is_deleted,
          tags_json, concepts_json, files_json, categories_json,
          memory_type, context, confidence, segment_id, scope_path, scope_module
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          id,
          projectId,
          input.content,
          summary,
          contentHash,
          sector,
          tier,
          importance,
          simhash,
          simhashPrefix,
          1.0,
          0,
          now,
          now,
          now,
          input.validFrom ?? null,
          0,
          JSON.stringify(input.tags || []),
          JSON.stringify(concepts),
          JSON.stringify(input.files || []),
          JSON.stringify([]),
          input.memoryType ?? null,
          input.context ?? null,
          confidence,
          input.segmentId ?? null,
          input.scopePath ?? null,
          input.scopeModule ?? null,
        ],
      );

      if (sessionId) {
        await store.linkToSession(id, sessionId, 'created');
      }

      log.info('memory', 'Memory created', {
        id,
        sector,
        tier,
        concepts: concepts.length,
      });

      const created = await store.get(id);
      if (!created) {
        throw new Error('Failed to get created memory');
      }

      publishEvent({
        type: 'memory:created',
        memoryId: id,
        projectId,
        timestamp: now,
      }).catch((err: unknown) => {
        log.debug('memory', 'Failed to publish memory:created event', {
          memoryId: id,
          error: err instanceof Error ? err.message : String(err),
        });
      });

      return created;
    },

    async get(id: string): Promise<Memory | null> {
      const db = await getDatabase();

      const result = await db.execute('SELECT * FROM memories WHERE id = ?', [id]);

      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;
      return rowToMemory(row);
    },

    async update(id: string, updates: Partial<MemoryInput>): Promise<Memory> {
      const db = await getDatabase();
      const now = Date.now();

      const setClauses: string[] = [];
      const args: (string | number | null)[] = [];

      if (updates.content !== undefined) {
        setClauses.push('content = ?');
        args.push(updates.content);

        const contentHash = await computeMD5(updates.content);
        setClauses.push('content_hash = ?');
        args.push(contentHash);

        const simhash = computeSimhash(updates.content);
        setClauses.push('simhash = ?');
        args.push(simhash);

        const concepts = extractConcepts(updates.content);
        setClauses.push('concepts_json = ?');
        args.push(JSON.stringify(concepts));
      }

      if (updates.sector !== undefined) {
        setClauses.push('sector = ?');
        args.push(updates.sector);
      }

      if (updates.tier !== undefined) {
        setClauses.push('tier = ?');
        args.push(updates.tier);
      }

      if (updates.importance !== undefined) {
        setClauses.push('importance = ?');
        args.push(updates.importance);
      }

      if (updates.tags !== undefined) {
        setClauses.push('tags_json = ?');
        args.push(JSON.stringify(updates.tags));
      }

      if (updates.files !== undefined) {
        setClauses.push('files_json = ?');
        args.push(JSON.stringify(updates.files));
      }

      if (updates.validFrom !== undefined) {
        setClauses.push('valid_from = ?');
        args.push(updates.validFrom);
      }

      if (updates.scopePath !== undefined) {
        setClauses.push('scope_path = ?');
        args.push(updates.scopePath);
      }

      if (updates.scopeModule !== undefined) {
        setClauses.push('scope_module = ?');
        args.push(updates.scopeModule);
      }

      setClauses.push('updated_at = ?');
      args.push(now);
      args.push(id);

      await db.execute(`UPDATE memories SET ${setClauses.join(', ')} WHERE id = ?`, args);

      log.info('memory', 'Memory updated', { id });

      const updated = await store.get(id);
      if (!updated) {
        throw new Error('Failed to get updated memory');
      }
      return updated;
    },

    async delete(id: string, hard = false): Promise<void> {
      const db = await getDatabase();
      log.info('memory', 'Deleting memory', { id, hard });

      const memory = await store.get(id);
      const projectId = memory?.projectId ?? '';

      if (hard) {
        await db.execute('DELETE FROM memories WHERE id = ?', [id]);
      } else {
        const now = Date.now();
        await db.execute(`UPDATE memories SET is_deleted = 1, deleted_at = ?, updated_at = ? WHERE id = ?`, [
          now,
          now,
          id,
        ]);
      }

      if (projectId) {
        publishEvent({
          type: 'memory:deleted',
          memoryId: id,
          projectId,
          timestamp: Date.now(),
        }).catch((err: unknown) => {
          log.debug('memory', 'Failed to publish memory:deleted event', {
            memoryId: id,
            error: err instanceof Error ? err.message : String(err),
          });
        });
      }
    },

    async restore(id: string): Promise<Memory> {
      const db = await getDatabase();
      log.info('memory', 'Restoring memory', { id });

      const now = Date.now();
      await db.execute(`UPDATE memories SET is_deleted = 0, deleted_at = NULL, updated_at = ? WHERE id = ?`, [now, id]);

      const restored = await store.get(id);
      if (!restored) {
        throw new Error('Failed to get restored memory');
      }
      return restored;
    },

    async list(options: ListOptions): Promise<Memory[]> {
      const db = await getDatabase();

      let sql = 'SELECT * FROM memories WHERE 1=1';
      const args: (string | number)[] = [];

      if (options.projectId) {
        sql += ' AND project_id = ?';
        args.push(options.projectId);
      }

      if (!options.includeDeleted) {
        sql += ' AND is_deleted = 0';
      }

      if (options.sector) {
        sql += ' AND sector = ?';
        args.push(options.sector);
      }

      if (options.tier) {
        sql += ' AND tier = ?';
        args.push(options.tier);
      }

      if (options.memoryType) {
        sql += ' AND memory_type = ?';
        args.push(options.memoryType);
      }

      if (options.minSalience !== undefined) {
        sql += ' AND salience >= ?';
        args.push(options.minSalience);
      }

      if (options.scopePath) {
        sql += ' AND scope_path = ?';
        args.push(options.scopePath);
      }

      if (options.scopeModule) {
        sql += ' AND scope_module = ?';
        args.push(options.scopeModule);
      }

      const orderBy = options.orderBy || 'created_at';
      const order = options.order || 'desc';
      sql += ` ORDER BY ${orderBy} ${order.toUpperCase()}`;

      if (options.limit !== undefined) {
        sql += ' LIMIT ?';
        args.push(options.limit);
      }

      if (options.offset !== undefined) {
        sql += ' OFFSET ?';
        args.push(options.offset);
      }

      const result = await db.execute(sql, args);
      return result.rows.map(rowToMemory);
    },

    async touch(id: string): Promise<void> {
      const db = await getDatabase();
      const now = Date.now();

      await db.execute(
        `UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ? AND is_deleted = 0`,
        [now, id],
      );

      log.debug('memory', 'Touched memory', { id });
    },

    async reinforce(id: string, amount = 0.1): Promise<Memory> {
      const db = await getDatabase();
      const now = Date.now();

      log.debug('memory', 'Reinforcing memory', { id, amount });

      await db.execute(
        `UPDATE memories
         SET salience = MIN(1.0, salience + ? * (1.0 - salience)),
             last_accessed = ?,
             access_count = access_count + 1,
             updated_at = ?
         WHERE id = ? AND is_deleted = 0`,
        [amount, now, now, id],
      );

      const reinforced = await store.get(id);
      if (!reinforced) {
        throw new Error('Failed to get reinforced memory');
      }

      publishEvent({
        type: 'memory:reinforced',
        memoryId: id,
        projectId: reinforced.projectId,
        timestamp: now,
      }).catch((err: unknown) => {
        log.debug('memory', 'Failed to publish memory:reinforced event', {
          memoryId: id,
          error: err instanceof Error ? err.message : String(err),
        });
      });

      return reinforced;
    },

    async deemphasize(id: string, amount = 0.2): Promise<Memory> {
      const db = await getDatabase();
      const now = Date.now();

      log.debug('memory', 'De-emphasizing memory', { id, amount });

      await db.execute(
        `UPDATE memories
         SET salience = MAX(0.05, salience - ?),
             updated_at = ?
         WHERE id = ? AND is_deleted = 0`,
        [amount, now, id],
      );

      const deemphasized = await store.get(id);
      if (!deemphasized) {
        throw new Error('Failed to get de-emphasized memory');
      }
      return deemphasized;
    },

    async linkToSession(memoryId: string, sessionId: string, usageType: UsageType): Promise<void> {
      const db = await getDatabase();
      log.debug('memory', 'Linking memory to session', {
        memoryId,
        sessionId,
        usageType,
      });

      const now = Date.now();
      await db.execute(
        `INSERT OR IGNORE INTO session_memories (session_id, memory_id, created_at, usage_type)
         VALUES (?, ?, ?, ?)`,
        [sessionId, memoryId, now, usageType],
      );
    },

    async getBySession(sessionId: string): Promise<Memory[]> {
      const db = await getDatabase();

      const result = await db.execute(
        `SELECT m.* FROM memories m
         JOIN session_memories sm ON sm.memory_id = m.id
         WHERE sm.session_id = ?
         ORDER BY sm.created_at DESC`,
        [sessionId],
      );

      return result.rows.map(rowToMemory);
    },
  };

  return store;
}
