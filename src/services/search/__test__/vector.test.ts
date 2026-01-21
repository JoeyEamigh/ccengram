import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../../db/database.js';
import type { EmbeddingResult, EmbeddingService } from '../../embedding/types.js';
import { searchVector } from '../vector.js';

function createMockEmbeddingService(): EmbeddingService {
  const mockVectors: Record<string, number[]> = {
    default: Array(128).fill(0.1),
    'login system security': [0.5, 0.8, 0.3, ...Array(125).fill(0.1)],
    'database design': [0.1, 0.2, 0.9, ...Array(125).fill(0.1)],
    authentication: [0.6, 0.7, 0.2, ...Array(125).fill(0.1)],
  };

  return {
    getProvider: () => ({
      name: 'mock',
      model: 'test-model',
      dimensions: 128,
      embed: async () => mockVectors['default'] ?? [],
      embedBatch: async () => [],
      isAvailable: async () => true,
    }),
    embed: async (text: string): Promise<EmbeddingResult> => {
      const vector = mockVectors[text.toLowerCase()] || mockVectors['default'];
      return {
        vector: vector ?? [],
        model: 'test-model',
        dimensions: 128,
        cached: false,
      };
    },
    embedBatch: async (texts: string[]): Promise<EmbeddingResult[]> => {
      return texts.map(text => ({
        vector: mockVectors[text.toLowerCase()] ?? mockVectors['default'] ?? [],
        model: 'test-model',
        dimensions: 128,
        cached: false,
      }));
    },
    getActiveModelId: () => 'mock:test-model',
    switchProvider: async () => {},
  };
}

describe('Vector Search', () => {
  let db: Database;
  let embeddingService: EmbeddingService;

  beforeEach(async () => {
    db = await createDatabase(':memory:');
    setDatabase(db);
    embeddingService = createMockEmbeddingService();

    const now = Date.now();
    await db.execute(`INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      'proj1',
      '/test/path',
      'Test Project',
      now,
      now,
    ]);
    await db.execute(`INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      'proj2',
      '/test/path2',
      'Test Project 2',
      now,
      now,
    ]);

    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ['mock:test-model', 'test-model', 'mock', 128, 1, now],
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  async function insertMemoryWithVector(
    id: string,
    content: string,
    projectId: string,
    vector: number[],
  ): Promise<void> {
    const now = Date.now();
    await db.execute(
      `INSERT INTO memories (
        id, project_id, content, sector, tier, importance,
        salience, access_count, created_at, updated_at, last_accessed,
        is_deleted, tags_json, concepts_json, files_json, categories_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [id, projectId, content, 'semantic', 'project', 0.5, 1.0, 0, now, now, now, 0, '[]', '[]', '[]', '[]'],
    );

    const vectorBuffer = new Float32Array(vector).buffer;
    await db.execute(
      `INSERT INTO memory_vectors (memory_id, model_id, vector, dim, created_at)
       VALUES (?, ?, ?, ?, ?)`,
      [id, 'mock:test-model', new Uint8Array(vectorBuffer), vector.length, now],
    );
  }

  test('finds semantically similar memories', async () => {
    const authVector = [0.6, 0.7, 0.2, ...Array(125).fill(0.1)];
    const dbVector = [0.1, 0.2, 0.9, ...Array(125).fill(0.1)];

    await insertMemoryWithVector('mem1', 'User authentication with JWT', 'proj1', authVector);
    await insertMemoryWithVector('mem2', 'Database schema design', 'proj1', dbVector);

    const results = await searchVector('login system security', embeddingService, 'proj1');

    expect(results.length).toBe(2);
    expect(results[0]?.memoryId).toBe('mem1');
  });

  test('returns similarity scores between 0 and 1', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Test memory', 'proj1', vector);

    const results = await searchVector('authentication', embeddingService, 'proj1');

    expect(results.length).toBeGreaterThan(0);
    expect(results[0]?.similarity).toBeGreaterThan(0);
    expect(results[0]?.similarity).toBeLessThanOrEqual(1);
  });

  test('filters by project', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Memory in proj1', 'proj1', vector);
    await insertMemoryWithVector('mem2', 'Memory in proj2', 'proj2', vector);

    const results = await searchVector('authentication', embeddingService, 'proj1');

    expect(results.length).toBe(1);
    expect(results[0]?.memoryId).toBe('mem1');
  });

  test('searches all projects when no filter', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Memory in proj1', 'proj1', vector);
    await insertMemoryWithVector('mem2', 'Memory in proj2', 'proj2', vector);

    const results = await searchVector('authentication', embeddingService);

    expect(results.length).toBe(2);
  });

  test('excludes deleted memories', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Active memory', 'proj1', vector);
    await insertMemoryWithVector('mem2', 'Deleted memory', 'proj1', vector);
    await db.execute('UPDATE memories SET is_deleted = 1 WHERE id = ?', ['mem2']);

    const results = await searchVector('authentication', embeddingService, 'proj1');

    expect(results.length).toBe(1);
    expect(results[0]?.memoryId).toBe('mem1');
  });

  test('respects limit parameter', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Memory 1', 'proj1', vector);
    await insertMemoryWithVector('mem2', 'Memory 2', 'proj1', vector);
    await insertMemoryWithVector('mem3', 'Memory 3', 'proj1', vector);

    const results = await searchVector('authentication', embeddingService, 'proj1', 2);

    expect(results.length).toBe(2);
  });

  test('returns empty array when no vectors exist', async () => {
    const results = await searchVector('authentication', embeddingService, 'proj1');

    expect(results).toEqual([]);
  });

  test('orders results by similarity descending', async () => {
    const highSimilarity = [0.1, 0.1, 0.1, ...Array(125).fill(0.1)];
    const lowSimilarity = [0.9, 0.9, 0.9, ...Array(125).fill(0.9)];

    await insertMemoryWithVector('mem1', 'Low similarity', 'proj1', lowSimilarity);
    await insertMemoryWithVector('mem2', 'High similarity', 'proj1', highSimilarity);

    const results = await searchVector('authentication', embeddingService, 'proj1');

    expect(results[0]?.memoryId).toBe('mem2');
    expect(results[0]?.similarity).toBeGreaterThan(results[1]?.similarity ?? 0);
  });
});
