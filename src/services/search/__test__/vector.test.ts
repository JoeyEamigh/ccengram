import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../../db/database.js';
import type { EmbeddingResult, EmbeddingService } from '../../embedding/types.js';
import { getCandidateIdsFromFTS, searchVector, searchVectorBatched, searchVectorOptimized } from '../vector.js';

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

describe('Batched Vector Search', () => {
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

  test('processes vectors in batches', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    for (let i = 0; i < 5; i++) {
      await insertMemoryWithVector(`mem${i}`, `Memory ${i}`, 'proj1', vector);
    }

    const results = await searchVectorBatched('authentication', embeddingService, 'proj1', 10, {
      batchSize: 2,
    });

    expect(results.length).toBe(5);
  });

  test('respects batch size limit', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    for (let i = 0; i < 10; i++) {
      await insertMemoryWithVector(`mem${i}`, `Memory ${i}`, 'proj1', vector);
    }

    const results = await searchVectorBatched('authentication', embeddingService, 'proj1', 5, {
      batchSize: 3,
    });

    expect(results.length).toBe(5);
  });

  test('early termination triggers on high quality result', async () => {
    const defaultQueryVector = Array(128).fill(0.1);
    const orthogonalVector = Array(64).fill(1).concat(Array(64).fill(-1));

    await insertMemoryWithVector('mem_high', 'High match', 'proj1', defaultQueryVector);
    for (let i = 0; i < 20; i++) {
      await insertMemoryWithVector(`mem_low_${i}`, `Low match ${i}`, 'proj1', orthogonalVector);
    }

    const results = await searchVectorBatched('authentication', embeddingService, 'proj1', 5, {
      batchSize: 5,
      earlyTerminationThreshold: 0.9,
    });

    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0]?.memoryId).toBe('mem_high');
    expect(results[0]?.similarity).toBeGreaterThanOrEqual(0.9);
  });

  test('respects maxBatches limit', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    for (let i = 0; i < 100; i++) {
      await insertMemoryWithVector(`mem${i}`, `Memory ${i}`, 'proj1', vector);
    }

    const results = await searchVectorBatched('authentication', embeddingService, 'proj1', 50, {
      batchSize: 10,
      maxBatches: 3,
    });

    expect(results.length).toBeLessThanOrEqual(30);
  });

  test('filters by candidate IDs when provided', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Memory 1', 'proj1', vector);
    await insertMemoryWithVector('mem2', 'Memory 2', 'proj1', vector);
    await insertMemoryWithVector('mem3', 'Memory 3', 'proj1', vector);

    const results = await searchVectorBatched('authentication', embeddingService, 'proj1', 10, {
      candidateIds: ['mem1', 'mem3'],
    });

    expect(results.length).toBe(2);
    const memoryIds = results.map(r => r.memoryId);
    expect(memoryIds).toContain('mem1');
    expect(memoryIds).toContain('mem3');
    expect(memoryIds).not.toContain('mem2');
  });

  test('returns same results as non-batched search', async () => {
    const vectors = [
      [0.6, 0.7, 0.2, ...Array(125).fill(0.1)],
      [0.1, 0.2, 0.9, ...Array(125).fill(0.1)],
      [0.5, 0.5, 0.5, ...Array(125).fill(0.1)],
    ];

    await insertMemoryWithVector('mem1', 'Auth memory', 'proj1', vectors[0]!);
    await insertMemoryWithVector('mem2', 'DB memory', 'proj1', vectors[1]!);
    await insertMemoryWithVector('mem3', 'Mixed memory', 'proj1', vectors[2]!);

    const regularResults = await searchVector('authentication', embeddingService, 'proj1', 3);
    const batchedResults = await searchVectorBatched('authentication', embeddingService, 'proj1', 3);

    expect(batchedResults.map(r => r.memoryId)).toEqual(regularResults.map(r => r.memoryId));
  });
});

describe('FTS Pre-filtering', () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(':memory:');
    setDatabase(db);

    const now = Date.now();
    await db.execute(`INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      'proj1',
      '/test/path',
      'Test Project',
      now,
      now,
    ]);
  });

  afterEach(() => {
    closeDatabase();
  });

  async function insertMemory(id: string, content: string, projectId: string): Promise<void> {
    const now = Date.now();
    await db.execute(
      `INSERT INTO memories (
        id, project_id, content, sector, tier, importance,
        salience, access_count, created_at, updated_at, last_accessed,
        is_deleted, tags_json, concepts_json, files_json, categories_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [id, projectId, content, 'semantic', 'project', 0.5, 1.0, 0, now, now, now, 0, '[]', '[]', '[]', '[]'],
    );
  }

  test('returns matching memory IDs from FTS', async () => {
    await insertMemory('mem1', 'User authentication with JWT tokens', 'proj1');
    await insertMemory('mem2', 'Database schema design patterns', 'proj1');
    await insertMemory('mem3', 'Authentication security best practices', 'proj1');

    const candidates = await getCandidateIdsFromFTS('authentication', 'proj1');

    expect(candidates).toContain('mem1');
    expect(candidates).toContain('mem3');
    expect(candidates).not.toContain('mem2');
  });

  test('returns empty array for query with no matches', async () => {
    await insertMemory('mem1', 'User authentication with JWT tokens', 'proj1');

    const candidates = await getCandidateIdsFromFTS('nonexistent', 'proj1');

    expect(candidates).toEqual([]);
  });

  test('respects maxCandidates limit', async () => {
    for (let i = 0; i < 10; i++) {
      await insertMemory(`mem${i}`, `Authentication method ${i}`, 'proj1');
    }

    const candidates = await getCandidateIdsFromFTS('authentication', 'proj1', 5);

    expect(candidates.length).toBeLessThanOrEqual(5);
  });

  test('filters by project', async () => {
    const now = Date.now();
    await db.execute(`INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      'proj2',
      '/test/path2',
      'Test Project 2',
      now,
      now,
    ]);

    await insertMemory('mem1', 'Authentication in project 1', 'proj1');
    await insertMemory('mem2', 'Authentication in project 2', 'proj2');

    const candidates = await getCandidateIdsFromFTS('authentication', 'proj1');

    expect(candidates).toContain('mem1');
    expect(candidates).not.toContain('mem2');
  });

  test('returns empty for short query words only', async () => {
    await insertMemory('mem1', 'User authentication', 'proj1');

    const candidates = await getCandidateIdsFromFTS('a to', 'proj1');

    expect(candidates).toEqual([]);
  });
});

describe('Optimized Vector Search', () => {
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

  test('uses FTS pre-filtering when candidates available', async () => {
    const authVector = [0.6, 0.7, 0.2, ...Array(125).fill(0.1)];
    const dbVector = [0.1, 0.2, 0.9, ...Array(125).fill(0.1)];

    await insertMemoryWithVector('mem1', 'User authentication module', 'proj1', authVector);
    await insertMemoryWithVector('mem2', 'Database optimization guide', 'proj1', dbVector);

    const results = await searchVectorOptimized('authentication', embeddingService, 'proj1');

    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0]?.memoryId).toBe('mem1');
  });

  test('falls back to full batched search when no FTS matches', async () => {
    const vector = [0.5, 0.5, ...Array(126).fill(0.1)];
    await insertMemoryWithVector('mem1', 'Some content here', 'proj1', vector);

    const results = await searchVectorOptimized('completely different query', embeddingService, 'proj1');

    expect(results.length).toBe(1);
  });

  test('returns accurate results for mixed queries', async () => {
    const vectors = [
      [0.6, 0.7, 0.2, ...Array(125).fill(0.1)],
      [0.5, 0.6, 0.3, ...Array(125).fill(0.1)],
      [0.1, 0.2, 0.9, ...Array(125).fill(0.1)],
    ];

    await insertMemoryWithVector('mem1', 'User login authentication flow', 'proj1', vectors[0]!);
    await insertMemoryWithVector('mem2', 'Session authentication tokens', 'proj1', vectors[1]!);
    await insertMemoryWithVector('mem3', 'Database design patterns', 'proj1', vectors[2]!);

    const results = await searchVectorOptimized('authentication', embeddingService, 'proj1', 3);

    const memoryIds = results.map(r => r.memoryId);
    expect(memoryIds).toContain('mem1');
    expect(memoryIds).toContain('mem2');
  });
});
