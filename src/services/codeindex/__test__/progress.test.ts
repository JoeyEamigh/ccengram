import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../../db/database.js';
import type { EmbeddingResult, EmbeddingService } from '../../embedding/types.js';
import { createCodeIndexService, type CodeIndexService } from '../index.js';
import type { IndexProgress, IndexStatistics } from '../types.js';

function createMockEmbeddingService(): EmbeddingService {
  return {
    getProvider: () => ({
      name: 'mock',
      model: 'test-model',
      dimensions: 128,
      embed: async () => Array(128).fill(0.1),
      embedBatch: async () => [],
      isAvailable: async () => true,
    }),
    embed: async (): Promise<EmbeddingResult> => ({
      vector: Array(128).fill(0.1),
      model: 'test-model',
      dimensions: 128,
      cached: false,
    }),
    embedBatch: async (texts: string[]): Promise<EmbeddingResult[]> => {
      return texts.map(() => ({
        vector: Array(128).fill(0.1),
        model: 'test-model',
        dimensions: 128,
        cached: false,
      }));
    },
    getActiveModelId: () => 'mock:test-model',
    switchProvider: async () => {},
  };
}

describe('Code Index Progress and Statistics', () => {
  let db: Database;
  let service: CodeIndexService;

  beforeEach(async () => {
    db = await createDatabase(':memory:');
    setDatabase(db);
    service = createCodeIndexService(createMockEmbeddingService());

    const now = Date.now();
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ['test-project', '/test/path', 'Test Project', now, now],
    );

    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ['mock:test-model', 'test-model', 'mock', 128, 1, now],
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  describe('IndexProgress', () => {
    test('progress includes timing information', async () => {
      let capturedProgress: IndexProgress | null = null;

      await service.index('/tmp', 'test-project', {
        dryRun: true,
        onProgress: (p) => {
          capturedProgress = p;
        },
      });

      expect(capturedProgress).not.toBeNull();
      expect(capturedProgress!.startedAt).toBeDefined();
      expect(capturedProgress!.phase).toBe('complete');
    });

    test('progress tracks bytes and tokens', async () => {
      let capturedProgress: IndexProgress | null = null;

      await service.index('/tmp', 'test-project', {
        dryRun: true,
        onProgress: (p) => {
          capturedProgress = p;
        },
      });

      expect(capturedProgress).not.toBeNull();
      expect(capturedProgress!.bytesProcessed).toBeDefined();
      expect(capturedProgress!.totalBytes).toBeDefined();
    });
  });

  describe('Checkpoints', () => {
    test('saveCheckpoint creates a new checkpoint', async () => {
      const progress: IndexProgress = {
        phase: 'indexing',
        scannedFiles: 10,
        indexedFiles: 5,
        totalFiles: 10,
        errors: [],
        startedAt: Date.now(),
      };

      const id = await service.saveCheckpoint('test-project', {
        projectId: 'test-project',
        phase: 'indexing',
        processedFiles: ['/file1.ts', '/file2.ts'],
        pendingFiles: ['/file3.ts'],
        progress,
      });

      expect(id).toBeDefined();
      expect(typeof id).toBe('string');
    });

    test('loadCheckpoint retrieves saved checkpoint', async () => {
      const progress: IndexProgress = {
        phase: 'indexing',
        scannedFiles: 10,
        indexedFiles: 5,
        totalFiles: 10,
        errors: ['test error'],
        startedAt: Date.now(),
      };

      await service.saveCheckpoint('test-project', {
        projectId: 'test-project',
        phase: 'indexing',
        processedFiles: ['/file1.ts', '/file2.ts'],
        pendingFiles: ['/file3.ts'],
        progress,
      });

      const loaded = await service.loadCheckpoint('test-project');

      expect(loaded).not.toBeNull();
      expect(loaded!.projectId).toBe('test-project');
      expect(loaded!.phase).toBe('indexing');
      expect(loaded!.processedFiles).toEqual(['/file1.ts', '/file2.ts']);
      expect(loaded!.pendingFiles).toEqual(['/file3.ts']);
      expect(loaded!.progress.indexedFiles).toBe(5);
    });

    test('loadCheckpoint returns null when no checkpoint exists', async () => {
      const loaded = await service.loadCheckpoint('nonexistent-project');
      expect(loaded).toBeNull();
    });

    test('clearCheckpoint removes checkpoint', async () => {
      const progress: IndexProgress = {
        phase: 'indexing',
        scannedFiles: 10,
        indexedFiles: 5,
        totalFiles: 10,
        errors: [],
      };

      await service.saveCheckpoint('test-project', {
        projectId: 'test-project',
        phase: 'indexing',
        processedFiles: ['/file1.ts'],
        pendingFiles: [],
        progress,
      });

      await service.clearCheckpoint('test-project');

      const loaded = await service.loadCheckpoint('test-project');
      expect(loaded).toBeNull();
    });

    test('saveCheckpoint replaces existing checkpoint', async () => {
      const progress1: IndexProgress = {
        phase: 'indexing',
        scannedFiles: 5,
        indexedFiles: 2,
        totalFiles: 10,
        errors: [],
      };

      const progress2: IndexProgress = {
        phase: 'indexing',
        scannedFiles: 10,
        indexedFiles: 8,
        totalFiles: 10,
        errors: [],
      };

      await service.saveCheckpoint('test-project', {
        projectId: 'test-project',
        phase: 'indexing',
        processedFiles: ['/file1.ts'],
        pendingFiles: ['/file2.ts'],
        progress: progress1,
      });

      await service.saveCheckpoint('test-project', {
        projectId: 'test-project',
        phase: 'indexing',
        processedFiles: ['/file1.ts', '/file2.ts', '/file3.ts'],
        pendingFiles: [],
        progress: progress2,
      });

      const loaded = await service.loadCheckpoint('test-project');
      expect(loaded!.processedFiles.length).toBe(3);
      expect(loaded!.progress.indexedFiles).toBe(8);
    });
  });

  describe('getStatistics', () => {
    test('returns null for non-indexed project', async () => {
      const stats = await service.getStatistics('nonexistent');
      expect(stats).toBeNull();
    });

    test('returns statistics for indexed project', async () => {
      const now = Date.now();
      await db.execute(
        `INSERT INTO code_index_state (project_id, last_indexed_at, indexed_files, gitignore_hash, total_bytes, total_tokens, total_chunks)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        ['test-project', now, 10, 'abc123', 50000, 12500, 25],
      );

      await db.execute(
        `INSERT INTO documents (id, project_id, source_path, source_type, title, full_content, checksum, created_at, updated_at, language, line_count, is_code)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`,
        ['doc1', 'test-project', '/test/file.ts', 'code', 'file.ts', 'content', 'hash1', now, now, 'ts', 100],
      );

      const stats = await service.getStatistics('test-project');

      expect(stats).not.toBeNull();
      expect(stats!.projectId).toBe('test-project');
      expect(stats!.totalFiles).toBe(10);
      expect(stats!.totalBytes).toBe(50000);
      expect(stats!.totalTokens).toBe(12500);
      expect(stats!.totalChunks).toBe(25);
      expect(stats!.averageChunksPerFile).toBe(2.5);
      expect(stats!.indexHealthScore).toBeGreaterThan(0);
    });

    test('returns language breakdown', async () => {
      const now = Date.now();
      await db.execute(
        `INSERT INTO code_index_state (project_id, last_indexed_at, indexed_files)
         VALUES (?, ?, ?)`,
        ['test-project', now, 3],
      );

      await db.execute(
        `INSERT INTO documents (id, project_id, source_path, source_type, title, full_content, checksum, created_at, updated_at, language, is_code)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`,
        ['doc1', 'test-project', '/test/file1.ts', 'code', 'file1.ts', '', 'h1', now, now, 'ts'],
      );
      await db.execute(
        `INSERT INTO documents (id, project_id, source_path, source_type, title, full_content, checksum, created_at, updated_at, language, is_code)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`,
        ['doc2', 'test-project', '/test/file2.ts', 'code', 'file2.ts', '', 'h2', now, now, 'ts'],
      );
      await db.execute(
        `INSERT INTO documents (id, project_id, source_path, source_type, title, full_content, checksum, created_at, updated_at, language, is_code)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`,
        ['doc3', 'test-project', '/test/file3.py', 'code', 'file3.py', '', 'h3', now, now, 'py'],
      );

      const stats = await service.getStatistics('test-project');

      expect(stats!.languageBreakdown['ts']).toBe(2);
      expect(stats!.languageBreakdown['py']).toBe(1);
    });
  });
});

describe('File Filtering', () => {
  let db: Database;
  let service: CodeIndexService;

  beforeEach(async () => {
    db = await createDatabase(':memory:');
    setDatabase(db);
    service = createCodeIndexService(createMockEmbeddingService());

    const now = Date.now();
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ['test-project', '/test/path', 'Test Project', now, now],
    );

    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ['mock:test-model', 'test-model', 'mock', 128, 1, now],
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  test('include paths filters to specified directories', async () => {
    const progress = await service.index('/tmp', 'test-project', {
      dryRun: true,
      includePaths: ['src/'],
    });

    expect(progress.phase).toBe('complete');
  });

  test('exclude paths removes specified directories', async () => {
    const progress = await service.index('/tmp', 'test-project', {
      dryRun: true,
      excludePaths: ['node_modules', 'dist'],
    });

    expect(progress.phase).toBe('complete');
  });
});
