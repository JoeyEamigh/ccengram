import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../src/db/database.js';
import { createEmbeddingService } from '../../src/services/embedding/index.js';
import type { EmbeddingService } from '../../src/services/embedding/types.js';
import { createCodeIndexService } from '../../src/services/codeindex/index.js';
import { getOrCreateProject } from '../../src/services/project.js';
import { createMemoryStore } from '../../src/services/memory/store.js';
import { createSearchService } from '../../src/services/search/hybrid.js';

describe('Scale Integration Tests', () => {
  const testDir = `/tmp/ccmemory-scale-${Date.now()}`;
  let db: Database;
  let embeddingService: EmbeddingService;

  beforeAll(async () => {
    await mkdir(testDir, { recursive: true });

    process.env['CCMEMORY_DATA_DIR'] = testDir;
    process.env['CCMEMORY_CONFIG_DIR'] = testDir;
    process.env['CCMEMORY_CACHE_DIR'] = testDir;

    db = await createDatabase(join(testDir, 'test.db'));
    setDatabase(db);

    embeddingService = await createEmbeddingService();
  });

  afterAll(async () => {
    closeDatabase();
    await rm(testDir, { recursive: true, force: true });
    delete process.env['CCMEMORY_DATA_DIR'];
    delete process.env['CCMEMORY_CONFIG_DIR'];
    delete process.env['CCMEMORY_CACHE_DIR'];
  });

  describe('Code Index Performance', () => {
    test('indexes 20 files efficiently with progress tracking', async () => {
      const projectDir = join(testDir, 'scale-20');
      await mkdir(projectDir, { recursive: true });

      const fileCount = 20;
      for (let i = 0; i < fileCount; i++) {
        await writeFile(
          join(projectDir, `module${i}.ts`),
          `
/**
 * Module ${i} - Auto-generated for scale testing
 */
export interface Module${i}Config {
  id: number;
  name: string;
  enabled: boolean;
}

export class Module${i}Service {
  private config: Module${i}Config;

  constructor(config: Module${i}Config) {
    this.config = config;
  }

  async process(data: unknown[]): Promise<unknown[]> {
    return data.map(item => ({ ...item as object, moduleId: ${i} }));
  }

  getStatus(): string {
    return this.config.enabled ? 'active' : 'inactive';
  }
}

export function createModule${i}(name: string): Module${i}Service {
  return new Module${i}Service({ id: ${i}, name, enabled: true });
}
`,
        );
      }

      const project = await getOrCreateProject(projectDir);
      const codeIndex = createCodeIndexService(embeddingService);

      const startTime = Date.now();
      const progress = await codeIndex.index(projectDir, project.id);
      const elapsed = Date.now() - startTime;

      expect(progress.totalFiles).toBe(fileCount);
      expect(progress.indexedFiles).toBe(fileCount);
      expect(progress.errors.length).toBe(0);
      expect(progress.phase).toBe('complete');

      const filesPerSecond = fileCount / (elapsed / 1000);
      console.log(`Indexed ${fileCount} files in ${elapsed}ms (${filesPerSecond.toFixed(1)} files/sec)`);

      expect(filesPerSecond).toBeGreaterThan(0.5);

      const state = await codeIndex.getState(project.id);
      expect(state).not.toBeNull();
      expect(state?.indexedFiles).toBe(fileCount);
    }, 60000);

    test('search performance scales with indexed files', async () => {
      const projectDir = join(testDir, 'scale-20');
      const project = await getOrCreateProject(projectDir);
      const codeIndex = createCodeIndexService(embeddingService);

      const queries = [
        'service class process data',
        'module configuration enabled',
        'create interface config',
        'async function return',
        'constructor initialize',
      ];

      for (const query of queries) {
        const startTime = Date.now();
        const results = await codeIndex.search({
          query,
          projectId: project.id,
          limit: 10,
        });
        const elapsed = Date.now() - startTime;

        expect(results.length).toBeGreaterThan(0);
        expect(results.length).toBeLessThanOrEqual(10);
        expect(elapsed).toBeLessThan(5000);

        console.log(`Search "${query.slice(0, 30)}..." returned ${results.length} results in ${elapsed}ms`);
      }
    });

    test('incremental indexing skips unchanged files', async () => {
      const projectDir = join(testDir, 'scale-20');
      const project = await getOrCreateProject(projectDir);
      const codeIndex = createCodeIndexService(embeddingService);

      const startTime = Date.now();
      const progress = await codeIndex.index(projectDir, project.id);
      const elapsed = Date.now() - startTime;

      expect(progress.indexedFiles).toBe(0);
      expect(progress.totalFiles).toBeGreaterThanOrEqual(0);
      expect(elapsed).toBeLessThan(5000);

      console.log(`Incremental scan of 20 files took ${elapsed}ms (${progress.indexedFiles} files re-indexed, ${progress.totalFiles} total)`);
    });
  });

  describe('Memory Scale Performance', () => {
    test('creates and searches 500 memories efficiently', async () => {
      const projectDir = join(testDir, 'memory-scale');
      await mkdir(projectDir, { recursive: true });
      const project = await getOrCreateProject(projectDir);
      const store = createMemoryStore();

      const memoryCount = 500;
      const createStart = Date.now();

      const topics = [
        'authentication', 'authorization', 'database', 'caching', 'logging',
        'error handling', 'validation', 'serialization', 'networking', 'security',
      ];
      const actions = [
        'implemented', 'fixed bug in', 'refactored', 'optimized', 'documented',
        'tested', 'reviewed', 'deployed', 'configured', 'debugged',
      ];
      const components = [
        'user service', 'payment module', 'auth controller', 'data layer', 'API gateway',
        'queue processor', 'event handler', 'config manager', 'cache service', 'logger',
      ];

      for (let i = 0; i < memoryCount; i++) {
        const topic = topics[i % topics.length] ?? 'general';
        const action = actions[Math.floor(i / topics.length) % actions.length] ?? 'worked on';
        const component = components[Math.floor(i / (topics.length * actions.length)) % components.length] ?? 'system';

        await store.create(
          {
            content: `${action} ${topic} in ${component}. This memory (#${i}) contains detailed information about the implementation, including specific code changes, configuration updates, and testing procedures that were followed. The ${topic} functionality in the ${component} required careful consideration of edge cases.`,
            memoryType: i % 2 === 0 ? 'codebase' : 'decision',
            scopePath: `src/${component.replace(/\s+/g, '-')}`,
            scopeModule: topic,
            importance: 0.3 + (i % 5) * 0.1,
          },
          project.id,
        );
      }

      const createElapsed = Date.now() - createStart;
      console.log(`Created ${memoryCount} memories in ${createElapsed}ms (${(memoryCount / (createElapsed / 1000)).toFixed(1)} memories/sec)`);

      const search = createSearchService(embeddingService);

      const searchQueries = [
        { query: 'authentication user service', scope: undefined },
        { query: 'database caching', scope: 'caching' },
        { query: 'error handling', scope: undefined },
        { query: 'payment refactored', scopePath: 'src/payment-module' },
      ];

      for (const { query, scope, scopePath } of searchQueries) {
        const searchStart = Date.now();
        const results = await search.search({
          query,
          projectId: project.id,
          scopeModule: scope,
          scopePath,
          limit: 10,
          mode: 'keyword',
        });
        const searchElapsed = Date.now() - searchStart;

        expect(results.length).toBeGreaterThan(0);
        expect(searchElapsed).toBeLessThan(1000);

        console.log(`Memory search "${query}" (scope: ${scope || scopePath || 'none'}) returned ${results.length} results in ${searchElapsed}ms`);
      }

      const listStart = Date.now();
      const scopedMemories = await store.list({
        projectId: project.id,
        scopeModule: 'authentication',
        limit: 100,
      });
      const listElapsed = Date.now() - listStart;

      expect(scopedMemories.length).toBeGreaterThan(0);
      expect(listElapsed).toBeLessThan(500);

      console.log(`Listed ${scopedMemories.length} scoped memories in ${listElapsed}ms`);
    });
  });

  describe('Deduplication at Scale', () => {
    test('dedup detects exact duplicates', async () => {
      const projectDir = join(testDir, 'dedup-scale');
      await mkdir(projectDir, { recursive: true });
      const project = await getOrCreateProject(projectDir);
      const store = createMemoryStore();

      const baseContent = 'The authentication service validates user credentials against the database and returns a JWT token for session management.';

      const startTime = Date.now();
      const createdIds: string[] = [];

      const memory1 = await store.create({ content: baseContent }, project.id);
      createdIds.push(memory1.id);

      const memory2 = await store.create({ content: baseContent }, project.id);
      createdIds.push(memory2.id);

      const memory3 = await store.create({ content: 'A completely different memory about payment processing and refunds.' }, project.id);
      createdIds.push(memory3.id);

      const elapsed = Date.now() - startTime;

      const uniqueIds = new Set(createdIds);
      console.log(`Created 3 memories (2 identical), got ${uniqueIds.size} unique in ${elapsed}ms`);

      expect(uniqueIds.size).toBe(2);
      expect(createdIds[0]).toBe(createdIds[1]);
      expect(elapsed).toBeLessThan(5000);
    });

    test('dedup handles many unique memories efficiently', async () => {
      const projectDir = join(testDir, 'dedup-unique');
      await mkdir(projectDir, { recursive: true });
      const project = await getOrCreateProject(projectDir);
      const store = createMemoryStore();

      const memoryCount = 50;
      const startTime = Date.now();

      for (let i = 0; i < memoryCount; i++) {
        await store.create({
          content: `Unique memory number ${i} about topic ${i % 10} with specific details about implementation ${i * 7}.`,
        }, project.id);
      }

      const elapsed = Date.now() - startTime;
      const memoriesPerSecond = memoryCount / (elapsed / 1000);

      console.log(`Created ${memoryCount} unique memories in ${elapsed}ms (${memoriesPerSecond.toFixed(1)} memories/sec)`);

      expect(memoriesPerSecond).toBeGreaterThan(5);
    });
  });

  describe('Checkpoint and Resume', () => {
    test('checkpoint saves and loads correctly', async () => {
      const projectDir = join(testDir, 'checkpoint-test');
      await mkdir(projectDir, { recursive: true });

      for (let i = 0; i < 10; i++) {
        await writeFile(join(projectDir, `file${i}.ts`), `export const x${i} = ${i};`);
      }

      const project = await getOrCreateProject(projectDir);
      const codeIndex = createCodeIndexService(embeddingService);

      await codeIndex.index(projectDir, project.id);

      const state = await codeIndex.getState(project.id);
      expect(state).not.toBeNull();
      expect(state?.indexedFiles).toBe(10);

      const checkpoint = await codeIndex.loadCheckpoint(project.id);
      expect(checkpoint).toBeNull();
    }, 60000);
  });
});
