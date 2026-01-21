import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../../db/database.js';
import { createMemoryStore, type MemoryStore } from '../../memory/store.js';
import { createSearchService, type SearchService } from '../hybrid.js';

describe('Scoped Search Operations', () => {
  let db: Database;
  let store: MemoryStore;
  let search: SearchService;
  const projectId = 'test-scoped-search-project';

  beforeEach(async () => {
    db = await createDatabase(':memory:');
    setDatabase(db);
    store = createMemoryStore();
    search = createSearchService(null);

    const now = Date.now();
    await db.execute(`INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      projectId,
      '/test/scoped-search-path',
      'Test Scoped Search Project',
      now,
      now,
    ]);
  });

  afterEach(() => {
    closeDatabase();
  });

  describe('Search with scope filtering', () => {
    it('should filter search results by scope_path', async () => {
      await store.create(
        { content: 'Authentication uses JWT tokens for secure sessions', scopePath: 'src/auth' },
        projectId,
      );
      await store.create(
        { content: 'Authentication validates user credentials against database', scopePath: 'src/auth' },
        projectId,
      );
      await store.create(
        { content: 'Authentication tokens are refreshed every 15 minutes', scopePath: 'src/payments' },
        projectId,
      );

      const results = await search.search({
        query: 'authentication',
        projectId,
        scopePath: 'src/auth',
        mode: 'keyword',
      });

      expect(results.length).toBe(2);
      for (const r of results) {
        expect(r.memory.scopePath).toBe('src/auth');
      }
    });

    it('should filter search results by scope_module', async () => {
      await store.create(
        { content: 'Frontend component handles user input validation', scopeModule: 'frontend' },
        projectId,
      );
      await store.create(
        { content: 'Frontend state management uses Redux for global state', scopeModule: 'frontend' },
        projectId,
      );
      await store.create(
        { content: 'Backend validation occurs before database writes', scopeModule: 'backend' },
        projectId,
      );

      const results = await search.search({
        query: 'validation',
        projectId,
        scopeModule: 'frontend',
        mode: 'keyword',
      });

      expect(results.length).toBe(1);
      expect(results[0]?.memory.scopeModule).toBe('frontend');
    });

    it('should combine scope_path and scope_module filters', async () => {
      await store.create(
        { content: 'API endpoint handles user registration flow', scopePath: 'src/api', scopeModule: 'user' },
        projectId,
      );
      await store.create(
        { content: 'API endpoint processes payment transactions', scopePath: 'src/api', scopeModule: 'payment' },
        projectId,
      );
      await store.create(
        { content: 'Service layer manages user profile updates', scopePath: 'src/services', scopeModule: 'user' },
        projectId,
      );

      const results = await search.search({
        query: 'user',
        projectId,
        scopePath: 'src/api',
        scopeModule: 'user',
        mode: 'keyword',
      });

      expect(results.length).toBe(1);
      expect(results[0]?.memory.scopePath).toBe('src/api');
      expect(results[0]?.memory.scopeModule).toBe('user');
    });

    it('should return empty results when no memories match scope', async () => {
      await store.create(
        { content: 'Database connection pooling configuration', scopePath: 'src/db' },
        projectId,
      );
      await store.create(
        { content: 'Cache invalidation strategy for Redis', scopeModule: 'cache' },
        projectId,
      );

      const results = await search.search({
        query: 'database',
        projectId,
        scopePath: 'src/nonexistent',
        mode: 'keyword',
      });

      expect(results.length).toBe(0);
    });

    it('should search across all memories when no scope specified', async () => {
      await store.create(
        { content: 'Config file parsing in the configuration module', scopePath: 'src/config' },
        projectId,
      );
      await store.create(
        { content: 'Config validation before application startup', scopeModule: 'startup' },
        projectId,
      );
      await store.create(
        { content: 'Config defaults are loaded from environment variables' },
        projectId,
      );

      const results = await search.search({
        query: 'config',
        projectId,
        mode: 'keyword',
      });

      expect(results.length).toBe(3);
    });
  });

  describe('Scope filtering combined with other filters', () => {
    it('should combine scope with sector filter', async () => {
      await store.create(
        { content: 'Learned that auth module prefers bcrypt for password hashing', sector: 'reflective', scopePath: 'src/auth' },
        projectId,
      );
      await store.create(
        { content: 'Auth module API endpoint documentation is at /docs/auth', sector: 'semantic', scopePath: 'src/auth' },
        projectId,
      );
      await store.create(
        { content: 'Realized caching helps auth module performance', sector: 'reflective', scopePath: 'src/cache' },
        projectId,
      );

      const results = await search.search({
        query: 'auth',
        projectId,
        sector: 'reflective',
        scopePath: 'src/auth',
        mode: 'keyword',
      });

      expect(results.length).toBe(1);
      expect(results[0]?.memory.sector).toBe('reflective');
      expect(results[0]?.memory.scopePath).toBe('src/auth');
    });

    it('should combine scope with memory type filter', async () => {
      await store.create(
        { content: 'Gotcha: Payment API requires idempotency keys', memoryType: 'gotcha', scopeModule: 'payments' },
        projectId,
      );
      await store.create(
        { content: 'Decision: Use Stripe for payment processing', memoryType: 'decision', scopeModule: 'payments' },
        projectId,
      );
      await store.create(
        { content: 'Gotcha: Auth tokens expire silently without refresh', memoryType: 'gotcha', scopeModule: 'auth' },
        projectId,
      );

      const results = await search.search({
        query: 'payment OR stripe',
        projectId,
        memoryType: 'gotcha',
        scopeModule: 'payments',
        mode: 'keyword',
      });

      expect(results.length).toBe(1);
      expect(results[0]?.memory.memoryType).toBe('gotcha');
      expect(results[0]?.memory.scopeModule).toBe('payments');
    });
  });
});
