import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../../db/database.js';
import { createMemoryStore, type MemoryStore } from '../store.js';

describe('Scoped Memory Operations', () => {
  let db: Database;
  let store: MemoryStore;
  const projectId = 'test-scoped-project';

  beforeEach(async () => {
    db = await createDatabase(':memory:');
    setDatabase(db);
    store = createMemoryStore();

    const now = Date.now();
    await db.execute(`INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      projectId,
      '/test/scoped-path',
      'Test Scoped Project',
      now,
      now,
    ]);
  });

  afterEach(() => {
    closeDatabase();
  });

  describe('Creating memories with scope', () => {
    it('should create memory with scope_path', async () => {
      const memory = await store.create(
        {
          content: 'Authentication logic uses JWT tokens for session management',
          scopePath: 'src/services/auth',
        },
        projectId,
      );

      expect(memory.scopePath).toBe('src/services/auth');
      expect(memory.scopeModule).toBeUndefined();
    });

    it('should create memory with scope_module', async () => {
      const memory = await store.create(
        {
          content: 'Payment processing handles refunds via Stripe API',
          scopeModule: 'payments',
        },
        projectId,
      );

      expect(memory.scopePath).toBeUndefined();
      expect(memory.scopeModule).toBe('payments');
    });

    it('should create memory with both scope_path and scope_module', async () => {
      const memory = await store.create(
        {
          content: 'User service validates email format before registration',
          scopePath: 'src/services/user',
          scopeModule: 'user-management',
        },
        projectId,
      );

      expect(memory.scopePath).toBe('src/services/user');
      expect(memory.scopeModule).toBe('user-management');
    });

    it('should create memory without any scope', async () => {
      const memory = await store.create(
        {
          content: 'General project configuration uses YAML files',
        },
        projectId,
      );

      expect(memory.scopePath).toBeUndefined();
      expect(memory.scopeModule).toBeUndefined();
    });
  });

  describe('Listing memories with scope filter', () => {
    it('should filter memories by scope_path', async () => {
      await store.create({ content: 'Auth memory about JWT token handling and validation', scopePath: 'src/auth' }, projectId);
      await store.create({ content: 'Auth memory about session management with Redis', scopePath: 'src/auth' }, projectId);
      await store.create({ content: 'Payment memory about Stripe integration', scopePath: 'src/payments' }, projectId);
      await store.create({ content: 'No scope memory about general configuration' }, projectId);

      const authMemories = await store.list({
        projectId,
        scopePath: 'src/auth',
      });

      expect(authMemories.length).toBe(2);
      for (const m of authMemories) {
        expect(m.scopePath).toBe('src/auth');
      }
    });

    it('should filter memories by scope_module', async () => {
      await store.create({ content: 'Core module handles database connection pooling', scopeModule: 'core' }, projectId);
      await store.create({ content: 'Core module provides logging infrastructure', scopeModule: 'core' }, projectId);
      await store.create({ content: 'API module exposes REST endpoints for users', scopeModule: 'api' }, projectId);
      await store.create({ content: 'No module memory about project documentation' }, projectId);

      const coreMemories = await store.list({
        projectId,
        scopeModule: 'core',
      });

      expect(coreMemories.length).toBe(2);
      for (const m of coreMemories) {
        expect(m.scopeModule).toBe('core');
      }
    });

    it('should filter memories by both scope_path and scope_module', async () => {
      await store.create({ content: 'Both scopes', scopePath: 'src/auth', scopeModule: 'security' }, projectId);
      await store.create({ content: 'Only path', scopePath: 'src/auth' }, projectId);
      await store.create({ content: 'Only module', scopeModule: 'security' }, projectId);
      await store.create({ content: 'No scope' }, projectId);

      const filtered = await store.list({
        projectId,
        scopePath: 'src/auth',
        scopeModule: 'security',
      });

      expect(filtered.length).toBe(1);
      expect(filtered[0]?.content).toBe('Both scopes');
    });

    it('should return empty when no memories match scope filter', async () => {
      await store.create({ content: 'Memory 1', scopePath: 'src/auth' }, projectId);
      await store.create({ content: 'Memory 2', scopeModule: 'core' }, projectId);

      const filtered = await store.list({
        projectId,
        scopePath: 'src/nonexistent',
      });

      expect(filtered.length).toBe(0);
    });
  });

  describe('Updating memory scope', () => {
    it('should update scope_path', async () => {
      const original = await store.create(
        { content: 'Memory to update', scopePath: 'src/old' },
        projectId,
      );

      const updated = await store.update(original.id, {
        scopePath: 'src/new',
      });

      expect(updated.scopePath).toBe('src/new');
    });

    it('should update scope_module', async () => {
      const original = await store.create(
        { content: 'Memory to update', scopeModule: 'old-module' },
        projectId,
      );

      const updated = await store.update(original.id, {
        scopeModule: 'new-module',
      });

      expect(updated.scopeModule).toBe('new-module');
    });

    it('should add scope to memory that had none', async () => {
      const original = await store.create(
        { content: 'Memory without scope' },
        projectId,
      );

      expect(original.scopePath).toBeUndefined();

      const updated = await store.update(original.id, {
        scopePath: 'src/added',
        scopeModule: 'added-module',
      });

      expect(updated.scopePath).toBe('src/added');
      expect(updated.scopeModule).toBe('added-module');
    });
  });

  describe('Monorepo scoping scenarios', () => {
    it('should support deeply nested path scopes', async () => {
      const memory = await store.create(
        {
          content: 'Component uses React hooks for state management',
          scopePath: 'packages/web/src/components/dashboard/widgets',
          scopeModule: 'frontend',
        },
        projectId,
      );

      const filtered = await store.list({
        projectId,
        scopePath: 'packages/web/src/components/dashboard/widgets',
      });

      expect(filtered.length).toBe(1);
      expect(filtered[0]?.id).toBe(memory.id);
    });

    it('should handle multiple modules in same project', async () => {
      await store.create({ content: 'Frontend memory', scopeModule: 'frontend' }, projectId);
      await store.create({ content: 'Backend memory', scopeModule: 'backend' }, projectId);
      await store.create({ content: 'Shared memory', scopeModule: 'shared' }, projectId);
      await store.create({ content: 'Mobile memory', scopeModule: 'mobile' }, projectId);

      const frontendMemories = await store.list({ projectId, scopeModule: 'frontend' });
      const backendMemories = await store.list({ projectId, scopeModule: 'backend' });
      const allMemories = await store.list({ projectId });

      expect(frontendMemories.length).toBe(1);
      expect(backendMemories.length).toBe(1);
      expect(allMemories.length).toBe(4);
    });

    it('should work with memory types combined with scope', async () => {
      await store.create(
        { content: 'Auth gotcha', memoryType: 'gotcha', scopePath: 'src/auth' },
        projectId,
      );
      await store.create(
        { content: 'Auth decision', memoryType: 'decision', scopePath: 'src/auth' },
        projectId,
      );
      await store.create(
        { content: 'Payment gotcha', memoryType: 'gotcha', scopePath: 'src/payments' },
        projectId,
      );

      const authGotchas = await store.list({
        projectId,
        memoryType: 'gotcha',
        scopePath: 'src/auth',
      });

      expect(authGotchas.length).toBe(1);
      expect(authGotchas[0]?.content).toBe('Auth gotcha');
    });
  });
});
