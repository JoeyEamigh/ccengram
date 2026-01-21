import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../src/db/database.js';
import { createEmbeddingService } from '../../src/services/embedding/index.js';
import type { EmbeddingService } from '../../src/services/embedding/types.js';
import { getOrCreateSession } from '../../src/services/memory/sessions.js';
import { createMemoryStore } from '../../src/services/memory/store.js';
import { getOrCreateProject } from '../../src/services/project.js';
import { createSearchService } from '../../src/services/search/hybrid.js';

describe('Search Quality Integration', () => {
  const testDir = `/tmp/ccmemory-search-integration-${Date.now()}`;
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

  test('FTS search finds exact keyword matches', async () => {
    const projectPath = '/test/fts-project';
    const sessionId = `fts-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: 'The PostgreSQL database configuration uses connection pooling',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'Redis is used for session caching and rate limiting',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'The frontend uses React with TypeScript for type safety',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    const searchService = createSearchService(embeddingService);

    const postgresResults = await searchService.search({
      query: 'PostgreSQL',
      projectId: project.id,
      mode: 'keyword',
      limit: 10,
    });

    expect(postgresResults.length).toBe(1);
    expect(postgresResults[0]?.memory.content).toContain('PostgreSQL');

    const redisResults = await searchService.search({
      query: 'Redis caching',
      projectId: project.id,
      mode: 'keyword',
      limit: 10,
    });

    expect(redisResults.length).toBe(1);
    expect(redisResults[0]?.memory.content).toContain('Redis');
  });

  test('semantic search finds conceptually related content with embeddings', async () => {
    const projectPath = '/test/semantic-project';
    const sessionId = `semantic-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const memories = [
      await store.create(
        {
          content: 'User authentication is handled using JWT tokens with refresh token rotation',
          sector: 'semantic',
          tier: 'project',
        },
        project.id,
        sessionId,
      ),
      await store.create(
        {
          content: 'The application uses bcrypt for secure password hashing',
          sector: 'semantic',
          tier: 'project',
        },
        project.id,
        sessionId,
      ),
      await store.create(
        {
          content: 'API rate limiting prevents abuse with a sliding window algorithm',
          sector: 'semantic',
          tier: 'project',
        },
        project.id,
        sessionId,
      ),
    ];

    const modelId = embeddingService.getActiveModelId();
    for (const memory of memories) {
      const result = await embeddingService.embed(memory.content);
      const vectorBlob = new Float32Array(result.vector).buffer;
      await db.execute(
        `INSERT OR REPLACE INTO memory_vectors (memory_id, model_id, vector, dim, created_at)
         VALUES (?, ?, ?, ?, ?)`,
        [memory.id, modelId, vectorBlob, result.vector.length, Date.now()],
      );
    }

    const searchService = createSearchService(embeddingService);

    const results = await searchService.search({
      query: 'security login password',
      projectId: project.id,
      mode: 'semantic',
      limit: 10,
    });

    expect(results.length).toBeGreaterThan(0);
    const topResult = results[0];
    expect(topResult?.memory.content.includes('authentication') || topResult?.memory.content.includes('password')).toBe(
      true,
    );
  });

  test('hybrid search combines keyword and semantic results', async () => {
    const projectPath = '/test/hybrid-project';
    const sessionId = `hybrid-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: 'The GraphQL API uses Apollo Server with DataLoader for batching',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'REST endpoints are implemented using Express.js middleware pattern',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'API documentation is generated using OpenAPI specifications',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    const searchService = createSearchService(embeddingService);

    const results = await searchService.search({
      query: 'GraphQL API',
      projectId: project.id,
      mode: 'hybrid',
      limit: 10,
    });

    expect(results.length).toBeGreaterThan(0);
    const graphqlResult = results.find(r => r.memory.content.includes('GraphQL'));
    expect(graphqlResult).toBeDefined();
    expect(graphqlResult?.matchType === 'keyword' || graphqlResult?.matchType === 'both').toBe(true);
  });

  test('sector filtering limits results to specific sectors', async () => {
    const projectPath = '/test/sector-project';
    const sessionId = `sector-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: 'Tool: Bash\nCommand: npm run build\nOutput: Build succeeded',
        sector: 'episodic',
        tier: 'session',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'The build process uses Webpack for bundling JavaScript modules',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'To build the project, run npm run build from the root directory',
        sector: 'procedural',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    const searchService = createSearchService(embeddingService);

    const semanticResults = await searchService.search({
      query: 'build',
      projectId: project.id,
      sector: 'semantic',
      limit: 10,
    });

    expect(semanticResults.every(r => r.memory.sector === 'semantic')).toBe(true);
    expect(semanticResults.length).toBe(1);

    const proceduralResults = await searchService.search({
      query: 'build',
      projectId: project.id,
      sector: 'procedural',
      limit: 10,
    });

    expect(proceduralResults.every(r => r.memory.sector === 'procedural')).toBe(true);
    expect(proceduralResults.length).toBe(1);
  });

  test('tier filtering separates session and project memories', async () => {
    const projectPath = '/test/tier-project';
    const sessionId = `tier-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: 'Debugging: Found null pointer exception in user service',
        sector: 'episodic',
        tier: 'session',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'The user service handles all user-related operations including CRUD',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    const searchService = createSearchService(embeddingService);

    const sessionResults = await searchService.search({
      query: 'user service',
      projectId: project.id,
      tier: 'session',
      limit: 10,
    });

    expect(sessionResults.every(r => r.memory.tier === 'session')).toBe(true);

    const projectResults = await searchService.search({
      query: 'user service',
      projectId: project.id,
      tier: 'project',
      limit: 10,
    });

    expect(projectResults.every(r => r.memory.tier === 'project')).toBe(true);
  });

  test('salience filtering excludes low-salience memories', async () => {
    const projectPath = '/test/salience-project';
    const sessionId = `salience-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    const highSalience = await store.create(
      {
        content: 'Critical: Production database uses read replicas for scaling',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    const lowSalience = await store.create(
      {
        content: 'Note: Database performance metrics are logged hourly',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.deemphasize(lowSalience.id, 0.8);

    const searchService = createSearchService(embeddingService);

    const allResults = await searchService.search({
      query: 'database',
      projectId: project.id,
      limit: 10,
    });

    expect(allResults.length).toBe(2);

    const highSalienceResults = await searchService.search({
      query: 'database',
      projectId: project.id,
      minSalience: 0.5,
      limit: 10,
    });

    expect(highSalienceResults.length).toBe(1);
    expect(highSalienceResults[0]?.memory.id).toBe(highSalience.id);
  });

  test('search results are scored and ranked correctly', async () => {
    const projectPath = '/test/ranking-project';
    const sessionId = `ranking-session-${Date.now()}`;

    const project = await getOrCreateProject(projectPath);
    await getOrCreateSession(sessionId, project.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: 'React components are organized using atomic design principles',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'The React application uses hooks extensively for state management',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    await store.create(
      {
        content: 'CSS modules provide scoped styling for React components',
        sector: 'semantic',
        tier: 'project',
      },
      project.id,
      sessionId,
    );

    const searchService = createSearchService(embeddingService);

    const results = await searchService.search({
      query: 'React components',
      projectId: project.id,
      limit: 10,
    });

    expect(results.length).toBeGreaterThan(0);
    for (let i = 1; i < results.length; i++) {
      const prev = results[i - 1];
      const curr = results[i];
      if (prev && curr) {
        expect(prev.score).toBeGreaterThanOrEqual(curr.score);
      }
    }
  });

  test('project isolation prevents cross-project search', async () => {
    const projectPath1 = '/test/project-a';
    const projectPath2 = '/test/project-b';
    const sessionId = `isolation-session-${Date.now()}`;

    const project1 = await getOrCreateProject(projectPath1);
    const project2 = await getOrCreateProject(projectPath2);
    await getOrCreateSession(sessionId, project1.id);
    await getOrCreateSession(`${sessionId}-2`, project2.id);

    const store = createMemoryStore();

    await store.create(
      {
        content: 'Project A uses MongoDB for data storage',
        sector: 'semantic',
        tier: 'project',
      },
      project1.id,
      sessionId,
    );

    await store.create(
      {
        content: 'Project B uses PostgreSQL for data storage',
        sector: 'semantic',
        tier: 'project',
      },
      project2.id,
      `${sessionId}-2`,
    );

    const searchService = createSearchService(embeddingService);

    const project1Results = await searchService.search({
      query: 'data storage',
      projectId: project1.id,
      limit: 10,
    });

    expect(project1Results.length).toBe(1);
    expect(project1Results[0]?.memory.content).toContain('MongoDB');

    const project2Results = await searchService.search({
      query: 'data storage',
      projectId: project2.id,
      limit: 10,
    });

    expect(project2Results.length).toBe(1);
    expect(project2Results[0]?.memory.content).toContain('PostgreSQL');
  });
});
