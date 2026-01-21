import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { closeDatabase, createDatabase, setDatabase, type Database } from '../../../db/database.js';
import { getSectorForMemoryType, saveExtractionSegment } from '../extractor.js';
import type { SegmentAccumulator } from '../types.js';

describe('Extractor utilities', () => {
  describe('getSectorForMemoryType', () => {
    test('preference type maps to emotional sector', () => {
      expect(getSectorForMemoryType('preference')).toBe('emotional');
    });

    test('codebase type maps to semantic sector', () => {
      expect(getSectorForMemoryType('codebase')).toBe('semantic');
    });

    test('decision type maps to reflective sector', () => {
      expect(getSectorForMemoryType('decision')).toBe('reflective');
    });

    test('gotcha type maps to procedural sector', () => {
      expect(getSectorForMemoryType('gotcha')).toBe('procedural');
    });

    test('pattern type maps to procedural sector', () => {
      expect(getSectorForMemoryType('pattern')).toBe('procedural');
    });
  });
});

describe('saveExtractionSegment', () => {
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

    await db.execute(`INSERT INTO sessions (id, project_id, started_at) VALUES (?, ?, ?)`, [
      'sess1',
      'proj1',
      now,
    ]);
  });

  afterEach(() => {
    closeDatabase();
  });

  function createMockAccumulator(overrides: Partial<SegmentAccumulator> = {}): SegmentAccumulator {
    return {
      sessionId: 'sess1',
      projectId: 'proj1',
      segmentId: 'seg1',
      segmentStart: Date.now() - 10000,
      userPrompts: [],
      filesRead: [],
      filesModified: [],
      commandsRun: [],
      errorsEncountered: [],
      searchesPerformed: [],
      completedTasks: [],
      toolCallCount: 5,
      ...overrides,
    };
  }

  test('saves extraction segment record', async () => {
    const accumulator = createMockAccumulator({
      filesRead: ['/src/index.ts', '/src/utils.ts'],
      filesModified: ['/src/app.ts'],
      toolCallCount: 10,
    });

    await saveExtractionSegment(accumulator, 'user_prompt', 2, 500);

    const result = await db.execute('SELECT * FROM extraction_segments WHERE id = ?', ['seg1']);
    expect(result.rows).toHaveLength(1);

    const row = result.rows[0];
    expect(row?.['trigger']).toBe('user_prompt');
    expect(row?.['memories_extracted']).toBe(2);
    expect(row?.['tool_call_count']).toBe(10);
  });

  test('saves extraction with user prompts', async () => {
    const accumulator = createMockAccumulator({
      userPrompts: [
        { content: 'Fix the bug', timestamp: Date.now() },
        { content: 'Use async/await', timestamp: Date.now(), signal: { category: 'preference', extractable: true, summary: null } },
      ],
    });

    await saveExtractionSegment(accumulator, 'stop', 1, 300);

    const result = await db.execute('SELECT user_prompts_json FROM extraction_segments WHERE id = ?', ['seg1']);
    const prompts = JSON.parse(String(result.rows[0]?.['user_prompts_json'])) as unknown[];
    expect(prompts).toHaveLength(2);
  });

  test('records extraction duration', async () => {
    const accumulator = createMockAccumulator();

    await saveExtractionSegment(accumulator, 'pre_compact', 0, 1234);

    const result = await db.execute('SELECT extraction_duration_ms FROM extraction_segments WHERE id = ?', ['seg1']);
    expect(result.rows[0]?.['extraction_duration_ms']).toBe(1234);
  });

  test('records extraction tokens when provided', async () => {
    const accumulator = createMockAccumulator();

    await saveExtractionSegment(accumulator, 'user_prompt', 1, 500, 150);

    const result = await db.execute('SELECT extraction_tokens FROM extraction_segments WHERE id = ?', ['seg1']);
    expect(result.rows[0]?.['extraction_tokens']).toBe(150);
  });

  test('handles null extraction tokens', async () => {
    const accumulator = createMockAccumulator();

    await saveExtractionSegment(accumulator, 'user_prompt', 1, 500);

    const result = await db.execute('SELECT extraction_tokens FROM extraction_segments WHERE id = ?', ['seg1']);
    expect(result.rows[0]?.['extraction_tokens']).toBeNull();
  });

  test('stores segment time range', async () => {
    const segmentStart = Date.now() - 30000;
    const accumulator = createMockAccumulator({ segmentStart });

    await saveExtractionSegment(accumulator, 'stop', 0, 100);

    const result = await db.execute('SELECT segment_start, segment_end FROM extraction_segments WHERE id = ?', ['seg1']);
    expect(Number(result.rows[0]?.['segment_start'])).toBe(segmentStart);
    expect(Number(result.rows[0]?.['segment_end'])).toBeGreaterThan(segmentStart);
  });
});
