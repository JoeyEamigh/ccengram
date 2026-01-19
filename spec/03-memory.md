# Memory System Specification

## Overview

The memory system stores, classifies, and manages memories using a 5-sector model based on OpenMemory research. Supports deduplication, time-based decay with access reinforcement, explicit relationships, and soft delete.

## Files to Create

- `src/services/memory/store.ts` - Memory CRUD operations
- `src/services/memory/sectors.ts` - Memory sector classification
- `src/services/memory/dedup.ts` - Simhash deduplication
- `src/services/memory/decay.ts` - Salience-based decay
- `src/services/memory/relationships.ts` - Memory relationship management
- `src/services/memory/sessions.ts` - Session-memory tracking

## Memory Sectors (5-Sector Model)

Based on OpenMemory research, memories are classified into 5 sectors with different decay characteristics:

| Sector | Description | Decay Rate | Examples |
|--------|-------------|------------|----------|
| `episodic` | Events, conversations, specific interactions | 0.02/day | "User asked about auth flow" |
| `semantic` | Facts, knowledge, learned information | 0.005/day | "API endpoint is /api/users" |
| `procedural` | Skills, workflows, how-to knowledge | 0.01/day | "Deploy via `bun run deploy`" |
| `emotional` | Sentiments, frustrations, satisfactions | 0.003/day | "Frustrated by slow tests" |
| `reflective` | Insights, patterns, lessons learned | 0.008/day | "This codebase favors composition" |

### Type Definitions

```typescript
// src/services/memory/sectors.ts
type MemorySector = 'episodic' | 'semantic' | 'procedural' | 'emotional' | 'reflective';
type MemoryTier = 'session' | 'project' | 'global';

type Memory = {
  id: string;
  projectId: string;
  content: string;
  summary?: string;
  contentHash?: string;

  // Classification
  sector: MemorySector;
  tier: MemoryTier;
  importance: number;         // Base importance 0-1
  categories: string[];

  // Deduplication
  simhash?: string;

  // Salience/Reinforcement
  salience: number;           // Current strength 0-1 (decays, boosted on access)
  accessCount: number;

  // Timestamps (bi-temporal)
  createdAt: number;
  updatedAt: number;
  lastAccessed: number;
  validFrom?: number;         // When fact became true
  validUntil?: number;        // When fact ceased being true (null = current)

  // Soft delete
  isDeleted: boolean;
  deletedAt?: number;

  // Metadata
  embeddingModelId?: string;
  tags: string[];
  concepts: string[];
  files: string[];
};

type MemoryInput = {
  content: string;
  sector?: MemorySector;      // Auto-classify if not provided
  tier?: MemoryTier;          // Default: 'project'
  importance?: number;        // Default: 0.5
  tags?: string[];
  files?: string[];
  validFrom?: number;
};
```

### Sector Classification

```typescript
// Sector patterns for auto-classification
const SECTOR_PATTERNS: Record<MemorySector, RegExp[]> = {
  episodic: [
    /\b(asked|said|mentioned|discussed|talked about)\b/i,
    /\b(session|conversation|earlier|just now)\b/i,
    /\b(user (wanted|requested|asked for))\b/i,
  ],
  semantic: [
    /\b(is|are|was|were|has|have|contains)\b/i,
    /\b(located (at|in)|defined in|implemented in)\b/i,
    /\b(file|function|class|module|component|endpoint)\b/i,
    /\b(fact|information|knowledge)\b/i,
  ],
  procedural: [
    /\b(how to|steps to|process|workflow|procedure)\b/i,
    /\b(first|then|next|finally|step \d+)\b/i,
    /\b(run|execute|build|deploy|test)\b/i,
    /\b(command|script|recipe)\b/i,
  ],
  emotional: [
    /\b(frustrated|annoyed|happy|satisfied|confused)\b/i,
    /\b(love|hate|prefer|dislike)\b/i,
    /\b(pain point|struggle|enjoy)\b/i,
  ],
  reflective: [
    /\b(learned|realized|noticed|insight|pattern)\b/i,
    /\b(better to|should have|next time)\b/i,
    /\b(observation|conclusion|takeaway)\b/i,
    /\b(this codebase|this project|in general)\b/i,
  ],
};

const SECTOR_DECAY_RATES: Record<MemorySector, number> = {
  emotional: 0.003,   // Emotions persist longest
  semantic: 0.005,    // Facts persist well
  reflective: 0.008,  // Insights persist
  procedural: 0.01,   // Procedures change moderately
  episodic: 0.02,     // Events fade fastest
};

function classifyMemorySector(content: string): MemorySector {
  const scores: Record<MemorySector, number> = {
    episodic: 0, semantic: 0, procedural: 0, emotional: 0, reflective: 0,
  };

  for (const [sector, patterns] of Object.entries(SECTOR_PATTERNS)) {
    for (const pattern of patterns) {
      const matches = content.match(pattern);
      if (matches) {
        scores[sector as MemorySector] += matches.length;
      }
    }
  }

  let maxSector: MemorySector = 'semantic';  // Default
  let maxScore = 0;
  for (const [sector, score] of Object.entries(scores)) {
    if (score > maxScore) {
      maxScore = score;
      maxSector = sector as MemorySector;
    }
  }

  return maxSector;
}
```

### Test Specification

```typescript
// src/services/memory/__test__/sectors.test.ts (colocated)
describe('Memory Sector Classification', () => {
  test('classifies conversation events as episodic', () => {
    const content = 'User asked about the authentication flow earlier';
    expect(classifyMemorySector(content)).toBe('episodic');
  });

  test('classifies facts as semantic', () => {
    const content = 'The auth handler is located in src/auth/handler.ts';
    expect(classifyMemorySector(content)).toBe('semantic');
  });

  test('classifies workflows as procedural', () => {
    const content = 'To deploy: first run build, then push to main';
    expect(classifyMemorySector(content)).toBe('procedural');
  });

  test('classifies feelings as emotional', () => {
    const content = 'Frustrated by the slow test suite';
    expect(classifyMemorySector(content)).toBe('emotional');
  });

  test('classifies insights as reflective', () => {
    const content = 'This codebase favors composition over inheritance';
    expect(classifyMemorySector(content)).toBe('reflective');
  });

  test('defaults to semantic for ambiguous content', () => {
    const content = 'The function returns a string';
    expect(classifyMemorySector(content)).toBe('semantic');
  });
});
```

## Simhash Deduplication

### Interface

```typescript
// src/services/memory/dedup.ts
function computeSimhash(text: string): string;
function hammingDistance(hash1: string, hash2: string): number;
function isDuplicate(hash1: string, hash2: string, threshold?: number): boolean;
function findSimilarMemory(simhash: string, projectId: string): Promise<Memory | null>;
```

### Implementation Notes

Simhash is a locality-sensitive hash that produces similar hashes for similar text:

```typescript
function computeSimhash(text: string): string {
  const tokens = text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 2);

  const vector = new Array(64).fill(0);

  for (const token of tokens) {
    const hash = fnv1a64(token);
    for (let i = 0; i < 64; i++) {
      if ((hash >> BigInt(i)) & 1n) {
        vector[i]++;
      } else {
        vector[i]--;
      }
    }
  }

  let result = 0n;
  for (let i = 0; i < 64; i++) {
    if (vector[i] > 0) {
      result |= (1n << BigInt(i));
    }
  }

  return result.toString(16).padStart(16, '0');
}

function fnv1a64(str: string): bigint {
  let hash = 14695981039346656037n;
  for (let i = 0; i < str.length; i++) {
    hash ^= BigInt(str.charCodeAt(i));
    hash = (hash * 1099511628211n) & ((1n << 64n) - 1n);
  }
  return hash;
}

function hammingDistance(hash1: string, hash2: string): number {
  const h1 = BigInt('0x' + hash1);
  const h2 = BigInt('0x' + hash2);
  const xor = h1 ^ h2;

  let count = 0;
  let n = xor;
  while (n > 0n) {
    count += Number(n & 1n);
    n >>= 1n;
  }
  return count;
}

function isDuplicate(hash1: string, hash2: string, threshold = 3): boolean {
  return hammingDistance(hash1, hash2) <= threshold;
}
```

### Test Specification

```typescript
// src/services/memory/__test__/dedup.test.ts (colocated)
describe('Simhash Deduplication', () => {
  test('identical text produces identical hash', () => {
    const text = 'The quick brown fox jumps over the lazy dog';
    expect(computeSimhash(text)).toBe(computeSimhash(text));
  });

  test('similar text produces similar hash', () => {
    const text1 = 'The quick brown fox jumps over the lazy dog';
    const text2 = 'The quick brown fox leaps over the lazy dog';
    const distance = hammingDistance(computeSimhash(text1), computeSimhash(text2));
    expect(distance).toBeLessThan(10);
  });

  test('different text produces different hash', () => {
    const text1 = 'The quick brown fox';
    const text2 = 'A completely different sentence about programming';
    const distance = hammingDistance(computeSimhash(text1), computeSimhash(text2));
    expect(distance).toBeGreaterThan(20);
  });

  test('isDuplicate with default threshold', () => {
    const hash1 = '0000000000000000';
    const hash2 = '0000000000000007';  // 3 bits different
    expect(isDuplicate(hash1, hash2)).toBe(true);
  });
});
```

## Memory Store

### Interface

```typescript
// src/services/memory/store.ts
type MemoryStore = {
  create(input: MemoryInput, projectId: string, sessionId?: string): Promise<Memory>;
  get(id: string): Promise<Memory | null>;
  update(id: string, updates: Partial<MemoryInput>): Promise<Memory>;
  delete(id: string, hard?: boolean): Promise<void>;  // Soft delete by default
  restore(id: string): Promise<Memory>;               // Restore soft-deleted
  list(options: ListOptions): Promise<Memory[]>;
  touch(id: string): Promise<void>;                   // Update last_accessed

  // Reinforcement
  reinforce(id: string, amount?: number): Promise<Memory>;
  deemphasize(id: string, amount?: number): Promise<Memory>;

  // Session tracking
  linkToSession(memoryId: string, sessionId: string, usageType: UsageType): Promise<void>;
  getBySession(sessionId: string): Promise<Memory[]>;
};

type ListOptions = {
  limit?: number;
  offset?: number;
  sector?: MemorySector;
  tier?: MemoryTier;
  minSalience?: number;
  includeDeleted?: boolean;
  orderBy?: 'created_at' | 'salience' | 'last_accessed';
  order?: 'asc' | 'desc';
};

type UsageType = 'created' | 'recalled' | 'updated' | 'reinforced';

function createMemoryStore(): MemoryStore;
```

### Implementation Notes

```typescript
import { log } from "../../utils/log.js";

function createMemoryStore(): MemoryStore {
  const db = getDatabase();
  const embedding = getEmbeddingService();

  return {
    async create(input: MemoryInput, projectId: string, sessionId?: string): Promise<Memory> {
      const id = crypto.randomUUID();
      const now = Date.now();

      const sector = input.sector || classifyMemorySector(input.content);
      const tier = input.tier || 'project';
      const importance = input.importance ?? 0.5;

      const simhash = computeSimhash(input.content);
      const contentHash = await computeMD5(input.content);

      log.debug("memory", "Creating memory", { sector, tier, projectId, simhash: simhash.slice(0, 8) });

      // Check for duplicates
      const existing = await findSimilarMemory(simhash, projectId);
      if (existing && !existing.isDeleted) {
        log.info("memory", "Duplicate detected, reinforcing existing", {
          existingId: existing.id,
          similarity: "high"
        });
        await this.reinforce(existing.id, 0.1);
        if (sessionId) {
          await this.linkToSession(existing.id, sessionId, 'reinforced');
        }
        return this.get(existing.id) as Promise<Memory>;
      }

      const concepts = extractConcepts(input.content);

      await db.execute(`
        INSERT INTO memories (
          id, project_id, content, content_hash, sector, tier, importance,
          simhash, salience, access_count, created_at, updated_at,
          last_accessed, valid_from, is_deleted,
          tags_json, concepts_json, files_json, categories_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `, [
        id, projectId, input.content, contentHash, sector, tier, importance,
        simhash, 1.0, 0, now, now, now, input.validFrom || null, 0,
        JSON.stringify(input.tags || []),
        JSON.stringify(concepts),
        JSON.stringify(input.files || []),
        JSON.stringify([]),
      ]);

      // Generate and store embedding
      const embeddingResult = await embedding.embed(input.content);
      await db.execute(`
        INSERT INTO memory_vectors (memory_id, model_id, vector, dim)
        VALUES (?, ?, vector(?), ?)
      `, [id, embeddingResult.model, JSON.stringify(embeddingResult.vector), embeddingResult.dimensions]);

      // Track in session
      if (sessionId) {
        await this.linkToSession(id, sessionId, 'created');
      }

      log.info("memory", "Memory created", { id, sector, tier, concepts: concepts.length });
      return this.get(id) as Promise<Memory>;
    },

    async reinforce(id: string, amount = 0.1): Promise<Memory> {
      const now = Date.now();
      log.debug("memory", "Reinforcing memory", { id, amount });
      // Diminishing returns: boost more when salience is low
      await db.execute(`
        UPDATE memories
        SET salience = MIN(1.0, salience + ? * (1.0 - salience)),
            last_accessed = ?,
            access_count = access_count + 1,
            updated_at = ?
        WHERE id = ? AND is_deleted = 0
      `, [amount, now, now, id]);
      return this.get(id) as Promise<Memory>;
    },

    async deemphasize(id: string, amount = 0.2): Promise<Memory> {
      const now = Date.now();
      log.debug("memory", "De-emphasizing memory", { id, amount });
      await db.execute(`
        UPDATE memories
        SET salience = MAX(0.05, salience - ?),
            updated_at = ?
        WHERE id = ? AND is_deleted = 0
      `, [amount, now, id]);
      return this.get(id) as Promise<Memory>;
    },

    async delete(id: string, hard = false): Promise<void> {
      log.info("memory", "Deleting memory", { id, hard });
      if (hard) {
        await db.execute('DELETE FROM memories WHERE id = ?', [id]);
      } else {
        const now = Date.now();
        await db.execute(`
          UPDATE memories
          SET is_deleted = 1, deleted_at = ?, updated_at = ?
          WHERE id = ?
        `, [now, now, id]);
      }
    },

    async restore(id: string): Promise<Memory> {
      log.info("memory", "Restoring memory", { id });
      const now = Date.now();
      await db.execute(`
        UPDATE memories
        SET is_deleted = 0, deleted_at = NULL, updated_at = ?
        WHERE id = ?
      `, [now, id]);
      return this.get(id) as Promise<Memory>;
    },

    async linkToSession(memoryId: string, sessionId: string, usageType: UsageType): Promise<void> {
      log.debug("memory", "Linking memory to session", { memoryId, sessionId, usageType });
      const now = Date.now();
      await db.execute(`
        INSERT INTO session_memories (session_id, memory_id, created_at, usage_type)
        VALUES (?, ?, ?, ?)
      `, [sessionId, memoryId, now, usageType]);
    },

    // ... other methods
  };
}
```

### Test Specification

```typescript
// src/services/memory/__test__/store.test.ts (colocated)
describe('MemoryStore', () => {
  let store: MemoryStore;

  beforeEach(async () => {
    await setupTestDatabase();
    store = createMemoryStore();
  });

  test('creates memory with auto-classification', async () => {
    const memory = await store.create({
      content: 'The API endpoint is /api/users',
    }, 'proj1');

    expect(memory.sector).toBe('semantic');
    expect(memory.simhash).toBeDefined();
    expect(memory.salience).toBe(1.0);
  });

  test('deduplicates similar content', async () => {
    const mem1 = await store.create({
      content: 'The auth module is in src/auth/index.ts',
    }, 'proj1');

    const mem2 = await store.create({
      content: 'The authentication module is located in src/auth/index.ts',
    }, 'proj1');

    expect(mem2.id).toBe(mem1.id);
    expect(mem2.salience).toBeGreaterThan(1.0 - 0.1);  // Reinforced
  });

  test('reinforce increases salience with diminishing returns', async () => {
    const memory = await store.create({ content: 'Test' }, 'proj1');
    expect(memory.salience).toBe(1.0);

    const after = await store.reinforce(memory.id, 0.5);
    expect(after.salience).toBe(1.0);  // Already at max

    // Create new memory with lower salience
    const mem2 = await store.create({ content: 'Another test' }, 'proj1');
    await store.deemphasize(mem2.id, 0.5);
    const lowSalience = await store.get(mem2.id);
    expect(lowSalience!.salience).toBe(0.5);

    const boosted = await store.reinforce(mem2.id, 0.5);
    expect(boosted.salience).toBe(0.75);  // 0.5 + 0.5 * (1 - 0.5)
  });

  test('deemphasize reduces salience with floor', async () => {
    const memory = await store.create({ content: 'Test' }, 'proj1');

    await store.deemphasize(memory.id, 2.0);  // Try to reduce by more than possible
    const after = await store.get(memory.id);
    expect(after!.salience).toBe(0.05);  // Floor
  });

  test('soft delete marks as deleted', async () => {
    const memory = await store.create({ content: 'Test' }, 'proj1');
    await store.delete(memory.id);

    const deleted = await store.get(memory.id);
    expect(deleted!.isDeleted).toBe(true);
    expect(deleted!.deletedAt).toBeDefined();
  });

  test('restore brings back soft-deleted memory', async () => {
    const memory = await store.create({ content: 'Test' }, 'proj1');
    await store.delete(memory.id);
    await store.restore(memory.id);

    const restored = await store.get(memory.id);
    expect(restored!.isDeleted).toBe(false);
  });

  test('hard delete removes permanently', async () => {
    const memory = await store.create({ content: 'Test' }, 'proj1');
    await store.delete(memory.id, true);

    const gone = await store.get(memory.id);
    expect(gone).toBeNull();
  });

  test('linkToSession tracks memory usage', async () => {
    const memory = await store.create({ content: 'Test' }, 'proj1', 'sess1');

    const sessionMemories = await store.getBySession('sess1');
    expect(sessionMemories).toHaveLength(1);
    expect(sessionMemories[0].id).toBe(memory.id);
  });

  test('list excludes deleted by default', async () => {
    await store.create({ content: 'Visible' }, 'proj1');
    const toDelete = await store.create({ content: 'Hidden' }, 'proj1');
    await store.delete(toDelete.id);

    const list = await store.list({ limit: 10 });
    expect(list).toHaveLength(1);

    const listWithDeleted = await store.list({ limit: 10, includeDeleted: true });
    expect(listWithDeleted).toHaveLength(2);
  });
});
```

## Memory Relationships

### Interface

```typescript
// src/services/memory/relationships.ts
type RelationshipType =
  | 'SUPERSEDES'      // New info replaces old
  | 'CONTRADICTS'     // Conflicting information
  | 'RELATED_TO'      // General semantic connection
  | 'BUILDS_ON'       // Extends previous knowledge
  | 'CONFIRMS'        // Reinforces existing info
  | 'APPLIES_TO'      // Memory applies to specific context
  | 'DEPENDS_ON'      // Prerequisite relationship
  | 'ALTERNATIVE_TO'; // Different approach to same problem

type MemoryRelationship = {
  id: string;
  sourceMemoryId: string;
  targetMemoryId: string;
  relationshipType: RelationshipType;
  createdAt: number;
  validFrom: number;
  validUntil?: number;
  confidence: number;
  extractedBy: 'user' | 'llm' | 'system';
};

function createRelationship(
  sourceId: string,
  targetId: string,
  type: RelationshipType,
  extractedBy?: 'user' | 'llm' | 'system'
): Promise<MemoryRelationship>;

function supersede(oldMemoryId: string, newMemoryId: string): Promise<void>;
function getRelationships(memoryId: string): Promise<MemoryRelationship[]>;
function getSupersedingMemory(memoryId: string): Promise<Memory | null>;
```

### Implementation Notes

```typescript
async function supersede(oldMemoryId: string, newMemoryId: string): Promise<void> {
  const db = getDatabase();
  const now = Date.now();

  // Mark old memory as no longer valid
  await db.execute(`
    UPDATE memories SET valid_until = ?, updated_at = ?
    WHERE id = ? AND valid_until IS NULL
  `, [now, now, oldMemoryId]);

  // Create SUPERSEDES relationship
  await createRelationship(newMemoryId, oldMemoryId, 'SUPERSEDES', 'system');
}

async function getSupersedingMemory(memoryId: string): Promise<Memory | null> {
  const db = getDatabase();
  const result = await db.execute(`
    SELECT m.* FROM memories m
    JOIN memory_relationships r ON r.source_memory_id = m.id
    WHERE r.target_memory_id = ?
      AND r.relationship_type = 'SUPERSEDES'
      AND r.valid_until IS NULL
      AND m.is_deleted = 0
    ORDER BY r.created_at DESC
    LIMIT 1
  `, [memoryId]);

  if (result.rows.length === 0) return null;
  return rowToMemory(result.rows[0]);
}
```

### Test Specification

```typescript
// src/services/memory/__test__/relationships.test.ts (colocated)
describe('Memory Relationships', () => {
  test('supersede marks old memory with valid_until', async () => {
    const old = await store.create({ content: 'Old fact' }, 'proj1');
    const newMem = await store.create({ content: 'New fact' }, 'proj1');

    await supersede(old.id, newMem.id);

    const updated = await store.get(old.id);
    expect(updated!.validUntil).toBeDefined();
  });

  test('getSupersedingMemory returns newer version', async () => {
    const old = await store.create({ content: 'Old fact' }, 'proj1');
    const newMem = await store.create({ content: 'New fact' }, 'proj1');
    await supersede(old.id, newMem.id);

    const superseding = await getSupersedingMemory(old.id);
    expect(superseding!.id).toBe(newMem.id);
  });

  test('createRelationship stores edge', async () => {
    const mem1 = await store.create({ content: 'Fact 1' }, 'proj1');
    const mem2 = await store.create({ content: 'Fact 2' }, 'proj1');

    const rel = await createRelationship(mem1.id, mem2.id, 'RELATED_TO', 'user');

    expect(rel.relationshipType).toBe('RELATED_TO');
    expect(rel.confidence).toBe(1.0);
  });
});
```

## Salience Decay

### Interface

```typescript
// src/services/memory/decay.ts
type DecayConfig = {
  enabled: boolean;
  interval: number;  // ms between decay runs
  batchSize: number;
};

function calculateDecay(memory: Memory): number;
function applyDecay(memories: Memory[]): Promise<void>;
function startDecayProcess(config?: DecayConfig): () => void;  // Returns stop function
```

### Implementation Notes

```typescript
function calculateDecay(memory: Memory): number {
  const daysSinceAccess = (Date.now() - memory.lastAccessed) / (1000 * 60 * 60 * 24);
  const decayRate = SECTOR_DECAY_RATES[memory.sector];

  // Exponential decay modulated by importance
  const effectiveDecayRate = decayRate / (memory.importance + 0.1);
  const decayed = memory.salience * Math.exp(-effectiveDecayRate * daysSinceAccess);

  // Access count provides protection (diminishing)
  const accessProtection = Math.min(0.1, Math.log1p(memory.accessCount) * 0.02);

  return Math.max(0.05, Math.min(1.0, decayed + accessProtection));
}

function startDecayProcess(config: DecayConfig = {
  enabled: true,
  interval: 60 * 60 * 1000,  // 1 hour
  batchSize: 100,
}): () => void {
  if (!config.enabled) return () => {};

  const runDecay = async () => {
    const db = getDatabase();

    const result = await db.execute(`
      SELECT * FROM memories
      WHERE salience > 0.05 AND is_deleted = 0
      ORDER BY updated_at ASC
      LIMIT ?
    `, [config.batchSize]);

    const memories = result.rows.map(rowToMemory);
    await applyDecay(memories);
  };

  const interval = setInterval(runDecay, config.interval);
  runDecay();

  return () => clearInterval(interval);
}
```

### Test Specification

```typescript
// src/services/memory/__test__/decay.test.ts (colocated)
describe('Salience Decay', () => {
  test('emotional memories decay slowest', () => {
    const emotional: Memory = {
      ...mockMemory,
      sector: 'emotional',
      salience: 1.0,
      lastAccessed: Date.now() - 7 * 24 * 60 * 60 * 1000,
    };

    const episodic: Memory = {
      ...mockMemory,
      sector: 'episodic',
      salience: 1.0,
      lastAccessed: Date.now() - 7 * 24 * 60 * 60 * 1000,
    };

    expect(calculateDecay(emotional)).toBeGreaterThan(calculateDecay(episodic));
  });

  test('higher importance slows decay', () => {
    const highImportance: Memory = {
      ...mockMemory,
      importance: 0.9,
      salience: 1.0,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
    };

    const lowImportance: Memory = {
      ...mockMemory,
      importance: 0.1,
      salience: 1.0,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
    };

    expect(calculateDecay(highImportance)).toBeGreaterThan(calculateDecay(lowImportance));
  });

  test('access count provides protection', () => {
    const highAccess: Memory = {
      ...mockMemory,
      accessCount: 50,
      salience: 0.5,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
    };

    const lowAccess: Memory = {
      ...mockMemory,
      accessCount: 1,
      salience: 0.5,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
    };

    expect(calculateDecay(highAccess)).toBeGreaterThan(calculateDecay(lowAccess));
  });

  test('salience has minimum floor', () => {
    const ancient: Memory = {
      ...mockMemory,
      salience: 1.0,
      lastAccessed: Date.now() - 365 * 24 * 60 * 60 * 1000,
    };

    expect(calculateDecay(ancient)).toBeGreaterThanOrEqual(0.05);
  });
});
```

## Acceptance Criteria

- [ ] Memory CRUD operations work correctly
- [ ] Auto-classification assigns appropriate sectors
- [ ] Simhash deduplication prevents duplicates
- [ ] Similar content reinforces existing memory salience
- [ ] Concepts extracted from content
- [ ] Decay reduces salience over time based on sector
- [ ] Importance modulates decay rate
- [ ] Access count protects from decay
- [ ] Reinforce and deemphasize adjust salience correctly
- [ ] Soft delete works (is_deleted flag)
- [ ] Hard delete removes permanently
- [ ] Restore recovers soft-deleted memories
- [ ] Session-memory links track usage types
- [ ] Supersede creates relationship and marks valid_until
- [ ] getSupersedingMemory returns newer version
- [ ] Bi-temporal queries work (valid_from/valid_until)
