import { getDatabase } from '../../db/database.js';
import { log } from '../../utils/log.js';
import type { Memory, MemorySector } from './types.js';
import { SECTOR_DECAY_RATES } from './types.js';
import { rowToMemory } from './utils.js';

export type DecayConfig = {
  enabled: boolean;
  interval: number;
  batchSize: number;
};

const DEFAULT_DECAY_CONFIG: DecayConfig = {
  enabled: true,
  interval: 60 * 60 * 1000,
  batchSize: 100,
};

const DEFAULT_DECAY_INTERVAL_HOURS = 24;
const MIN_SALIENCE = 0.05;

export function calculateDecay(memory: Memory): number {
  const daysSinceAccess = (Date.now() - memory.lastAccessed) / (1000 * 60 * 60 * 24);
  const decayRate = SECTOR_DECAY_RATES[memory.sector];

  const effectiveDecayRate = decayRate / (memory.importance + 0.1);
  const decayed = memory.salience * Math.exp(-effectiveDecayRate * daysSinceAccess);

  const accessProtection = Math.min(0.1, Math.log1p(memory.accessCount) * 0.02);

  const finalSalience = Math.max(0.05, Math.min(1.0, decayed + accessProtection));

  return finalSalience;
}

export function calculateSalienceBoost(currentSalience: number, amount: number): number {
  const boosted = currentSalience + amount * (1.0 - currentSalience);
  return Math.min(1.0, boosted);
}

export async function applyDecay(memories: Memory[]): Promise<void> {
  if (memories.length === 0) return;

  const db = await getDatabase();
  const now = Date.now();

  log.debug('decay', 'Applying decay', { count: memories.length });

  const statements = memories.map(memory => {
    const newSalience = calculateDecay(memory);
    return {
      sql: `UPDATE memories SET salience = ?, updated_at = ? WHERE id = ?`,
      args: [newSalience, now, memory.id],
    };
  });

  await db.batch(statements);

  log.info('decay', 'Decay applied', { count: memories.length });
}

export async function getMemoriesForDecay(batchSize: number): Promise<Memory[]> {
  const db = await getDatabase();

  const result = await db.execute(
    `SELECT * FROM memories
     WHERE salience > 0.05 AND is_deleted = 0
     ORDER BY updated_at ASC
     LIMIT ?`,
    [batchSize],
  );

  return result.rows.map(rowToMemory);
}

export function startDecayProcess(config: Partial<DecayConfig> = {}): () => void {
  const finalConfig: DecayConfig = { ...DEFAULT_DECAY_CONFIG, ...config };

  if (!finalConfig.enabled) {
    log.info('decay', 'Decay process disabled');
    return () => {};
  }

  log.info('decay', 'Starting decay process', {
    interval: finalConfig.interval,
    batchSize: finalConfig.batchSize,
  });

  let stopped = false;

  const runDecay = async (): Promise<void> => {
    if (stopped) return;

    try {
      const memories = await getMemoriesForDecay(finalConfig.batchSize);
      if (memories.length > 0) {
        await applyDecay(memories);
      }
    } catch (error) {
      log.error('decay', 'Decay process error', {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  };

  runDecay();

  const interval = setInterval(runDecay, finalConfig.interval);

  return () => {
    stopped = true;
    clearInterval(interval);
    log.info('decay', 'Decay process stopped');
  };
}

export function getDecayRateForSector(sector: MemorySector): number {
  return SECTOR_DECAY_RATES[sector];
}

export function estimateTimeToDecay(memory: Memory, targetSalience: number): number {
  if (memory.salience <= targetSalience) {
    return 0;
  }

  const decayRate = SECTOR_DECAY_RATES[memory.sector];
  const effectiveDecayRate = decayRate / (memory.importance + 0.1);

  const accessProtection = Math.min(0.1, Math.log1p(memory.accessCount) * 0.02);
  const adjustedTarget = targetSalience - accessProtection;

  if (adjustedTarget <= 0) {
    return Infinity;
  }

  const daysToDecay = Math.log(memory.salience / adjustedTarget) / effectiveDecayRate;

  return daysToDecay * 24 * 60 * 60 * 1000;
}

export function calculateEffectiveDecayRate(sector: MemorySector, importance: number): number {
  const baseRate = SECTOR_DECAY_RATES[sector];
  return baseRate / (importance + 0.1);
}

export function calculateNextDecayAt(
  currentSalience: number,
  decayRate: number,
  accessCount: number,
  intervalHours = DEFAULT_DECAY_INTERVAL_HOURS,
): number {
  if (currentSalience <= MIN_SALIENCE) {
    return Date.now() + 365 * 24 * 60 * 60 * 1000;
  }

  const accessProtection = Math.min(0.1, Math.log1p(accessCount) * 0.02);
  const targetSalience = Math.max(MIN_SALIENCE, currentSalience - 0.05);
  const adjustedTarget = targetSalience - accessProtection;

  if (adjustedTarget <= 0 || decayRate <= 0) {
    return Date.now() + 365 * 24 * 60 * 60 * 1000;
  }

  const daysUntilTarget = Math.log(currentSalience / adjustedTarget) / decayRate;
  const hoursUntilTarget = daysUntilTarget * 24;

  const intervalMs = Math.max(intervalHours, Math.min(hoursUntilTarget, 168)) * 60 * 60 * 1000;

  return Date.now() + intervalMs;
}

export async function applyDecayOptimized(batchSize = 100): Promise<number> {
  const db = await getDatabase();
  const now = Date.now();

  log.debug('decay', 'Starting optimized decay', { batchSize });

  const dueMemories = await db.execute(
    `SELECT id, sector, importance, salience, access_count, last_accessed
     FROM memories
     WHERE is_deleted = 0
       AND salience > ?
       AND (next_decay_at IS NULL OR next_decay_at <= ?)
     ORDER BY next_decay_at ASC NULLS FIRST
     LIMIT ?`,
    [MIN_SALIENCE, now, batchSize],
  );

  if (dueMemories.rows.length === 0) {
    log.debug('decay', 'No memories due for decay');
    return 0;
  }

  log.debug('decay', 'Processing decay batch', { count: dueMemories.rows.length });

  const updates: { sql: string; args: (string | number)[] }[] = [];

  for (const row of dueMemories.rows) {
    const id = String(row['id']);
    const sector = String(row['sector']) as MemorySector;
    const importance = Number(row['importance'] ?? 0.5);
    const currentSalience = Number(row['salience'] ?? 1.0);
    const accessCount = Number(row['access_count'] ?? 0);
    const lastAccessed = Number(row['last_accessed'] ?? now);

    const daysSinceAccess = (now - lastAccessed) / (1000 * 60 * 60 * 24);
    const effectiveDecayRate = calculateEffectiveDecayRate(sector, importance);

    const decayedSalience = currentSalience * Math.exp(-effectiveDecayRate * daysSinceAccess);
    const accessProtection = Math.min(0.1, Math.log1p(accessCount) * 0.02);
    const newSalience = Math.max(MIN_SALIENCE, Math.min(1.0, decayedSalience + accessProtection));

    const nextDecayAt = calculateNextDecayAt(newSalience, effectiveDecayRate, accessCount);

    updates.push({
      sql: `UPDATE memories
            SET salience = ?,
                decay_rate = ?,
                next_decay_at = ?,
                updated_at = ?
            WHERE id = ?`,
      args: [newSalience, effectiveDecayRate, nextDecayAt, now, id],
    });
  }

  await db.batch(updates);

  log.info('decay', 'Optimized decay complete', {
    processed: updates.length,
    ms: Date.now() - now,
  });

  return updates.length;
}

export async function initializeDecayScheduling(projectId?: string): Promise<number> {
  const db = await getDatabase();
  const now = Date.now();

  log.info('decay', 'Initializing decay scheduling', { projectId });

  let sql = `
    SELECT id, sector, importance, salience, access_count
    FROM memories
    WHERE is_deleted = 0
      AND next_decay_at IS NULL
      AND salience > ?
  `;
  const args: (string | number)[] = [MIN_SALIENCE];

  if (projectId) {
    sql += ' AND project_id = ?';
    args.push(projectId);
  }

  sql += ' LIMIT 1000';

  const memories = await db.execute(sql, args);

  if (memories.rows.length === 0) {
    return 0;
  }

  const updates: { sql: string; args: (string | number)[] }[] = [];

  for (const row of memories.rows) {
    const id = String(row['id']);
    const sector = String(row['sector']) as MemorySector;
    const importance = Number(row['importance'] ?? 0.5);
    const currentSalience = Number(row['salience'] ?? 1.0);
    const accessCount = Number(row['access_count'] ?? 0);

    const effectiveDecayRate = calculateEffectiveDecayRate(sector, importance);
    const nextDecayAt = calculateNextDecayAt(currentSalience, effectiveDecayRate, accessCount);

    updates.push({
      sql: `UPDATE memories SET decay_rate = ?, next_decay_at = ? WHERE id = ?`,
      args: [effectiveDecayRate, nextDecayAt, id],
    });
  }

  await db.batch(updates);

  log.info('decay', 'Decay scheduling initialized', {
    processed: updates.length,
    ms: Date.now() - now,
  });

  return updates.length;
}

export async function getMemoriesDueForDecay(limit = 100): Promise<number> {
  const db = await getDatabase();
  const now = Date.now();

  const result = await db.execute(
    `SELECT COUNT(*) as count
     FROM memories
     WHERE is_deleted = 0
       AND salience > ?
       AND (next_decay_at IS NULL OR next_decay_at <= ?)`,
    [MIN_SALIENCE, now],
  );

  const row = result.rows[0];
  return row ? Number(row['count']) : 0;
}
