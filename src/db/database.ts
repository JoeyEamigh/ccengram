import { createClient, type Client, type InArgs, type ResultSet } from '@libsql/client';
import { log } from '../utils/log.js';
import { ensureDirectories, getPaths } from '../utils/paths.js';
import { runMigrations } from './migrations.js';

export type DatabaseStatement = {
  sql: string;
  args?: InArgs;
};

export type TransactionClient = {
  execute(sql: string, args?: InArgs): Promise<ResultSet>;
};

export type Database = {
  client: Client;
  execute(sql: string, args?: InArgs): Promise<ResultSet>;
  batch(statements: DatabaseStatement[]): Promise<ResultSet[]>;
  transaction<T>(fn: (tx: TransactionClient) => Promise<T>): Promise<T>;
  close(): void;
};

let singleton: Database | null = null;

function wrapClient(client: Client): Database {
  return {
    client,
    async execute(sql: string, args?: InArgs): Promise<ResultSet> {
      return client.execute({ sql, args: args ?? [] });
    },
    async batch(statements: DatabaseStatement[]): Promise<ResultSet[]> {
      return client.batch(
        statements.map(s => ({ sql: s.sql, args: s.args ?? [] })),
        'write',
      );
    },
    async transaction<T>(fn: (tx: TransactionClient) => Promise<T>): Promise<T> {
      const tx = await client.transaction('write');
      try {
        const wrappedTx: TransactionClient = {
          async execute(sql: string, args?: InArgs): Promise<ResultSet> {
            return tx.execute({ sql, args: args ?? [] });
          },
        };
        const result = await fn(wrappedTx);
        await tx.commit();
        return result;
      } catch (err) {
        await tx.rollback();
        throw err;
      }
    },
    close(): void {
      client.close();
    },
  };
}

export async function createDatabase(dbPath?: string): Promise<Database> {
  const paths = getPaths();
  const actualPath = dbPath ?? paths.db;

  if (actualPath !== ':memory:') {
    await ensureDirectories();
  }

  const client = createClient({
    url: actualPath === ':memory:' ? ':memory:' : `file:${actualPath}`,
  });

  await client.execute('PRAGMA journal_mode=WAL');
  await client.execute('PRAGMA busy_timeout=5000');
  await client.execute('PRAGMA synchronous=NORMAL');
  await client.execute('PRAGMA foreign_keys=ON');

  await runMigrations(client);

  return wrapClient(client);
}

export async function getDatabase(): Promise<Database> {
  if (!singleton) {
    singleton = await createDatabase();
  }
  return singleton;
}

export function closeDatabase(): void {
  if (singleton) {
    singleton.close();
    singleton = null;
  }
}

export function setDatabase(db: Database): void {
  singleton = db;
}

export type IntegrityCheckResult = {
  ok: boolean;
  errors: string[];
};

export async function checkIntegrity(db: Database): Promise<IntegrityCheckResult> {
  const errors: string[] = [];

  try {
    const result = await db.execute('PRAGMA integrity_check');
    for (const row of result.rows) {
      const value = String(row['integrity_check'] ?? row[0]);
      if (value !== 'ok') {
        errors.push(value);
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    errors.push(`Integrity check failed: ${message}`);
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}

export async function checkpointWAL(db: Database): Promise<boolean> {
  try {
    await db.execute('PRAGMA wal_checkpoint(TRUNCATE)');
    log.debug('db', 'WAL checkpoint completed');
    return true;
  } catch (error) {
    log.warn('db', 'WAL checkpoint failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

export type RecoveryResult = {
  success: boolean;
  message: string;
  recoveredRows?: number;
};

export async function recoverDatabase(dbPath: string): Promise<RecoveryResult> {
  log.info('db', 'Starting database recovery', { path: dbPath });

  const backupPath = `${dbPath}.backup-${Date.now()}`;
  const recoveryPath = `${dbPath}.recovery`;

  try {
    const { copyFile, unlink } = await import('node:fs/promises');

    await copyFile(dbPath, backupPath);
    log.debug('db', 'Created backup', { backupPath });

    const sourceClient = createClient({ url: `file:${dbPath}` });
    const recoveryClient = createClient({ url: `file:${recoveryPath}` });

    await recoveryClient.execute('PRAGMA journal_mode=WAL');

    const tableResult = await sourceClient.execute(
      `SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`,
    );

    let recoveredRows = 0;
    const failedTables: string[] = [];

    for (const tableRow of tableResult.rows) {
      const tableName = String(tableRow['name']);
      const createSql = String(tableRow['sql']);

      try {
        await recoveryClient.execute(createSql);

        const dataResult = await sourceClient.execute(`SELECT * FROM "${tableName}"`);
        for (const row of dataResult.rows) {
          const columns = Object.keys(row);
          const placeholders = columns.map(() => '?').join(', ');
          const values = columns.map(c => row[c]);

          await recoveryClient.execute({
            sql: `INSERT OR IGNORE INTO "${tableName}" (${columns.join(', ')}) VALUES (${placeholders})`,
            args: values as InArgs,
          });
          recoveredRows++;
        }

        log.debug('db', 'Recovered table', { table: tableName, rows: dataResult.rows.length });
      } catch (error) {
        log.warn('db', 'Failed to recover table', {
          table: tableName,
          error: error instanceof Error ? error.message : String(error),
        });
        failedTables.push(tableName);
      }
    }

    sourceClient.close();
    recoveryClient.close();

    try {
      await copyFile(recoveryPath, dbPath);
    } finally {
      try {
        await unlink(recoveryPath);
      } catch (cleanupError) {
        log.warn('db', 'Failed to clean up recovery file', {
          path: recoveryPath,
          error: cleanupError instanceof Error ? cleanupError.message : String(cleanupError),
        });
      }
    }

    log.info('db', 'Database recovery completed', { recoveredRows, failedTables });

    return {
      success: true,
      message:
        failedTables.length > 0
          ? `Recovery completed with ${failedTables.length} failed tables: ${failedTables.join(', ')}`
          : 'Recovery completed successfully',
      recoveredRows,
    };
  } catch (error) {
    log.error('db', 'Database recovery failed', {
      error: error instanceof Error ? error.message : String(error),
    });

    return {
      success: false,
      message: `Recovery failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

const MAX_BUSY_RETRIES = 5;

export async function createDatabaseWithRecovery(dbPath?: string, retryCount = 0): Promise<Database> {
  const paths = getPaths();
  const actualPath = dbPath ?? paths.db;

  if (actualPath === ':memory:') {
    return createDatabase(actualPath);
  }

  try {
    const db = await createDatabase(actualPath);
    const integrity = await checkIntegrity(db);

    if (!integrity.ok) {
      log.warn('db', 'Database integrity check failed, attempting recovery', {
        errors: integrity.errors,
      });
      db.close();

      const recovery = await recoverDatabase(actualPath);
      if (!recovery.success) {
        throw new Error(`Database recovery failed: ${recovery.message}`);
      }

      return createDatabase(actualPath);
    }

    return db;
  } catch (error) {
    if (error instanceof Error && error.message.includes('SQLITE_BUSY')) {
      if (retryCount >= MAX_BUSY_RETRIES) {
        log.error('db', 'Database still locked after max retries', {
          path: actualPath,
          retries: retryCount,
        });
        throw new Error(`Database locked after ${MAX_BUSY_RETRIES} retries: ${actualPath}`);
      }

      log.warn('db', 'Database is locked, waiting for availability', {
        path: actualPath,
        retry: retryCount + 1,
        maxRetries: MAX_BUSY_RETRIES,
      });
      await new Promise(r => setTimeout(r, 1000));
      return createDatabaseWithRecovery(dbPath, retryCount + 1);
    }

    throw error;
  }
}
