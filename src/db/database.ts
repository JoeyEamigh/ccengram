import { createClient, type Client, type InArgs, type ResultSet } from "@libsql/client";
import { getPaths, ensureDirectories } from "../utils/paths.js";
import { runMigrations } from "./migrations.js";

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
        statements.map((s) => ({ sql: s.sql, args: s.args ?? [] })),
        "write"
      );
    },
    async transaction<T>(fn: (tx: TransactionClient) => Promise<T>): Promise<T> {
      const tx = await client.transaction("write");
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

  if (actualPath !== ":memory:") {
    await ensureDirectories();
  }

  const client = createClient({
    url: actualPath === ":memory:" ? ":memory:" : `file:${actualPath}`,
  });

  await client.execute("PRAGMA journal_mode=WAL");
  await client.execute("PRAGMA busy_timeout=5000");
  await client.execute("PRAGMA synchronous=NORMAL");
  await client.execute("PRAGMA foreign_keys=ON");

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
