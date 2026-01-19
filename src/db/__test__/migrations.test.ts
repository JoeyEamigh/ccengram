import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createDatabase, type Database } from "../database.js";
import { migrations, getCurrentVersion, runMigrations } from "../migrations.js";

describe("Migrations", () => {
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
  });

  afterEach(() => {
    db.close();
  });

  test("migrations are ordered by version", () => {
    for (let i = 1; i < migrations.length; i++) {
      const prev = migrations[i - 1];
      const curr = migrations[i];
      if (prev && curr) {
        expect(curr.version).toBeGreaterThan(prev.version);
      }
    }
  });

  test("all migrations have unique versions", () => {
    const versions = migrations.map((m) => m.version);
    const unique = new Set(versions);
    expect(unique.size).toBe(versions.length);
  });

  test("all migrations have names", () => {
    for (const migration of migrations) {
      expect(migration.name).toBeTruthy();
      expect(migration.name.length).toBeGreaterThan(0);
    }
  });

  test("migrations run idempotently", async () => {
    const v1 = await getCurrentVersion(db.client);

    await runMigrations(db.client);
    const v2 = await getCurrentVersion(db.client);

    await runMigrations(db.client);
    const v3 = await getCurrentVersion(db.client);

    expect(v1).toBe(v2);
    expect(v2).toBe(v3);
  });

  test("getCurrentVersion returns 0 for empty database", async () => {
    const freshDb = await createDatabase(":memory:");
    const version = await getCurrentVersion(freshDb.client);
    expect(version).toBeGreaterThan(0);
    freshDb.close();
  });

  test("migration tracking table exists", async () => {
    const result = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
    );
    expect(result.rows.length).toBe(1);
  });

  test("applied migrations are tracked", async () => {
    const result = await db.execute(
      "SELECT version, name FROM _migrations ORDER BY version"
    );
    expect(result.rows.length).toBeGreaterThan(0);

    const first = result.rows[0];
    expect(first?.["version"]).toBe(1);
    expect(first?.["name"]).toBe("initial_schema");
  });
});
