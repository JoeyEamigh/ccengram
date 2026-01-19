import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { homedir } from "node:os";
import { join } from "node:path";
import { getPaths, ensureDirectories } from "../paths.js";
import { rm, stat } from "node:fs/promises";

describe("XDG Paths", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    delete process.env["XDG_DATA_HOME"];
    delete process.env["XDG_CONFIG_HOME"];
    delete process.env["XDG_CACHE_HOME"];
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  test("uses XDG_DATA_HOME when set", () => {
    process.env["XDG_DATA_HOME"] = "/tmp/test-data";
    const paths = getPaths();
    expect(paths.data).toBe("/tmp/test-data/ccmemory");
  });

  test("uses XDG_CONFIG_HOME when set", () => {
    process.env["XDG_CONFIG_HOME"] = "/tmp/test-config";
    const paths = getPaths();
    expect(paths.config).toBe("/tmp/test-config/ccmemory");
  });

  test("uses XDG_CACHE_HOME when set", () => {
    process.env["XDG_CACHE_HOME"] = "/tmp/test-cache";
    const paths = getPaths();
    expect(paths.cache).toBe("/tmp/test-cache/ccmemory");
  });

  test("falls back to platform defaults on Linux", () => {
    const home = homedir();
    const paths = getPaths();

    if (process.platform === "linux") {
      expect(paths.data).toBe(join(home, ".local", "share", "ccmemory"));
      expect(paths.config).toBe(join(home, ".config", "ccmemory"));
      expect(paths.cache).toBe(join(home, ".cache", "ccmemory"));
    }
  });

  test("database path is under data directory", () => {
    const paths = getPaths();
    expect(paths.db).toBe(`${paths.data}/memories.db`);
  });

  test("all paths include ccmemory suffix", () => {
    const paths = getPaths();
    expect(paths.config).toMatch(/ccmemory$/);
    expect(paths.data).toMatch(/ccmemory$/);
    expect(paths.cache).toMatch(/ccmemory$/);
  });
});

describe("ensureDirectories", () => {
  const testBase = "/tmp/ccmemory-test-" + Date.now();

  beforeEach(() => {
    process.env["XDG_DATA_HOME"] = join(testBase, "data");
    process.env["XDG_CONFIG_HOME"] = join(testBase, "config");
    process.env["XDG_CACHE_HOME"] = join(testBase, "cache");
  });

  afterEach(async () => {
    await rm(testBase, { recursive: true, force: true });
  });

  test("creates all directories", async () => {
    const paths = getPaths();
    await ensureDirectories();

    const configStat = await stat(paths.config);
    const dataStat = await stat(paths.data);
    const cacheStat = await stat(paths.cache);

    expect(configStat.isDirectory()).toBe(true);
    expect(dataStat.isDirectory()).toBe(true);
    expect(cacheStat.isDirectory()).toBe(true);
  });

  test("is idempotent", async () => {
    await ensureDirectories();
    await ensureDirectories();
    const paths = getPaths();
    const dataStat = await stat(paths.data);
    expect(dataStat.isDirectory()).toBe(true);
  });
});
