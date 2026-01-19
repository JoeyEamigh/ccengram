import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { rm, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { log, getLogPath, setLogPath } from "../log.js";

describe("Logger", () => {
  const testDir = `/tmp/ccmemory-log-test-${Date.now()}`;
  const testLogPath = join(testDir, "test.log");
  const originalEnv = { ...process.env };

  beforeEach(async () => {
    setLogPath(testLogPath);
    log.setLevel("debug");
    await rm(testDir, { recursive: true, force: true });
  });

  afterEach(async () => {
    await log.flush();
    await rm(testDir, { recursive: true, force: true });
    process.env = { ...originalEnv };
  });

  test("writes to log file", async () => {
    log.info("test", "hello world");
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    expect(content).toContain("[INFO ]");
    expect(content).toContain("hello world");
  });

  test("includes module name", async () => {
    log.info("mymodule", "test message");
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    expect(content).toContain(":mymodule]");
  });

  test("respects log level - filters debug when level is warn", async () => {
    log.setLevel("warn");
    log.debug("test", "should not appear");
    log.info("test", "should not appear either");
    log.warn("test", "should appear");
    await log.flush();

    let content = "";
    try {
      content = await readFile(testLogPath, "utf-8");
    } catch {
      content = "";
    }
    expect(content).not.toContain("should not appear");
    expect(content).toContain("should appear");
  });

  test("includes PID for multi-instance support", async () => {
    log.info("test", "pid test");
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    expect(content).toContain(`[${process.pid}:test]`);
  });

  test("serializes context as JSON", async () => {
    log.info("test", "with context", { count: 5, name: "foo" });
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    expect(content).toContain('{"count":5,"name":"foo"}');
  });

  test("handles all log levels", async () => {
    log.debug("test", "debug msg");
    log.info("test", "info msg");
    log.warn("test", "warn msg");
    log.error("test", "error msg");
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    expect(content).toContain("[DEBUG]");
    expect(content).toContain("[INFO ]");
    expect(content).toContain("[WARN ]");
    expect(content).toContain("[ERROR]");
  });

  test("creates log directory if missing", async () => {
    const nestedPath = join(testDir, "nested", "deep", "test.log");
    setLogPath(nestedPath);

    log.info("test", "creating nested");
    await log.flush();

    const dirStat = await stat(join(testDir, "nested", "deep"));
    expect(dirStat.isDirectory()).toBe(true);
  });

  test("getLevel returns current level", () => {
    log.setLevel("error");
    expect(log.getLevel()).toBe("error");
    log.setLevel("debug");
    expect(log.getLevel()).toBe("debug");
  });

  test("flush waits for pending writes", async () => {
    for (let i = 0; i < 10; i++) {
      log.info("test", `message ${i}`);
    }
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    const lines = content.trim().split("\n");
    expect(lines.length).toBe(10);
  });

  test("includes ISO timestamp", async () => {
    log.info("test", "timestamp check");
    await log.flush();

    const content = await readFile(testLogPath, "utf-8");
    expect(content).toMatch(/\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\]/);
  });
});

describe("Logger Level from Environment", () => {
  test("getLogPath returns path under data directory", () => {
    const path = getLogPath();
    expect(path).toContain("ccmemory");
    expect(path).toMatch(/\.log$/);
  });
});
