import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { getPaths } from "./paths.js";

export type LogLevel = "debug" | "info" | "warn" | "error";

export type LogContext = Record<string, unknown>;

export type Logger = {
  debug(module: string, message: string, context?: LogContext): void;
  info(module: string, message: string, context?: LogContext): void;
  warn(module: string, message: string, context?: LogContext): void;
  error(module: string, message: string, context?: LogContext): void;
  setLevel(level: LogLevel): void;
  getLevel(): LogLevel;
  flush(): Promise<void>;
};

const LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

function parseLogLevel(str: string | undefined): LogLevel {
  if (str && str in LEVELS) {
    return str as LogLevel;
  }
  return "info";
}

let currentLevel: LogLevel = parseLogLevel(process.env["LOG_LEVEL"]);
let logPath: string | null = null;
let pendingWrites: Promise<void>[] = [];
let directoryEnsured = false;

export function getLogPath(): string {
  if (!logPath) {
    const paths = getPaths();
    logPath = `${paths.data}/ccmemory.log`;
  }
  return logPath;
}

export function setLogPath(path: string): void {
  logPath = path;
  directoryEnsured = false;
}

async function ensureLogDirectory(): Promise<void> {
  if (directoryEnsured) return;
  const path = getLogPath();
  await mkdir(dirname(path), { recursive: true });
  directoryEnsured = true;
}

function formatLine(
  level: LogLevel,
  module: string,
  message: string,
  context?: LogContext
): string {
  const timestamp = new Date().toISOString();
  const pid = process.pid;
  const contextStr = context ? ` ${JSON.stringify(context)}` : "";
  return `[${timestamp}] [${level.toUpperCase().padEnd(5)}] [${pid}:${module}] ${message}${contextStr}\n`;
}

function shouldLog(level: LogLevel): boolean {
  return LEVELS[level] >= LEVELS[currentLevel];
}

async function writeLog(
  level: LogLevel,
  module: string,
  message: string,
  context?: LogContext
): Promise<void> {
  if (!shouldLog(level)) return;

  const line = formatLine(level, module, message, context);

  try {
    await ensureLogDirectory();
    const path = getLogPath();
    await appendFile(path, line);
  } catch (err) {
    process.stderr.write(`[LOG WRITE FAILED] ${line}`);
  }
}

function logAsync(
  level: LogLevel,
  module: string,
  message: string,
  context?: LogContext
): void {
  const promise = writeLog(level, module, message, context);
  pendingWrites.push(promise);
  promise.finally(() => {
    const idx = pendingWrites.indexOf(promise);
    if (idx >= 0) pendingWrites.splice(idx, 1);
  });
}

export const log: Logger = {
  debug(module, message, context) {
    logAsync("debug", module, message, context);
  },
  info(module, message, context) {
    logAsync("info", module, message, context);
  },
  warn(module, message, context) {
    logAsync("warn", module, message, context);
  },
  error(module, message, context) {
    logAsync("error", module, message, context);
  },
  setLevel(level) {
    currentLevel = level;
  },
  getLevel() {
    return currentLevel;
  },
  async flush() {
    await Promise.all(pendingWrites);
  },
};
