# CLI Specification

## Overview

The CLI provides standalone access to CCMemory for searching, managing, and diagnosing the memory system.

## Files to Create

- `src/cli/index.ts` - Entry point
- `src/cli/commands/search.ts` - Search command
- `src/cli/commands/config.ts` - Config management
- `src/cli/commands/import.ts` - Import/export
- `src/cli/commands/health.ts` - Diagnostics

## CLI Interface

```bash
ccmemory <command> [options]

Commands:
  search <query>           Search memories
  show <id>               Show memory details
  delete <id>             Delete a memory
  archive                 Archive old memories

  import <file>           Import document
  export                  Export memories

  config [key] [value]    View/set configuration
  health                  Check system health
  stats                   Show statistics

  serve                   Start WebUI server
```

## Command Implementations

### Search Command

```typescript
// src/cli/commands/search.ts
import { parseArgs } from "util";
import { createSearchService } from "../../services/search/hybrid";
import { getOrCreateProject } from "../../services/project";
import { log } from "../../utils/log";

export async function searchCommand(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      project: { type: "string", short: "p" },
      type: { type: "string", short: "t" },
      limit: { type: "string", short: "l", default: "10" },
      semantic: { type: "boolean" },
      keywords: { type: "boolean" },
      json: { type: "boolean" }
    },
    allowPositionals: true
  });

  const query = positionals.join(" ");
  if (!query) {
    console.error("Usage: ccmemory search <query> [-p project] [-t type]");
    process.exit(1);
  }

  const search = createSearchService();
  const projectId = values.project
    ? (await getOrCreateProject(values.project)).id
    : undefined;

  const mode = values.semantic ? "semantic"
    : values.keywords ? "keyword"
    : "hybrid";

  log.debug("cli", "Search command", { query: query.slice(0, 50), mode, projectId, limit: values.limit });

  const results = await search.search({
    query,
    projectId,
    type: values.type as any,
    limit: parseInt(values.limit as string),
    mode
  });

  log.info("cli", "Search complete", { results: results.length, query: query.slice(0, 30) });

  if (values.json) {
    console.log(JSON.stringify(results, null, 2));
  } else {
    if (results.length === 0) {
      console.log("No memories found.");
      return;
    }

    for (const result of results) {
      const mem = result.memory;
      console.log(`\n${"─".repeat(60)}`);
      console.log(`ID: ${mem.id}`);
      console.log(`Type: ${mem.type} | Score: ${result.score.toFixed(3)} | Salience: ${mem.salience.toFixed(2)}`);
      console.log(`Created: ${new Date(mem.createdAt).toLocaleString()}`);
      console.log(`\n${mem.content}`);
    }
  }
}
```

### Show Command

```typescript
// src/cli/commands/show.ts
import { parseArgs } from "util";
import { createMemoryStore } from "../../services/memory/store";
import { createSearchService } from "../../services/search/hybrid";
import { log } from "../../utils/log";

export async function showCommand(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      related: { type: "boolean", short: "r" },
      json: { type: "boolean" }
    },
    allowPositionals: true
  });

  const id = positionals[0];
  if (!id) {
    console.error("Usage: ccmemory show <id> [--related]");
    process.exit(1);
  }

  log.debug("cli", "Show command", { id, related: values.related });

  const store = createMemoryStore();
  const memory = await store.get(id);

  if (!memory) {
    log.warn("cli", "Memory not found", { id });
    console.error(`Memory not found: ${id}`);
    process.exit(1);
  }

  log.debug("cli", "Memory retrieved", { id, sector: memory.sector });

  if (values.json) {
    console.log(JSON.stringify(memory, null, 2));
  } else {
    console.log(`ID: ${memory.id}`);
    console.log(`Type: ${memory.type}`);
    console.log(`Tier: ${memory.tier}`);
    console.log(`Salience: ${memory.salience.toFixed(3)}`);
    console.log(`Access Count: ${memory.accessCount}`);
    console.log(`Created: ${new Date(memory.createdAt).toLocaleString()}`);
    console.log(`Last Accessed: ${new Date(memory.lastAccessed).toLocaleString()}`);
    console.log(`\nContent:\n${memory.content}`);

    if (memory.tags.length > 0) {
      console.log(`\nTags: ${memory.tags.join(", ")}`);
    }
    if (memory.concepts.length > 0) {
      console.log(`Concepts: ${memory.concepts.join(", ")}`);
    }
    if (memory.files.length > 0) {
      console.log(`Files: ${memory.files.join(", ")}`);
    }
  }

  if (values.related) {
    console.log(`\n${"─".repeat(40)}\nRelated memories:\n`);
    const search = createSearchService();
    const timeline = await search.timeline(id, 3, 3);

    for (const mem of timeline) {
      const marker = mem.id === id ? ">>>" : "   ";
      console.log(`${marker} [${new Date(mem.createdAt).toISOString().slice(0, 16)}] ${mem.type}`);
      console.log(`    ${mem.content.slice(0, 100)}...`);
    }
  }
}
```

### Config Command

```typescript
// src/cli/commands/config.ts
import { getPaths } from "../../utils/paths";
import { log } from "../../utils/log";

interface Config {
  embedding: {
    provider: "ollama" | "openrouter";
    ollama: { baseUrl: string; model: string };
    openrouter: { apiKey?: string; model: string };
  };
  capture: {
    enabled: boolean;
    toolMatcher: string;
    maxResultSize: number;
  };
}

export async function configCommand(args: string[]) {
  const paths = getPaths();
  const configPath = `${paths.config}/config.json`;

  // Load existing config
  let config: Config;
  try {
    config = await Bun.file(configPath).json();
  } catch {
    config = getDefaultConfig();
  }

  if (args.length === 0) {
    // Show all config
    console.log(JSON.stringify(config, null, 2));
    return;
  }

  const key = args[0];
  const value = args[1];

  if (!value) {
    // Get specific key
    const val = getNestedValue(config, key);
    console.log(val !== undefined ? JSON.stringify(val) : "Not set");
    return;
  }

  // Set value
  setNestedValue(config, key, parseValue(value));
  await Bun.write(configPath, JSON.stringify(config, null, 2));
  log.info("cli", "Config updated", { key, value });
  console.log(`Set ${key} = ${value}`);
}

function getNestedValue(obj: any, path: string): any {
  return path.split(".").reduce((o, k) => o?.[k], obj);
}

function setNestedValue(obj: any, path: string, value: any): void {
  const keys = path.split(".");
  const last = keys.pop()!;
  const target = keys.reduce((o, k) => o[k] = o[k] || {}, obj);
  target[last] = value;
}

function parseValue(str: string): any {
  if (str === "true") return true;
  if (str === "false") return false;
  if (/^\d+$/.test(str)) return parseInt(str);
  if (/^\d+\.\d+$/.test(str)) return parseFloat(str);
  return str;
}
```

### Health Command

```typescript
// src/cli/commands/health.ts
import { parseArgs } from "util";
import { getDatabase } from "../../db/database";
import { createEmbeddingService } from "../../services/embedding";
import { log } from "../../utils/log";

export async function healthCommand(args: string[]) {
  const { values } = parseArgs({
    args,
    options: { verbose: { type: "boolean", short: "v" } }
  });

  log.debug("cli", "Health check starting");

  const checks: Array<{ name: string; status: "ok" | "warn" | "fail"; message: string }> = [];

  // Check database
  try {
    const db = getDatabase();
    await db.execute("SELECT 1");
    checks.push({ name: "Database", status: "ok", message: "Connected" });

    // Check WAL mode
    const journal = await db.execute("PRAGMA journal_mode");
    if (journal.rows[0][0] === "wal") {
      checks.push({ name: "WAL Mode", status: "ok", message: "Enabled" });
    } else {
      checks.push({ name: "WAL Mode", status: "warn", message: `Mode: ${journal.rows[0][0]}` });
    }
  } catch (err) {
    checks.push({ name: "Database", status: "fail", message: String(err) });
  }

  // Check Ollama
  try {
    const response = await fetch("http://localhost:11434/api/tags");
    if (response.ok) {
      const data = await response.json();
      const models = data.models || [];
      checks.push({
        name: "Ollama",
        status: "ok",
        message: `${models.length} models available`
      });
    } else {
      checks.push({ name: "Ollama", status: "fail", message: "Not responding" });
    }
  } catch {
    checks.push({ name: "Ollama", status: "warn", message: "Not running (OpenRouter fallback available)" });
  }

  // Check embedding model
  try {
    const embedding = await createEmbeddingService();
    checks.push({
      name: "Embedding",
      status: "ok",
      message: `Provider: ${embedding.getProvider().name}, Model: ${embedding.getProvider().model}`
    });
  } catch (err) {
    checks.push({ name: "Embedding", status: "fail", message: String(err) });
  }

  // Display results
  const icons = { ok: "✓", warn: "⚠", fail: "✗" };
  const colors = { ok: "\x1b[32m", warn: "\x1b[33m", fail: "\x1b[31m" };
  const reset = "\x1b[0m";

  console.log("\nCCMemory Health Check\n");

  for (const check of checks) {
    const icon = icons[check.status];
    const color = colors[check.status];
    console.log(`${color}${icon}${reset} ${check.name}: ${check.message}`);
  }

  const failed = checks.filter(c => c.status === "fail").length;
  const warned = checks.filter(c => c.status === "warn").length;

  log.info("cli", "Health check complete", { passed: checks.length - failed - warned, warned, failed });

  console.log();
  if (failed > 0) {
    console.log(`${colors.fail}${failed} check(s) failed${reset}`);
    process.exit(1);
  } else if (warned > 0) {
    console.log(`${colors.warn}${warned} warning(s)${reset}`);
  } else {
    console.log(`${colors.ok}All checks passed${reset}`);
  }
}
```

### Stats Command

```typescript
// src/cli/commands/stats.ts
import { parseArgs } from "util";
import { getDatabase } from "../../db/database";
import { log } from "../../utils/log";

export async function statsCommand(args: string[]) {
  const { values } = parseArgs({
    args,
    options: { project: { type: "string", short: "p" } }
  });

  log.debug("cli", "Stats command", { project: values.project });

  const db = getDatabase();

  // Memory counts by type
  const byType = await db.execute(`
    SELECT memory_type, COUNT(*) as count
    FROM memories
    ${values.project ? "WHERE project_id = ?" : ""}
    GROUP BY memory_type
  `, values.project ? [values.project] : []);

  // Memory counts by tier
  const byTier = await db.execute(`
    SELECT tier, COUNT(*) as count
    FROM memories
    ${values.project ? "WHERE project_id = ?" : ""}
    GROUP BY tier
  `, values.project ? [values.project] : []);

  // Total counts
  const totals = await db.execute(`
    SELECT
      (SELECT COUNT(*) FROM memories) as memories,
      (SELECT COUNT(*) FROM documents) as documents,
      (SELECT COUNT(*) FROM document_chunks) as chunks,
      (SELECT COUNT(*) FROM projects) as projects,
      (SELECT COUNT(*) FROM sessions) as sessions
  `);

  // Salience distribution
  const salience = await db.execute(`
    SELECT
      COUNT(CASE WHEN salience >= 0.8 THEN 1 END) as high,
      COUNT(CASE WHEN salience >= 0.5 AND salience < 0.8 THEN 1 END) as medium,
      COUNT(CASE WHEN salience >= 0.2 AND salience < 0.5 THEN 1 END) as low,
      COUNT(CASE WHEN salience < 0.2 THEN 1 END) as very_low
    FROM memories
  `);

  console.log("\nCCMemory Statistics\n");

  console.log("Totals:");
  console.log(`  Memories: ${totals.rows[0][0]}`);
  console.log(`  Documents: ${totals.rows[0][1]}`);
  console.log(`  Document Chunks: ${totals.rows[0][2]}`);
  console.log(`  Projects: ${totals.rows[0][3]}`);
  console.log(`  Sessions: ${totals.rows[0][4]}`);

  console.log("\nMemories by Type:");
  for (const row of byType.rows) {
    console.log(`  ${row[0]}: ${row[1]}`);
  }

  console.log("\nMemories by Tier:");
  for (const row of byTier.rows) {
    console.log(`  ${row[0]}: ${row[1]}`);
  }

  console.log("\nSalience Distribution:");
  console.log(`  High (≥0.8): ${salience.rows[0][0]}`);
  console.log(`  Medium (0.5-0.8): ${salience.rows[0][1]}`);
  console.log(`  Low (0.2-0.5): ${salience.rows[0][2]}`);
  console.log(`  Very Low (<0.2): ${salience.rows[0][3]}`);
}
```

### Import/Export Commands

```typescript
// src/cli/commands/import.ts
import { parseArgs } from "util";
import { getOrCreateProject } from "../../services/project";
import { createDocumentService } from "../../services/documents/ingest";
import { log } from "../../utils/log";

export async function importCommand(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      project: { type: "string", short: "p" },
      tags: { type: "string", short: "t" }
    },
    allowPositionals: true
  });

  const filePath = positionals[0];
  if (!filePath) {
    console.error("Usage: ccmemory import <file> [-p project] [--tags tag1,tag2]");
    process.exit(1);
  }

  const cwd = values.project || process.cwd();
  const project = await getOrCreateProject(cwd);
  const docs = createDocumentService();

  const tags = values.tags?.split(",").map(t => t.trim());

  log.info("cli", "Importing document", { path: filePath, project: project.id });

  const doc = await docs.ingest({
    projectId: project.id,
    path: filePath
  });

  log.info("cli", "Document imported", { id: doc.id, title: doc.title });
  console.log(`Imported: ${doc.title || doc.id}`);
  console.log(`Chunks: ${await countChunks(doc.id)}`);
}

// src/cli/commands/export.ts
import { parseArgs } from "util";
import { getDatabase } from "../../db/database";
import { getOrCreateProject } from "../../services/project";
import { log } from "../../utils/log";

export async function exportCommand(args: string[]) {
  const { values } = parseArgs({
    args,
    options: {
      project: { type: "string", short: "p" },
      format: { type: "string", short: "f", default: "json" },
      output: { type: "string", short: "o" }
    }
  });

  log.debug("cli", "Export command", { project: values.project, format: values.format });

  const db = getDatabase();

  let sql = "SELECT * FROM memories";
  const sqlArgs: any[] = [];

  if (values.project) {
    const project = await getOrCreateProject(values.project);
    sql += " WHERE project_id = ?";
    sqlArgs.push(project.id);
  }

  sql += " ORDER BY created_at DESC";

  const result = await db.execute(sql, sqlArgs);
  const memories = result.rows.map(rowToMemory);

  let output: string;
  if (values.format === "json") {
    output = JSON.stringify(memories, null, 2);
  } else if (values.format === "csv") {
    const headers = ["id", "type", "tier", "salience", "content", "created_at"];
    const rows = memories.map(m => [
      m.id, m.type, m.tier, m.salience, `"${m.content.replace(/"/g, '""')}"`, m.createdAt
    ]);
    output = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  } else {
    console.error("Unsupported format. Use: json, csv");
    process.exit(1);
  }

  if (values.output) {
    await Bun.write(values.output, output);
    log.info("cli", "Memories exported to file", { count: memories.length, path: values.output });
    console.log(`Exported ${memories.length} memories to ${values.output}`);
  } else {
    log.info("cli", "Memories exported to stdout", { count: memories.length });
    console.log(output);
  }
}
```

## CLI Entry Point

```typescript
// src/cli/index.ts
#!/usr/bin/env bun

import { searchCommand } from "./commands/search";
import { showCommand } from "./commands/show";
import { configCommand } from "./commands/config";
import { healthCommand } from "./commands/health";
import { statsCommand } from "./commands/stats";
import { importCommand } from "./commands/import";
import { exportCommand } from "./commands/export";
import { serveCommand } from "./commands/serve";
import { log } from "../utils/log";

const commands: Record<string, (args: string[]) => Promise<void>> = {
  search: searchCommand,
  show: showCommand,
  delete: deleteCommand,
  archive: archiveCommand,
  import: importCommand,
  export: exportCommand,
  config: configCommand,
  health: healthCommand,
  stats: statsCommand,
  serve: serveCommand
};

async function main() {
  const [command, ...args] = process.argv.slice(2);

  if (!command || command === "help" || command === "--help") {
    printHelp();
    return;
  }

  const handler = commands[command];
  if (!handler) {
    log.warn("cli", "Unknown command", { command });
    console.error(`Unknown command: ${command}`);
    console.error(`Run 'ccmemory help' for usage.`);
    process.exit(1);
  }

  log.debug("cli", "Executing command", { command, args: args.length });
  await handler(args);
}

function printHelp() {
  console.log(`
CCMemory - Claude Code Memory System

Usage: ccmemory <command> [options]

Commands:
  search <query>           Search memories
    -p, --project <path>   Filter by project
    -t, --type <type>      Filter by type (decision|procedure|discovery|preference)
    -l, --limit <n>        Max results (default: 10)
    --semantic             Semantic search only
    --keywords             Keyword search only
    --json                 JSON output

  show <id>               Show memory details
    -r, --related          Show timeline context

  delete <id>             Delete a memory
    --force                Skip confirmation

  archive                 Archive old low-salience memories
    --before <date>        Archive before date
    --dry-run              Preview without changes

  import <file>           Import document
    -p, --project <path>   Associate with project
    --tags <t1,t2>         Add tags

  export                  Export memories
    -p, --project <path>   Filter by project
    -f, --format <fmt>     Format: json, csv
    -o, --output <file>    Output file

  config [key] [value]    View/set configuration
    Examples:
      ccmemory config
      ccmemory config embedding.provider
      ccmemory config embedding.provider ollama

  health                  Check system health
    -v, --verbose          Detailed output

  stats                   Show statistics
    -p, --project <path>   Filter by project

  serve                   Start WebUI server
    --port <port>          Port (default: 37778)
    --open                 Open in browser
`);
}

main().catch(err => {
  log.error("cli", "Command failed", { error: err.message });
  console.error(err);
  process.exit(1);
});
```

## Build Configuration

```json
// package.json (partial)
{
  "bin": {
    "ccmemory": "./dist/cli/index.js"
  },
  "scripts": {
    "build:cli": "bun build src/cli/index.ts --outdir dist/cli --target bun"
  }
}
```

## Test Specification

```typescript
// src/cli/commands/__test__/search.test.ts (colocated unit test)
import { test, expect, describe, beforeEach } from "bun:test";

describe("CLI Search Command", () => {
  beforeEach(async () => {
    await setupTestDatabase();
    await store.create({ content: "Test memory about React", sector: "semantic" }, "proj1");
  });

  test("searches and displays results", async () => {
    const output = await runCLI(["search", "React"]);
    expect(output).toContain("React");
    expect(output).toContain("ID:");
  });

  test("filters by sector", async () => {
    await store.create({ content: "Learning pattern", sector: "procedural" }, "proj1");

    const output = await runCLI(["search", "pattern", "-s", "procedural"]);
    expect(output).toContain("procedural");
  });

  test("outputs JSON when requested", async () => {
    const output = await runCLI(["search", "React", "--json"]);
    const parsed = JSON.parse(output);
    expect(Array.isArray(parsed)).toBe(true);
  });
});

// src/cli/commands/__test__/config.test.ts (colocated unit test)
describe("CLI Config Command", () => {
  test("shows all config", async () => {
    const output = await runCLI(["config"]);
    expect(output).toContain("embedding");
  });

  test("gets specific key", async () => {
    const output = await runCLI(["config", "embedding.provider"]);
    expect(output).toMatch(/ollama|openrouter/);
  });

  test("sets value", async () => {
    await runCLI(["config", "embedding.provider", "openrouter"]);
    const output = await runCLI(["config", "embedding.provider"]);
    expect(output).toContain("openrouter");
  });
});
```

## Acceptance Criteria

- [ ] `search` finds and displays memories
- [ ] `search` supports type/project filters
- [ ] `show` displays full memory details
- [ ] `show --related` shows timeline
- [ ] `delete` removes memories (with confirmation)
- [ ] `import` ingests documents
- [ ] `export` outputs JSON/CSV
- [ ] `config` views and sets configuration
- [ ] `health` reports system status
- [ ] `stats` shows usage statistics
- [ ] `serve` starts WebUI (see 08-webui.md)
- [ ] CLI is installable via `bun link`
