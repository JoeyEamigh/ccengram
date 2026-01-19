#!/usr/bin/env bun

import { searchCommand } from "./cli/commands/search.js";
import { showCommand } from "./cli/commands/show.js";
import { deleteCommand, archiveCommand } from "./cli/commands/delete.js";
import { importCommand, exportCommand } from "./cli/commands/import.js";
import { configCommand } from "./cli/commands/config.js";
import { healthCommand } from "./cli/commands/health.js";
import { statsCommand } from "./cli/commands/stats.js";
import { serveCommand } from "./cli/commands/serve.js";
import { shutdownCommand } from "./cli/commands/shutdown.js";
import { updateCommand } from "./cli/commands/update.js";
import { log } from "./utils/log.js";

const VERSION = "0.1.0";

const cliCommands: Record<string, (args: string[]) => Promise<void>> = {
  search: searchCommand,
  show: showCommand,
  delete: deleteCommand,
  archive: archiveCommand,
  import: importCommand,
  export: exportCommand,
  config: configCommand,
  health: healthCommand,
  stats: statsCommand,
  serve: serveCommand,
  shutdown: shutdownCommand,
  update: updateCommand,
};

async function runMcpServer(): Promise<void> {
  const { runMcpServer: mcpMain } = await import("./mcp/index.js");
  await mcpMain();
}

async function runHook(hookName: string): Promise<void> {
  switch (hookName) {
    case "capture": {
      const { captureHook } = await import("./hooks/capture.js");
      await captureHook();
      break;
    }
    case "summarize": {
      const { summarizeHook } = await import("./hooks/summarize.js");
      await summarizeHook();
      break;
    }
    case "cleanup": {
      const { cleanupHook } = await import("./hooks/cleanup.js");
      await cleanupHook();
      break;
    }
    default:
      console.error(`Unknown hook: ${hookName}`);
      console.error("Available hooks: capture, summarize, cleanup");
      process.exit(1);
  }
}

function printHelp(): void {
  console.log(`
CCMemory - Claude Code Memory System v${VERSION}

Usage: ccmemory <command> [options]

Commands:
  search <query>           Search memories
    -p, --project <path>   Filter by project
    -s, --sector <sector>  Filter by sector
    -l, --limit <n>        Max results (default: 10)
    --semantic             Semantic search only
    --keywords             Keyword search only
    --json                 JSON output

  show <id>               Show memory details
    -r, --related          Show related memories

  delete <id>             Delete a memory
    --force                Skip confirmation

  archive                 Archive old low-salience memories
    --before <date>        Archive before date
    --dry-run              Preview without changes

  import <file>           Import document
    -p, --project <path>   Associate with project
    -t, --title <title>    Document title

  export                  Export memories
    -p, --project <path>   Filter by project
    -f, --format <fmt>     Format: json, csv (default: json)
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
    -p, --port <port>      Port (default: 37778)
    --open                 Open in browser

  shutdown                Stop the WebUI server
    -p, --port <port>      Port (default: 37778)

  update                  Check for and install updates
    -c, --check            Check only, don't install
    -f, --force            Force update even if current

Internal commands (used by Claude Code plugin):
  mcp-server              Start MCP server (stdio transport)
  hook <name>             Run a hook (capture, summarize, cleanup)
`);
}

async function main(): Promise<void> {
  const [command, ...args] = process.argv.slice(2);

  if (!command || command === "help" || command === "--help") {
    printHelp();
    return;
  }

  if (command === "--version" || command === "-v") {
    console.log(`ccmemory ${VERSION}`);
    return;
  }

  if (command === "mcp-server") {
    log.debug("main", "Starting MCP server");
    await runMcpServer();
    return;
  }

  if (command === "hook") {
    const hookName = args[0];
    if (!hookName) {
      console.error("Hook name required: ccmemory hook <capture|summarize|cleanup>");
      process.exit(1);
    }
    await runHook(hookName);
    return;
  }

  const handler = cliCommands[command];
  if (!handler) {
    log.warn("main", "Unknown command", { command });
    console.error(`Unknown command: ${command}`);
    console.error(`Run 'ccmemory help' for usage.`);
    process.exit(1);
  }

  log.debug("main", "Executing command", { command, args: args.length });
  await handler(args);
}

main().catch((err: Error) => {
  log.error("main", "Command failed", { error: err.message });
  console.error(err);
  process.exit(1);
});
