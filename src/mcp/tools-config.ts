import { join } from "node:path";
import { log } from "../utils/log.js";
import { getDatabase } from "../db/database.js";
import { getPaths } from "../utils/paths.js";

export type ToolMode = "full" | "recall" | "custom";

export type ToolsConfig = {
  mode: ToolMode;
  enabledTools: string[];
};

const RECALL_TOOLS = ["memory_search", "memory_timeline", "docs_search"];

const ALL_TOOLS = [
  "memory_search",
  "memory_timeline",
  "memory_add",
  "memory_reinforce",
  "memory_deemphasize",
  "memory_delete",
  "memory_supersede",
  "docs_search",
  "docs_ingest",
];

async function getConfigFromDb(): Promise<ToolsConfig | null> {
  try {
    const db = await getDatabase();
    const result = await db.execute(
      "SELECT value FROM config WHERE key = ?",
      ["toolMode"]
    );

    const row = result.rows[0];
    if (!row) {
      return null;
    }

    const mode = String(row["value"]) as ToolMode;

    if (mode === "custom") {
      const customResult = await db.execute(
        "SELECT value FROM config WHERE key = ?",
        ["enabledTools"]
      );
      const enabledTools = customResult.rows[0]
        ? String(customResult.rows[0]["value"]).split(",").filter(Boolean)
        : ALL_TOOLS;
      return { mode, enabledTools };
    }

    return { mode, enabledTools: [] };
  } catch {
    return null;
  }
}

async function getConfigFromProjectFile(projectDir: string): Promise<ToolsConfig | null> {
  const configPath = join(projectDir, ".claude", "ccmemory.local.md");

  try {
    const file = Bun.file(configPath);
    if (!(await file.exists())) {
      return null;
    }

    const content = await file.text();
    const config = parseYamlFrontmatter(content);

    if (!config) {
      return null;
    }

    const mode = config["toolMode"] as ToolMode | undefined;
    if (!mode || !["full", "recall", "custom"].includes(mode)) {
      return null;
    }

    if (mode === "custom") {
      const enabledTools = config["enabledTools"] as string[] | undefined;
      return {
        mode,
        enabledTools: enabledTools ?? ALL_TOOLS,
      };
    }

    return { mode, enabledTools: [] };
  } catch {
    return null;
  }
}

function parseYamlFrontmatter(content: string): Record<string, unknown> | null {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) {
    return null;
  }

  const yamlContent = match[1];
  if (!yamlContent) return null;

  const result: Record<string, unknown> = {};

  for (const line of yamlContent.split("\n")) {
    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) continue;

    const key = line.slice(0, colonIndex).trim();
    let value: unknown = line.slice(colonIndex + 1).trim();

    // Handle arrays (simple inline format: [a, b, c])
    if (typeof value === "string" && value.startsWith("[") && value.endsWith("]")) {
      value = value
        .slice(1, -1)
        .split(",")
        .map((s) => s.trim().replace(/^["']|["']$/g, ""))
        .filter(Boolean);
    }

    result[key] = value;
  }

  return result;
}

export async function getToolsConfig(projectDir?: string): Promise<ToolsConfig> {
  // Priority:
  // 1. Environment variable CCMEMORY_TOOL_MODE
  // 2. Per-project config (.claude/ccmemory.local.md)
  // 3. Global config (database)
  // 4. Default (full)

  // Check environment variable
  const envMode = process.env["CCMEMORY_TOOL_MODE"] as ToolMode | undefined;
  if (envMode && ["full", "recall", "custom"].includes(envMode)) {
    log.debug("mcp", "Using tool mode from environment", { mode: envMode });

    if (envMode === "custom") {
      const envTools = process.env["CCMEMORY_ENABLED_TOOLS"];
      return {
        mode: envMode,
        enabledTools: envTools ? envTools.split(",").filter(Boolean) : ALL_TOOLS,
      };
    }

    return { mode: envMode, enabledTools: [] };
  }

  // Check per-project config
  if (projectDir) {
    const projectConfig = await getConfigFromProjectFile(projectDir);
    if (projectConfig) {
      log.debug("mcp", "Using tool mode from project config", {
        mode: projectConfig.mode,
        projectDir,
      });
      return projectConfig;
    }
  }

  // Check global config from file
  const fileConfig = await getConfigFromFile();
  if (fileConfig) {
    log.debug("mcp", "Using tool mode from config file", { mode: fileConfig.mode });
    return fileConfig;
  }

  // Check global config from database
  const dbConfig = await getConfigFromDb();
  if (dbConfig) {
    log.debug("mcp", "Using tool mode from database", { mode: dbConfig.mode });
    return dbConfig;
  }

  // Default: full
  return { mode: "full", enabledTools: [] };
}

async function getConfigFromFile(): Promise<ToolsConfig | null> {
  try {
    const paths = getPaths();
    const configPath = join(paths.config, "config.json");

    const file = Bun.file(configPath);
    if (!(await file.exists())) {
      return null;
    }

    const config = (await file.json()) as Record<string, unknown>;
    const tools = config["tools"] as Record<string, unknown> | undefined;

    if (!tools || typeof tools !== "object") {
      return null;
    }

    const mode = tools["mode"] as ToolMode | undefined;
    if (!mode || !["full", "recall", "custom"].includes(mode)) {
      return null;
    }

    if (mode === "custom") {
      const enabledTools = tools["enabledTools"] as string[] | undefined;
      return {
        mode,
        enabledTools: enabledTools ?? ALL_TOOLS,
      };
    }

    return { mode, enabledTools: [] };
  } catch {
    return null;
  }
}

export function filterTools<T extends { name: string }>(
  tools: T[],
  config: ToolsConfig
): T[] {
  switch (config.mode) {
    case "full":
      return tools;

    case "recall":
      return tools.filter((t) => RECALL_TOOLS.includes(t.name));

    case "custom":
      return tools.filter((t) => config.enabledTools.includes(t.name));

    default:
      return tools;
  }
}

export async function setToolMode(mode: ToolMode, enabledTools?: string[]): Promise<void> {
  const db = await getDatabase();
  const now = Date.now();

  await db.execute(
    `INSERT INTO config (key, value, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
    ["toolMode", mode, now]
  );

  if (mode === "custom" && enabledTools) {
    await db.execute(
      `INSERT INTO config (key, value, updated_at)
       VALUES (?, ?, ?)
       ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
      ["enabledTools", enabledTools.join(","), now]
    );
  }

  log.info("mcp", "Tool mode updated", { mode, enabledTools });
}
