import { getPaths, ensureDirectories } from "../../utils/paths.js";
import { log } from "../../utils/log.js";

type ToolMode = "full" | "recall" | "custom";

type Config = {
  embedding: {
    provider: "ollama" | "openrouter";
    ollama: { baseUrl: string; model: string };
    openrouter: { apiKey?: string; model: string };
  };
  capture: {
    enabled: boolean;
    maxResultSize: number;
  };
  tools: {
    mode: ToolMode;
    enabledTools: string[];
  };
};

function getDefaultConfig(): Config {
  return {
    embedding: {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { model: "openai/text-embedding-3-small" },
    },
    capture: {
      enabled: true,
      maxResultSize: 10000,
    },
    tools: {
      mode: "full",
      enabledTools: [],
    },
  };
}

function getNestedValue(obj: unknown, path: string): unknown {
  const keys = path.split(".");
  let current: unknown = obj;

  for (const key of keys) {
    if (typeof current !== "object" || current === null) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }

  return current;
}

function setNestedValue(obj: unknown, path: string, value: unknown): void {
  const keys = path.split(".");
  const lastKey = keys.pop();
  if (!lastKey) return;

  let current: unknown = obj;
  for (const key of keys) {
    if (typeof current !== "object" || current === null) return;
    const record = current as Record<string, unknown>;
    if (!(key in record)) {
      record[key] = {};
    }
    current = record[key];
  }

  if (typeof current === "object" && current !== null) {
    (current as Record<string, unknown>)[lastKey] = value;
  }
}

function parseValue(str: string): unknown {
  if (str === "true") return true;
  if (str === "false") return false;
  if (/^\d+$/.test(str)) return parseInt(str, 10);
  if (/^\d+\.\d+$/.test(str)) return parseFloat(str);
  return str;
}

export async function configCommand(args: string[]): Promise<void> {
  const paths = getPaths();
  await ensureDirectories();
  const configPath = `${paths.config}/config.json`;

  let config: Config;
  try {
    config = await Bun.file(configPath).json();
  } catch {
    config = getDefaultConfig();
  }

  if (args.length === 0) {
    console.log(JSON.stringify(config, null, 2));
    return;
  }

  const key = args[0];
  const value = args[1];

  if (!key) {
    console.log(JSON.stringify(config, null, 2));
    return;
  }

  if (!value) {
    const val = getNestedValue(config, key);
    console.log(val !== undefined ? JSON.stringify(val) : "Not set");
    return;
  }

  setNestedValue(config, key, parseValue(value));
  await Bun.write(configPath, JSON.stringify(config, null, 2));
  log.info("cli", "Config updated", { key, value });
  console.log(`Set ${key} = ${value}`);
}
