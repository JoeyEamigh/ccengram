import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export type Paths = {
  config: string;
  data: string;
  cache: string;
  db: string;
};

type Platform = "linux" | "darwin" | "win32";

function getPlatform(): Platform {
  const p = process.platform;
  if (p === "linux" || p === "darwin" || p === "win32") {
    return p;
  }
  return "linux";
}

function getConfigDir(platform: Platform): string {
  const env = process.env["XDG_CONFIG_HOME"];
  if (env) return env;

  const home = homedir();
  switch (platform) {
    case "darwin":
      return join(home, "Library", "Application Support");
    case "win32":
      return process.env["APPDATA"] ?? join(home, "AppData", "Roaming");
    default:
      return join(home, ".config");
  }
}

function getDataDir(platform: Platform): string {
  const env = process.env["XDG_DATA_HOME"];
  if (env) return env;

  const home = homedir();
  switch (platform) {
    case "darwin":
      return join(home, "Library", "Application Support");
    case "win32":
      return process.env["LOCALAPPDATA"] ?? join(home, "AppData", "Local");
    default:
      return join(home, ".local", "share");
  }
}

function getCacheDir(platform: Platform): string {
  const env = process.env["XDG_CACHE_HOME"];
  if (env) return env;

  const home = homedir();
  switch (platform) {
    case "darwin":
      return join(home, "Library", "Caches");
    case "win32": {
      const local = process.env["LOCALAPPDATA"] ?? join(home, "AppData", "Local");
      return join(local, "cache");
    }
    default:
      return join(home, ".cache");
  }
}

const APP_NAME = "ccmemory";

export function getPaths(): Paths {
  const platform = getPlatform();
  const config = join(getConfigDir(platform), APP_NAME);
  const data = join(getDataDir(platform), APP_NAME);
  const cache = join(getCacheDir(platform), APP_NAME);
  const db = join(data, "memories.db");

  return { config, data, cache, db };
}

export async function ensureDirectories(): Promise<void> {
  const paths = getPaths();
  await Promise.all([
    mkdir(paths.config, { recursive: true }),
    mkdir(paths.data, { recursive: true }),
    mkdir(paths.cache, { recursive: true }),
  ]);
}
