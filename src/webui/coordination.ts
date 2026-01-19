import { join } from "node:path";
import { getPaths } from "../utils/paths.js";
import { log } from "../utils/log.js";

function getRuntimeDir(): string {
  const xdgRuntime = process.env["XDG_RUNTIME_DIR"];
  if (xdgRuntime) {
    return join(xdgRuntime, "ccmemory");
  }
  const paths = getPaths();
  return join(paths.cache, "runtime");
}

const RUNTIME_DIR = getRuntimeDir();
const LOCK_FILE = join(RUNTIME_DIR, "webui.lock");
const CLIENTS_FILE = join(RUNTIME_DIR, "clients.txt");

export async function tryAcquireLock(): Promise<boolean> {
  try {
    await Bun.$`mkdir -p ${RUNTIME_DIR}`.quiet();

    const lockFile = Bun.file(LOCK_FILE);
    if (await lockFile.exists()) {
      const pid = parseInt(await lockFile.text());
      if (isProcessAlive(pid)) {
        log.debug("webui", "Lock held by another process", { pid });
        return false;
      }
      log.debug("webui", "Stale lock file found, cleaning up", { stalePid: pid });
    }

    await Bun.write(LOCK_FILE, String(process.pid));
    log.debug("webui", "Lock acquired", { pid: process.pid });
    return true;
  } catch (err) {
    log.error("webui", "Failed to acquire lock", {
      error: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
}

export async function releaseLock(): Promise<void> {
  try {
    await Bun.$`rm -f ${LOCK_FILE}`.quiet();
    log.debug("webui", "Lock released");
  } catch {
    // Ignore errors on cleanup
  }
}

export async function registerClient(sessionId: string): Promise<void> {
  if (!sessionId) return;
  try {
    await Bun.$`mkdir -p ${RUNTIME_DIR}`.quiet();
    const clients = await getActiveClients();
    if (!clients.includes(sessionId)) {
      clients.push(sessionId);
      await Bun.write(CLIENTS_FILE, clients.join("\n"));
      log.debug("webui", "Client registered", {
        sessionId,
        totalClients: clients.length,
      });
    }
  } catch (err) {
    log.warn("webui", "Failed to register client", {
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

export async function unregisterClient(sessionId: string): Promise<void> {
  try {
    const clients = await getActiveClients();
    const filtered = clients.filter((c) => c !== sessionId);
    log.debug("webui", "Client unregistered", {
      sessionId,
      remainingClients: filtered.length,
    });
    if (filtered.length === 0) {
      await Bun.$`rm -f ${CLIENTS_FILE}`.quiet();
    } else {
      await Bun.write(CLIENTS_FILE, filtered.join("\n"));
    }
  } catch {
    // Ignore errors on cleanup
  }
}

export async function getActiveClients(): Promise<string[]> {
  try {
    const clientsFile = Bun.file(CLIENTS_FILE);
    if (!(await clientsFile.exists())) return [];
    const content = await clientsFile.text();
    return content.split("\n").filter(Boolean);
  } catch {
    return [];
  }
}

export async function isServerRunning(port: number): Promise<boolean> {
  try {
    const res = await fetch(`http://localhost:${port}/api/health`, {
      signal: AbortSignal.timeout(1000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
