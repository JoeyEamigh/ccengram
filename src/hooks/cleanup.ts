import { getDatabase, closeDatabase } from "../db/database.js";
import { unregisterClient, getActiveClients } from "../webui/coordination.js";
import { getPort } from "../utils/paths.js";
import { log } from "../utils/log.js";

type HookInput = {
  session_id: string;
};

const TIMEOUT_MS = 10000;

function parseInput(text: string): HookInput | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (typeof parsed !== "object" || parsed === null) return null;
    const obj = parsed as Record<string, unknown>;
    if (typeof obj["session_id"] !== "string") return null;
    return obj as unknown as HookInput;
  } catch {
    return null;
  }
}

export async function cleanupHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn("cleanup", "Cleanup hook timed out");
    closeDatabase();
    process.exit(0);
  }, TIMEOUT_MS);

  const inputText = await Bun.stdin.text();
  const input = parseInput(inputText);

  if (!input) {
    log.warn("cleanup", "Invalid hook input, skipping");
    clearTimeout(timeoutId);
    process.exit(0);
  }

  const { session_id } = input;

  log.info("cleanup", "Starting session cleanup", { session_id });

  const db = await getDatabase();

  await db.execute(
    "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
    [Date.now(), session_id]
  );

  const promoted = await db.execute(
    `UPDATE memories
     SET tier = 'project', updated_at = ?
     WHERE id IN (
       SELECT m.id FROM memories m
       JOIN session_memories sm ON sm.memory_id = m.id
       WHERE sm.session_id = ? AND m.tier = 'session' AND m.salience > 0.7
     )`,
    [Date.now(), session_id]
  );

  log.debug("cleanup", "Promoted high-salience memories", {
    session_id,
    count: promoted.rowsAffected,
  });

  closeDatabase();

  // Unregister this session and check if we should shut down server
  await unregisterClient(session_id);
  const remainingClients = await getActiveClients();

  if (remainingClients.length === 0) {
    log.info("cleanup", "No active clients, signaling server shutdown");
    await signalServerShutdown();
  }

  clearTimeout(timeoutId);
  log.info("cleanup", "Session cleanup complete", { session_id });
  process.exit(0);
}

async function signalServerShutdown(): Promise<void> {
  try {
    const res = await fetch(`http://localhost:${getPort()}/api/shutdown`, {
      method: "POST",
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      log.info("cleanup", "Server shutdown signal sent");
    }
  } catch {
    log.debug("cleanup", "Server shutdown signal failed (may already be down)");
  }
}
