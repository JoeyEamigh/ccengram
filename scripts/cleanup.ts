import { getDatabase, closeDatabase } from "../src/db/database.js";
import { log } from "../src/utils/log.js";

type HookInput = {
  session_id: string;
};

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();
  const input: HookInput = JSON.parse(inputText);
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

  log.info("cleanup", "Session cleanup complete", { session_id });
  process.exit(0);
}

main().catch((err: Error) => {
  log.error("cleanup", "Cleanup hook failed", { error: err.message });
  closeDatabase();
  process.exit(0);
});
