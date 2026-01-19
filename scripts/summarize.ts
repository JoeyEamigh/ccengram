import { getDatabase, closeDatabase } from "../src/db/database.js";
import { createMemoryStore } from "../src/services/memory/store.js";
import { getOrCreateProject } from "../src/services/project.js";
import { createSessionService } from "../src/services/memory/sessions.js";
import { log } from "../src/utils/log.js";

type HookInput = {
  session_id: string;
  cwd: string;
  transcript_path?: string;
};

const abortController = new AbortController();

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();
  const input: HookInput = JSON.parse(inputText);
  const { session_id, cwd } = input;

  log.info("summarize", "Starting session summary", { session_id });

  const db = await getDatabase();
  const sessionService = createSessionService();
  const store = createMemoryStore();

  const memories = await db.execute(
    `SELECT m.content FROM memories m
     JOIN session_memories sm ON sm.memory_id = m.id
     WHERE sm.session_id = ? AND sm.usage_type = 'created'
     ORDER BY m.created_at ASC
     LIMIT 50`,
    [session_id]
  );

  if (memories.rows.length === 0) {
    log.debug("summarize", "No memories to summarize", { session_id });
    closeDatabase();
    process.exit(0);
  }

  log.debug("summarize", "Found session memories", {
    session_id,
    count: memories.rows.length,
  });

  const observations = memories.rows
    .map((r) => String(r["content"]))
    .join("\n---\n");

  const summaryContent = createBasicSummary(observations, memories.rows.length);

  log.info("summarize", "Summary generated", {
    session_id,
    length: summaryContent.length,
  });

  const project = await getOrCreateProject(cwd);

  await store.create(
    {
      content: `Session Summary:\n${summaryContent}`,
      sector: "reflective",
      tier: "project",
    },
    project.id,
    session_id
  );

  await sessionService.end(session_id, summaryContent);

  log.info("summarize", "Session summary stored", { session_id });

  closeDatabase();
  process.exit(0);
}

function createBasicSummary(observations: string, count: number): string {
  const lines = [
    `Session completed with ${count} tool observations.`,
    "",
  ];

  const fileMatches = observations.match(/(?:Read|Wrote|Edited) file: ([^\n]+)/g);
  if (fileMatches && fileMatches.length > 0) {
    const uniqueFiles = [...new Set(fileMatches)].slice(0, 10);
    lines.push("Files accessed:");
    for (const file of uniqueFiles) {
      lines.push(`  - ${file}`);
    }
    lines.push("");
  }

  const commandMatches = observations.match(/Command: ([^\n]+)/g);
  if (commandMatches && commandMatches.length > 0) {
    const uniqueCommands = [...new Set(commandMatches)].slice(0, 5);
    lines.push("Commands run:");
    for (const cmd of uniqueCommands) {
      lines.push(`  - ${cmd}`);
    }
  }

  return lines.join("\n");
}

process.on("SIGTERM", () => {
  abortController.abort();
  closeDatabase();
  process.exit(0);
});

process.on("SIGINT", () => {
  abortController.abort();
  closeDatabase();
  process.exit(0);
});

main().catch((err: Error) => {
  log.error("summarize", "Summarize hook failed", { error: err.message });
  closeDatabase();
  process.exit(0);
});
