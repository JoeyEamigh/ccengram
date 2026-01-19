import { getDatabase, closeDatabase } from "../db/database.js";
import { createMemoryStore } from "../services/memory/store.js";
import { getOrCreateProject } from "../services/project.js";
import { createSessionService } from "../services/memory/sessions.js";
import { getPort } from "../utils/paths.js";
import { log } from "../utils/log.js";

type HookInput = {
  session_id: string;
  cwd: string;
  transcript_path?: string;
};

const TIMEOUT_MS = 30000;
const abortController = new AbortController();

function parseInput(text: string): HookInput | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (typeof parsed !== "object" || parsed === null) return null;
    const obj = parsed as Record<string, unknown>;
    if (typeof obj["session_id"] !== "string") return null;
    if (typeof obj["cwd"] !== "string") return null;
    return obj as unknown as HookInput;
  } catch {
    return null;
  }
}

export async function summarizeHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn("summarize", "Summarize hook timed out");
    abortController.abort();
    closeDatabase();
    process.exit(0);
  }, TIMEOUT_MS);

  const inputText = await Bun.stdin.text();
  const input = parseInput(inputText);

  if (!input) {
    log.warn("summarize", "Invalid hook input, skipping");
    clearTimeout(timeoutId);
    process.exit(0);
  }

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
    clearTimeout(timeoutId);
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

  const memory = await store.create(
    {
      content: `Session Summary:\n${summaryContent}`,
      sector: "reflective",
      tier: "project",
    },
    project.id,
    session_id
  );

  await notifyMemoryCreated(memory.id, project.id, session_id);

  await sessionService.end(session_id, summaryContent);

  log.info("summarize", "Session summary stored", { session_id });

  clearTimeout(timeoutId);
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

async function notifyMemoryCreated(
  memoryId: string,
  projectId: string,
  sessionId: string
): Promise<void> {
  try {
    const res = await fetch(`http://localhost:${getPort()}/api/hooks/memory-created`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ memoryId, projectId, sessionId }),
    });
    if (!res.ok) {
      log.debug("summarize", "WebUI notification failed (server may not be running)", {
        status: res.status,
      });
    }
  } catch {
    log.debug("summarize", "WebUI notification skipped (server not running)");
  }
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
