import { createMemoryStore } from "../src/services/memory/store.js";
import { getOrCreateProject } from "../src/services/project.js";
import { getOrCreateSession } from "../src/services/memory/sessions.js";
import { log } from "../src/utils/log.js";

type HookInput = {
  session_id: string;
  cwd: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_result: unknown;
};

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();
  const input: HookInput = JSON.parse(inputText);

  const { session_id, cwd, tool_name, tool_input, tool_result } = input;
  log.debug("capture", "Processing tool observation", { session_id, tool_name });

  const resultStr = JSON.stringify(tool_result);
  if (resultStr.length > 10000) {
    log.debug("capture", "Skipping large tool result", {
      tool_name,
      bytes: resultStr.length,
    });
    process.exit(0);
  }

  const project = await getOrCreateProject(cwd);
  await getOrCreateSession(session_id, project.id);

  const content = formatToolObservation(tool_name, tool_input, tool_result);
  const files = extractFilePaths(tool_input, tool_result);

  const store = createMemoryStore();

  const memory = await store.create(
    {
      content,
      sector: "episodic",
      tier: "session",
      files,
    },
    project.id,
    session_id
  );

  log.debug("capture", "Captured tool observation", {
    session_id,
    tool_name,
    memoryId: memory.id,
  });

  process.exit(0);
}

function formatToolObservation(
  toolName: string,
  input: Record<string, unknown>,
  result: unknown
): string {
  const lines: string[] = [`Tool: ${toolName}`];

  switch (toolName) {
    case "Read":
      lines.push(`Read file: ${String(input["file_path"] ?? "")}`);
      break;
    case "Write":
      lines.push(`Wrote file: ${String(input["file_path"] ?? "")}`);
      break;
    case "Edit":
      lines.push(`Edited file: ${String(input["file_path"] ?? "")}`);
      break;
    case "Bash": {
      const command = String(input["command"] ?? "").slice(0, 200);
      lines.push(`Command: ${command}`);
      if (typeof result === "string" && result.length < 500) {
        lines.push(`Output: ${result}`);
      }
      break;
    }
    case "Grep":
    case "Glob":
      lines.push(`Pattern: ${String(input["pattern"] ?? "")}`);
      break;
    default:
      lines.push(`Input: ${JSON.stringify(input).slice(0, 300)}`);
  }

  return lines.join("\n");
}

function extractFilePaths(
  input: Record<string, unknown>,
  result: unknown
): string[] {
  const paths: string[] = [];

  if (typeof input["file_path"] === "string") {
    paths.push(input["file_path"]);
  }
  if (typeof input["path"] === "string") {
    paths.push(input["path"]);
  }

  if (Array.isArray(result)) {
    const filePaths = result.filter(
      (r): r is string =>
        typeof r === "string" && (r.includes("/") || r.includes("\\"))
    );
    paths.push(...filePaths.slice(0, 10));
  }

  return [...new Set(paths)];
}

main().catch((err: Error) => {
  log.error("capture", "Capture hook failed", { error: err.message });
  process.exit(0);
});
