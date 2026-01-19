import { parseArgs } from "util";
import { isServerRunning } from "../../webui/coordination.js";
import { getPort } from "../../utils/paths.js";
import { log } from "../../utils/log.js";

export async function shutdownCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      port: { type: "string", short: "p" },
      force: { type: "boolean", short: "f" },
    },
  });

  const port = values.port ? parseInt(values.port, 10) : getPort();

  if (!(await isServerRunning(port))) {
    console.log("CCMemory server is not running.");
    return;
  }

  log.info("cli", "Requesting server shutdown", { port });

  try {
    const res = await fetch(`http://localhost:${port}/api/shutdown`, {
      method: "POST",
      signal: AbortSignal.timeout(5000),
    });

    if (res.ok) {
      console.log("CCMemory server shutting down.");
    } else {
      const body = await res.text();
      console.error(`Failed to shutdown server: ${body}`);
      process.exit(1);
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      console.log("CCMemory server shutdown initiated.");
    } else {
      log.error("cli", "Failed to shutdown server", {
        error: err instanceof Error ? err.message : String(err),
      });
      console.error("Failed to connect to server.");
      process.exit(1);
    }
  }
}
