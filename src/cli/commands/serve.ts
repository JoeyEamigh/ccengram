import { parseArgs } from "util";
import { startServer } from "../../webui/server.js";
import { log } from "../../utils/log.js";

export async function serveCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      port: { type: "string", short: "p", default: "37778" },
      open: { type: "boolean" },
    },
  });

  const port = parseInt(values.port as string, 10);
  const sessionId = `cli-${Date.now()}`;

  log.info("cli", "Starting WebUI server", { port });

  const result = await startServer({
    port,
    sessionId,
    open: values.open,
  });

  if (result.alreadyRunning) {
    return;
  }

  console.log("\nPress Ctrl+C to stop the server.\n");

  process.on("SIGINT", async () => {
    log.info("cli", "Shutting down WebUI server");
    if (result.server) {
      result.server.stop();
    }
    if (result.checkInterval) {
      clearInterval(result.checkInterval);
    }
    process.exit(0);
  });

  await new Promise(() => {});
}
