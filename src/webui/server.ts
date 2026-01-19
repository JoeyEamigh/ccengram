import { renderToString } from "react-dom/server";
import { createElement } from "react";
import { App } from "./components/App.js";
import { handleAPI } from "./api/routes.js";
import { handleWebSocket, setServer, broadcastToRoom } from "./ws/handler.js";
import {
  registerClient,
  unregisterClient,
  tryAcquireLock,
  releaseLock,
  isServerRunning,
  getActiveClients,
} from "./coordination.js";
import { buildAssets, type BuildOutput } from "./build.js";
import { log } from "../utils/log.js";
import type { Memory } from "../services/memory/types.js";
import type { Session } from "../services/memory/sessions.js";

const DEFAULT_PORT = 37778;

type WebSocketData = { projectId?: string };

let buildOutput: BuildOutput | null = null;

type StartServerOptions = {
  port?: number;
  sessionId: string;
  open?: boolean;
};

export type ServerResult = {
  alreadyRunning?: boolean;
  server?: ReturnType<typeof Bun.serve>;
  checkInterval?: ReturnType<typeof setInterval>;
};

export async function startServer(
  options: StartServerOptions
): Promise<ServerResult> {
  const port = options.port ?? DEFAULT_PORT;

  log.info("webui", "Starting WebUI server", {
    port,
    sessionId: options.sessionId,
  });

  if (await isServerRunning(port)) {
    await registerClient(options.sessionId);
    log.debug("webui", "Server already running, registering as client");
    console.log(`CCMemory WebUI already running at http://localhost:${port}`);
    return { alreadyRunning: true };
  }

  const acquired = await tryAcquireLock();
  if (!acquired) {
    log.debug("webui", "Lock not acquired, another server is starting");
    await registerClient(options.sessionId);
    return { alreadyRunning: true };
  }

  log.info("webui", "Lock acquired, starting server");
  await registerClient(options.sessionId);

  log.debug("webui", "Building client assets");
  buildOutput = await buildAssets();
  log.debug("webui", "Client assets built", {
    jsSize: buildOutput.clientJs.length,
    cssSize: buildOutput.css.length,
  });

  const server = Bun.serve<WebSocketData>({
    port,

    async fetch(req, server) {
      const url = new URL(req.url);
      const path = url.pathname;

      if (path === "/ws") {
        const upgraded = server.upgrade(req, {
          data: { projectId: url.searchParams.get("project") ?? undefined },
        });
        if (upgraded) return undefined;
        return new Response("WebSocket upgrade failed", { status: 400 });
      }

      if (path.startsWith("/api/")) {
        return handleAPI(req, path);
      }

      if (path === "/client.js" && buildOutput) {
        return new Response(buildOutput.clientJs, {
          headers: { "Content-Type": "application/javascript" },
        });
      }

      if (path === "/styles.css" && buildOutput) {
        return new Response(buildOutput.css, {
          headers: { "Content-Type": "text/css" },
        });
      }

      return renderPage(url);
    },

    websocket: {
      open(ws) {
        const { projectId } = ws.data;
        const room = projectId ?? "global";
        ws.subscribe(room);
        log.debug("webui", "WebSocket connected", { room });
      },

      message(ws, message) {
        handleWebSocket(ws, message);
      },

      close(ws) {
        const { projectId } = ws.data;
        const room = projectId ?? "global";
        ws.unsubscribe(room);
        log.debug("webui", "WebSocket disconnected", { room });
      },
    },
  });

  setServer(server);

  log.info("webui", "WebUI server started", { port });
  console.log(`CCMemory WebUI running at http://localhost:${port}`);

  if (options.open) {
    openBrowser(`http://localhost:${port}`);
  }

  const checkInterval = setInterval(async () => {
    const clients = await getActiveClients();
    if (clients.length === 0) {
      log.info("webui", "No active clients, shutting down server");
      clearInterval(checkInterval);
      server.stop();
      await releaseLock();
    }
  }, 5000);

  return { server, checkInterval };
}

async function renderPage(url: URL): Promise<Response> {
  const initialData = await fetchInitialData(url);
  const appElement = createElement(App, {
    url: url.pathname,
    initialData,
  });
  const html = renderToString(appElement);

  return new Response(
    `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CCMemory</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body class="min-h-screen bg-background font-sans antialiased">
  <div id="root">${html}</div>
  <script>window.__INITIAL_DATA__ = ${JSON.stringify(initialData)};</script>
  <script src="/client.js"></script>
</body>
</html>`,
    {
      headers: { "Content-Type": "text/html; charset=utf-8" },
    }
  );
}

async function fetchInitialData(url: URL): Promise<unknown> {
  const path = url.pathname;

  if (path === "/" || path === "/search") {
    return { type: "search", results: [] };
  }

  if (path === "/agents") {
    return { type: "agents", sessions: [] };
  }

  if (path === "/timeline") {
    return { type: "timeline", data: null };
  }

  return { type: "home" };
}

function openBrowser(url: string): void {
  const cmd =
    process.platform === "darwin"
      ? "open"
      : process.platform === "win32"
        ? "start"
        : "xdg-open";
  Bun.spawn([cmd, url]);
}

export function notifyMemoryChange(projectId: string, memory: Memory): void {
  broadcastToRoom(projectId, { type: "memory:created", memory });
  broadcastToRoom("global", { type: "memory:created", memory, projectId });
}

export function notifySessionChange(projectId: string, session: Session): void {
  broadcastToRoom(projectId, { type: "session:updated", session });
  broadcastToRoom("global", { type: "session:updated", session, projectId });
}

export async function stopServer(sessionId: string): Promise<void> {
  await unregisterClient(sessionId);
}
