import type { ServerWebSocket } from "bun";
import { createMemoryStore } from "../../services/memory/store.js";
import { log } from "../../utils/log.js";

type WebSocketData = { projectId?: string };
type WebSocketMessage = {
  type: string;
  memoryId?: string;
  projectId?: string;
  amount?: number;
  hard?: boolean;
};

type BunServer = {
  publish(topic: string, data: string | ArrayBuffer | Uint8Array): void;
};

let serverRef: BunServer | null = null;

export function setServer(server: BunServer): void {
  serverRef = server;
}

export function broadcastToRoom(room: string, message: unknown): void {
  if (serverRef) {
    serverRef.publish(room, JSON.stringify(message));
  }
}

export async function handleWebSocket(
  ws: ServerWebSocket<WebSocketData>,
  message: string | Buffer
): Promise<void> {
  try {
    const data = JSON.parse(message.toString()) as WebSocketMessage;
    log.debug("webui", "WebSocket message received", { type: data.type });
    const store = createMemoryStore();

    switch (data.type) {
      case "memory:reinforce": {
        if (!data.memoryId) {
          ws.send(JSON.stringify({ type: "error", message: "Missing memoryId" }));
          return;
        }
        const memory = await store.reinforce(data.memoryId, data.amount ?? 0.1);
        log.debug("webui", "Memory reinforced via WebSocket", {
          memoryId: data.memoryId,
        });
        ws.send(JSON.stringify({ type: "memory:updated", memory }));
        break;
      }

      case "memory:deemphasize": {
        if (!data.memoryId) {
          ws.send(JSON.stringify({ type: "error", message: "Missing memoryId" }));
          return;
        }
        const memory = await store.deemphasize(data.memoryId, data.amount ?? 0.2);
        log.debug("webui", "Memory de-emphasized via WebSocket", {
          memoryId: data.memoryId,
        });
        ws.send(JSON.stringify({ type: "memory:updated", memory }));
        break;
      }

      case "memory:delete": {
        if (!data.memoryId) {
          ws.send(JSON.stringify({ type: "error", message: "Missing memoryId" }));
          return;
        }
        await store.delete(data.memoryId, data.hard ?? false);
        log.info("webui", "Memory deleted via WebSocket", {
          memoryId: data.memoryId,
          hard: data.hard,
        });
        ws.send(
          JSON.stringify({
            type: "memory:deleted",
            memoryId: data.memoryId,
            hard: data.hard,
          })
        );
        break;
      }

      case "subscribe:project": {
        if (data.projectId) {
          ws.subscribe(data.projectId);
          log.debug("webui", "Client subscribed to project", {
            projectId: data.projectId,
          });
          ws.send(JSON.stringify({ type: "subscribed", room: data.projectId }));
        }
        break;
      }

      case "unsubscribe:project": {
        if (data.projectId) {
          ws.unsubscribe(data.projectId);
          log.debug("webui", "Client unsubscribed from project", {
            projectId: data.projectId,
          });
        }
        break;
      }

      case "ping": {
        ws.send(JSON.stringify({ type: "pong" }));
        break;
      }
    }
  } catch (err) {
    log.error("webui", "WebSocket handler error", {
      error: err instanceof Error ? err.message : String(err),
    });
    ws.send(
      JSON.stringify({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      })
    );
  }
}
