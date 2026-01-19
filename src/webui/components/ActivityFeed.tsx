import { useEffect, useState } from "react";
import { Activity, Zap, Minus, Trash2 } from "lucide-react";
import { Badge } from "./ui/badge.js";
import { cn } from "../lib/utils.js";
import type { Memory, MemorySector } from "../../services/memory/types.js";

export type ActivityEvent = {
  id: string;
  type: "created" | "updated" | "deleted";
  memory: {
    id: string;
    content: string;
    sector: MemorySector;
    salience: number;
    summary?: string;
  };
  projectId?: string;
  timestamp: number;
};

type WebSocketMessage = {
  type: string;
  memory?: Memory;
  projectId?: string;
};

type ActivityFeedProps = {
  messages: WebSocketMessage[];
  maxItems?: number;
  onSelectMemory?: (memory: Memory) => void;
  compact?: boolean;
  initialEvents?: ActivityEvent[];
};

const sectorVariant: Record<
  MemorySector,
  "episodic" | "semantic" | "procedural" | "emotional" | "reflective"
> = {
  episodic: "episodic",
  semantic: "semantic",
  procedural: "procedural",
  emotional: "emotional",
  reflective: "reflective",
};

const eventIcon: Record<string, typeof Activity> = {
  created: Zap,
  updated: Activity,
  deleted: Trash2,
};

export function ActivityFeed({
  messages,
  maxItems = 20,
  onSelectMemory,
  compact = false,
  initialEvents = [],
}: ActivityFeedProps): JSX.Element {
  const [events, setEvents] = useState<ActivityEvent[]>(initialEvents);

  useEffect(() => {
    if (initialEvents.length > 0) {
      setEvents(initialEvents);
    }
  }, [initialEvents]);

  useEffect(() => {
    for (const msg of messages) {
      if (
        msg.type === "memory:created" ||
        msg.type === "memory:updated" ||
        msg.type === "memory:deleted"
      ) {
        const eventType = msg.type.split(":")[1] as ActivityEvent["type"];
        const memory = msg.memory;
        if (!memory) continue;

        const event: ActivityEvent = {
          id: `${memory.id}-${Date.now()}`,
          type: eventType,
          memory: {
            id: memory.id,
            content: memory.content,
            sector: memory.sector,
            salience: memory.salience,
            summary: memory.summary,
          },
          projectId: msg.projectId,
          timestamp: Date.now(),
        };

        setEvents((prev) => [event, ...prev].slice(0, maxItems));
      }
    }
  }, [messages, maxItems]);

  if (events.length === 0) {
    return (
      <div className="text-center py-6 text-sm text-muted-foreground">
        <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No recent activity</p>
        <p className="text-xs">Memory events from the last 24 hours will appear here</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {events.map((event) => {
        const Icon = eventIcon[event.type] ?? Activity;
        return (
          <div
            key={event.id}
            className={cn(
              "flex gap-3 p-2 rounded-md transition-colors",
              onSelectMemory && "cursor-pointer hover:bg-accent/50",
              event.type === "created" && "animate-in slide-in-from-top-2 duration-300"
            )}
            onClick={() => onSelectMemory?.(event.memory as unknown as Memory)}
          >
            <div className="flex-shrink-0 mt-0.5">
              <Icon
                className={cn(
                  "h-4 w-4",
                  event.type === "created" && "text-green-500",
                  event.type === "updated" && "text-blue-500",
                  event.type === "deleted" && "text-red-500"
                )}
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge
                  variant={sectorVariant[event.memory.sector]}
                  className="text-[10px] px-1.5 py-0"
                >
                  {event.memory.sector}
                </Badge>
                {!compact && (
                  <div
                    className="flex items-center gap-1"
                    title={`Salience: ${(event.memory.salience * 100).toFixed(0)}%`}
                  >
                    <Minus className="h-3 w-3 text-muted-foreground" />
                    <div className="h-1 w-8 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full bg-primary/60 rounded-full"
                        style={{ width: `${event.memory.salience * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                <span className="text-[10px] text-muted-foreground ml-auto">
                  {formatTime(event.timestamp)}
                </span>
              </div>
              <p className={cn("text-sm mt-1", compact ? "line-clamp-1" : "line-clamp-2")}>
                {event.memory.summary ?? event.memory.content}
              </p>
            </div>
          </div>
        );
      })}
      {events.length >= maxItems && (
        <p className="text-xs text-center text-muted-foreground pt-2">
          Showing {maxItems} most recent events
        </p>
      )}
    </div>
  );
}

function formatTime(ts: number): string {
  const now = Date.now();
  const diffMs = now - ts;
  const diffSecs = Math.floor(diffMs / 1000);

  if (diffSecs < 5) return "just now";
  if (diffSecs < 60) return `${diffSecs}s ago`;

  const diffMins = Math.floor(diffSecs / 60);
  if (diffMins < 60) return `${diffMins}m ago`;

  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}
