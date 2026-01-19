import { useState, useEffect } from "react";
import { Clock, Brain, Activity, ChevronDown, ChevronUp, Loader2, Zap } from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader } from "./ui/card.js";
import { Badge } from "./ui/badge.js";
import { Button } from "./ui/button.js";
import { cn } from "../lib/utils.js";
import { RelativeTime } from "./RelativeTime.js";
import type { MemorySector, Memory } from "../../services/memory/types.js";

type SessionMemory = {
  id: string;
  content: string;
  summary?: string;
  sector: MemorySector;
  salience: number;
  createdAt: number;
};

type Session = {
  id: string;
  startedAt: number;
  endedAt?: number;
  summary?: string;
  memoryCount?: number;
  lastActivity?: number;
};

type WebSocketMessage = {
  type: string;
  memory?: Memory;
  sessionId?: string;
  projectId?: string;
};

type SessionCardProps = {
  session: Session;
  onViewMemories: () => void;
  onViewTimeline: () => void;
  onSelectMemory?: (memory: SessionMemory) => void;
  messages?: WebSocketMessage[];
};

const sectorColors: Record<MemorySector, string> = {
  episodic: "bg-blue-500/10 text-blue-700 dark:text-blue-400",
  semantic: "bg-green-500/10 text-green-700 dark:text-green-400",
  procedural: "bg-purple-500/10 text-purple-700 dark:text-purple-400",
  emotional: "bg-red-500/10 text-red-700 dark:text-red-400",
  reflective: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
};

export function SessionCard({
  session,
  onViewMemories,
  onViewTimeline,
  onSelectMemory,
  messages = [],
}: SessionCardProps): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const [memories, setMemories] = useState<SessionMemory[]>([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [memoryCount, setMemoryCount] = useState(session.memoryCount ?? 0);
  const [hasNewMemory, setHasNewMemory] = useState(false);

  const isActive = !session.endedAt;
  const duration = session.endedAt
    ? formatDuration(session.endedAt - session.startedAt)
    : formatDuration(Date.now() - session.startedAt) + " (active)";

  useEffect(() => {
    for (const msg of messages) {
      if (msg.type === "memory:created" && msg.sessionId === session.id && msg.memory) {
        const newMemory: SessionMemory = {
          id: msg.memory.id,
          content: msg.memory.content,
          summary: msg.memory.summary,
          sector: msg.memory.sector as MemorySector,
          salience: msg.memory.salience,
          createdAt: msg.memory.createdAt,
        };

        setMemories((prev) => {
          if (prev.some((m) => m.id === newMemory.id)) return prev;
          return [newMemory, ...prev].slice(0, 5);
        });
        setMemoryCount((prev) => prev + 1);
        setHasNewMemory(true);

        setTimeout(() => setHasNewMemory(false), 2000);
      }
    }
  }, [messages, session.id]);

  const toggleExpand = async (): Promise<void> => {
    if (!expanded && !loaded) {
      setLoading(true);
      try {
        const res = await fetch(`/api/sessions/${session.id}/memories?limit=5`);
        const data = (await res.json()) as { memories: SessionMemory[] };
        setMemories(data.memories);
        setLoaded(true);
      } finally {
        setLoading(false);
      }
    }
    setExpanded(!expanded);
  };

  return (
    <Card className={cn(
      isActive && "ring-2 ring-green-500/50",
      hasNewMemory && "ring-2 ring-yellow-500/50 transition-all"
    )}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <span
            className="font-mono text-sm text-muted-foreground"
            title={session.id}
          >
            {session.id.slice(0, 8)}...
          </span>
          <div className="flex items-center gap-2">
            {hasNewMemory && (
              <Badge className="bg-yellow-500 text-white animate-pulse flex items-center gap-1">
                <Zap className="h-3 w-3" />
                NEW
              </Badge>
            )}
            {isActive && (
              <Badge className="bg-green-500 text-white animate-pulse">
                ACTIVE
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span>{duration}</span>
          </div>
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span>{memoryCount} memories</span>
          </div>
        </div>
        {session.lastActivity && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="h-4 w-4" />
            <span>Last: <RelativeTime timestamp={session.lastActivity} /></span>
          </div>
        )}
        {session.summary && (
          <p className="text-sm text-muted-foreground line-clamp-2">
            {session.summary}
          </p>
        )}

        {expanded && (
          <div className="mt-3 pt-3 border-t space-y-2">
            {loading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : memories.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-2">
                No memories created yet
              </p>
            ) : (
              <>
                <p className="text-xs font-medium text-muted-foreground">
                  Recent Memories
                </p>
                {memories.map((memory) => (
                  <div
                    key={memory.id}
                    className={cn(
                      "p-2 rounded-md text-xs cursor-pointer hover:bg-accent/50 transition-colors",
                      sectorColors[memory.sector]
                    )}
                    onClick={() => onSelectMemory?.(memory as unknown as SessionMemory)}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <Badge variant="outline" className="text-[10px] px-1 py-0">
                        {memory.sector}
                      </Badge>
                      <RelativeTime
                        timestamp={memory.createdAt}
                        className="text-[10px] opacity-70"
                      />
                    </div>
                    <p className="line-clamp-2">
                      {memory.summary ?? memory.content}
                    </p>
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </CardContent>

      <CardFooter className="gap-2 flex-wrap">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleExpand}
          className="gap-1"
        >
          {loading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : expanded ? (
            <ChevronUp className="h-3 w-3" />
          ) : (
            <ChevronDown className="h-3 w-3" />
          )}
          {expanded ? "Collapse" : "Expand"}
        </Button>
        <Button variant="outline" size="sm" onClick={onViewMemories}>
          All Memories
        </Button>
        <Button variant="outline" size="sm" onClick={onViewTimeline}>
          Timeline
        </Button>
      </CardFooter>
    </Card>
  );
}

function formatDuration(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  const hours = Math.floor(minutes / 60);
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  return `${minutes}m`;
}
