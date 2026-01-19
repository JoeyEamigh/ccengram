import { useState, useEffect } from "react";
import { Users, RefreshCw, Activity, ChevronDown, ChevronUp, Clock } from "lucide-react";
import { SessionCard } from "./SessionCard.js";
import { ActivityFeed } from "./ActivityFeed.js";
import type { ActivityEvent } from "./ActivityFeed.js";
import { Badge } from "./ui/badge.js";
import { Button } from "./ui/button.js";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card.js";
import { cn } from "../lib/utils.js";
import type { Memory } from "../../services/memory/types.js";

type Session = {
  id: string;
  projectId: string;
  startedAt: number;
  endedAt?: number;
  summary?: string;
  memoryCount?: number;
  lastActivity?: number;
};

type WebSocketMessage = {
  type: string;
  memory?: Memory;
  projectId?: string;
  session?: Session;
};

type AgentViewProps = {
  initialSessions: unknown[];
  wsConnected: boolean;
  onNavigate: (path: string) => void;
  messages?: WebSocketMessage[];
  onSelectMemory?: (memory: Memory) => void;
  initialActivity?: ActivityEvent[];
};

type ParallelGroup = {
  sessions: Session[];
  startTime: number;
  endTime: number;
};

export function AgentView({
  initialSessions,
  wsConnected,
  onNavigate,
  messages = [],
  onSelectMemory,
  initialActivity = [],
}: AgentViewProps): JSX.Element {
  const [sessions, setSessions] = useState(initialSessions as Session[]);
  const [groups, setGroups] = useState<ParallelGroup[]>([]);
  const [loading, setLoading] = useState(false);
  const [activityExpanded, setActivityExpanded] = useState(true);

  useEffect(() => {
    setSessions(initialSessions as Session[]);
  }, [initialSessions]);

  useEffect(() => {
    for (const msg of messages) {
      if (msg.type === "session:updated" && msg.session) {
        setSessions((prev) =>
          prev.map((s) => (s.id === msg.session!.id ? msg.session! : s))
        );
      }
    }
  }, [messages]);

  useEffect(() => {
    const sorted = [...sessions].sort((a, b) => b.startedAt - a.startedAt);
    const newGroups: ParallelGroup[] = [];

    for (const session of sorted) {
      const endTime = session.endedAt ?? Date.now();
      const overlappingGroup = newGroups.find(
        (g) => session.startedAt < g.endTime && endTime > g.startTime
      );

      if (overlappingGroup) {
        overlappingGroup.sessions.push(session);
        overlappingGroup.startTime = Math.min(
          overlappingGroup.startTime,
          session.startedAt
        );
        overlappingGroup.endTime = Math.max(overlappingGroup.endTime, endTime);
      } else {
        newGroups.push({
          sessions: [session],
          startTime: session.startedAt,
          endTime,
        });
      }
    }

    setGroups(newGroups);
  }, [sessions]);

  const refresh = async (): Promise<void> => {
    setLoading(true);
    try {
      const res = await fetch("/api/sessions");
      const data = (await res.json()) as { sessions: Session[] };
      setSessions(data.sessions);
    } finally {
      setLoading(false);
    }
  };

  const activeSessions = sessions.filter((s) => !s.endedAt);
  const totalMemories = sessions.reduce((sum, s) => sum + (s.memoryCount ?? 0), 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
            <Users className="h-6 w-6" />
            Agent Sessions
          </h2>
          <p className="text-muted-foreground">
            {sessions.length} sessions ({activeSessions.length} active) with{" "}
            {totalMemories} memories
          </p>
        </div>
        <div className="flex items-center gap-2">
          {wsConnected ? (
            <span className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
              <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              Live
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-muted" />
              Offline
            </span>
          )}
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader
          className="cursor-pointer"
          onClick={() => setActivityExpanded(!activityExpanded)}
        >
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Recent Activity
              <span className="text-xs font-normal text-muted-foreground">
                (last 24h)
              </span>
              {wsConnected && (
                <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              )}
            </CardTitle>
            <Button variant="ghost" size="icon">
              {activityExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardHeader>
        {activityExpanded && (
          <CardContent>
            <ActivityFeed
              messages={messages}
              maxItems={15}
              onSelectMemory={onSelectMemory}
              compact
              initialEvents={initialActivity}
            />
          </CardContent>
        )}
      </Card>

      {groups.length === 0 ? (
        <div className="text-center py-12">
          <Clock className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
          <p className="text-muted-foreground mb-4">
            No sessions in the last 24 hours.
          </p>
          <Button variant="outline" onClick={() => onNavigate("/timeline")}>
            Browse Timeline
          </Button>
        </div>
      ) : (
        <div className="space-y-6">
          {groups.map((group, i) => (
            <div
              key={i}
              className={cn(
                "rounded-lg border p-4",
                group.sessions.length > 1 && "border-primary/50 bg-primary/5"
              )}
            >
              <div className="flex items-center gap-3 mb-4">
                <span className="text-sm font-medium">
                  {formatDate(group.startTime)}
                </span>
                {group.sessions.length > 1 && (
                  <Badge variant="default" className="flex items-center gap-1">
                    <Users className="h-3 w-3" />
                    {group.sessions.length} parallel agents
                  </Badge>
                )}
                {group.sessions.some((s) => !s.endedAt) && (
                  <Badge className="bg-green-500 text-white animate-pulse">
                    ACTIVE
                  </Badge>
                )}
              </div>
              <div
                className={cn(
                  group.sessions.length > 1 && "grid gap-4 md:grid-cols-2"
                )}
              >
                {group.sessions.map((session) => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    onViewMemories={() =>
                      onNavigate(`/search?session=${session.id}`)
                    }
                    onViewTimeline={() =>
                      onNavigate(`/timeline?session=${session.id}`)
                    }
                    onSelectMemory={onSelectMemory as (memory: unknown) => void}
                    messages={messages}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatDate(ts: number): string {
  if (!ts || !Number.isFinite(ts)) return "";
  const t = ts < 1e12 ? ts * 1000 : ts;
  const date = new Date(t);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString();
}
