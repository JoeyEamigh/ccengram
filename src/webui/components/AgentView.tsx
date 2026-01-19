import { useState, useEffect } from "react";
import { Users } from "lucide-react";
import { SessionCard } from "./SessionCard.js";
import { Badge } from "./ui/badge.js";
import { cn } from "../lib/utils.js";

type Session = {
  id: string;
  projectId: string;
  startedAt: number;
  endedAt?: number;
  summary?: string;
  memoryCount?: number;
  lastActivity?: number;
};

type AgentViewProps = {
  initialSessions: unknown[];
  wsConnected: boolean;
  onNavigate: (path: string) => void;
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
}: AgentViewProps): JSX.Element {
  const [sessions] = useState(initialSessions as Session[]);
  const [groups, setGroups] = useState<ParallelGroup[]>([]);

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

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Agent Sessions</h2>
        <p className="text-muted-foreground">
          View parallel and recent Claude Code sessions
        </p>
      </div>

      <div className="flex items-center gap-2 text-sm">
        {wsConnected ? (
          <span className="flex items-center gap-1.5 text-green-600 dark:text-green-400">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            Live updates enabled
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-muted" />
            Connecting...
          </span>
        )}
      </div>

      {groups.length === 0 ? (
        <p className="text-center text-muted-foreground py-12">
          No sessions in the last 24 hours.
        </p>
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
  return new Date(ts).toLocaleString();
}
