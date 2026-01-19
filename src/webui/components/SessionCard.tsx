import { Clock, Brain, Activity } from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader } from "./ui/card.js";
import { Badge } from "./ui/badge.js";
import { Button } from "./ui/button.js";
import { cn } from "../lib/utils.js";

type Session = {
  id: string;
  startedAt: number;
  endedAt?: number;
  summary?: string;
  memoryCount?: number;
  lastActivity?: number;
};

type SessionCardProps = {
  session: Session;
  onViewMemories: () => void;
  onViewTimeline: () => void;
};

export function SessionCard({
  session,
  onViewMemories,
  onViewTimeline,
}: SessionCardProps): JSX.Element {
  const isActive = !session.endedAt;
  const duration = session.endedAt
    ? formatDuration(session.endedAt - session.startedAt)
    : formatDuration(Date.now() - session.startedAt) + " (active)";

  return (
    <Card className={cn(isActive && "ring-2 ring-green-500/50")}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <span
            className="font-mono text-sm text-muted-foreground"
            title={session.id}
          >
            {session.id.slice(0, 8)}...
          </span>
          {isActive && (
            <Badge className="bg-green-500 text-white animate-pulse">
              ACTIVE
            </Badge>
          )}
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
            <span>{session.memoryCount ?? 0} memories</span>
          </div>
        </div>
        {session.lastActivity && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="h-4 w-4" />
            <span>Last: {formatDate(session.lastActivity)}</span>
          </div>
        )}
        {session.summary && (
          <p className="text-sm text-muted-foreground line-clamp-2">
            {session.summary}
          </p>
        )}
      </CardContent>

      <CardFooter className="gap-2">
        <Button variant="outline" size="sm" onClick={onViewMemories}>
          View Memories
        </Button>
        <Button variant="outline" size="sm" onClick={onViewTimeline}>
          View Timeline
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

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
