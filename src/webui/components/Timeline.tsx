import { useState, useCallback } from "react";
import { Clock, Search } from "lucide-react";
import { Card, CardContent, CardHeader } from "./ui/card.js";
import { Badge } from "./ui/badge.js";
import { Button } from "./ui/button.js";
import { Input } from "./ui/input.js";
import { cn } from "../lib/utils.js";
import type { Memory, MemorySector } from "../../services/memory/types.js";
import type { TimelineResult } from "../../services/search/hybrid.js";

type TimelineProps = {
  initialData: unknown;
  onSelectMemory: (memory: Memory) => void;
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

function MemoryTimelineCard({
  memory,
  onClick,
  isAnchor,
}: {
  memory: Memory;
  onClick: () => void;
  isAnchor?: boolean;
}): JSX.Element {
  return (
    <Card
      className={cn(
        "cursor-pointer transition-colors hover:bg-accent/50",
        isAnchor && "ring-2 ring-primary"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={sectorVariant[memory.sector]}>{memory.sector}</Badge>
          <span className="text-sm text-muted-foreground">
            Salience: {(memory.salience * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-muted-foreground ml-auto">
            {formatDate(memory.createdAt)}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm leading-relaxed">
          {memory.content.slice(0, 300)}
          {memory.content.length > 300 ? "..." : ""}
        </p>
      </CardContent>
    </Card>
  );
}

export function Timeline({
  initialData,
  onSelectMemory,
}: TimelineProps): JSX.Element {
  const [anchorId, setAnchorId] = useState("");
  const [data, setData] = useState<TimelineResult | null>(
    initialData as TimelineResult | null
  );
  const [loading, setLoading] = useState(false);

  const handleLoad = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!anchorId.trim()) return;

      setLoading(true);
      try {
        const res = await fetch(
          `/api/timeline?anchor=${encodeURIComponent(anchorId)}`
        );
        const json = (await res.json()) as { data: TimelineResult };
        setData(json.data);
      } finally {
        setLoading(false);
      }
    },
    [anchorId]
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Timeline</h2>
        <p className="text-muted-foreground">
          View memories around a specific point in time
        </p>
      </div>

      <form onSubmit={handleLoad} className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            value={anchorId}
            onChange={(e) => setAnchorId(e.target.value)}
            placeholder="Enter memory ID to anchor timeline..."
            className="pl-10"
          />
        </div>
        <Button type="submit" disabled={loading}>
          {loading ? "Loading..." : "Load Timeline"}
        </Button>
      </form>

      {data ? (
        <div className="space-y-6">
          {data.before.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-muted-foreground">
                Before
              </h3>
              {data.before.map((memory: Memory) => (
                <MemoryTimelineCard
                  key={memory.id}
                  memory={memory}
                  onClick={() => onSelectMemory(memory)}
                />
              ))}
            </div>
          )}

          {data.anchor && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Anchor Memory
              </h3>
              <MemoryTimelineCard
                memory={data.anchor}
                onClick={() => onSelectMemory(data.anchor)}
                isAnchor
              />
            </div>
          )}

          {data.after.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-muted-foreground">
                After
              </h3>
              {data.after.map((memory: Memory) => (
                <MemoryTimelineCard
                  key={memory.id}
                  memory={memory}
                  onClick={() => onSelectMemory(memory)}
                />
              ))}
            </div>
          )}
        </div>
      ) : (
        <p className="text-center text-muted-foreground py-12">
          Enter a memory ID to view its timeline context.
        </p>
      )}
    </div>
  );
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
