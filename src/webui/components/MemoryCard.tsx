import { Link2 } from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader } from "./ui/card.js";
import { Badge } from "./ui/badge.js";
import { cn } from "../lib/utils.js";
import type { SearchResult } from "../../services/search/hybrid.js";
import type { MemorySector } from "../../services/memory/types.js";

type MemoryCardProps = {
  result: SearchResult;
  onClick: () => void;
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

export function MemoryCard({ result, onClick }: MemoryCardProps): JSX.Element {
  const { memory, score, sourceSession, isSuperseded, supersededBy, relatedMemoryCount } =
    result;

  return (
    <Card
      className={cn(
        "cursor-pointer transition-colors hover:bg-accent/50",
        isSuperseded && "opacity-60"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={sectorVariant[memory.sector]}>{memory.sector}</Badge>
          <span className="text-sm text-muted-foreground">
            Score: {(score * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-muted-foreground">
            Salience: {(memory.salience * 100).toFixed(0)}%
          </span>

          {sourceSession && (
            <span
              className="text-sm text-muted-foreground ml-auto"
              title={sourceSession.summary ?? ""}
            >
              {formatDate(sourceSession.startedAt)}
            </span>
          )}

          {isSuperseded && (
            <Badge variant="destructive" title={`Superseded by ${supersededBy?.id}`}>
              SUPERSEDED
            </Badge>
          )}

          {relatedMemoryCount > 0 && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Link2 className="h-3 w-3" />
              {relatedMemoryCount} related
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm leading-relaxed">
          {memory.content.slice(0, 300)}
          {memory.content.length > 300 ? "..." : ""}
        </p>
      </CardContent>

      <CardFooter className="pt-2 text-xs text-muted-foreground">
        <span>{formatDate(memory.createdAt)}</span>
        {memory.tags && memory.tags.length > 0 && (
          <span className="ml-auto">{memory.tags.join(", ")}</span>
        )}
      </CardFooter>
    </Card>
  );
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
