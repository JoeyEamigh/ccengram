import { useState, useCallback, useEffect } from "react";
import { Clock, Search, Calendar, ChevronLeft, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader } from "./ui/card.js";
import { Badge } from "./ui/badge.js";
import { Button } from "./ui/button.js";
import { Input } from "./ui/input.js";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "./ui/select.js";
import { cn } from "../lib/utils.js";
import { RelativeTime } from "./RelativeTime.js";
import type { Memory, MemorySector } from "../../services/memory/types.js";
import type { TimelineResult } from "../../services/search/hybrid.js";

type BrowseMemory = Memory & {
  source_session_id?: string;
  source_session_summary?: string;
  created_at?: number;
};

type DateAggregate = {
  date: string;
  count: number;
};

type Project = {
  id: string;
  path: string;
  name?: string;
};

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

const sectors: MemorySector[] = [
  "episodic",
  "semantic",
  "procedural",
  "emotional",
  "reflective",
];

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
          <RelativeTime
            timestamp={memory.createdAt}
            className="text-sm text-muted-foreground ml-auto"
          />
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

function BrowseMemoryCard({
  memory,
  onClick,
  onViewTimeline,
}: {
  memory: BrowseMemory;
  onClick: () => void;
  onViewTimeline: () => void;
}): JSX.Element {
  return (
    <Card className="cursor-pointer transition-colors hover:bg-accent/50">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={sectorVariant[memory.sector]}>{memory.sector}</Badge>
          <span className="text-xs text-muted-foreground">
            {(memory.salience * 100).toFixed(0)}%
          </span>
          <RelativeTime
            timestamp={memory.created_at ?? memory.createdAt}
            className="text-xs text-muted-foreground ml-auto"
          />
        </div>
      </CardHeader>
      <CardContent className="space-y-2" onClick={onClick}>
        <p className="text-sm leading-relaxed line-clamp-2">
          {memory.summary ?? memory.content}
        </p>
        {memory.source_session_summary && (
          <p className="text-xs text-muted-foreground line-clamp-1">
            Session: {memory.source_session_summary}
          </p>
        )}
      </CardContent>
      <div className="px-4 pb-3">
        <Button
          variant="ghost"
          size="sm"
          className="text-xs"
          onClick={(e) => {
            e.stopPropagation();
            onViewTimeline();
          }}
        >
          <Clock className="h-3 w-3 mr-1" />
          View Timeline
        </Button>
      </div>
    </Card>
  );
}

export function Timeline({
  initialData,
  onSelectMemory,
}: TimelineProps): JSX.Element {
  const parsed = initialData as {
    browseMode?: boolean;
    data?: TimelineResult;
    memories?: BrowseMemory[];
    dateAggregates?: DateAggregate[];
    projects?: Project[];
    hasMore?: boolean;
  } | null;

  const [browseMode, setBrowseMode] = useState(parsed?.browseMode ?? true);
  const [anchorId, setAnchorId] = useState("");
  const [timelineData, setTimelineData] = useState<TimelineResult | null>(
    parsed?.data ?? null
  );

  const [memories, setMemories] = useState<BrowseMemory[]>(
    parsed?.memories ?? []
  );
  const [dateAggregates] = useState<DateAggregate[]>(
    parsed?.dateAggregates ?? []
  );
  const [projects] = useState<Project[]>(parsed?.projects ?? []);
  const [hasMore, setHasMore] = useState(parsed?.hasMore ?? false);

  const [selectedProject, setSelectedProject] = useState<string>("");
  const [selectedSector, setSelectedSector] = useState<string>("");
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const newParsed = initialData as {
      browseMode?: boolean;
      data?: TimelineResult;
      memories?: BrowseMemory[];
      dateAggregates?: DateAggregate[];
      projects?: Project[];
      hasMore?: boolean;
    } | null;

    if (newParsed?.data) {
      setTimelineData(newParsed.data);
      setBrowseMode(false);
    } else if (newParsed?.memories) {
      setMemories(newParsed.memories);
      setHasMore(newParsed.hasMore ?? false);
      setBrowseMode(true);
    }
  }, [initialData]);

  const loadTimeline = useCallback(
    async (id: string) => {
      if (!id.trim()) return;
      setLoading(true);
      try {
        const res = await fetch(`/api/timeline?anchor=${encodeURIComponent(id)}`);
        const json = (await res.json()) as { data: TimelineResult };
        setTimelineData(json.data);
        setBrowseMode(false);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const loadBrowse = useCallback(
    async (date?: string, project?: string, sector?: string) => {
      setLoading(true);
      setOffset(0);
      try {
        const params = new URLSearchParams();
        if (date) params.set("date", date);
        if (project) params.set("project", project);
        if (sector) params.set("sector", sector);

        const res = await fetch(`/api/timeline/browse?${params.toString()}`);
        const json = (await res.json()) as {
          memories: BrowseMemory[];
          hasMore: boolean;
        };
        setMemories(json.memories);
        setHasMore(json.hasMore);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    try {
      const nextOffset = offset + 50;
      const params = new URLSearchParams();
      if (selectedDate) params.set("date", selectedDate);
      if (selectedProject) params.set("project", selectedProject);
      if (selectedSector) params.set("sector", selectedSector);
      params.set("offset", String(nextOffset));

      const res = await fetch(`/api/timeline/browse?${params.toString()}`);
      const json = (await res.json()) as {
        memories: BrowseMemory[];
        hasMore: boolean;
      };
      setMemories((prev) => [...prev, ...json.memories]);
      setHasMore(json.hasMore);
      setOffset(nextOffset);
    } finally {
      setLoadingMore(false);
    }
  }, [loadingMore, hasMore, offset, selectedDate, selectedProject, selectedSector]);

  const handleDateSelect = (dateStr: string): void => {
    setSelectedDate(dateStr);
    loadBrowse(dateStr, selectedProject, selectedSector);
  };

  const handleQuickDate = (days: number): void => {
    const d = new Date();
    d.setDate(d.getDate() - days);
    const dateStr = d.toISOString().split("T")[0] ?? "";
    setSelectedDate(dateStr);
    loadBrowse(dateStr, selectedProject, selectedSector);
  };

  const handleProjectChange = (value: string): void => {
    const projectId = value === "all" ? "" : value;
    setSelectedProject(projectId);
    loadBrowse(selectedDate, projectId, selectedSector);
  };

  const handleSectorChange = (value: string): void => {
    const sector = value === "all" ? "" : value;
    setSelectedSector(sector);
    loadBrowse(selectedDate, selectedProject, sector);
  };

  const handleSubmit = (e: React.FormEvent): void => {
    e.preventDefault();
    if (anchorId.trim()) {
      loadTimeline(anchorId);
    }
  };

  const backToBrowse = (): void => {
    setBrowseMode(true);
    setTimelineData(null);
    setAnchorId("");
  };

  const groupMemoriesByDate = (
    mems: BrowseMemory[]
  ): Map<string, BrowseMemory[]> => {
    const groups = new Map<string, BrowseMemory[]>();
    for (const mem of mems) {
      const rawTs = mem.created_at ?? mem.createdAt ?? 0;
      const ts = rawTs < 1e12 ? rawTs * 1000 : rawTs;
      const dateKey = new Date(ts).toLocaleDateString(undefined, {
        weekday: "short",
        month: "short",
        day: "numeric",
      });
      if (!groups.has(dateKey)) {
        groups.set(dateKey, []);
      }
      groups.get(dateKey)!.push(mem);
    }
    return groups;
  };

  if (!browseMode && timelineData) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={backToBrowse}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Browse
          </Button>
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Timeline</h2>
            <p className="text-muted-foreground">
              Memories around anchor point
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {timelineData.before.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-muted-foreground">
                Before
              </h3>
              {timelineData.before.map((memory: Memory) => (
                <MemoryTimelineCard
                  key={memory.id}
                  memory={memory}
                  onClick={() => onSelectMemory(memory)}
                />
              ))}
            </div>
          )}

          {timelineData.anchor && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Anchor Memory
              </h3>
              <MemoryTimelineCard
                memory={timelineData.anchor}
                onClick={() => onSelectMemory(timelineData.anchor)}
                isAnchor
              />
            </div>
          )}

          {timelineData.after.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-muted-foreground">
                After
              </h3>
              {timelineData.after.map((memory: Memory) => (
                <MemoryTimelineCard
                  key={memory.id}
                  memory={memory}
                  onClick={() => onSelectMemory(memory)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  const groupedMemories = groupMemoriesByDate(memories);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Calendar className="h-6 w-6" />
          Timeline
        </h2>
        <p className="text-muted-foreground">
          Browse memories by date or search by ID
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            value={anchorId}
            onChange={(e) => setAnchorId(e.target.value)}
            placeholder="Jump to memory ID..."
            className="pl-10"
          />
        </div>
        <Button type="submit" disabled={loading || !anchorId.trim()}>
          Go
        </Button>
      </form>

      <div className="flex flex-wrap gap-4">
        <div className="flex gap-2">
          <Button
            variant={selectedDate === new Date().toISOString().split("T")[0] ? "default" : "outline"}
            size="sm"
            onClick={() => handleQuickDate(0)}
          >
            Today
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleQuickDate(1)}
          >
            Yesterday
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSelectedDate("");
              loadBrowse("", selectedProject, selectedSector);
            }}
          >
            Last 7 Days
          </Button>
        </div>

        <div className="flex gap-2 ml-auto">
          {projects.length > 0 && (
            <Select value={selectedProject || "all"} onValueChange={handleProjectChange}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Projects" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Projects</SelectItem>
                {projects.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name ?? p.path.split("/").pop()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          <Select value={selectedSector || "all"} onValueChange={handleSectorChange}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="All Sectors" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sectors</SelectItem>
              {sectors.map((s) => (
                <SelectItem key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant="outline"
            size="icon"
            onClick={() => loadBrowse(selectedDate, selectedProject, selectedSector)}
            disabled={loading}
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          </Button>
        </div>
      </div>

      {dateAggregates.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {dateAggregates.slice(0, 14).map((agg) => (
            <Button
              key={agg.date}
              variant={selectedDate === agg.date ? "default" : "ghost"}
              size="sm"
              className="text-xs"
              onClick={() => handleDateSelect(agg.date)}
            >
              {formatShortDate(agg.date)} ({agg.count})
            </Button>
          ))}
        </div>
      )}

      {memories.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          {loading ? "Loading..." : "No memories found for this time range."}
        </div>
      ) : (
        <div className="space-y-8">
          {Array.from(groupedMemories.entries()).map(([dateKey, mems]) => (
            <div key={dateKey}>
              <h3 className="text-sm font-medium text-muted-foreground mb-4 sticky top-14 bg-background py-2 z-10 border-b">
                {dateKey}
              </h3>
              <div className="grid gap-4 sm:grid-cols-2">
                {mems.map((memory) => (
                  <BrowseMemoryCard
                    key={memory.id}
                    memory={memory}
                    onClick={() => onSelectMemory(memory)}
                    onViewTimeline={() => loadTimeline(memory.id)}
                  />
                ))}
              </div>
            </div>
          ))}
          {hasMore && (
            <div className="text-center">
              <Button
                variant="outline"
                onClick={loadMore}
                disabled={loadingMore}
              >
                {loadingMore ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Loading...
                  </>
                ) : (
                  "Load More"
                )}
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatShortDate(dateStr: string): string {
  const d = new Date(dateStr + "T00:00:00");
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
