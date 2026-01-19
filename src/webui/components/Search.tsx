import { useState, useCallback } from "react";
import { Search as SearchIcon, Loader2 } from "lucide-react";
import { MemoryCard } from "./MemoryCard.js";
import { Button } from "./ui/button.js";
import { Input } from "./ui/input.js";
import { Checkbox } from "./ui/checkbox.js";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select.js";
import type { SearchResult } from "../../services/search/hybrid.js";
import type { Memory, MemorySector } from "../../services/memory/types.js";

type SearchProps = {
  initialResults: SearchResult[];
  onSelectMemory: (memory: Memory) => void;
  wsConnected: boolean;
};

export function Search({
  initialResults,
  onSelectMemory,
  wsConnected,
}: SearchProps): JSX.Element {
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState<MemorySector | "all">("all");
  const [includeSuperseded, setIncludeSuperseded] = useState(false);
  const [results, setResults] = useState(initialResults);
  const [loading, setLoading] = useState(false);

  const handleSearch = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!query.trim()) return;

      setLoading(true);
      try {
        const params = new URLSearchParams({ q: query });
        if (sector !== "all") params.set("sector", sector);
        if (includeSuperseded) params.set("include_superseded", "true");

        const res = await fetch(`/api/search?${params}`);
        const data = (await res.json()) as { results: SearchResult[] };
        setResults(data.results);
        if (typeof window !== "undefined") {
          window.history.pushState({}, "", `/search?${params}`);
        }
      } finally {
        setLoading(false);
      }
    },
    [query, sector, includeSuperseded]
  );

  return (
    <div className="space-y-6">
      <form
        onSubmit={handleSearch}
        className="flex flex-col gap-4 sm:flex-row sm:items-center"
      >
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search memories..."
            className="pl-10"
            autoFocus
          />
        </div>

        <Select
          value={sector}
          onValueChange={(v) => setSector(v as MemorySector | "all")}
        >
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="All Sectors" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Sectors</SelectItem>
            <SelectItem value="episodic">Episodic</SelectItem>
            <SelectItem value="semantic">Semantic</SelectItem>
            <SelectItem value="procedural">Procedural</SelectItem>
            <SelectItem value="emotional">Emotional</SelectItem>
            <SelectItem value="reflective">Reflective</SelectItem>
          </SelectContent>
        </Select>

        <div className="flex items-center space-x-2">
          <Checkbox
            id="superseded"
            checked={includeSuperseded}
            onCheckedChange={(checked) => setIncludeSuperseded(checked === true)}
          />
          <label
            htmlFor="superseded"
            className="text-sm text-muted-foreground cursor-pointer"
          >
            Include superseded
          </label>
        </div>

        <Button type="submit" disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Searching...
            </>
          ) : (
            "Search"
          )}
        </Button>
      </form>

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

      <div className="space-y-4">
        {results.length === 0 ? (
          <p className="text-center text-muted-foreground py-12">
            {query
              ? "No memories found."
              : "Enter a search query to find memories."}
          </p>
        ) : (
          results.map((r) => (
            <MemoryCard
              key={r.memory.id}
              result={r}
              onClick={() => onSelectMemory(r.memory)}
            />
          ))
        )}
      </div>
    </div>
  );
}
