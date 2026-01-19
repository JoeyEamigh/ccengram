import { useState, useEffect } from "react";
import {
  Settings as SettingsIcon,
  RefreshCw,
  Brain,
  FolderOpen,
  FileText,
  Clock,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card.js";
import { Button } from "./ui/button.js";
import { Badge } from "./ui/badge.js";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select.js";
import { Checkbox } from "./ui/checkbox.js";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs.js";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "./ui/dialog.js";
import { cn } from "../lib/utils.js";

type Stats = {
  totals: {
    memories: number;
    projectMemories: number;
    documents: number;
    sessions: number;
  };
  bySector: Record<string, number>;
};

type ConfigMap = {
  embeddingProvider: string;
  captureEnabled: string;
  captureThreshold: string;
};

type Project = {
  id: string;
  path: string;
  name?: string;
  memory_count: number;
};

const sectorColors: Record<string, string> = {
  episodic: "bg-sector-episodic",
  semantic: "bg-sector-semantic",
  procedural: "bg-sector-procedural",
  emotional: "bg-sector-emotional",
  reflective: "bg-sector-reflective",
};

const sectorVariants: Record<
  string,
  "episodic" | "semantic" | "procedural" | "emotional" | "reflective"
> = {
  episodic: "episodic",
  semantic: "semantic",
  procedural: "procedural",
  emotional: "emotional",
  reflective: "reflective",
};

export function Settings(): JSX.Element {
  const [activeTab, setActiveTab] = useState("stats");
  const [stats, setStats] = useState<Stats | null>(null);
  const [config, setConfig] = useState<ConfigMap | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [selectedProjectForClear, setSelectedProjectForClear] = useState<string>("all");
  const [clearing, setClearing] = useState(false);

  const loadStats = async (): Promise<void> => {
    setLoading(true);
    try {
      const res = await fetch("/api/stats");
      const data = (await res.json()) as Stats;
      setStats(data);
    } finally {
      setLoading(false);
    }
  };

  const loadConfig = async (): Promise<void> => {
    try {
      const res = await fetch("/api/config");
      const data = (await res.json()) as { config: ConfigMap };
      setConfig(data.config);
    } catch {
      setConfig({
        embeddingProvider: "ollama",
        captureEnabled: "true",
        captureThreshold: "0.3",
      });
    }
  };

  const loadProjects = async (): Promise<void> => {
    try {
      const res = await fetch("/api/projects");
      const data = (await res.json()) as { projects: Project[] };
      setProjects(data.projects);
    } catch {
      setProjects([]);
    }
  };

  const updateConfig = async (key: keyof ConfigMap, value: string): Promise<void> => {
    setSaving(true);
    try {
      await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, value }),
      });
      setConfig((prev) => (prev ? { ...prev, [key]: value } : null));
    } finally {
      setSaving(false);
    }
  };

  const clearMemories = async (): Promise<void> => {
    setClearing(true);
    try {
      const body = selectedProjectForClear === "all" ? {} : { projectId: selectedProjectForClear };
      await fetch("/api/memories/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setClearDialogOpen(false);
      await loadStats();
      await loadProjects();
    } finally {
      setClearing(false);
    }
  };

  useEffect(() => {
    loadStats();
    loadConfig();
    loadProjects();
  }, []);

  const totalSectorCount = stats
    ? Object.values(stats.bySector).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
            <SettingsIcon className="h-6 w-6" />
            Settings
          </h2>
          <p className="text-muted-foreground">
            System statistics and configuration
          </p>
        </div>
        <Button variant="outline" onClick={() => { loadStats(); loadConfig(); loadProjects(); }} disabled={loading}>
          <RefreshCw
            className={cn("h-4 w-4 mr-2", loading && "animate-spin")}
          />
          Refresh
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="management">Memory Management</TabsTrigger>
        </TabsList>

        <TabsContent value="stats">
          {stats && (
            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card className="relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-16 h-16 -mr-4 -mt-4 bg-primary/5 rounded-full" />
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                      <Brain className="h-4 w-4" />
                      Total Memories
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold tabular-nums">
                      {stats.totals.memories.toLocaleString()}
                    </div>
                  </CardContent>
                </Card>

                <Card className="relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-16 h-16 -mr-4 -mt-4 bg-primary/5 rounded-full" />
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                      <FolderOpen className="h-4 w-4" />
                      Project Memories
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold tabular-nums">
                      {stats.totals.projectMemories.toLocaleString()}
                    </div>
                  </CardContent>
                </Card>

                <Card className="relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-16 h-16 -mr-4 -mt-4 bg-primary/5 rounded-full" />
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      Documents
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold tabular-nums">
                      {stats.totals.documents.toLocaleString()}
                    </div>
                  </CardContent>
                </Card>

                <Card className="relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-16 h-16 -mr-4 -mt-4 bg-primary/5 rounded-full" />
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      Sessions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold tabular-nums">
                      {stats.totals.sessions.toLocaleString()}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {Object.keys(stats.bySector).length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Memories by Sector</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {Object.entries(stats.bySector).map(([sector, count]) => {
                      const percentage =
                        totalSectorCount > 0 ? (count / totalSectorCount) * 100 : 0;
                      return (
                        <div key={sector} className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Badge variant={sectorVariants[sector] ?? "default"}>
                              {sector}
                            </Badge>
                            <span className="text-sm font-medium tabular-nums">
                              {count.toLocaleString()}{" "}
                              <span className="text-muted-foreground">
                                ({percentage.toFixed(1)}%)
                              </span>
                            </span>
                          </div>
                          <div className="h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full transition-all duration-500",
                                sectorColors[sector] ?? "bg-primary"
                              )}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="config">
          {config && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Embedding Provider</CardTitle>
                  <CardDescription>
                    Choose how memories are embedded for semantic search
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Select
                    value={config.embeddingProvider}
                    onValueChange={(v) => updateConfig("embeddingProvider", v)}
                    disabled={saving}
                  >
                    <SelectTrigger className="w-[200px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ollama">Ollama (Local)</SelectItem>
                      <SelectItem value="openrouter">OpenRouter (Cloud)</SelectItem>
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Memory Capture</CardTitle>
                  <CardDescription>
                    Configure automatic memory capture from tool observations
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="captureEnabled"
                      checked={config.captureEnabled === "true"}
                      onCheckedChange={(checked) =>
                        updateConfig("captureEnabled", checked ? "true" : "false")
                      }
                      disabled={saving}
                    />
                    <label
                      htmlFor="captureEnabled"
                      className="text-sm font-medium leading-none cursor-pointer"
                    >
                      Enable automatic memory capture
                    </label>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Capture Threshold</label>
                    <Select
                      value={config.captureThreshold}
                      onValueChange={(v) => updateConfig("captureThreshold", v)}
                      disabled={saving || config.captureEnabled !== "true"}
                    >
                      <SelectTrigger className="w-[200px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0.1">Low (0.1)</SelectItem>
                        <SelectItem value="0.3">Medium (0.3)</SelectItem>
                        <SelectItem value="0.5">High (0.5)</SelectItem>
                        <SelectItem value="0.7">Very High (0.7)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Only capture memories with importance above this threshold
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="management">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <Trash2 className="h-5 w-5" />
                Clear Memories
              </CardTitle>
              <CardDescription>
                Permanently delete all memories for a project or all projects
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Select Project</label>
                <Select
                  value={selectedProjectForClear}
                  onValueChange={setSelectedProjectForClear}
                >
                  <SelectTrigger className="w-[300px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Projects</SelectItem>
                    {projects.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name ?? p.path.split("/").pop()} ({p.memory_count} memories)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Button
                variant="destructive"
                onClick={() => setClearDialogOpen(true)}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Clear Memories
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              Confirm Memory Deletion
            </DialogTitle>
            <DialogDescription>
              {selectedProjectForClear === "all"
                ? "This will delete ALL memories across ALL projects. This action cannot be undone."
                : `This will delete all memories for the selected project. This action cannot be undone.`}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setClearDialogOpen(false)}
              disabled={clearing}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={clearMemories}
              disabled={clearing}
            >
              {clearing ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete All Memories"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
