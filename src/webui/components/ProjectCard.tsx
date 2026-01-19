import { FolderGit2, Brain, Users, Activity, Search, Clock } from "lucide-react";
import { Card, CardContent, CardHeader } from "./ui/card.js";
import { Button } from "./ui/button.js";
import { cn } from "../lib/utils.js";
import { RelativeTime } from "./RelativeTime.js";

type Project = {
  id: string;
  path: string;
  name?: string;
  memory_count: number;
  session_count: number;
  last_activity?: number;
  created_at: number;
};

type ProjectCardProps = {
  project: Project;
  onClick: () => void;
  onViewMemories: () => void;
  onViewTimeline: () => void;
};

export function ProjectCard({ project, onClick, onViewMemories, onViewTimeline }: ProjectCardProps): JSX.Element {
  const displayName = project.name ?? project.path.split("/").pop() ?? project.path;
  const hasActivity = project.memory_count > 0 || project.session_count > 0;

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all duration-200 hover:shadow-md hover:bg-accent/30",
        !hasActivity && "opacity-60"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-md bg-primary/10">
            <FolderGit2 className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold truncate" title={displayName}>
              {displayName}
            </h3>
            <p
              className="text-xs text-muted-foreground truncate"
              title={project.path}
            >
              {project.path}
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span>{project.memory_count.toLocaleString()} memories</span>
          </div>
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <span>{project.session_count.toLocaleString()} sessions</span>
          </div>
        </div>
        {project.last_activity && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-3">
            <Activity className="h-3 w-3" />
            <span>Last activity: <RelativeTime timestamp={project.last_activity} /></span>
          </div>
        )}
        <div className="flex gap-2 mt-3 pt-3 border-t">
          <Button
            variant="ghost"
            size="sm"
            className="flex-1 text-xs"
            onClick={(e) => {
              e.stopPropagation();
              onViewMemories();
            }}
          >
            <Search className="h-3 w-3 mr-1" /> Memories
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="flex-1 text-xs"
            onClick={(e) => {
              e.stopPropagation();
              onViewTimeline();
            }}
          >
            <Clock className="h-3 w-3 mr-1" /> Timeline
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
