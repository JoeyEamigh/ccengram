import {
  Plus,
  Minus,
  Clock,
  Eye,
  History,
  Archive,
  Trash2,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "./ui/dialog.js";
import { Badge } from "./ui/badge.js";
import { Button } from "./ui/button.js";
import type { Memory, MemorySector } from "../../services/memory/types.js";

type MemoryDetailProps = {
  memory: Memory;
  onClose: () => void;
  onReinforce: (id: string) => void;
  onDeemphasize: (id: string) => void;
  onDelete: (id: string, hard: boolean) => void;
  onViewTimeline: (id: string) => void;
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

export function MemoryDetail({
  memory,
  onClose,
  onReinforce,
  onDeemphasize,
  onDelete,
  onViewTimeline,
}: MemoryDetailProps): JSX.Element {
  const handleDelete = (hard: boolean): void => {
    const msg = hard
      ? "Permanently delete this memory? This cannot be undone."
      : "Archive this memory?";
    if (typeof window !== "undefined" && window.confirm(msg)) {
      onDelete(memory.id, hard);
      onClose();
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant={sectorVariant[memory.sector]}>{memory.sector}</Badge>
            <Badge variant="outline">{memory.tier}</Badge>
            {memory.isDeleted && <Badge variant="destructive">DELETED</Badge>}
            {memory.validUntil && <Badge variant="destructive">SUPERSEDED</Badge>}
          </div>
          <DialogTitle className="sr-only">Memory Details</DialogTitle>
          <DialogDescription className="sr-only">
            View and manage memory details
          </DialogDescription>
        </DialogHeader>

        <div className="prose prose-sm dark:prose-invert max-w-none py-4">
          <p className="whitespace-pre-wrap">{memory.content}</p>
        </div>

        <div className="space-y-3 border-t pt-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Salience</span>
            <div className="flex items-center gap-2">
              <span className="font-medium">
                {(memory.salience * 100).toFixed(0)}%
              </span>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => onReinforce(memory.id)}
              >
                <Plus className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => onDeemphasize(memory.id)}
              >
                <Minus className="h-3 w-3" />
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Created: {formatDate(memory.createdAt)}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Eye className="h-4 w-4" />
              <span>Accessed: {memory.accessCount} times</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground col-span-2">
              <History className="h-4 w-4" />
              <span>Last accessed: {formatDate(memory.lastAccessed)}</span>
            </div>
          </div>
        </div>

        <DialogFooter className="flex-col sm:flex-row gap-2">
          <Button variant="outline" onClick={() => onViewTimeline(memory.id)}>
            <History className="mr-2 h-4 w-4" />
            View Timeline
          </Button>
          <div className="flex gap-2 ml-auto">
            <Button variant="secondary" onClick={() => handleDelete(false)}>
              <Archive className="mr-2 h-4 w-4" />
              Archive
            </Button>
            <Button variant="destructive" onClick={() => handleDelete(true)}>
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Forever
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
