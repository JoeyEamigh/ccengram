import { useState, useEffect } from "react";

type RelativeTimeProps = {
  timestamp: number;
  className?: string;
};

export function RelativeTime({ timestamp, className }: RelativeTimeProps): JSX.Element {
  const [, setTick] = useState(0);

  useEffect(() => {
    const getInterval = (): number => {
      const diffMs = Date.now() - timestamp;
      const diffSecs = Math.floor(diffMs / 1000);

      // For very recent times (< 1 min), update every second
      if (diffSecs < 60) return 1000;
      // For times 1-60 min ago, update every 30 seconds
      if (diffSecs < 3600) return 30000;
      // For times 1-24 hours ago, update every minute
      if (diffSecs < 86400) return 60000;
      // For older times, no need to update
      return 0;
    };

    const interval = getInterval();
    if (interval === 0) return;

    // Synchronize to wall clock boundaries
    const now = Date.now();
    const delay = interval - (now % interval);

    const timeoutId = setTimeout(() => {
      setTick((t) => t + 1);
      const intervalId = setInterval(() => {
        setTick((t) => t + 1);
      }, interval);

      // Store interval ID for cleanup
      (timeoutId as any).intervalId = intervalId;
    }, delay);

    return () => {
      clearTimeout(timeoutId);
      if ((timeoutId as any).intervalId) {
        clearInterval((timeoutId as any).intervalId);
      }
    };
  }, [timestamp]);

  return <span className={className}>{formatRelativeTime(timestamp)}</span>;
}

export function formatRelativeTime(ts: number): string {
  if (!ts || !Number.isFinite(ts)) return "";

  const now = Date.now();
  const diffMs = now - ts;
  const diffSecs = Math.floor(diffMs / 1000);

  if (diffSecs < 5) return "just now";
  if (diffSecs < 60) return `${diffSecs}s ago`;

  const diffMins = Math.floor(diffSecs / 60);
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;

  const date = new Date(ts);
  const nowDate = new Date(now);

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() !== nowDate.getFullYear() ? "numeric" : undefined,
  });
}

export function formatRelativeTimeShort(ts: number): string {
  if (!ts || !Number.isFinite(ts)) return "";

  const now = Date.now();
  const diffMs = now - ts;
  const diffSecs = Math.floor(diffMs / 1000);

  if (diffSecs < 5) return "just now";
  if (diffSecs < 60) return `${diffSecs}s ago`;

  const diffMins = Math.floor(diffSecs / 60);
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}
