import { useState, useEffect } from "react";
import { Layout } from "./Layout.js";
import { Projects } from "./Projects.js";
import { Search } from "./Search.js";
import { Timeline } from "./Timeline.js";
import { AgentView } from "./AgentView.js";
import { Settings } from "./Settings.js";
import { MemoryDetail } from "./MemoryDetail.js";
import { useWebSocket } from "../hooks/useWebSocket.js";
import type { Memory } from "../../services/memory/types.js";
import type { SearchResult } from "../../services/search/hybrid.js";
import type { ActivityEvent } from "./ActivityFeed.js";

type Project = {
  id: string;
  path: string;
  name?: string;
  memory_count: number;
  session_count: number;
  last_activity?: number;
  created_at: number;
};

type InitialData = {
  type: string;
  results?: SearchResult[];
  sessions?: unknown[];
  projects?: Project[];
  memories?: unknown[];
  data?: unknown;
  recentActivity?: unknown[];
};

type AppProps = {
  url: string;
  initialData: unknown;
};

export function App({ url, initialData }: AppProps): JSX.Element {
  const [currentPath, setCurrentPath] = useState(() => {
    if (typeof window !== "undefined") {
      return window.location.pathname + window.location.search;
    }
    return url;
  });
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [data, setData] = useState<InitialData>(initialData as InitialData);

  const { connected, messages, send } = useWebSocket();

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handlePopState = (): void => {
      setCurrentPath(window.location.pathname + window.location.search);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    fetchPageData(currentPath).then(setData);
  }, [currentPath]);

  useEffect(() => {
    for (const msg of messages) {
      switch (msg.type) {
        case "memory:created":
          if (currentPath === "/" || currentPath === "/search") {
            setData((prev) => ({
              ...prev,
              results: [
                msg as unknown as SearchResult,
                ...(prev.results ?? []),
              ],
            }));
          }
          break;
        case "memory:updated":
          setData((prev) => ({
            ...prev,
            results: prev.results?.map((r) =>
              r.memory.id === (msg.memory as Memory)?.id
                ? { ...r, memory: msg.memory as Memory }
                : r
            ),
          }));
          if (selectedMemory?.id === (msg.memory as Memory)?.id) {
            setSelectedMemory(msg.memory as Memory);
          }
          break;
        case "session:updated":
          if (currentPath === "/agents") {
            setData((prev) => ({
              ...prev,
              sessions: prev.sessions?.map((s) =>
                (s as { id: string }).id === (msg.session as { id: string })?.id
                  ? msg.session
                  : s
              ),
            }));
          }
          break;
      }
    }
  }, [messages, currentPath, selectedMemory?.id]);

  const navigate = (path: string): void => {
    if (typeof window !== "undefined") {
      window.history.pushState({}, "", path);
    }
    setCurrentPath(path);
  };

  const renderPage = (): JSX.Element => {
    const pathname = currentPath.split("?")[0] ?? currentPath;

    if (pathname === "/projects") {
      return (
        <Projects
          initialProjects={(data.projects ?? []) as Project[]}
          onSelectProject={(projectId) => navigate(`/search?project=${projectId}`)}
          onNavigate={navigate}
          wsConnected={connected}
        />
      );
    }
    if (pathname === "/" || pathname === "/search") {
      return (
        <Search
          initialResults={data.results ?? []}
          onSelectMemory={setSelectedMemory}
          wsConnected={connected}
        />
      );
    }
    if (pathname === "/timeline") {
      return (
        <Timeline
          initialData={data}
          onSelectMemory={setSelectedMemory}
        />
      );
    }
    if (pathname === "/agents") {
      return (
        <AgentView
          initialSessions={(data.sessions ?? []) as unknown[]}
          wsConnected={connected}
          onNavigate={navigate}
          messages={messages}
          onSelectMemory={setSelectedMemory}
          initialActivity={(data.recentActivity ?? []) as ActivityEvent[]}
        />
      );
    }
    if (pathname === "/settings") {
      return <Settings />;
    }
    return (
      <Projects
          initialProjects={(data.projects ?? []) as Project[]}
          onSelectProject={(projectId) => navigate(`/search?project=${projectId}`)}
          onNavigate={navigate}
          wsConnected={connected}
        />
    );
  };

  return (
    <Layout
      currentPath={currentPath}
      onNavigate={navigate}
      wsConnected={connected}
    >
      {renderPage()}
      {selectedMemory && (
        <MemoryDetail
          memory={selectedMemory}
          onClose={() => setSelectedMemory(null)}
          onReinforce={(id) => send({ type: "memory:reinforce", memoryId: id })}
          onDeemphasize={(id) =>
            send({ type: "memory:deemphasize", memoryId: id })
          }
          onDelete={(id, hard) =>
            send({ type: "memory:delete", memoryId: id, hard })
          }
          onViewTimeline={(id) => {
            setSelectedMemory(null);
            navigate(`/timeline?anchor=${id}`);
          }}
        />
      )}
    </Layout>
  );
}

async function fetchPageData(path: string): Promise<InitialData> {
  if (typeof window === "undefined") {
    return { type: "home" };
  }
  const res = await fetch(`/api/page-data?path=${encodeURIComponent(path)}`);
  return (await res.json()) as InitialData;
}
