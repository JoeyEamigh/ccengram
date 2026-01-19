import { useState, useEffect } from "react";
import { Layout } from "./Layout.js";
import { Search } from "./Search.js";
import { Timeline } from "./Timeline.js";
import { AgentView } from "./AgentView.js";
import { Settings } from "./Settings.js";
import { MemoryDetail } from "./MemoryDetail.js";
import { useWebSocket } from "../hooks/useWebSocket.js";
import type { Memory } from "../../services/memory/types.js";
import type { SearchResult } from "../../services/search/hybrid.js";

type InitialData = {
  type: string;
  results?: SearchResult[];
  sessions?: unknown[];
  data?: unknown;
};

type AppProps = {
  url: string;
  initialData: unknown;
};

export function App({ url, initialData }: AppProps): JSX.Element {
  const [currentPath, setCurrentPath] = useState(url);
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [data, setData] = useState<InitialData>(initialData as InitialData);

  const { connected, messages, send } = useWebSocket();

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handlePopState = (): void => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

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
    fetchPageData(path).then(setData);
  };

  const renderPage = (): JSX.Element => {
    if (currentPath === "/" || currentPath.startsWith("/search")) {
      return (
        <Search
          initialResults={data.results ?? []}
          onSelectMemory={setSelectedMemory}
          wsConnected={connected}
        />
      );
    }
    if (currentPath === "/timeline") {
      return (
        <Timeline
          initialData={data.data}
          onSelectMemory={setSelectedMemory}
        />
      );
    }
    if (currentPath === "/agents") {
      return (
        <AgentView
          initialSessions={(data.sessions ?? []) as unknown[]}
          wsConnected={connected}
          onNavigate={navigate}
        />
      );
    }
    if (currentPath === "/settings") {
      return <Settings />;
    }
    return (
      <Search
        initialResults={[]}
        onSelectMemory={setSelectedMemory}
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
