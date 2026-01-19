import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { createDatabase, closeDatabase, setDatabase, type Database } from "../../../db/database.js";
import { createEmbeddingService, createEmbeddingServiceOptional } from "../index.js";
import type { EmbeddingConfig } from "../types.js";

function mockFetch(fn: (url: string | URL | Request) => Promise<Response>): void {
  globalThis.fetch = mock(fn) as unknown as typeof fetch;
}

describe("EmbeddingService", () => {
  const originalFetch = globalThis.fetch;
  let db: Database;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
    setDatabase(db);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    closeDatabase();
  });

  function mockOllamaAvailable(embedding: number[] = new Array(4096).fill(0.1)) {
    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("localhost:11434/api/tags")) {
        return new Response(
          JSON.stringify({ models: [{ name: "qwen3-embedding:latest" }] }),
          { status: 200 }
        );
      }

      if (urlStr.includes("localhost:11434/api/embeddings")) {
        return new Response(JSON.stringify({ embedding }), { status: 200 });
      }

      return new Response("Not found", { status: 404 });
    });
  }

  function mockOllamaUnavailable() {
    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("localhost:11434")) {
        throw new Error("ECONNREFUSED");
      }

      if (urlStr.includes("openrouter.ai/api/v1/models")) {
        return new Response(JSON.stringify({ data: [] }), { status: 200 });
      }

      if (urlStr.includes("openrouter.ai/api/v1/embeddings")) {
        return new Response(
          JSON.stringify({
            data: [{ embedding: new Array(1536).fill(0.2), index: 0 }],
            model: "openai/text-embedding-3-small",
            usage: { prompt_tokens: 5, total_tokens: 5 },
          }),
          { status: 200 }
        );
      }

      return new Response("Not found", { status: 404 });
    });
  }

  test("initializes with Ollama when available", async () => {
    mockOllamaAvailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingService(config);
    expect(service.getProvider().name).toBe("ollama");
    expect(service.getActiveModelId()).toBe("ollama:qwen3-embedding");
  });

  test("falls back to OpenRouter when Ollama unavailable", async () => {
    mockOllamaUnavailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingService(config);
    expect(service.getProvider().name).toBe("openrouter");
  });

  test("throws when no providers available", async () => {
    mockFetch(async () => {
      throw new Error("Network error");
    });

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { model: "openai/text-embedding-3-small" },
    };

    await expect(createEmbeddingService(config)).rejects.toThrow(
      "No embedding provider available"
    );
  });

  test("registers model in database", async () => {
    mockOllamaAvailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    await createEmbeddingService(config);

    const result = await db.execute(
      "SELECT * FROM embedding_models WHERE is_active = 1"
    );
    expect(result.rows.length).toBe(1);
    expect(result.rows[0]?.["id"]).toBe("ollama:qwen3-embedding");
    expect(result.rows[0]?.["dimensions"]).toBe(4096);
  });

  test("embed returns consistent structure", async () => {
    mockOllamaAvailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingService(config);
    const result = await service.embed("test text");

    expect(result).toHaveProperty("vector");
    expect(result).toHaveProperty("model");
    expect(result).toHaveProperty("dimensions");
    expect(result).toHaveProperty("cached");
    expect(result.vector.length).toBe(result.dimensions);
    expect(result.model).toBe("qwen3-embedding");
  });

  test("embedBatch processes multiple texts", async () => {
    mockOllamaAvailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingService(config);
    const results = await service.embedBatch(["text1", "text2", "text3"]);

    expect(results.length).toBe(3);
    expect(results.every((r) => r.vector.length === 4096)).toBe(true);
  });

  test("switchProvider changes active provider", async () => {
    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("localhost:11434/api/tags")) {
        return new Response(
          JSON.stringify({ models: [{ name: "qwen3-embedding:latest" }] }),
          { status: 200 }
        );
      }

      if (urlStr.includes("localhost:11434/api/embeddings")) {
        return new Response(
          JSON.stringify({ embedding: new Array(4096).fill(0.1) }),
          { status: 200 }
        );
      }

      if (urlStr.includes("openrouter.ai/api/v1/models")) {
        return new Response(JSON.stringify({ data: [] }), { status: 200 });
      }

      if (urlStr.includes("openrouter.ai/api/v1/embeddings")) {
        return new Response(
          JSON.stringify({
            data: [{ embedding: new Array(1536).fill(0.2), index: 0 }],
            model: "openai/text-embedding-3-small",
            usage: { prompt_tokens: 5, total_tokens: 5 },
          }),
          { status: 200 }
        );
      }

      return new Response("Not found", { status: 404 });
    });

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingService(config);
    expect(service.getProvider().name).toBe("ollama");

    await service.switchProvider("openrouter");
    expect(service.getProvider().name).toBe("openrouter");

    const result = await db.execute(
      "SELECT * FROM embedding_models WHERE is_active = 1"
    );
    expect(result.rows[0]?.["id"]).toBe("openrouter:openai/text-embedding-3-small");
  });

  test("switchProvider throws if provider unavailable", async () => {
    mockOllamaAvailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingService(config);

    await expect(service.switchProvider("openrouter")).rejects.toThrow(
      "Provider openrouter not available"
    );
  });

  test("createEmbeddingServiceOptional returns null when no providers available", async () => {
    mockFetch(async () => {
      throw new Error("Network error");
    });

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingServiceOptional(config);
    expect(service).toBeNull();
  });

  test("createEmbeddingServiceOptional returns service when provider available", async () => {
    mockOllamaAvailable();

    const config: EmbeddingConfig = {
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test-key", model: "openai/text-embedding-3-small" },
    };

    const service = await createEmbeddingServiceOptional(config);
    expect(service).not.toBeNull();
    expect(service?.getProvider().name).toBe("ollama");
  });
});
