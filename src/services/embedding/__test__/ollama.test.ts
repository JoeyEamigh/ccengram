import { afterEach, describe, expect, mock, test } from "bun:test";
import { OllamaProvider } from "../ollama.js";

function mockFetch(fn: (url: string | URL | Request) => Promise<Response>): void {
  globalThis.fetch = mock(fn) as unknown as typeof fetch;
}

describe("OllamaProvider", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("checks availability when model exists", async () => {
    const mockEmbedding = new Array(4096).fill(0).map(() => Math.random());

    let callCount = 0;
    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();
      callCount++;

      if (urlStr.includes("/api/tags")) {
        return new Response(
          JSON.stringify({
            models: [{ name: "qwen3-embedding:latest" }],
          }),
          { status: 200 }
        );
      }

      if (urlStr.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: mockEmbedding }), { status: 200 });
      }

      return new Response("Not found", { status: 404 });
    });

    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(true);
    expect(provider.dimensions).toBe(4096);
    expect(callCount).toBe(2);
  });

  test("returns false when Ollama is not running", async () => {
    mockFetch(async () => {
      throw new Error("ECONNREFUSED");
    });

    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("returns false when model is not installed", async () => {
    mockFetch(async () => {
      return new Response(
        JSON.stringify({
          models: [{ name: "llama3:latest" }],
        }),
        { status: 200 }
      );
    });

    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("embed returns vector of correct dimensions", async () => {
    const mockEmbedding = new Array(4096).fill(0).map(() => Math.random());

    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("/api/tags")) {
        return new Response(
          JSON.stringify({ models: [{ name: "qwen3-embedding:latest" }] }),
          { status: 200 }
        );
      }

      return new Response(JSON.stringify({ embedding: mockEmbedding }), { status: 200 });
    });

    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    await provider.isAvailable();

    const vector = await provider.embed("test text");
    expect(vector.length).toBe(4096);
    expect(Array.isArray(vector)).toBe(true);
  });

  test("embedBatch processes multiple texts", async () => {
    const mockEmbedding = new Array(4096).fill(0).map(() => Math.random());

    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("/api/tags")) {
        return new Response(
          JSON.stringify({ models: [{ name: "qwen3-embedding:latest" }] }),
          { status: 200 }
        );
      }

      return new Response(JSON.stringify({ embedding: mockEmbedding }), { status: 200 });
    });

    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    await provider.isAvailable();

    const texts = ["text 1", "text 2", "text 3"];
    const vectors = await provider.embedBatch(texts);

    expect(vectors.length).toBe(3);
    expect(vectors.every((v) => v.length === 4096)).toBe(true);
  });

  test("throws error when dimensions not detected", () => {
    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    expect(() => provider.dimensions).toThrow("Dimensions not yet detected");
  });

  test("handles Ollama API errors", async () => {
    mockFetch(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("/api/tags")) {
        return new Response(
          JSON.stringify({ models: [{ name: "qwen3-embedding:latest" }] }),
          { status: 200 }
        );
      }

      return new Response("Internal Server Error", { status: 500 });
    });

    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("normalizes base URL by removing trailing slash", async () => {
    const provider = new OllamaProvider({
      baseUrl: "http://localhost:11434/",
      model: "qwen3-embedding",
    });

    let calledUrl = "";
    mockFetch(async (url: string | URL | Request) => {
      calledUrl = typeof url === "string" ? url : url.toString();
      return new Response(JSON.stringify({ models: [] }), { status: 200 });
    });

    await provider.isAvailable();
    expect(calledUrl).toBe("http://localhost:11434/api/tags");
  });
});
