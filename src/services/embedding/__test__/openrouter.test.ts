import { afterEach, describe, expect, mock, test } from "bun:test";
import { OpenRouterProvider } from "../openrouter.js";

function mockFetch(fn: (url: string | URL | Request) => Promise<Response>): void {
  globalThis.fetch = mock(fn) as unknown as typeof fetch;
}

describe("OpenRouterProvider", () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };

  afterEach(() => {
    globalThis.fetch = originalFetch;
    process.env = { ...originalEnv };
  });

  test("requires API key", async () => {
    delete process.env["OPENROUTER_API_KEY"];
    const provider = new OpenRouterProvider({
      model: "openai/text-embedding-3-small",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("uses API key from config", async () => {
    delete process.env["OPENROUTER_API_KEY"];

    mockFetch(async () => {
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    });

    const provider = new OpenRouterProvider({
      apiKey: "test-key",
      model: "openai/text-embedding-3-small",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(true);
  });

  test("uses API key from environment", async () => {
    process.env["OPENROUTER_API_KEY"] = "env-test-key";

    mockFetch(async () => {
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    });

    const provider = new OpenRouterProvider({
      model: "openai/text-embedding-3-small",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(true);
  });

  test("uses correct dimensions for known models", () => {
    const provider3Small = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-3-small",
    });
    expect(provider3Small.dimensions).toBe(1536);

    const provider3Large = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-3-large",
    });
    expect(provider3Large.dimensions).toBe(3072);

    const providerAda = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-ada-002",
    });
    expect(providerAda.dimensions).toBe(1536);
  });

  test("defaults to 1536 dimensions for unknown models", () => {
    const provider = new OpenRouterProvider({
      apiKey: "test",
      model: "unknown/model",
    });
    expect(provider.dimensions).toBe(1536);
  });

  test("embed returns vector", async () => {
    const mockEmbedding = new Array(1536).fill(0).map(() => Math.random());

    mockFetch(async () => {
      return new Response(
        JSON.stringify({
          data: [{ embedding: mockEmbedding, index: 0 }],
          model: "openai/text-embedding-3-small",
          usage: { prompt_tokens: 5, total_tokens: 5 },
        }),
        { status: 200 }
      );
    });

    const provider = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-3-small",
    });

    const vector = await provider.embed("test text");
    expect(vector.length).toBe(1536);
    expect(Array.isArray(vector)).toBe(true);
  });

  test("embedBatch returns correct structure", async () => {
    const mockEmbeddings = [
      { embedding: new Array(1536).fill(0.1), index: 0 },
      { embedding: new Array(1536).fill(0.2), index: 1 },
    ];

    mockFetch(async () => {
      return new Response(
        JSON.stringify({
          data: mockEmbeddings,
          model: "openai/text-embedding-3-small",
          usage: { prompt_tokens: 10, total_tokens: 10 },
        }),
        { status: 200 }
      );
    });

    const provider = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-3-small",
    });

    const vectors = await provider.embedBatch(["text1", "text2"]);
    expect(vectors.length).toBe(2);
    expect(vectors[0]?.length).toBe(1536);
    expect(vectors[1]?.length).toBe(1536);
  });

  test("handles API errors", async () => {
    mockFetch(async () => {
      return new Response("Unauthorized", { status: 401 });
    });

    const provider = new OpenRouterProvider({
      apiKey: "invalid-key",
      model: "openai/text-embedding-3-small",
    });

    await expect(provider.embed("test")).rejects.toThrow("OpenRouter embed failed");
  });

  test("handles network errors during availability check", async () => {
    mockFetch(async () => {
      throw new Error("Network error");
    });

    const provider = new OpenRouterProvider({
      apiKey: "test-key",
      model: "openai/text-embedding-3-small",
    });

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });
});
