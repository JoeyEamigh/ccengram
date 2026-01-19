# Embedding Service Specification

## Overview

The embedding service provides vector embeddings for text using Ollama as the primary provider with OpenRouter as a fallback for users without local GPU.

## Dependencies

```json
{
  // No external dependencies - uses native fetch
}
```

## Files to Create

- `src/services/embedding/index.ts` - Provider selection and interface
- `src/services/embedding/ollama.ts` - Ollama provider
- `src/services/embedding/openrouter.ts` - OpenRouter fallback
- `src/services/embedding/cache.ts` - Optional embedding cache

## Configuration

### User Config Schema

```typescript
// Part of $XDG_CONFIG_HOME/ccmemory/config.json
interface EmbeddingConfig {
  provider: "ollama" | "openrouter";
  ollama: {
    baseUrl: string;      // Default: "http://localhost:11434"
    model: string;        // Default: "qwen3-embedding"
  };
  openrouter: {
    apiKey?: string;      // From env OPENROUTER_API_KEY if not set
    model: string;        // Default: "openai/text-embedding-3-small"
  };
}
```

### Default Configuration

```typescript
const DEFAULT_CONFIG: EmbeddingConfig = {
  provider: "ollama",
  ollama: {
    baseUrl: "http://localhost:11434",
    model: "qwen3-embedding"  // 32k context, 4096 dimensions
  },
  openrouter: {
    model: "openai/text-embedding-3-small"  // 1536 dimensions
  }
};
```

## Embedding Provider Interface

### Interface

```typescript
// src/services/embedding/index.ts
export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;

  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  isAvailable(): Promise<boolean>;
}

export interface EmbeddingService {
  getProvider(): EmbeddingProvider;
  embed(text: string): Promise<EmbeddingResult>;
  embedBatch(texts: string[]): Promise<EmbeddingResult[]>;
  getActiveModelId(): string;
  switchProvider(provider: "ollama" | "openrouter"): Promise<void>;
}

export interface EmbeddingResult {
  vector: number[];
  model: string;
  dimensions: number;
  cached: boolean;
}

export function createEmbeddingService(config?: EmbeddingConfig): Promise<EmbeddingService>;
```

### Implementation Notes

```typescript
import { log } from "../../utils/log.js";

export async function createEmbeddingService(
  config: EmbeddingConfig = DEFAULT_CONFIG
): Promise<EmbeddingService> {
  const providers: Record<string, EmbeddingProvider> = {
    ollama: new OllamaProvider(config.ollama),
    openrouter: new OpenRouterProvider(config.openrouter)
  };

  // Try preferred provider, fall back if unavailable
  let active = providers[config.provider];
  log.debug("embedding", "Checking provider availability", { provider: config.provider });

  if (!(await active.isAvailable())) {
    const fallback = config.provider === "ollama" ? "openrouter" : "ollama";
    log.warn("embedding", "Primary provider unavailable, trying fallback", {
      primary: config.provider,
      fallback
    });

    if (await providers[fallback].isAvailable()) {
      active = providers[fallback];
      log.info("embedding", "Using fallback provider", { provider: fallback });
    } else {
      log.error("embedding", "No embedding provider available");
      throw new Error("No embedding provider available");
    }
  }

  // Register model in database
  await registerModel(active);
  log.info("embedding", "Embedding service initialized", {
    provider: active.name,
    model: active.model,
    dimensions: active.dimensions
  });

  return {
    getProvider: () => active,
    embed: async (text) => ({
      vector: await active.embed(text),
      model: active.model,
      dimensions: active.dimensions,
      cached: false
    }),
    embedBatch: async (texts) => {
      const vectors = await active.embedBatch(texts);
      return vectors.map(v => ({
        vector: v,
        model: active.model,
        dimensions: active.dimensions,
        cached: false
      }));
    },
    getActiveModelId: () => `${active.name}:${active.model}`,
    switchProvider: async (provider) => {
      if (!(await providers[provider].isAvailable())) {
        throw new Error(`Provider ${provider} not available`);
      }
      active = providers[provider];
      await registerModel(active);
    }
  };
}

async function registerModel(provider: EmbeddingProvider): Promise<void> {
  const db = getDatabase();
  const modelId = `${provider.name}:${provider.model}`;

  // Upsert model and set as active
  await db.batch([
    {
      sql: `INSERT INTO embedding_models (id, name, provider, dimensions, is_active)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT (id) DO UPDATE SET is_active = 0`,
      args: [modelId, provider.model, provider.name, provider.dimensions]
    },
    {
      sql: `UPDATE embedding_models SET is_active = 0`,
      args: []
    },
    {
      sql: `UPDATE embedding_models SET is_active = 1 WHERE id = ?`,
      args: [modelId]
    }
  ]);
}
```

## Ollama Provider

### Interface

```typescript
// src/services/embedding/ollama.ts
export class OllamaProvider implements EmbeddingProvider {
  readonly name = "ollama";
  readonly model: string;
  readonly dimensions: number;

  constructor(config: OllamaConfig);
  async embed(text: string): Promise<number[]>;
  async embedBatch(texts: string[]): Promise<number[][]>;
  async isAvailable(): Promise<boolean>;
}
```

### Implementation Notes

```typescript
import { log } from "../../utils/log.js";

export class OllamaProvider implements EmbeddingProvider {
  readonly name = "ollama";
  private baseUrl: string;
  readonly model: string;
  private _dimensions: number | null = null;

  constructor(config: OllamaConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.model = config.model;
  }

  get dimensions(): number {
    if (!this._dimensions) {
      throw new Error("Call isAvailable() first to detect dimensions");
    }
    return this._dimensions;
  }

  async isAvailable(): Promise<boolean> {
    try {
      log.debug("embedding", "Checking Ollama availability", { url: this.baseUrl });

      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) {
        log.debug("embedding", "Ollama not responding", { status: response.status });
        return false;
      }

      const data = await response.json();
      const models = data.models || [];

      const hasModel = models.some((m: any) =>
        m.name === this.model || m.name.startsWith(`${this.model}:`)
      );

      if (!hasModel) {
        log.warn("embedding", "Model not found in Ollama", {
          model: this.model,
          available: models.map((m: any) => m.name)
        });
        return false;
      }

      // Detect dimensions by embedding a test string
      const testVec = await this.embed("dimension test");
      this._dimensions = testVec.length;

      log.info("embedding", "Ollama provider ready", {
        model: this.model,
        dimensions: this._dimensions
      });
      return true;
    } catch (e) {
      log.debug("embedding", "Ollama check failed", { error: (e as Error).message });
      return false;
    }
  }

  async embed(text: string): Promise<number[]> {
    const start = Date.now();
    const response = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        prompt: text
      })
    });

    if (!response.ok) {
      log.error("embedding", "Ollama embed failed", { status: response.statusText });
      throw new Error(`Ollama embed failed: ${response.statusText}`);
    }

    const data = await response.json();
    log.debug("embedding", "Embedded text", { length: text.length, ms: Date.now() - start });
    return data.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const start = Date.now();
    log.debug("embedding", "Batch embedding", { count: texts.length });
    const results = await Promise.all(texts.map(t => this.embed(t)));
    log.info("embedding", "Batch complete", { count: texts.length, ms: Date.now() - start });
    return results;
  }
}
```

### Test Specification

```typescript
// src/services/embedding/__test__/ollama.test.ts (colocated unit test)
import { describe, test, expect, mock, beforeEach } from "bun:test";

describe("OllamaProvider", () => {
  let provider: OllamaProvider;

  beforeEach(() => {
    provider = new OllamaProvider({
      baseUrl: "http://localhost:11434",
      model: "qwen3-embedding"
    });
  });

  test("checks availability correctly", async () => {
    // Mock fetch to return available models
    global.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        models: [{ name: "qwen3-embedding:latest" }]
      })
    }));

    // Also mock the dimension detection call
    // ... setup mock for embeddings endpoint

    const available = await provider.isAvailable();
    expect(available).toBe(true);
  });

  test("returns false when Ollama is not running", async () => {
    global.fetch = mock(() => Promise.reject(new Error("ECONNREFUSED")));

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("returns false when model is not installed", async () => {
    global.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        models: [{ name: "llama3:latest" }]  // Different model
      })
    }));

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("embed returns vector of correct dimensions", async () => {
    // Mock successful embedding
    const mockVector = new Array(4096).fill(0).map(() => Math.random());
    global.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ embedding: mockVector })
    }));

    const vector = await provider.embed("test text");
    expect(vector.length).toBe(4096);
    expect(Array.isArray(vector)).toBe(true);
  });

  test("embedBatch processes multiple texts", async () => {
    const mockVector = new Array(4096).fill(0).map(() => Math.random());
    global.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ embedding: mockVector })
    }));

    const texts = ["text 1", "text 2", "text 3"];
    const vectors = await provider.embedBatch(texts);

    expect(vectors.length).toBe(3);
    expect(vectors.every(v => v.length === 4096)).toBe(true);
  });
});
```

## OpenRouter Provider

### Interface

```typescript
// src/services/embedding/openrouter.ts
export class OpenRouterProvider implements EmbeddingProvider {
  readonly name = "openrouter";
  readonly model: string;
  readonly dimensions: number;

  constructor(config: OpenRouterConfig);
  async embed(text: string): Promise<number[]>;
  async embedBatch(texts: string[]): Promise<number[][]>;
  async isAvailable(): Promise<boolean>;
}
```

### Implementation Notes

```typescript
import { log } from "../../utils/log.js";

const MODEL_DIMENSIONS: Record<string, number> = {
  "openai/text-embedding-3-small": 1536,
  "openai/text-embedding-3-large": 3072,
  "openai/text-embedding-ada-002": 1536
};

export class OpenRouterProvider implements EmbeddingProvider {
  readonly name = "openrouter";
  private apiKey: string;
  readonly model: string;
  readonly dimensions: number;

  constructor(config: OpenRouterConfig) {
    this.apiKey = config.apiKey || process.env.OPENROUTER_API_KEY || "";
    this.model = config.model;
    this.dimensions = MODEL_DIMENSIONS[this.model] || 1536;
  }

  async isAvailable(): Promise<boolean> {
    if (!this.apiKey) {
      log.warn("embedding", "OpenRouter API key not configured");
      return false;
    }

    try {
      log.debug("embedding", "Checking OpenRouter availability");
      const response = await fetch("https://openrouter.ai/api/v1/models", {
        headers: { "Authorization": `Bearer ${this.apiKey}` }
      });
      if (response.ok) {
        log.info("embedding", "OpenRouter provider ready", { model: this.model });
      }
      return response.ok;
    } catch (e) {
      log.debug("embedding", "OpenRouter check failed", { error: (e as Error).message });
      return false;
    }
  }

  async embed(text: string): Promise<number[]> {
    const start = Date.now();
    const response = await fetch("https://openrouter.ai/api/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`,
        "HTTP-Referer": "https://github.com/user/ccmemory",
        "X-Title": "CCMemory"
      },
      body: JSON.stringify({
        model: this.model,
        input: text
      })
    });

    if (!response.ok) {
      log.error("embedding", "OpenRouter embed failed", { status: response.statusText });
      throw new Error(`OpenRouter embed failed: ${response.statusText}`);
    }

    const data = await response.json();
    log.debug("embedding", "OpenRouter embedded", { length: text.length, ms: Date.now() - start });
    return data.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const start = Date.now();
    log.debug("embedding", "OpenRouter batch embedding", { count: texts.length });
    const response = await fetch("https://openrouter.ai/api/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`,
        "HTTP-Referer": "https://github.com/user/ccmemory",
        "X-Title": "CCMemory"
      },
      body: JSON.stringify({
        model: this.model,
        input: texts
      })
    });

    if (!response.ok) {
      throw new Error(`OpenRouter embed batch failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.data.map((d: any) => d.embedding);
  }
}
```

### Test Specification

```typescript
// src/services/embedding/__test__/openrouter.test.ts (colocated unit test)
describe("OpenRouterProvider", () => {
  test("requires API key", async () => {
    delete process.env.OPENROUTER_API_KEY;
    const provider = new OpenRouterProvider({ model: "openai/text-embedding-3-small" });

    const available = await provider.isAvailable();
    expect(available).toBe(false);
  });

  test("uses correct dimensions for known models", () => {
    const provider = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-3-large"
    });
    expect(provider.dimensions).toBe(3072);
  });

  test("embedBatch returns correct structure", async () => {
    const mockEmbeddings = [
      { embedding: new Array(1536).fill(0.1) },
      { embedding: new Array(1536).fill(0.2) }
    ];

    global.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ data: mockEmbeddings })
    }));

    const provider = new OpenRouterProvider({
      apiKey: "test",
      model: "openai/text-embedding-3-small"
    });

    const vectors = await provider.embedBatch(["text1", "text2"]);
    expect(vectors.length).toBe(2);
    expect(vectors[0].length).toBe(1536);
  });
});
```

## Model Switching and Re-embedding

### Handling Different Dimensions

When the embedding model changes, existing vectors become incompatible. Strategy:

1. **Lazy re-embedding**: Mark old vectors as stale, re-embed on access
2. **Track model ID**: Store `embedding_model_id` with each vector
3. **Mixed results**: Search only vectors from current model

```typescript
// src/services/embedding/reembed.ts
export async function reembedStaleMemories(
  service: EmbeddingService,
  batchSize: number = 50
): Promise<number> {
  const db = getDatabase();
  const currentModel = service.getActiveModelId();

  // Find memories with outdated embeddings
  const stale = await db.execute(`
    SELECT m.id, m.content
    FROM memories m
    LEFT JOIN memory_vectors mv ON m.id = mv.memory_id
    WHERE mv.model_id != ? OR mv.model_id IS NULL
    LIMIT ?
  `, [currentModel, batchSize]);

  if (stale.rows.length === 0) return 0;

  // Re-embed in batch
  const texts = stale.rows.map(r => r[1] as string);
  const results = await service.embedBatch(texts);

  // Update vectors
  const statements = stale.rows.map((row, i) => ({
    sql: `INSERT OR REPLACE INTO memory_vectors (memory_id, model_id, vector, dim)
          VALUES (?, ?, vector(?), ?)`,
    args: [row[0], currentModel, JSON.stringify(results[i].vector), results[i].dimensions]
  }));

  await db.batch(statements);
  return stale.rows.length;
}
```

## Embedding Service Test Specification

```typescript
// src/services/embedding/__test__/service.test.ts (colocated unit test)
describe("EmbeddingService", () => {
  test("falls back to OpenRouter when Ollama unavailable", async () => {
    // Mock Ollama as unavailable, OpenRouter as available
    const service = await createEmbeddingService({
      provider: "ollama",
      ollama: { baseUrl: "http://localhost:11434", model: "qwen3-embedding" },
      openrouter: { apiKey: "test", model: "openai/text-embedding-3-small" }
    });

    expect(service.getProvider().name).toBe("openrouter");
  });

  test("registers model in database", async () => {
    const service = await createEmbeddingService();
    const db = getDatabase();

    const active = await db.execute(
      "SELECT * FROM embedding_models WHERE is_active = 1"
    );
    expect(active.rows.length).toBe(1);
  });

  test("switchProvider updates active model", async () => {
    const service = await createEmbeddingService();
    await service.switchProvider("openrouter");

    expect(service.getProvider().name).toBe("openrouter");
  });

  test("embed returns consistent structure", async () => {
    const service = await createEmbeddingService();
    const result = await service.embed("test text");

    expect(result).toHaveProperty("vector");
    expect(result).toHaveProperty("model");
    expect(result).toHaveProperty("dimensions");
    expect(result.vector.length).toBe(result.dimensions);
  });
});
```

## Acceptance Criteria

- [ ] Ollama provider connects and embeds text
- [ ] OpenRouter provider works with API key
- [ ] Service falls back when primary unavailable
- [ ] Model dimensions detected automatically
- [ ] Models registered in database
- [ ] Batch embedding works efficiently
- [ ] Model switching triggers re-embedding
- [ ] Config persists in XDG config directory
- [ ] Error messages are clear and actionable
