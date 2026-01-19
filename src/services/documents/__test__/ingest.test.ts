import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { unlinkSync, writeFileSync } from "fs";
import {
  createDatabase,
  closeDatabase,
  setDatabase,
  getDatabase,
  type Database,
} from "../../../db/database.js";
import { createDocumentService, type DocumentService } from "../ingest.js";
import type { EmbeddingService, EmbeddingResult } from "../../embedding/types.js";

function createMockEmbeddingService(): EmbeddingService {
  const mockVector = Array(128).fill(0.1);

  return {
    getProvider: () => ({
      name: "mock",
      model: "test-model",
      dimensions: 128,
      embed: async () => mockVector,
      embedBatch: async () => [],
      isAvailable: async () => true,
    }),
    embed: async (): Promise<EmbeddingResult> => ({
      vector: mockVector,
      model: "test-model",
      dimensions: 128,
      cached: false,
    }),
    embedBatch: async (texts: string[]): Promise<EmbeddingResult[]> =>
      texts.map(() => ({
        vector: mockVector,
        model: "test-model",
        dimensions: 128,
        cached: false,
      })),
    getActiveModelId: () => "mock:test-model",
    switchProvider: async () => {},
  };
}

describe("Document Ingestion", () => {
  let db: Database;
  let docs: DocumentService;
  const tempFiles: string[] = [];

  beforeEach(async () => {
    db = await createDatabase(":memory:");
    setDatabase(db);
    docs = createDocumentService(createMockEmbeddingService());

    const now = Date.now();
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ["proj1", "/test/path", "Test Project", now, now]
    );

    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ["mock:test-model", "test-model", "mock", 128, 1, now]
    );
  });

  afterEach(() => {
    closeDatabase();
    for (const path of tempFiles) {
      try {
        unlinkSync(path);
      } catch {
        // Ignore errors
      }
    }
    tempFiles.length = 0;
  });

  function createTempFile(content: string, ext = "txt"): string {
    const path = `/tmp/test-doc-${Date.now()}-${Math.random()}.${ext}`;
    writeFileSync(path, content);
    tempFiles.push(path);
    return path;
  }

  test("ingests text file", async () => {
    const tempPath = createTempFile("This is test content for ingestion.");

    const doc = await docs.ingest({
      projectId: "proj1",
      path: tempPath,
    });

    expect(doc.sourceType).toBe("txt");
    expect(doc.fullContent).toContain("test content");
    expect(doc.checksum).toBeDefined();
    expect(doc.checksum.length).toBe(64);
  });

  test("ingests markdown with title extraction", async () => {
    const content = "# My Document\n\nSome content here.";
    const doc = await docs.ingest({
      projectId: "proj1",
      content,
      sourceType: "md",
    });

    expect(doc.title).toBe("My Document");
    expect(doc.sourceType).toBe("md");
  });

  test("ingests raw content", async () => {
    const content = "Raw content without a file.";
    const doc = await docs.ingest({
      projectId: "proj1",
      content,
    });

    expect(doc.fullContent).toBe(content);
    expect(doc.sourceType).toBe("txt");
    expect(doc.sourcePath).toBeUndefined();
  });

  test("chunks long documents", async () => {
    const longContent = "Paragraph of content. ".repeat(500);
    const doc = await docs.ingest({
      projectId: "proj1",
      content: longContent,
    });

    const database = await getDatabase();
    const chunks = await database.execute(
      "SELECT * FROM document_chunks WHERE document_id = ?",
      [doc.id]
    );
    expect(chunks.rows.length).toBeGreaterThan(1);
  });

  test("updates existing document by path", async () => {
    const tempPath = createTempFile("Original content");

    const doc1 = await docs.ingest({ projectId: "proj1", path: tempPath });

    await new Promise((resolve) => setTimeout(resolve, 10));
    writeFileSync(tempPath, "Updated content");
    const doc2 = await docs.ingest({ projectId: "proj1", path: tempPath });

    expect(doc2.id).toBe(doc1.id);
    expect(doc2.fullContent).toContain("Updated");
    expect(doc2.updatedAt).toBeGreaterThan(doc1.updatedAt);
  });

  test("skips unchanged documents", async () => {
    const tempPath = createTempFile("Same content");

    const doc1 = await docs.ingest({ projectId: "proj1", path: tempPath });
    const doc2 = await docs.ingest({ projectId: "proj1", path: tempPath });

    expect(doc2.updatedAt).toBe(doc1.updatedAt);
  });

  test("search finds relevant chunks", async () => {
    await docs.ingest({
      projectId: "proj1",
      content: "React is a JavaScript library for building user interfaces.",
    });

    const results = await docs.search("JavaScript UI framework", "proj1");
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]?.document.fullContent).toContain("React");
  });

  test("checkForUpdates detects changes", async () => {
    const tempPath = createTempFile("Original");

    await docs.ingest({ projectId: "proj1", path: tempPath });
    writeFileSync(tempPath, "Changed content now");

    const updated = await docs.checkForUpdates("proj1");
    expect(updated.length).toBe(1);
  });

  test("checkForUpdates detects missing files", async () => {
    const tempPath = createTempFile("Content");

    await docs.ingest({ projectId: "proj1", path: tempPath });
    unlinkSync(tempPath);
    const idx = tempFiles.indexOf(tempPath);
    if (idx !== -1) tempFiles.splice(idx, 1);

    const updated = await docs.checkForUpdates("proj1");
    expect(updated.length).toBe(1);
  });

  test("get returns null for non-existent document", async () => {
    const doc = await docs.get("non-existent-id");
    expect(doc).toBeNull();
  });

  test("list returns documents for project", async () => {
    await docs.ingest({ projectId: "proj1", content: "Doc 1" });
    await docs.ingest({ projectId: "proj1", content: "Doc 2" });

    const list = await docs.list("proj1");
    expect(list.length).toBe(2);
  });

  test("delete removes document and chunks", async () => {
    const doc = await docs.ingest({ projectId: "proj1", content: "To be deleted" });

    await docs.delete(doc.id);

    const fetched = await docs.get(doc.id);
    expect(fetched).toBeNull();

    const database = await getDatabase();
    const chunks = await database.execute(
      "SELECT * FROM document_chunks WHERE document_id = ?",
      [doc.id]
    );
    expect(chunks.rows.length).toBe(0);
  });

  test("getChunks returns all chunks for document", async () => {
    const longContent = "Sentence here. ".repeat(500);
    const doc = await docs.ingest({ projectId: "proj1", content: longContent });

    const chunks = await docs.getChunks(doc.id);
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0]?.documentId).toBe(doc.id);
    expect(chunks[0]?.chunkIndex).toBe(0);
  });

  test("uses custom title when provided", async () => {
    const doc = await docs.ingest({
      projectId: "proj1",
      content: "Content without header",
      title: "Custom Title",
    });

    expect(doc.title).toBe("Custom Title");
  });

  test("throws error when no content source provided", async () => {
    expect(
      docs.ingest({ projectId: "proj1" })
    ).rejects.toThrow("Must provide path, url, or content");
  });

  test("detects markdown file by extension", async () => {
    const tempPath = createTempFile("# Heading\n\nContent", "md");

    const doc = await docs.ingest({ projectId: "proj1", path: tempPath });

    expect(doc.sourceType).toBe("md");
    expect(doc.title).toBe("Heading");
  });

  test("extracts first line as title for txt without header", async () => {
    const doc = await docs.ingest({
      projectId: "proj1",
      content: "First line title\nSecond line content",
      sourceType: "txt",
    });

    expect(doc.title).toBe("First line title");
  });
});

describe("Document Search", () => {
  let db: Database;
  let docs: DocumentService;

  beforeEach(async () => {
    db = await createDatabase(":memory:");
    setDatabase(db);

    const queryVectors: Record<string, number[]> = {
      default: Array(128).fill(0.1),
    };

    const docVectors: number[][] = [];

    const embeddingService: EmbeddingService = {
      getProvider: () => ({
        name: "mock",
        model: "test-model",
        dimensions: 128,
        embed: async () => queryVectors["default"] ?? [],
        embedBatch: async () => [],
        isAvailable: async () => true,
      }),
      embed: async (): Promise<EmbeddingResult> => ({
        vector: queryVectors["default"] ?? [],
        model: "test-model",
        dimensions: 128,
        cached: false,
      }),
      embedBatch: async (texts: string[]): Promise<EmbeddingResult[]> =>
        texts.map((_, i) => {
          const vector = docVectors[i] ?? Array(128).fill(0.1 + i * 0.01);
          if (!docVectors[i]) docVectors.push(vector);
          return {
            vector,
            model: "test-model",
            dimensions: 128,
            cached: false,
          };
        }),
      getActiveModelId: () => "mock:test-model",
      switchProvider: async () => {},
    };

    docs = createDocumentService(embeddingService);

    const now = Date.now();
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ["proj1", "/test/path", "Test Project", now, now]
    );
    await db.execute(
      `INSERT INTO projects (id, path, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      ["proj2", "/test/path2", "Test Project 2", now, now]
    );
    await db.execute(
      `INSERT INTO embedding_models (id, name, provider, dimensions, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      ["mock:test-model", "test-model", "mock", 128, 1, now]
    );
  });

  afterEach(() => {
    closeDatabase();
  });

  test("filters search by project", async () => {
    await docs.ingest({ projectId: "proj1", content: "Content in project one" });
    await docs.ingest({ projectId: "proj2", content: "Content in project two" });

    const results = await docs.search("content", "proj1");
    expect(results.every((r) => r.document.projectId === "proj1")).toBe(true);
  });

  test("searches all projects when no filter", async () => {
    await docs.ingest({ projectId: "proj1", content: "Content A" });
    await docs.ingest({ projectId: "proj2", content: "Content B" });

    const results = await docs.search("content");
    expect(results.length).toBe(2);
  });

  test("respects limit parameter", async () => {
    await docs.ingest({ projectId: "proj1", content: "Doc 1 content" });
    await docs.ingest({ projectId: "proj1", content: "Doc 2 content" });
    await docs.ingest({ projectId: "proj1", content: "Doc 3 content" });

    const results = await docs.search("content", "proj1", 2);
    expect(results.length).toBeLessThanOrEqual(2);
  });

  test("returns empty array when no documents exist", async () => {
    const results = await docs.search("query", "proj1");
    expect(results).toEqual([]);
  });

  test("includes chunk information in results", async () => {
    await docs.ingest({ projectId: "proj1", content: "Some searchable content" });

    const results = await docs.search("searchable", "proj1");
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]?.chunk).toBeDefined();
    expect(results[0]?.chunk.content).toBeDefined();
  });
});
