# Documents Specification

## Overview

Documents are external txt/md files ingested for reference. They are stored separately from memories and chunked for efficient vector search.

## Files to Create

- `src/services/documents/ingest.ts` - Document ingestion
- `src/services/documents/chunk.ts` - Text chunking with overlap
- `src/services/documents/store.ts` - Document CRUD

## Document Interface

```typescript
// src/services/documents/types.ts
export interface Document {
  id: string;
  projectId: string;
  sourcePath?: string;
  sourceUrl?: string;
  sourceType: "txt" | "md" | "url";
  title?: string;
  fullContent: string;
  checksum: string;
  createdAt: number;
  updatedAt: number;
}

export interface DocumentChunk {
  id: string;
  documentId: string;
  chunkIndex: number;
  content: string;
  startOffset: number;
  endOffset: number;
  tokensEstimate: number;
}

export interface IngestOptions {
  projectId: string;
  path?: string;
  url?: string;
  content?: string;
  title?: string;
  sourceType?: "txt" | "md" | "url";
}
```

## Chunking Strategy

### Interface

```typescript
// src/services/documents/chunk.ts
export interface ChunkOptions {
  targetTokens?: number;  // Target chunk size (default: 768)
  overlap?: number;       // Overlap ratio (default: 0.1)
  minChunkSize?: number;  // Minimum chunk size (default: 100)
}

export interface Chunk {
  content: string;
  startOffset: number;
  endOffset: number;
  tokensEstimate: number;
}

export function chunkText(text: string, options?: ChunkOptions): Chunk[];
```

### Implementation Notes

```typescript
const CHARS_PER_TOKEN = 4;  // Rough estimate

export function chunkText(text: string, options: ChunkOptions = {}): Chunk[] {
  const {
    targetTokens = 768,
    overlap = 0.1,
    minChunkSize = 100
  } = options;

  const totalTokens = Math.ceil(text.length / CHARS_PER_TOKEN);

  // If small enough, return as single chunk
  if (totalTokens <= targetTokens) {
    return [{
      content: text,
      startOffset: 0,
      endOffset: text.length,
      tokensEstimate: totalTokens
    }];
  }

  const targetChars = targetTokens * CHARS_PER_TOKEN;
  const overlapChars = Math.floor(targetChars * overlap);

  // Split by paragraphs first
  const paragraphs = text.split(/\n\n+/);
  const chunks: Chunk[] = [];

  let currentChunk = "";
  let chunkStart = 0;
  let currentPos = 0;

  for (const para of paragraphs) {
    // Split paragraph into sentences
    const sentences = para.split(/(?<=[.!?])\s+/);

    for (const sentence of sentences) {
      const potentialChunk = currentChunk + (currentChunk ? " " : "") + sentence;

      if (potentialChunk.length > targetChars && currentChunk.length >= minChunkSize) {
        // Save current chunk
        chunks.push({
          content: currentChunk,
          startOffset: chunkStart,
          endOffset: chunkStart + currentChunk.length,
          tokensEstimate: Math.ceil(currentChunk.length / CHARS_PER_TOKEN)
        });

        // Start new chunk with overlap
        const overlapText = currentChunk.slice(-overlapChars);
        currentChunk = overlapText + " " + sentence;
        chunkStart = currentPos - overlapChars;
      } else {
        currentChunk = potentialChunk;
      }

      currentPos += sentence.length + 1;
    }

    currentPos++;  // For paragraph break
  }

  // Add final chunk
  if (currentChunk.length > 0) {
    chunks.push({
      content: currentChunk,
      startOffset: chunkStart,
      endOffset: chunkStart + currentChunk.length,
      tokensEstimate: Math.ceil(currentChunk.length / CHARS_PER_TOKEN)
    });
  }

  return chunks;
}
```

### Test Specification

```typescript
// src/services/documents/__test__/chunk.test.ts (colocated unit test)
describe("Text Chunking", () => {
  test("returns single chunk for small text", () => {
    const text = "Short text that fits in one chunk.";
    const chunks = chunkText(text);
    expect(chunks.length).toBe(1);
    expect(chunks[0].content).toBe(text);
  });

  test("splits long text into multiple chunks", () => {
    const text = "A".repeat(5000);  // ~1250 tokens
    const chunks = chunkText(text);
    expect(chunks.length).toBeGreaterThan(1);
  });

  test("chunks have overlap", () => {
    const text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.".repeat(50);
    const chunks = chunkText(text, { targetTokens: 100, overlap: 0.2 });

    // Check adjacent chunks share some content
    for (let i = 0; i < chunks.length - 1; i++) {
      const end = chunks[i].content.slice(-50);
      const start = chunks[i + 1].content.slice(0, 100);
      // Some overlap should exist
      expect(start.includes(end.slice(-20)) || end.includes(start.slice(0, 20))).toBe(true);
    }
  });

  test("respects minimum chunk size", () => {
    const text = "Word. ".repeat(100);
    const chunks = chunkText(text, { minChunkSize: 50 });
    expect(chunks.every(c => c.content.length >= 50)).toBe(true);
  });

  test("tracks offsets correctly", () => {
    const text = "First part. Second part. Third part.";
    const chunks = chunkText(text, { targetTokens: 5 });

    // Verify offsets are valid
    for (const chunk of chunks) {
      expect(chunk.startOffset).toBeGreaterThanOrEqual(0);
      expect(chunk.endOffset).toBeLessThanOrEqual(text.length);
      expect(chunk.endOffset).toBeGreaterThan(chunk.startOffset);
    }
  });

  test("estimates tokens correctly", () => {
    const text = "Word ".repeat(100);  // ~100 tokens
    const chunks = chunkText(text);
    const totalTokens = chunks.reduce((sum, c) => sum + c.tokensEstimate, 0);
    expect(totalTokens).toBeGreaterThanOrEqual(80);
    expect(totalTokens).toBeLessThanOrEqual(150);
  });
});
```

## Document Ingestion

### Interface

```typescript
// src/services/documents/ingest.ts
export interface DocumentService {
  ingest(options: IngestOptions): Promise<Document>;
  get(id: string): Promise<Document | null>;
  getByPath(path: string, projectId: string): Promise<Document | null>;
  delete(id: string): Promise<void>;
  list(projectId: string): Promise<Document[]>;
  search(query: string, projectId?: string, limit?: number): Promise<DocumentSearchResult[]>;
  checkForUpdates(projectId: string): Promise<string[]>;  // Returns IDs of updated docs
}

export interface DocumentSearchResult {
  document: Document;
  chunk: DocumentChunk;
  score: number;
  matchType: "semantic" | "keyword" | "both";
}

export function createDocumentService(): DocumentService;
```

### Implementation Notes

```typescript
import { createHash } from "crypto";
import { log } from "../../utils/log.js";

export function createDocumentService(): DocumentService {
  const db = getDatabase();
  const embedding = getEmbeddingService();

  return {
    async ingest(options: IngestOptions): Promise<Document> {
      const { projectId, path, url, content, title, sourceType } = options;
      const start = Date.now();

      log.info("docs", "Ingesting document", { projectId, path, url: url?.slice(0, 50) });

      // Get content
      let fullContent: string;
      let finalSourceType: "txt" | "md" | "url";

      if (content) {
        fullContent = content;
        finalSourceType = sourceType || "txt";
      } else if (path) {
        fullContent = await Bun.file(path).text();
        finalSourceType = path.endsWith(".md") ? "md" : "txt";
        log.debug("docs", "Read file", { path, bytes: fullContent.length });
      } else if (url) {
        const response = await fetch(url);
        fullContent = await response.text();
        finalSourceType = "url";
        log.debug("docs", "Fetched URL", { url, bytes: fullContent.length });
      } else {
        log.error("docs", "No content source provided");
        throw new Error("Must provide path, url, or content");
      }

      // Compute checksum
      const checksum = createHash("sha256").update(fullContent).digest("hex");

      // Check if document exists and hasn't changed
      const existing = path
        ? await this.getByPath(path, projectId)
        : null;

      if (existing?.checksum === checksum) {
        log.debug("docs", "Document unchanged, skipping", { id: existing.id });
        return existing;
      }

      const id = existing?.id || crypto.randomUUID();
      const now = Date.now();

      // Delete existing chunks if updating
      if (existing) {
        await db.execute("DELETE FROM document_chunks WHERE document_id = ?", [id]);
        await db.execute("DELETE FROM document_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)", [id]);
      }

      // Store document
      await db.execute(`
        INSERT OR REPLACE INTO documents (id, project_id, source_path, source_url, source_type, title, full_content, checksum, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `, [
        id, projectId, path || null, url || null, finalSourceType,
        title || extractTitle(fullContent, finalSourceType),
        fullContent, checksum,
        existing?.createdAt || now, now
      ]);

      // Chunk and embed
      const chunks = chunkText(fullContent);
      const embeddings = await embedding.embedBatch(chunks.map(c => c.content));

      for (let i = 0; i < chunks.length; i++) {
        const chunkId = `${id}-${i}`;
        const chunk = chunks[i];

        await db.execute(`
          INSERT INTO document_chunks (id, document_id, chunk_index, content, start_offset, end_offset, tokens_estimate)
          VALUES (?, ?, ?, ?, ?, ?, ?)
        `, [chunkId, id, i, chunk.content, chunk.startOffset, chunk.endOffset, chunk.tokensEstimate]);

        await db.execute(`
          INSERT INTO document_vectors (chunk_id, model_id, vector, dim)
          VALUES (?, ?, vector(?), ?)
        `, [chunkId, embeddings[i].model, JSON.stringify(embeddings[i].vector), embeddings[i].dimensions]);
      }

      return this.get(id) as Promise<Document>;
    },

    async search(query: string, projectId?: string, limit = 10): Promise<DocumentSearchResult[]> {
      const start = Date.now();
      log.debug("docs", "Document search starting", { query: query.slice(0, 50), projectId, limit });

      // Similar to memory search but for documents
      const queryEmbedding = await embedding.embed(query);
      const modelId = embedding.getActiveModelId();

      let sql = `
        SELECT
          dc.id as chunk_id,
          dc.document_id,
          dc.content,
          dc.chunk_index,
          vector_distance_cos(dv.vector, vector(?)) as distance
        FROM document_vectors dv
        JOIN document_chunks dc ON dv.chunk_id = dc.id
        JOIN documents d ON dc.document_id = d.id
        WHERE dv.model_id = ?
          AND dv.rowid IN (
            SELECT rowid FROM vector_top_k('document_vectors_idx', vector(?), ?)
          )
      `;
      const args: any[] = [
        JSON.stringify(queryEmbedding.vector),
        modelId,
        JSON.stringify(queryEmbedding.vector),
        limit * 3
      ];

      if (projectId) {
        sql += " AND d.project_id = ?";
        args.push(projectId);
      }

      sql += " ORDER BY distance ASC LIMIT ?";
      args.push(limit);

      const result = await db.execute(sql, args);

      // Fetch full documents and build results
      const results: DocumentSearchResult[] = [];
      for (const row of result.rows) {
        const doc = await this.get(row[1] as string);
        if (!doc) continue;

        results.push({
          document: doc,
          chunk: {
            id: row[0] as string,
            documentId: row[1] as string,
            content: row[2] as string,
            chunkIndex: row[3] as number,
            startOffset: 0,
            endOffset: 0,
            tokensEstimate: 0
          },
          score: 1 - (row[4] as number),
          matchType: "semantic"
        });
      }

      log.info("docs", "Document search complete", { results: results.length, ms: Date.now() - start });
      return results;
    },

    async checkForUpdates(projectId: string): Promise<string[]> {
      log.debug("docs", "Checking for document updates", { projectId });
      const docs = await this.list(projectId);
      const updated: string[] = [];

      for (const doc of docs) {
        if (!doc.sourcePath) continue;

        try {
          const currentContent = await Bun.file(doc.sourcePath).text();
          const currentChecksum = createHash("sha256").update(currentContent).digest("hex");

          if (currentChecksum !== doc.checksum) {
            log.debug("docs", "Document changed", { id: doc.id, path: doc.sourcePath });
            updated.push(doc.id);
          }
        } catch {
          // File doesn't exist anymore, mark as updated
          log.warn("docs", "Document file missing", { id: doc.id, path: doc.sourcePath });
          updated.push(doc.id);
        }
      }

      log.info("docs", "Update check complete", { projectId, checked: docs.length, updated: updated.length });
      return updated;
    }
  };
}

function extractTitle(content: string, type: "txt" | "md" | "url"): string {
  if (type === "md") {
    // Look for first H1
    const h1Match = content.match(/^#\s+(.+)$/m);
    if (h1Match) return h1Match[1].trim();
  }

  // Use first line as title
  const firstLine = content.split("\n")[0].trim();
  return firstLine.slice(0, 100);
}
```

### Test Specification

```typescript
// src/services/documents/__test__/ingest.test.ts (colocated unit test)
describe("Document Ingestion", () => {
  let docs: DocumentService;

  beforeEach(async () => {
    await setupTestDatabase();
    docs = createDocumentService();
  });

  test("ingests text file", async () => {
    // Create temp file
    const tempPath = "/tmp/test-doc.txt";
    await Bun.write(tempPath, "This is test content for ingestion.");

    const doc = await docs.ingest({
      projectId: "proj1",
      path: tempPath
    });

    expect(doc.sourceType).toBe("txt");
    expect(doc.fullContent).toContain("test content");
    expect(doc.checksum).toBeDefined();
  });

  test("ingests markdown with title extraction", async () => {
    const content = "# My Document\n\nSome content here.";
    const doc = await docs.ingest({
      projectId: "proj1",
      content,
      sourceType: "md"
    });

    expect(doc.title).toBe("My Document");
  });

  test("chunks long documents", async () => {
    const longContent = "Paragraph. ".repeat(500);
    const doc = await docs.ingest({
      projectId: "proj1",
      content: longContent
    });

    // Check chunks were created
    const chunks = await db.execute(
      "SELECT * FROM document_chunks WHERE document_id = ?",
      [doc.id]
    );
    expect(chunks.rows.length).toBeGreaterThan(1);
  });

  test("updates existing document by path", async () => {
    const tempPath = "/tmp/test-doc.txt";
    await Bun.write(tempPath, "Original content");

    const doc1 = await docs.ingest({ projectId: "proj1", path: tempPath });

    await Bun.write(tempPath, "Updated content");
    const doc2 = await docs.ingest({ projectId: "proj1", path: tempPath });

    expect(doc2.id).toBe(doc1.id);
    expect(doc2.fullContent).toContain("Updated");
    expect(doc2.updatedAt).toBeGreaterThan(doc1.updatedAt);
  });

  test("skips unchanged documents", async () => {
    const tempPath = "/tmp/test-doc.txt";
    await Bun.write(tempPath, "Same content");

    const doc1 = await docs.ingest({ projectId: "proj1", path: tempPath });
    const doc2 = await docs.ingest({ projectId: "proj1", path: tempPath });

    expect(doc2.updatedAt).toBe(doc1.updatedAt);  // Not updated
  });

  test("search finds relevant chunks", async () => {
    await docs.ingest({
      projectId: "proj1",
      content: "React is a JavaScript library for building user interfaces."
    });

    const results = await docs.search("JavaScript UI framework", "proj1");
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].document.fullContent).toContain("React");
  });

  test("checkForUpdates detects changes", async () => {
    const tempPath = "/tmp/test-doc.txt";
    await Bun.write(tempPath, "Original");

    await docs.ingest({ projectId: "proj1", path: tempPath });
    await Bun.write(tempPath, "Changed");

    const updated = await docs.checkForUpdates("proj1");
    expect(updated.length).toBe(1);
  });
});
```

## Acceptance Criteria

- [ ] Text chunking splits by sentences/paragraphs
- [ ] Chunks have configurable overlap
- [ ] Document ingestion from file path works
- [ ] Document ingestion from URL works
- [ ] Document ingestion from raw content works
- [ ] Markdown title extraction works
- [ ] Checksum prevents re-processing unchanged files
- [ ] Document updates replace old chunks
- [ ] Document search finds relevant chunks
- [ ] Change detection identifies updated files
