import { createHash } from "crypto";
import { getDatabase } from "../../db/database.js";
import { log } from "../../utils/log.js";
import type { EmbeddingService } from "../embedding/types.js";
import { chunkText } from "./chunk.js";
import type {
  Document,
  DocumentChunk,
  DocumentSourceType,
  IngestOptions,
} from "./types.js";

export type DocumentSearchResult = {
  document: Document;
  chunk: DocumentChunk;
  score: number;
  matchType: "semantic" | "keyword" | "both";
};

export type DocumentService = {
  ingest(options: IngestOptions): Promise<Document>;
  get(id: string): Promise<Document | null>;
  getByPath(path: string, projectId: string): Promise<Document | null>;
  delete(id: string): Promise<void>;
  list(projectId: string): Promise<Document[]>;
  search(
    query: string,
    projectId?: string,
    limit?: number
  ): Promise<DocumentSearchResult[]>;
  checkForUpdates(projectId: string): Promise<string[]>;
  getChunks(documentId: string): Promise<DocumentChunk[]>;
};

function rowToDocument(row: Record<string, unknown>): Document {
  return {
    id: String(row["id"]),
    projectId: String(row["project_id"]),
    sourcePath: row["source_path"] ? String(row["source_path"]) : undefined,
    sourceUrl: row["source_url"] ? String(row["source_url"]) : undefined,
    sourceType: String(row["source_type"]) as DocumentSourceType,
    title: row["title"] ? String(row["title"]) : undefined,
    fullContent: String(row["full_content"]),
    checksum: String(row["checksum"]),
    createdAt: Number(row["created_at"]),
    updatedAt: Number(row["updated_at"]),
  };
}

function rowToChunk(row: Record<string, unknown>): DocumentChunk {
  return {
    id: String(row["id"]),
    documentId: String(row["document_id"]),
    chunkIndex: Number(row["chunk_index"]),
    content: String(row["content"]),
    startOffset: Number(row["start_offset"]),
    endOffset: Number(row["end_offset"]),
    tokensEstimate: Number(row["tokens_estimate"]),
  };
}

function extractTitle(content: string, type: DocumentSourceType): string {
  if (type === "md") {
    const h1Match = content.match(/^#\s+(.+)$/m);
    if (h1Match && h1Match[1]) return h1Match[1].trim();
  }

  const firstLine = content.split("\n")[0]?.trim() ?? "";
  return firstLine.slice(0, 100);
}

export function createDocumentService(
  embeddingService: EmbeddingService
): DocumentService {
  const service: DocumentService = {
    async ingest(options: IngestOptions): Promise<Document> {
      const { projectId, path, url, content, title, sourceType } = options;
      const start = Date.now();
      const db = await getDatabase();

      log.info("docs", "Ingesting document", {
        projectId,
        path,
        url: url?.slice(0, 50),
      });

      let fullContent: string;
      let finalSourceType: DocumentSourceType;

      if (content !== undefined) {
        fullContent = content;
        finalSourceType = sourceType ?? "txt";
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

      const checksum = createHash("sha256").update(fullContent).digest("hex");

      const existing = path ? await service.getByPath(path, projectId) : null;

      if (existing?.checksum === checksum) {
        log.debug("docs", "Document unchanged, skipping", { id: existing.id });
        return existing;
      }

      const id = existing?.id ?? crypto.randomUUID();
      const now = Date.now();

      if (existing) {
        await db.execute(
          "DELETE FROM document_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)",
          [id]
        );
        await db.execute("DELETE FROM document_chunks WHERE document_id = ?", [
          id,
        ]);
      }

      const docTitle = title ?? extractTitle(fullContent, finalSourceType);

      await db.execute(
        `INSERT OR REPLACE INTO documents (id, project_id, source_path, source_url, source_type, title, full_content, checksum, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          id,
          projectId,
          path ?? null,
          url ?? null,
          finalSourceType,
          docTitle,
          fullContent,
          checksum,
          existing?.createdAt ?? now,
          now,
        ]
      );

      const chunks = chunkText(fullContent);
      const modelId = embeddingService.getActiveModelId();

      if (chunks.length > 0) {
        const embeddings = await embeddingService.embedBatch(
          chunks.map((c) => c.content)
        );

        for (let i = 0; i < chunks.length; i++) {
          const chunkId = `${id}-${i}`;
          const chunk = chunks[i];
          const embedding = embeddings[i];

          if (!chunk || !embedding) continue;

          await db.execute(
            `INSERT INTO document_chunks (id, document_id, chunk_index, content, start_offset, end_offset, tokens_estimate)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [
              chunkId,
              id,
              i,
              chunk.content,
              chunk.startOffset,
              chunk.endOffset,
              chunk.tokensEstimate,
            ]
          );

          const vectorBuffer = new Float32Array(embedding.vector).buffer;
          await db.execute(
            `INSERT INTO document_vectors (chunk_id, model_id, vector, dim, created_at)
             VALUES (?, ?, ?, ?, ?)`,
            [
              chunkId,
              modelId,
              new Uint8Array(vectorBuffer),
              embedding.dimensions,
              now,
            ]
          );
        }
      }

      log.info("docs", "Document ingested", {
        id,
        chunks: chunks.length,
        ms: Date.now() - start,
      });

      const result = await service.get(id);
      if (!result) throw new Error("Failed to retrieve ingested document");
      return result;
    },

    async get(id: string): Promise<Document | null> {
      const db = await getDatabase();
      const result = await db.execute("SELECT * FROM documents WHERE id = ?", [
        id,
      ]);
      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;
      return rowToDocument(row);
    },

    async getByPath(path: string, projectId: string): Promise<Document | null> {
      const db = await getDatabase();
      const result = await db.execute(
        "SELECT * FROM documents WHERE source_path = ? AND project_id = ?",
        [path, projectId]
      );
      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;
      return rowToDocument(row);
    },

    async delete(id: string): Promise<void> {
      const db = await getDatabase();
      await db.execute(
        "DELETE FROM document_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)",
        [id]
      );
      await db.execute("DELETE FROM document_chunks WHERE document_id = ?", [
        id,
      ]);
      await db.execute("DELETE FROM documents WHERE id = ?", [id]);
      log.info("docs", "Document deleted", { id });
    },

    async list(projectId: string): Promise<Document[]> {
      const db = await getDatabase();
      const result = await db.execute(
        "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at DESC",
        [projectId]
      );
      return result.rows.map(rowToDocument);
    },

    async search(
      query: string,
      projectId?: string,
      limit = 10
    ): Promise<DocumentSearchResult[]> {
      const start = Date.now();
      const db = await getDatabase();
      log.debug("docs", "Document search starting", {
        query: query.slice(0, 50),
        projectId,
        limit,
      });

      const queryEmbedding = await embeddingService.embed(query);
      const modelId = embeddingService.getActiveModelId();

      let sql: string;
      const args: (string | number | null)[] = [];

      if (projectId) {
        sql = `
          SELECT
            dc.id as chunk_id,
            dc.document_id,
            dc.content,
            dc.chunk_index,
            dc.start_offset,
            dc.end_offset,
            dc.tokens_estimate,
            dv.vector
          FROM document_vectors dv
          JOIN document_chunks dc ON dv.chunk_id = dc.id
          JOIN documents d ON dc.document_id = d.id
          WHERE dv.model_id = ? AND d.project_id = ?
        `;
        args.push(modelId, projectId);
      } else {
        sql = `
          SELECT
            dc.id as chunk_id,
            dc.document_id,
            dc.content,
            dc.chunk_index,
            dc.start_offset,
            dc.end_offset,
            dc.tokens_estimate,
            dv.vector
          FROM document_vectors dv
          JOIN document_chunks dc ON dv.chunk_id = dc.id
          WHERE dv.model_id = ?
        `;
        args.push(modelId);
      }

      const result = await db.execute(sql, args);

      const results: DocumentSearchResult[] = [];

      for (const row of result.rows) {
        const vectorData = row["vector"];
        const storedVector = parseVector(vectorData);

        if (storedVector.length !== queryEmbedding.dimensions) {
          continue;
        }

        const similarity = cosineSimilarity(
          queryEmbedding.vector,
          storedVector
        );

        const doc = await service.get(String(row["document_id"]));
        if (!doc) continue;

        results.push({
          document: doc,
          chunk: {
            id: String(row["chunk_id"]),
            documentId: String(row["document_id"]),
            content: String(row["content"]),
            chunkIndex: Number(row["chunk_index"]),
            startOffset: Number(row["start_offset"]),
            endOffset: Number(row["end_offset"]),
            tokensEstimate: Number(row["tokens_estimate"]),
          },
          score: similarity,
          matchType: "semantic",
        });
      }

      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, limit);

      log.info("docs", "Document search complete", {
        results: topResults.length,
        ms: Date.now() - start,
      });

      return topResults;
    },

    async checkForUpdates(projectId: string): Promise<string[]> {
      log.debug("docs", "Checking for document updates", { projectId });
      const docs = await service.list(projectId);
      const updated: string[] = [];

      for (const doc of docs) {
        if (!doc.sourcePath) continue;

        try {
          const currentContent = await Bun.file(doc.sourcePath).text();
          const currentChecksum = createHash("sha256")
            .update(currentContent)
            .digest("hex");

          if (currentChecksum !== doc.checksum) {
            log.debug("docs", "Document changed", {
              id: doc.id,
              path: doc.sourcePath,
            });
            updated.push(doc.id);
          }
        } catch {
          log.warn("docs", "Document file missing", {
            id: doc.id,
            path: doc.sourcePath,
          });
          updated.push(doc.id);
        }
      }

      log.info("docs", "Update check complete", {
        projectId,
        checked: docs.length,
        updated: updated.length,
      });
      return updated;
    },

    async getChunks(documentId: string): Promise<DocumentChunk[]> {
      const db = await getDatabase();
      const result = await db.execute(
        "SELECT * FROM document_chunks WHERE document_id = ? ORDER BY chunk_index",
        [documentId]
      );
      return result.rows.map(rowToChunk);
    },
  };

  return service;
}

function parseVector(blob: unknown): number[] {
  if (blob instanceof Uint8Array || blob instanceof ArrayBuffer) {
    const buffer = blob instanceof ArrayBuffer ? blob : blob.buffer;
    return Array.from(new Float32Array(buffer));
  }

  if (typeof blob === "string") {
    try {
      return JSON.parse(blob) as number[];
    } catch {
      return [];
    }
  }

  if (Array.isArray(blob)) {
    return blob as number[];
  }

  return [];
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}
