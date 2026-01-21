import { createHash } from 'crypto';
import { getDatabase } from '../../db/database.js';
import { log } from '../../utils/log.js';
import type { EmbeddingService } from '../embedding/types.js';
import { chunkCode } from './chunker.js';
import { getProjectHash } from './coordination.js';
import { fileExists, getFileMtime, loadGitignorePatterns, scanDirectory } from './scanner.js';
import type {
  ChunkExport,
  CodeIndexExport,
  CodeIndexOptions,
  CodeIndexState,
  CodeLanguage,
  CodeSearchOptions,
  CodeSearchResult,
  IndexCheckpoint,
  IndexedFile,
  IndexedFileExport,
  IndexProgress,
  IndexStatistics,
  ScannedFile,
  WatcherEvent,
} from './types.js';

const BATCH_SIZE = 50;
const PARALLEL_FILES = 5;

export type CodeIndexService = {
  index(projectPath: string, projectId: string, options?: CodeIndexOptions): Promise<IndexProgress>;
  search(options: CodeSearchOptions): Promise<CodeSearchResult[]>;
  getState(projectId: string): Promise<CodeIndexState | null>;
  getStatistics(projectId: string): Promise<IndexStatistics | null>;
  processFileChanges(projectPath: string, projectId: string, events: WatcherEvent[]): Promise<void>;
  cleanupDeletedFiles(projectId: string): Promise<number>;
  deleteFile(projectId: string, filePath: string): Promise<boolean>;
  exportIndex(projectPath: string, projectId: string): Promise<CodeIndexExport | null>;
  importIndex(projectPath: string, projectId: string, data: CodeIndexExport): Promise<{ imported: number; skipped: number }>;
  saveCheckpoint(projectId: string, checkpoint: Omit<IndexCheckpoint, 'id' | 'createdAt' | 'updatedAt'>): Promise<string>;
  loadCheckpoint(projectId: string): Promise<IndexCheckpoint | null>;
  clearCheckpoint(projectId: string): Promise<void>;
};

function rowToIndexedFile(row: Record<string, unknown>): IndexedFile {
  return {
    id: String(row['id']),
    projectId: String(row['project_id']),
    path: String(row['path']),
    checksum: String(row['checksum']),
    mtime: Number(row['mtime']),
    indexedAt: Number(row['indexed_at']),
  };
}

function rowToCodeSearchResult(row: Record<string, unknown>): CodeSearchResult {
  let symbols: string[] = [];
  const symbolsJson = row['symbols_json'];
  if (symbolsJson && typeof symbolsJson === 'string') {
    try {
      symbols = JSON.parse(symbolsJson) as string[];
    } catch {
      symbols = [];
    }
  }

  return {
    documentId: String(row['document_id']),
    chunkId: String(row['chunk_id']),
    path: String(row['source_path']),
    content: String(row['content']),
    startLine: Number(row['start_line'] ?? 0),
    endLine: Number(row['end_line'] ?? 0),
    language: (row['language'] as CodeLanguage) ?? 'unknown',
    chunkType: (row['chunk_type'] as CodeSearchResult['chunkType']) ?? 'block',
    symbols,
    score: Number(row['score'] ?? 0),
  };
}

async function getIndexedFile(projectId: string, path: string): Promise<IndexedFile | null> {
  const db = await getDatabase();
  const result = await db.execute('SELECT * FROM indexed_files WHERE project_id = ? AND path = ?', [projectId, path]);
  if (result.rows.length === 0) return null;
  const row = result.rows[0];
  if (!row) return null;
  return rowToIndexedFile(row);
}

async function indexFile(
  file: ScannedFile,
  projectId: string,
  embeddingService: EmbeddingService,
): Promise<{ chunks: number; error?: string }> {
  const db = await getDatabase();

  let content: string;
  try {
    content = await Bun.file(file.path).text();
  } catch (err) {
    return { chunks: 0, error: `Failed to read file: ${(err as Error).message}` };
  }

  const checksum = createHash('sha256').update(content).digest('hex');

  const existing = await getIndexedFile(projectId, file.path);
  if (existing && existing.checksum === checksum) {
    return { chunks: 0 };
  }

  const chunks = chunkCode(content, file.language);
  if (chunks.length === 0) {
    return { chunks: 0 };
  }

  const docId = existing?.id ?? crypto.randomUUID();
  const now = Date.now();
  const lineCount = content.split('\n').length;

  if (existing) {
    await db.execute(
      'DELETE FROM document_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)',
      [docId],
    );
    await db.execute('DELETE FROM document_chunks WHERE document_id = ?', [docId]);
    await db.execute('DELETE FROM indexed_files WHERE id = ?', [docId]);
  }

  await db.execute(
    `INSERT OR REPLACE INTO documents (id, project_id, source_path, source_type, title, full_content, checksum, created_at, updated_at, language, line_count, is_code)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`,
    [
      docId,
      projectId,
      file.path,
      'code',
      file.relativePath,
      content,
      checksum,
      existing ? (existing.indexedAt ?? now) : now,
      now,
      file.language,
      lineCount,
    ],
  );

  const modelId = embeddingService.getActiveModelId();
  const chunkContents = chunks.map(c => c.content);
  const embeddings = await embeddingService.embedBatch(chunkContents);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = embeddings[i];
    if (!chunk || !embedding) continue;

    const chunkId = `${docId}-${i}`;

    await db.execute(
      `INSERT INTO document_chunks (id, document_id, chunk_index, content, start_offset, end_offset, tokens_estimate, start_line, end_line, chunk_type, symbols_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        chunkId,
        docId,
        i,
        chunk.content,
        0,
        chunk.content.length,
        chunk.tokensEstimate,
        chunk.startLine,
        chunk.endLine,
        chunk.chunkType,
        JSON.stringify(chunk.symbols),
      ],
    );

    const vectorBuffer = new Float32Array(embedding.vector).buffer;
    await db.execute(
      `INSERT INTO document_vectors (chunk_id, model_id, vector, dim, created_at)
       VALUES (?, ?, ?, ?, ?)`,
      [chunkId, modelId, new Uint8Array(vectorBuffer), embedding.dimensions, now],
    );
  }

  await db.execute(
    `INSERT OR REPLACE INTO indexed_files (id, project_id, path, checksum, mtime, indexed_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [docId, projectId, file.path, checksum, file.mtime, now],
  );

  return { chunks: chunks.length };
}

function shouldIncludeFile(file: ScannedFile, includePaths?: string[], excludePaths?: string[]): boolean {
  if (excludePaths && excludePaths.length > 0) {
    for (const pattern of excludePaths) {
      if (file.relativePath.includes(pattern) || file.path.includes(pattern)) {
        return false;
      }
    }
  }

  if (includePaths && includePaths.length > 0) {
    for (const pattern of includePaths) {
      if (file.relativePath.includes(pattern) || file.path.includes(pattern)) {
        return true;
      }
    }
    return false;
  }

  return true;
}

export function createCodeIndexService(embeddingService: EmbeddingService): CodeIndexService {
  return {
    async index(projectPath: string, projectId: string, options?: CodeIndexOptions): Promise<IndexProgress> {
      const { force = false, dryRun = false, onProgress, includePaths, excludePaths, resumeFromCheckpoint } = options ?? {};
      const start = Date.now();

      log.info('codeindex', 'Starting code indexing', { projectPath, projectId, force, dryRun, includePaths, excludePaths });

      const progress: IndexProgress = {
        phase: 'scanning',
        scannedFiles: 0,
        indexedFiles: 0,
        totalFiles: 0,
        errors: [],
        startedAt: start,
        bytesProcessed: 0,
        totalBytes: 0,
        tokensProcessed: 0,
      };

      let checkpoint: IndexCheckpoint | null = null;
      if (resumeFromCheckpoint) {
        checkpoint = await this.loadCheckpoint(projectId);
        if (checkpoint) {
          log.info('codeindex', 'Resuming from checkpoint', {
            processedFiles: checkpoint.processedFiles.length,
            pendingFiles: checkpoint.pendingFiles.length,
          });
          progress.indexedFiles = checkpoint.processedFiles.length;
        }
      }

      const scanResult = await scanDirectory(projectPath, scanned => {
        progress.scannedFiles = scanned;
        onProgress?.(progress);
      });

      let filteredFiles = scanResult.files.filter(f => shouldIncludeFile(f, includePaths, excludePaths));

      if (checkpoint) {
        const processedSet = new Set(checkpoint.processedFiles);
        filteredFiles = filteredFiles.filter(f => !processedSet.has(f.path));
      }

      progress.phase = 'indexing';
      progress.totalFiles = filteredFiles.length + (checkpoint?.processedFiles.length ?? 0);
      progress.totalBytes = filteredFiles.reduce((sum, f) => sum + f.size, 0);
      onProgress?.(progress);

      if (dryRun) {
        log.info('codeindex', 'Dry run complete', {
          files: filteredFiles.length,
          skipped: scanResult.skippedCount,
          filtered: scanResult.files.length - filteredFiles.length,
        });
        progress.phase = 'complete';
        return progress;
      }

      const db = await getDatabase();

      const filesToIndex: ScannedFile[] = [];
      for (const file of filteredFiles) {
        if (force) {
          filesToIndex.push(file);
          continue;
        }

        const indexed = await getIndexedFile(projectId, file.path);
        if (!indexed || indexed.mtime < file.mtime) {
          filesToIndex.push(file);
        }
      }

      log.info('codeindex', 'Files to index', {
        total: filteredFiles.length,
        toIndex: filesToIndex.length,
        filtered: scanResult.files.length - filteredFiles.length,
      });

      const processedFiles: string[] = checkpoint?.processedFiles ?? [];
      let totalChunks = 0;
      let lastCheckpointTime = Date.now();
      const CHECKPOINT_INTERVAL = 30000;

      for (let i = 0; i < filesToIndex.length; i += BATCH_SIZE) {
        const batch = filesToIndex.slice(i, i + BATCH_SIZE);

        for (let j = 0; j < batch.length; j += PARALLEL_FILES) {
          const parallelBatch = batch.slice(j, j + PARALLEL_FILES);

          const results = await Promise.allSettled(
            parallelBatch.map(async file => {
              progress.currentFile = file.relativePath;
              onProgress?.(progress);
              return { file, result: await indexFile(file, projectId, embeddingService) };
            }),
          );

          for (const settled of results) {
            if (settled.status === 'fulfilled') {
              const { file, result } = settled.value;
              if (result.error) {
                progress.errors.push(`${file.relativePath}: ${result.error}`);
              } else if (result.chunks > 0) {
                progress.indexedFiles++;
                totalChunks += result.chunks;
                processedFiles.push(file.path);
                progress.bytesProcessed = (progress.bytesProcessed ?? 0) + file.size;
                progress.tokensProcessed = (progress.tokensProcessed ?? 0) + Math.ceil(file.size / 4);
              }
            } else {
              const err = settled.reason as Error;
              progress.errors.push(`Unknown file: ${err.message}`);
              log.error('codeindex', 'Failed to index file', { error: err.message });
            }
          }

          const elapsed = Date.now() - start;
          if (progress.indexedFiles > 0 && elapsed > 0) {
            progress.filesPerSecond = (progress.indexedFiles / elapsed) * 1000;
            const remaining = progress.totalFiles - progress.indexedFiles;
            progress.estimatedTimeRemaining = remaining / progress.filesPerSecond * 1000;
          }

          onProgress?.(progress);

          if (Date.now() - lastCheckpointTime > CHECKPOINT_INTERVAL && processedFiles.length > 0) {
            const pendingFiles = filesToIndex.slice(i + j + parallelBatch.length).map(f => f.path);
            await this.saveCheckpoint(projectId, {
              projectId,
              phase: 'indexing',
              processedFiles,
              pendingFiles,
              progress,
            });
            lastCheckpointTime = Date.now();
          }
        }
      }

      const gitignore = await loadGitignorePatterns(projectPath);
      const gitignoreHash = gitignore.hash;

      await db.execute(
        `INSERT OR REPLACE INTO code_index_state (project_id, last_indexed_at, indexed_files, gitignore_hash, total_bytes, total_tokens, total_chunks)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [projectId, Date.now(), progress.indexedFiles, gitignoreHash, progress.bytesProcessed ?? 0, progress.tokensProcessed ?? 0, totalChunks],
      );

      await this.clearCheckpoint(projectId);

      progress.phase = 'complete';
      progress.currentFile = undefined;
      onProgress?.(progress);

      log.info('codeindex', 'Code indexing complete', {
        indexed: progress.indexedFiles,
        chunks: totalChunks,
        bytes: progress.bytesProcessed,
        errors: progress.errors.length,
        ms: Date.now() - start,
      });

      return progress;
    },

    async search(options: CodeSearchOptions): Promise<CodeSearchResult[]> {
      const { query, projectId, language, limit = 10 } = options;
      const start = Date.now();

      log.debug('codeindex', 'Code search starting', { query: query.slice(0, 50), projectId, language, limit });

      const db = await getDatabase();

      const state = await this.getState(projectId);
      if (!state) {
        log.warn('codeindex', 'Project not indexed', { projectId });
        return [];
      }

      const instructedQuery = `Instruct: Find code that implements or relates to the following query\nQuery: ${query}`;
      const queryEmbedding = await embeddingService.embed(instructedQuery);
      const modelId = embeddingService.getActiveModelId();

      let sql = `
        SELECT
          dc.id as chunk_id,
          dc.document_id,
          dc.content,
          dc.start_line,
          dc.end_line,
          dc.chunk_type,
          dc.symbols_json,
          d.source_path,
          d.language,
          dv.vector
        FROM document_vectors dv
        JOIN document_chunks dc ON dv.chunk_id = dc.id
        JOIN documents d ON dc.document_id = d.id
        WHERE dv.model_id = ? AND d.project_id = ? AND d.is_code = 1
      `;

      const args: (string | number)[] = [modelId, projectId];

      if (language) {
        sql += ' AND d.language = ?';
        args.push(language);
      }

      const result = await db.execute(sql, args);

      const results: CodeSearchResult[] = [];

      for (const row of result.rows) {
        const vectorData = row['vector'];
        const storedVector = parseVector(vectorData);

        if (storedVector.length !== queryEmbedding.dimensions) {
          continue;
        }

        const similarity = cosineSimilarity(queryEmbedding.vector, storedVector);

        results.push({
          ...rowToCodeSearchResult(row),
          score: similarity,
        });
      }

      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, limit);

      log.info('codeindex', 'Code search complete', {
        results: topResults.length,
        ms: Date.now() - start,
      });

      return topResults;
    },

    async getState(projectId: string): Promise<CodeIndexState | null> {
      const db = await getDatabase();
      const result = await db.execute('SELECT * FROM code_index_state WHERE project_id = ?', [projectId]);

      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;

      return {
        projectId: String(row['project_id']),
        lastIndexedAt: Number(row['last_indexed_at']),
        indexedFiles: Number(row['indexed_files']),
        gitignoreHash: row['gitignore_hash'] ? String(row['gitignore_hash']) : null,
      };
    },

    async processFileChanges(projectPath: string, projectId: string, events: WatcherEvent[]): Promise<void> {
      log.debug('codeindex', 'Processing file changes', { count: events.length });

      for (const event of events) {
        if (event.type === 'delete') {
          await this.deleteFile(projectId, event.path);
          continue;
        }

        const mtime = await getFileMtime(event.path);
        if (mtime === null) continue;

        const file: ScannedFile = {
          path: event.path,
          relativePath: event.path.replace(projectPath + '/', ''),
          size: 0,
          mtime,
          language: getLanguageFromPath(event.path),
        };

        if (file.language === 'unknown') continue;

        try {
          const result = await indexFile(file, projectId, embeddingService);
          if (result.chunks > 0) {
            log.debug('codeindex', 'File indexed', { path: file.relativePath, chunks: result.chunks });
          }
        } catch (err) {
          log.error('codeindex', 'Failed to index file', {
            path: file.relativePath,
            error: (err as Error).message,
          });
        }
      }
    },

    async cleanupDeletedFiles(projectId: string): Promise<number> {
      const db = await getDatabase();
      const result = await db.execute('SELECT id, path FROM indexed_files WHERE project_id = ?', [projectId]);

      let deleted = 0;

      for (const row of result.rows) {
        const path = String(row['path']);
        const exists = await fileExists(path);
        if (!exists) {
          const didDelete = await this.deleteFile(projectId, path);
          if (didDelete) deleted++;
        }
      }

      if (deleted > 0) {
        log.info('codeindex', 'Cleaned up deleted files', { deleted });
      }

      return deleted;
    },

    async deleteFile(projectId: string, filePath: string): Promise<boolean> {
      const db = await getDatabase();

      const result = await db.execute(
        'SELECT id FROM indexed_files WHERE project_id = ? AND path = ?',
        [projectId, filePath],
      );

      if (result.rows.length === 0) {
        return false;
      }

      const id = String(result.rows[0]?.['id']);

      await db.execute(
        'DELETE FROM document_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)',
        [id],
      );
      await db.execute('DELETE FROM document_chunks WHERE document_id = ?', [id]);
      await db.execute('DELETE FROM documents WHERE id = ?', [id]);
      await db.execute('DELETE FROM indexed_files WHERE id = ?', [id]);

      log.debug('codeindex', 'Deleted file index', { path: filePath });
      return true;
    },

    async exportIndex(projectPath: string, projectId: string): Promise<CodeIndexExport | null> {
      const db = await getDatabase();
      const state = await this.getState(projectId);

      if (!state) {
        log.warn('codeindex', 'No index state found for export', { projectId });
        return null;
      }

      const modelId = embeddingService.getActiveModelId();

      const docsResult = await db.execute(
        `SELECT id, source_path, language, line_count, checksum
         FROM documents
         WHERE project_id = ? AND is_code = 1`,
        [projectId],
      );

      const files: IndexedFileExport[] = [];

      for (const docRow of docsResult.rows) {
        const docId = String(docRow['id']);
        const sourcePath = String(docRow['source_path']);
        const relativePath = sourcePath.startsWith(projectPath)
          ? sourcePath.slice(projectPath.length + 1)
          : sourcePath;

        const chunksResult = await db.execute(
          `SELECT dc.content, dc.start_line, dc.end_line, dc.chunk_type, dc.symbols_json, dv.vector
           FROM document_chunks dc
           JOIN document_vectors dv ON dc.id = dv.chunk_id
           WHERE dc.document_id = ? AND dv.model_id = ?
           ORDER BY dc.chunk_index`,
          [docId, modelId],
        );

        const chunks: ChunkExport[] = [];
        for (const chunkRow of chunksResult.rows) {
          let symbols: string[] = [];
          const symbolsJson = chunkRow['symbols_json'];
          if (symbolsJson && typeof symbolsJson === 'string') {
            try {
              symbols = JSON.parse(symbolsJson) as string[];
            } catch {
              symbols = [];
            }
          }

          chunks.push({
            content: String(chunkRow['content']),
            startLine: Number(chunkRow['start_line'] ?? 0),
            endLine: Number(chunkRow['end_line'] ?? 0),
            chunkType: (chunkRow['chunk_type'] as ChunkExport['chunkType']) ?? 'block',
            symbols,
            vector: parseVector(chunkRow['vector']),
          });
        }

        files.push({
          relativePath,
          language: (docRow['language'] as CodeLanguage) ?? 'unknown',
          lineCount: Number(docRow['line_count'] ?? 0),
          checksum: String(docRow['checksum']),
          chunks,
        });
      }

      log.info('codeindex', 'Index exported', { projectId, files: files.length });

      return {
        version: 1,
        exportedAt: Date.now(),
        projectPath,
        state,
        files,
      };
    },

    async importIndex(
      projectPath: string,
      projectId: string,
      data: CodeIndexExport,
    ): Promise<{ imported: number; skipped: number }> {
      const db = await getDatabase();
      const modelId = embeddingService.getActiveModelId();
      const now = Date.now();

      let imported = 0;
      let skipped = 0;

      for (const file of data.files) {
        const fullPath = `${projectPath}/${file.relativePath}`;

        const existing = await getIndexedFile(projectId, fullPath);
        if (existing && existing.checksum === file.checksum) {
          skipped++;
          continue;
        }

        const docId = existing?.id ?? crypto.randomUUID();

        if (existing) {
          await db.execute(
            'DELETE FROM document_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)',
            [docId],
          );
          await db.execute('DELETE FROM document_chunks WHERE document_id = ?', [docId]);
          await db.execute('DELETE FROM indexed_files WHERE id = ?', [docId]);
        }

        await db.execute(
          `INSERT OR REPLACE INTO documents (id, project_id, source_path, source_type, title, full_content, checksum, created_at, updated_at, language, line_count, is_code)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`,
          [
            docId,
            projectId,
            fullPath,
            'code',
            file.relativePath,
            '',
            file.checksum,
            now,
            now,
            file.language,
            file.lineCount,
          ],
        );

        for (let i = 0; i < file.chunks.length; i++) {
          const chunk = file.chunks[i];
          if (!chunk) continue;

          const chunkId = `${docId}-${i}`;

          await db.execute(
            `INSERT INTO document_chunks (id, document_id, chunk_index, content, start_offset, end_offset, tokens_estimate, start_line, end_line, chunk_type, symbols_json)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
              chunkId,
              docId,
              i,
              chunk.content,
              0,
              chunk.content.length,
              Math.ceil(chunk.content.length / 4),
              chunk.startLine,
              chunk.endLine,
              chunk.chunkType,
              JSON.stringify(chunk.symbols),
            ],
          );

          const vectorBuffer = new Float32Array(chunk.vector).buffer;
          await db.execute(
            `INSERT INTO document_vectors (chunk_id, model_id, vector, dim, created_at)
             VALUES (?, ?, ?, ?, ?)`,
            [chunkId, modelId, new Uint8Array(vectorBuffer), chunk.vector.length, now],
          );
        }

        await db.execute(
          `INSERT OR REPLACE INTO indexed_files (id, project_id, path, checksum, mtime, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?)`,
          [docId, projectId, fullPath, file.checksum, now, now],
        );

        imported++;
      }

      await db.execute(
        `INSERT OR REPLACE INTO code_index_state (project_id, last_indexed_at, indexed_files, gitignore_hash)
         VALUES (?, ?, ?, ?)`,
        [projectId, now, imported + skipped, data.state.gitignoreHash],
      );

      log.info('codeindex', 'Index imported', { projectId, imported, skipped });

      return { imported, skipped };
    },

    async getStatistics(projectId: string): Promise<IndexStatistics | null> {
      const db = await getDatabase();

      const state = await this.getState(projectId);
      if (!state) return null;

      const stateResult = await db.execute(
        `SELECT total_bytes, total_tokens, total_chunks FROM code_index_state WHERE project_id = ?`,
        [projectId],
      );
      const stateRow = stateResult.rows[0];
      const totalBytes = Number(stateRow?.['total_bytes'] ?? 0);
      const totalTokens = Number(stateRow?.['total_tokens'] ?? 0);
      const totalChunks = Number(stateRow?.['total_chunks'] ?? 0);

      const languageResult = await db.execute(
        `SELECT language, COUNT(*) as count FROM documents
         WHERE project_id = ? AND is_code = 1
         GROUP BY language`,
        [projectId],
      );

      const languageBreakdown: Record<CodeLanguage, number> = {} as Record<CodeLanguage, number>;
      for (const row of languageResult.rows) {
        const lang = String(row['language']) as CodeLanguage;
        languageBreakdown[lang] = Number(row['count']);
      }

      const avgChunks = state.indexedFiles > 0 ? totalChunks / state.indexedFiles : 0;

      let healthScore = 100;
      if (state.indexedFiles === 0) healthScore = 0;
      else if (avgChunks < 1) healthScore -= 20;
      if (totalBytes === 0) healthScore -= 30;

      return {
        projectId,
        totalFiles: state.indexedFiles,
        totalChunks,
        totalBytes,
        totalTokens,
        languageBreakdown,
        lastIndexedAt: state.lastIndexedAt,
        averageChunksPerFile: avgChunks,
        indexHealthScore: Math.max(0, healthScore),
      };
    },

    async saveCheckpoint(
      projectId: string,
      checkpoint: Omit<IndexCheckpoint, 'id' | 'createdAt' | 'updatedAt'>,
    ): Promise<string> {
      const db = await getDatabase();
      const id = crypto.randomUUID();
      const now = Date.now();

      await db.execute(
        `DELETE FROM index_checkpoints WHERE project_id = ?`,
        [projectId],
      );

      await db.execute(
        `INSERT INTO index_checkpoints (id, project_id, phase, processed_files_json, pending_files_json, progress_json, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          id,
          projectId,
          checkpoint.phase,
          JSON.stringify(checkpoint.processedFiles),
          JSON.stringify(checkpoint.pendingFiles),
          JSON.stringify(checkpoint.progress),
          now,
          now,
        ],
      );

      log.debug('codeindex', 'Checkpoint saved', { projectId, processedFiles: checkpoint.processedFiles.length });
      return id;
    },

    async loadCheckpoint(projectId: string): Promise<IndexCheckpoint | null> {
      const db = await getDatabase();

      const result = await db.execute(
        `SELECT * FROM index_checkpoints WHERE project_id = ? ORDER BY created_at DESC LIMIT 1`,
        [projectId],
      );

      if (result.rows.length === 0) return null;
      const row = result.rows[0];
      if (!row) return null;

      let processedFiles: string[] = [];
      let pendingFiles: string[] = [];
      let progress: IndexProgress = {
        phase: 'scanning',
        scannedFiles: 0,
        indexedFiles: 0,
        totalFiles: 0,
        errors: [],
      };

      try {
        const processedJson = row['processed_files_json'];
        if (typeof processedJson === 'string') {
          processedFiles = JSON.parse(processedJson) as string[];
        }
      } catch {}

      try {
        const pendingJson = row['pending_files_json'];
        if (typeof pendingJson === 'string') {
          pendingFiles = JSON.parse(pendingJson) as string[];
        }
      } catch {}

      try {
        const progressJson = row['progress_json'];
        if (typeof progressJson === 'string') {
          progress = JSON.parse(progressJson) as IndexProgress;
        }
      } catch {}

      return {
        id: String(row['id']),
        projectId: String(row['project_id']),
        phase: String(row['phase']) as IndexProgress['phase'],
        processedFiles,
        pendingFiles,
        progress,
        createdAt: Number(row['created_at']),
        updatedAt: Number(row['updated_at']),
      };
    },

    async clearCheckpoint(projectId: string): Promise<void> {
      const db = await getDatabase();
      await db.execute(`DELETE FROM index_checkpoints WHERE project_id = ?`, [projectId]);
      log.debug('codeindex', 'Checkpoint cleared', { projectId });
    },
  };
}

function parseVector(blob: unknown): number[] {
  if (blob instanceof Uint8Array || blob instanceof ArrayBuffer) {
    const buffer = blob instanceof ArrayBuffer ? blob : blob.buffer;
    return Array.from(new Float32Array(buffer));
  }

  if (typeof blob === 'string') {
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

function getLanguageFromPath(filePath: string): CodeLanguage {
  const ext = filePath.substring(filePath.lastIndexOf('.')).toLowerCase();
  const LANGUAGE_EXTENSIONS: Record<string, CodeLanguage> = {
    '.ts': 'ts',
    '.tsx': 'tsx',
    '.js': 'js',
    '.jsx': 'jsx',
    '.py': 'py',
    '.rs': 'rs',
    '.go': 'go',
    '.java': 'java',
  };
  return LANGUAGE_EXTENSIONS[ext] ?? 'unknown';
}

export { getProjectHash };
