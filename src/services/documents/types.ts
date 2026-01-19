export type DocumentSourceType = "txt" | "md" | "url";

export type Document = {
  id: string;
  projectId: string;
  sourcePath?: string;
  sourceUrl?: string;
  sourceType: DocumentSourceType;
  title?: string;
  fullContent: string;
  checksum: string;
  createdAt: number;
  updatedAt: number;
};

export type DocumentChunk = {
  id: string;
  documentId: string;
  chunkIndex: number;
  content: string;
  startOffset: number;
  endOffset: number;
  tokensEstimate: number;
};

export type IngestOptions = {
  projectId: string;
  path?: string;
  url?: string;
  content?: string;
  title?: string;
  sourceType?: DocumentSourceType;
};
