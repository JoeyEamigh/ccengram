export type CodeLanguage =
  | 'ts'
  | 'tsx'
  | 'js'
  | 'jsx'
  | 'py'
  | 'rs'
  | 'go'
  | 'java'
  | 'c'
  | 'cpp'
  | 'h'
  | 'hpp'
  | 'cs'
  | 'rb'
  | 'php'
  | 'swift'
  | 'kt'
  | 'scala'
  | 'sh'
  | 'bash'
  | 'zsh'
  | 'sql'
  | 'json'
  | 'yaml'
  | 'yml'
  | 'toml'
  | 'md'
  | 'css'
  | 'scss'
  | 'html'
  | 'vue'
  | 'svelte'
  | 'unknown';

export type ChunkType = 'function' | 'class' | 'imports' | 'block' | 'module';

export type CodeChunk = {
  content: string;
  startLine: number;
  endLine: number;
  chunkType: ChunkType;
  symbols: string[];
  tokensEstimate: number;
};

export type IndexedFile = {
  id: string;
  projectId: string;
  path: string;
  checksum: string;
  mtime: number;
  indexedAt: number;
};

export type CodeIndexState = {
  projectId: string;
  lastIndexedAt: number;
  indexedFiles: number;
  gitignoreHash: string | null;
};

export type CodeSearchResult = {
  documentId: string;
  chunkId: string;
  path: string;
  content: string;
  startLine: number;
  endLine: number;
  language: CodeLanguage;
  chunkType: ChunkType;
  symbols: string[];
  score: number;
};

export type ScanResult = {
  files: ScannedFile[];
  skippedCount: number;
  totalSize: number;
};

export type ScannedFile = {
  path: string;
  relativePath: string;
  size: number;
  mtime: number;
  language: CodeLanguage;
};

export type IndexProgress = {
  phase: 'scanning' | 'indexing' | 'complete' | 'paused';
  scannedFiles: number;
  indexedFiles: number;
  totalFiles: number;
  currentFile?: string;
  errors: string[];
  startedAt?: number;
  bytesProcessed?: number;
  totalBytes?: number;
  tokensProcessed?: number;
  filesPerSecond?: number;
  estimatedTimeRemaining?: number;
};

export type WatcherEvent = {
  type: 'add' | 'change' | 'delete';
  path: string;
  timestamp: number;
};

export type WatcherStatus = {
  projectId: string;
  projectPath: string;
  pid: number;
  startedAt: number;
  lastActivity: number;
  indexedFiles: number;
};

export type CodeIndexOptions = {
  force?: boolean;
  dryRun?: boolean;
  onProgress?: (progress: IndexProgress) => void;
  includePaths?: string[];
  excludePaths?: string[];
  resumeFromCheckpoint?: boolean;
};

export type IndexStatistics = {
  projectId: string;
  totalFiles: number;
  totalChunks: number;
  totalBytes: number;
  totalTokens: number;
  languageBreakdown: Record<CodeLanguage, number>;
  lastIndexedAt: number;
  averageChunksPerFile: number;
  indexHealthScore: number;
};

export type IndexCheckpoint = {
  id: string;
  projectId: string;
  phase: IndexProgress['phase'];
  processedFiles: string[];
  pendingFiles: string[];
  progress: IndexProgress;
  createdAt: number;
  updatedAt: number;
};

export type CodeSearchOptions = {
  query: string;
  projectId: string;
  language?: CodeLanguage;
  limit?: number;
};

export const LANGUAGE_EXTENSIONS: Record<string, CodeLanguage> = {
  '.ts': 'ts',
  '.tsx': 'tsx',
  '.js': 'js',
  '.jsx': 'jsx',
  '.mjs': 'js',
  '.cjs': 'js',
  '.py': 'py',
  '.pyw': 'py',
  '.rs': 'rs',
  '.go': 'go',
  '.java': 'java',
  '.c': 'c',
  '.cpp': 'cpp',
  '.cc': 'cpp',
  '.cxx': 'cpp',
  '.h': 'h',
  '.hpp': 'hpp',
  '.hxx': 'hpp',
  '.cs': 'cs',
  '.rb': 'rb',
  '.php': 'php',
  '.swift': 'swift',
  '.kt': 'kt',
  '.kts': 'kt',
  '.scala': 'scala',
  '.sc': 'scala',
  '.sh': 'sh',
  '.bash': 'bash',
  '.zsh': 'zsh',
  '.sql': 'sql',
  '.json': 'json',
  '.yaml': 'yaml',
  '.yml': 'yml',
  '.toml': 'toml',
  '.md': 'md',
  '.css': 'css',
  '.scss': 'scss',
  '.sass': 'scss',
  '.less': 'css',
  '.html': 'html',
  '.htm': 'html',
  '.vue': 'vue',
  '.svelte': 'svelte',
};

export type CodeIndexExport = {
  version: number;
  exportedAt: number;
  projectPath: string;
  state: CodeIndexState;
  files: IndexedFileExport[];
};

export type IndexedFileExport = {
  relativePath: string;
  language: CodeLanguage;
  lineCount: number;
  checksum: string;
  chunks: ChunkExport[];
};

export type ChunkExport = {
  content: string;
  startLine: number;
  endLine: number;
  chunkType: ChunkType;
  symbols: string[];
  vector: number[];
};

export const DEFAULT_IGNORE_PATTERNS = [
  'node_modules',
  '.git',
  'dist',
  'build',
  'out',
  '.next',
  '.nuxt',
  '.output',
  'coverage',
  '.nyc_output',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  'venv',
  '.venv',
  'env',
  '.env',
  'target',
  'vendor',
  '.idea',
  '.vscode',
  '*.min.js',
  '*.min.css',
  '*.bundle.js',
  '*.map',
  'package-lock.json',
  'yarn.lock',
  'pnpm-lock.yaml',
  'bun.lockb',
  'Cargo.lock',
  'Gemfile.lock',
  'poetry.lock',
  '*.log',
  '*.tmp',
  '*.temp',
  '.DS_Store',
  'Thumbs.db',
];
