import { closeDatabase } from '../db/database.js';
import { readLockFile } from '../services/codeindex/coordination.js';
import { createCodeIndexService } from '../services/codeindex/index.js';
import { createEmbeddingServiceOptional } from '../services/embedding/index.js';
import {
  addCommand,
  addCompletedTask,
  addError,
  addFileModified,
  addFileRead,
  addSearch,
  createExtractionService,
  getAccumulator,
  getOrCreateAccumulator,
  incrementToolCallCount,
} from '../services/extraction/index.js';
import { createSessionService, getOrCreateSession } from '../services/memory/sessions.js';
import { getOrCreateProject } from '../services/project.js';
import { log } from '../utils/log.js';
import { registerClient, unregisterClient } from '../webui/coordination.js';

type UserPromptInput = {
  session_id: string;
  cwd: string;
  prompt: string;
};

type PostToolInput = {
  session_id: string;
  cwd: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: unknown;
};

type PreCompactInput = {
  session_id: string;
  cwd: string;
  trigger?: string;
};

type StopInput = {
  session_id: string;
  cwd: string;
  transcript_path?: string;
};

type SessionStartInput = {
  session_id: string;
  cwd: string;
  source?: string;
};

type SessionEndInput = {
  session_id: string;
};

function parseInput<T>(text: string, requiredFields: string[]): T | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (typeof parsed !== 'object' || parsed === null) return null;
    const obj = parsed as Record<string, unknown>;
    for (const field of requiredFields) {
      if (typeof obj[field] !== 'string') return null;
    }
    return obj as unknown as T;
  } catch {
    return null;
  }
}

const TIMEOUT_MS = 30000;

async function registerSessionClient(sessionId: string): Promise<void> {
  try {
    await registerClient(sessionId);
    log.debug('hooks', 'Session registered', { sessionId });
  } catch (err) {
    log.debug('hooks', 'Session registration skipped', {
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

function spawnBackgroundExtraction(
  sessionId: string,
  cwd: string,
  trigger: string,
  transcriptPath?: string,
): void {
  const binaryPath = process.argv[0] ?? 'ccmemory';
  const args = ['extract-bg', sessionId, '--trigger', trigger, '--cwd', cwd];
  if (transcriptPath) {
    args.push('--transcript', transcriptPath);
  }

  const cleanEnv = Object.fromEntries(
    Object.entries(process.env).filter(([k]) => !k.startsWith('CLAUDE'))
  );

  const proc = Bun.spawn([binaryPath, ...args], {
    stdout: 'ignore',
    stderr: 'ignore',
    stdin: 'ignore',
    env: cleanEnv,
  });
  proc.unref();

  log.debug('hooks', 'Background extraction spawned', { sessionId, trigger });
}

export async function userPromptHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn('hooks', 'User prompt hook timed out');
    process.exit(0);
  }, TIMEOUT_MS);

  try {
    const inputText = await Bun.stdin.text();
    const input = parseInput<UserPromptInput>(inputText, ['session_id', 'cwd', 'prompt']);

    if (!input) {
      log.warn('hooks', 'Invalid user prompt hook input');
      process.exit(0);
    }

    const { session_id, cwd, prompt } = input;
    log.debug('hooks', 'Processing user prompt', { session_id });

    await registerSessionClient(session_id);

    const project = await getOrCreateProject(cwd);
    await getOrCreateSession(session_id, project.id);

    const accumulator = await getAccumulator(session_id);

    if (accumulator && accumulator.toolCallCount > 0) {
      log.info('hooks', 'Spawning background extraction for previous segment', {
        session_id,
        toolCallCount: accumulator.toolCallCount,
      });

      spawnBackgroundExtraction(session_id, cwd, 'user_prompt');
    }

    const embeddingService = await createEmbeddingServiceOptional();
    const extractionService = createExtractionService(embeddingService);

    const signal = await extractionService.classifySignal(prompt);

    const userPrompt = {
      content: prompt,
      timestamp: Date.now(),
      signal: signal ?? undefined,
    };

    await extractionService.startSegment(session_id, project.id, userPrompt);

    log.debug('hooks', 'User prompt processed, new segment started', { session_id });
  } finally {
    clearTimeout(timeoutId);
    closeDatabase();
  }
  process.exit(0);
}

export async function postToolHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn('hooks', 'Post tool hook timed out');
    process.exit(0);
  }, 10000);

  const inputText = await Bun.stdin.text();
  const input = parseInput<PostToolInput>(inputText, ['session_id', 'cwd', 'tool_name']);

  if (!input) {
    log.warn('hooks', 'Invalid post tool hook input');
    clearTimeout(timeoutId);
    process.exit(0);
  }

  const { session_id, cwd, tool_name, tool_input, tool_response } = input;
  log.debug('hooks', 'Processing tool observation', { session_id, tool_name });

  const project = await getOrCreateProject(cwd);
  await getOrCreateAccumulator(session_id, project.id);

  switch (tool_name) {
    case 'Read':
      if (typeof tool_input['file_path'] === 'string') {
        await addFileRead(session_id, tool_input['file_path']);
      }
      break;

    case 'Write':
    case 'Edit':
      if (typeof tool_input['file_path'] === 'string') {
        await addFileModified(session_id, tool_input['file_path']);
      }
      break;

    case 'Bash': {
      const command = String(tool_input['command'] ?? '').slice(0, 200);
      const responseObj = typeof tool_response === 'object' && tool_response !== null
        ? tool_response as Record<string, unknown>
        : {};
      const exitCode = typeof responseObj['exitCode'] === 'number' ? responseObj['exitCode'] : undefined;
      const stderr = typeof responseObj['stderr'] === 'string' ? responseObj['stderr'] : '';

      await addCommand(session_id, {
        command,
        exitCode,
        hasError: exitCode !== 0 && exitCode !== undefined,
      });

      if (stderr) {
        await addError(session_id, {
          source: 'Bash',
          message: stderr.slice(0, 500),
        });
      }
      break;
    }

    case 'Grep':
    case 'Glob': {
      const pattern = String(tool_input['pattern'] ?? '');
      const resultCount = Array.isArray(tool_response) ? tool_response.length : 0;
      await addSearch(session_id, {
        tool: tool_name as 'Grep' | 'Glob',
        pattern,
        resultCount,
      });
      break;
    }

    case 'TodoWrite': {
      const todos = tool_input['todos'] as Array<{ content?: string; status?: string }> | undefined;
      if (Array.isArray(todos)) {
        const now = Date.now();
        let newCompletedCount = 0;
        for (const todo of todos) {
          if (todo.status === 'completed' && typeof todo.content === 'string') {
            await addCompletedTask(session_id, {
              content: todo.content,
              timestamp: now,
            });
            newCompletedCount++;
          }
        }

        const accumulator = await getAccumulator(session_id);
        if (accumulator && newCompletedCount > 0) {
          const totalCompleted = accumulator.completedTasks.length;
          const toolCalls = accumulator.toolCallCount;

          if (totalCompleted >= 3 && toolCalls >= 5) {
            log.info('hooks', 'Spawning background extraction for todo completion', {
              session_id,
              completedTasks: totalCompleted,
              toolCalls,
            });
            spawnBackgroundExtraction(session_id, cwd, 'todo_completion');
          }
        }
      }
      await incrementToolCallCount(session_id);
      break;
    }

    default:
      await incrementToolCallCount(session_id);
  }

  clearTimeout(timeoutId);
  closeDatabase();
  process.exit(0);
}

export async function preCompactHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn('hooks', 'Pre-compact hook timed out');
    closeDatabase();
    process.exit(0);
  }, 10000);

  const inputText = await Bun.stdin.text();
  const input = parseInput<PreCompactInput>(inputText, ['session_id', 'cwd']);

  if (!input) {
    log.warn('hooks', 'Invalid pre-compact hook input');
    clearTimeout(timeoutId);
    process.exit(0);
  }

  const { session_id, cwd } = input;
  log.info('hooks', 'Processing pre-compact hook', { session_id });

  const project = await getOrCreateProject(cwd);
  await getOrCreateSession(session_id, project.id);

  const accumulator = await getAccumulator(session_id);

  if (accumulator && accumulator.toolCallCount > 0) {
    log.info('hooks', 'Spawning background extraction', { session_id });
    spawnBackgroundExtraction(session_id, cwd, 'pre_compact');
  }

  clearTimeout(timeoutId);
  closeDatabase();
  process.exit(0);
}

export async function stopHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn('hooks', 'Stop hook timed out');
    closeDatabase();
    process.exit(0);
  }, 10000);

  const inputText = await Bun.stdin.text();
  const input = parseInput<StopInput>(inputText, ['session_id', 'cwd']);

  if (!input) {
    log.warn('hooks', 'Invalid stop hook input');
    clearTimeout(timeoutId);
    process.exit(0);
  }

  const { session_id, cwd, transcript_path } = input;
  log.info('hooks', 'Processing stop hook', { session_id });

  const project = await getOrCreateProject(cwd);
  await getOrCreateSession(session_id, project.id);

  const accumulator = await getAccumulator(session_id);

  if (accumulator && accumulator.toolCallCount > 0) {
    log.info('hooks', 'Spawning background extraction', { session_id });
    spawnBackgroundExtraction(session_id, cwd, 'stop', transcript_path);
  }

  clearTimeout(timeoutId);
  closeDatabase();
  process.exit(0);
}

export async function sessionStartHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn('hooks', 'Session start hook timed out');
    process.exit(0);
  }, 10000);

  const inputText = await Bun.stdin.text();
  const input = parseInput<SessionStartInput>(inputText, ['session_id', 'cwd']);

  if (!input) {
    log.warn('hooks', 'Invalid session start hook input');
    clearTimeout(timeoutId);
    process.exit(0);
  }

  const { session_id, cwd } = input;
  log.info('hooks', 'Initializing session', { session_id });

  await registerSessionClient(session_id);

  const project = await getOrCreateProject(cwd);
  await getOrCreateSession(session_id, project.id);

  const sessionService = createSessionService();
  const cleaned = await sessionService.cleanupStaleSessions();
  if (cleaned > 0) {
    log.debug('hooks', 'Cleaned up stale sessions', { count: cleaned });
  }

  await maybeAutoStartWatcher(cwd, project.id);

  log.debug('hooks', 'Session initialized (no context injection)', { session_id });

  clearTimeout(timeoutId);
  closeDatabase();
  process.exit(0);
}

async function maybeAutoStartWatcher(projectPath: string, projectId: string): Promise<void> {
  try {
    const existingWatcher = await readLockFile(projectPath);
    if (existingWatcher) {
      log.debug('hooks', 'Watcher already running', { projectPath, pid: existingWatcher.pid });
      return;
    }

    const embeddingService = await createEmbeddingServiceOptional();
    if (!embeddingService) {
      log.debug('hooks', 'No embedding service available, skipping watcher auto-start');
      return;
    }

    const codeIndex = createCodeIndexService(embeddingService);
    const state = await codeIndex.getState(projectId);

    if (!state) {
      log.debug('hooks', 'No code index exists for project, skipping watcher auto-start');
      return;
    }

    log.info('hooks', 'Auto-starting code index watcher as background process', { projectPath });

    const ccmemoryPath = process.argv[1] ?? 'ccmemory';
    const child = Bun.spawn([ccmemoryPath, 'watch', projectPath], {
      stdio: ['ignore', 'ignore', 'ignore'],
      env: process.env,
    });

    child.unref();

    log.info('hooks', 'Spawned watcher background process', { projectPath, pid: child.pid });
  } catch (err) {
    log.error('hooks', 'Error during watcher auto-start', { error: (err as Error).message });
  }
}

export async function sessionEndHook(): Promise<void> {
  const timeoutId = setTimeout(() => {
    log.warn('hooks', 'Session end hook timed out');
    closeDatabase();
    process.exit(0);
  }, 10000);

  const inputText = await Bun.stdin.text();
  const input = parseInput<SessionEndInput>(inputText, ['session_id']);

  if (!input) {
    log.warn('hooks', 'Invalid session end hook input');
    clearTimeout(timeoutId);
    process.exit(0);
  }

  const { session_id } = input;
  log.info('hooks', 'Session ending', { session_id });

  const sessionService = createSessionService();
  try {
    await sessionService.end(session_id);
    log.debug('hooks', 'Session marked as ended in database', { session_id });
  } catch (err) {
    log.debug('hooks', 'Could not mark session as ended (may not exist)', {
      session_id,
      error: err instanceof Error ? err.message : String(err),
    });
  }

  closeDatabase();
  await unregisterClient(session_id);

  clearTimeout(timeoutId);
  process.exit(0);
}
