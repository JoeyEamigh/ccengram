import { log } from './log.js';

export type RetryConfig = {
  maxRetries?: number;
  baseDelay?: number;
  timeout?: number;
  onRetry?: (attempt: number, error: Error) => void;
};

const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_BASE_DELAY = 500;
const DEFAULT_TIMEOUT = 30000;

function isRetriableError(error: unknown): boolean {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    if (
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('econnreset') ||
      message.includes('econnrefused') ||
      message.includes('etimedout') ||
      message.includes('socket') ||
      message.includes('abort')
    ) {
      return true;
    }
  }
  return false;
}

function isRetriableStatus(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

function calculateBackoff(attempt: number, baseDelay: number, retryAfter?: number): number {
  if (retryAfter !== undefined && retryAfter > 0) {
    return retryAfter * 1000;
  }
  const exponentialDelay = baseDelay * Math.pow(2, attempt);
  const jitter = Math.random() * 0.2 * exponentialDelay;
  return exponentialDelay + jitter;
}

function parseRetryAfter(response: Response): number | undefined {
  const header = response.headers.get('Retry-After');
  if (!header) return undefined;

  const seconds = parseInt(header, 10);
  if (!Number.isNaN(seconds)) {
    return seconds;
  }

  const date = Date.parse(header);
  if (!Number.isNaN(date)) {
    const delayMs = date - Date.now();
    return Math.max(0, Math.ceil(delayMs / 1000));
  }

  return undefined;
}

export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  config: RetryConfig = {},
): Promise<Response> {
  const maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES;
  const baseDelay = config.baseDelay ?? DEFAULT_BASE_DELAY;
  const timeout = config.timeout ?? DEFAULT_TIMEOUT;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        return response;
      }

      if (isRetriableStatus(response.status) && attempt < maxRetries) {
        const retryAfter = parseRetryAfter(response);
        const delay = calculateBackoff(attempt, baseDelay, retryAfter);

        log.debug('fetch', 'Retrying after status', {
          url,
          status: response.status,
          attempt: attempt + 1,
          delayMs: Math.round(delay),
        });

        config.onRetry?.(attempt + 1, new Error(`HTTP ${response.status}`));
        await sleep(delay);
        continue;
      }

      return response;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (lastError.name === 'AbortError') {
        lastError = new Error(`Request timeout after ${timeout}ms`);
      }

      if (isRetriableError(lastError) && attempt < maxRetries) {
        const delay = calculateBackoff(attempt, baseDelay);

        log.debug('fetch', 'Retrying after error', {
          url,
          error: lastError.message,
          attempt: attempt + 1,
          delayMs: Math.round(delay),
        });

        config.onRetry?.(attempt + 1, lastError);
        await sleep(delay);
        continue;
      }

      throw lastError;
    }
  }

  throw lastError ?? new Error('Max retries exceeded');
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message = 'Operation timed out',
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => reject(new Error(message)), ms);

    promise
      .then(result => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch(error => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}
