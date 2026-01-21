import { afterEach, describe, expect, mock, test } from 'bun:test';
import { fetchWithRetry, withTimeout } from '../fetch-resilient.js';

function mockFetch(impl: (url: string, options?: RequestInit) => Promise<Response>): void {
  globalThis.fetch = mock(impl) as unknown as typeof fetch;
}

describe('fetchWithRetry', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test('succeeds on first attempt for OK response', async () => {
    mockFetch(() => Promise.resolve(new Response('ok', { status: 200 })));

    const response = await fetchWithRetry('https://example.com');
    expect(response.status).toBe(200);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  test('returns non-retriable error responses without retry', async () => {
    mockFetch(() => Promise.resolve(new Response('not found', { status: 404 })));

    const response = await fetchWithRetry('https://example.com');
    expect(response.status).toBe(404);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  test('retries on 503 status', async () => {
    let callCount = 0;
    mockFetch(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.resolve(new Response('unavailable', { status: 503 }));
      }
      return Promise.resolve(new Response('ok', { status: 200 }));
    });

    const response = await fetchWithRetry('https://example.com', {}, { baseDelay: 10 });
    expect(response.status).toBe(200);
    expect(callCount).toBe(3);
  });

  test('retries on 429 status with Retry-After header', async () => {
    let callCount = 0;
    mockFetch(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve(
          new Response('rate limited', {
            status: 429,
            headers: { 'Retry-After': '1' },
          }),
        );
      }
      return Promise.resolve(new Response('ok', { status: 200 }));
    });

    const start = Date.now();
    const response = await fetchWithRetry('https://example.com', {}, { baseDelay: 10 });
    const elapsed = Date.now() - start;

    expect(response.status).toBe(200);
    expect(callCount).toBe(2);
    expect(elapsed).toBeGreaterThanOrEqual(900);
  });

  test('retries on network error', async () => {
    let callCount = 0;
    mockFetch(() => {
      callCount++;
      if (callCount < 2) {
        return Promise.reject(new Error('Network error'));
      }
      return Promise.resolve(new Response('ok', { status: 200 }));
    });

    const response = await fetchWithRetry('https://example.com', {}, { baseDelay: 10 });
    expect(response.status).toBe(200);
    expect(callCount).toBe(2);
  });

  test('throws after max retries exceeded', async () => {
    mockFetch(() => Promise.reject(new Error('Network error')));

    await expect(
      fetchWithRetry('https://example.com', {}, { maxRetries: 2, baseDelay: 10 }),
    ).rejects.toThrow('Network error');

    expect(globalThis.fetch).toHaveBeenCalledTimes(3);
  });

  test('calls onRetry callback', async () => {
    let callCount = 0;
    mockFetch(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.reject(new Error('Network error'));
      }
      return Promise.resolve(new Response('ok', { status: 200 }));
    });

    const retries: number[] = [];
    await fetchWithRetry(
      'https://example.com',
      {},
      {
        baseDelay: 10,
        onRetry: attempt => retries.push(attempt),
      },
    );

    expect(retries).toEqual([1, 2]);
  });

  test('respects timeout configuration', async () => {
    mockFetch((_url, options) => {
      return new Promise((resolve, reject) => {
        const signal = options?.signal;
        if (signal) {
          signal.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        }
        setTimeout(() => resolve(new Response('ok')), 5000);
      });
    });

    await expect(
      fetchWithRetry('https://example.com', {}, { timeout: 100, maxRetries: 0 }),
    ).rejects.toThrow('timeout');
  });

  test('does not retry on non-retriable errors', async () => {
    mockFetch(() => Promise.reject(new Error('Invalid argument')));

    await expect(
      fetchWithRetry('https://example.com', {}, { maxRetries: 3, baseDelay: 10 }),
    ).rejects.toThrow('Invalid argument');

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  test('uses exponential backoff', async () => {
    let callCount = 0;
    const timestamps: number[] = [];
    mockFetch(() => {
      timestamps.push(Date.now());
      callCount++;
      if (callCount < 3) {
        return Promise.reject(new Error('ECONNRESET'));
      }
      return Promise.resolve(new Response('ok', { status: 200 }));
    });

    await fetchWithRetry('https://example.com', {}, { baseDelay: 50 });

    const delay1 = timestamps[1]! - timestamps[0]!;
    const delay2 = timestamps[2]! - timestamps[1]!;

    expect(delay1).toBeGreaterThanOrEqual(40);
    expect(delay2).toBeGreaterThan(delay1);
  });
});

describe('withTimeout', () => {
  test('resolves when promise completes before timeout', async () => {
    const result = await withTimeout(Promise.resolve('success'), 1000);
    expect(result).toBe('success');
  });

  test('rejects when promise times out', async () => {
    const slowPromise = new Promise(resolve => setTimeout(() => resolve('done'), 5000));

    await expect(withTimeout(slowPromise, 50)).rejects.toThrow('timed out');
  });

  test('uses custom timeout message', async () => {
    const slowPromise = new Promise(resolve => setTimeout(() => resolve('done'), 5000));

    await expect(withTimeout(slowPromise, 50, 'Custom timeout')).rejects.toThrow('Custom timeout');
  });

  test('propagates promise rejection', async () => {
    const failingPromise = Promise.reject(new Error('Original error'));

    await expect(withTimeout(failingPromise, 1000)).rejects.toThrow('Original error');
  });
});
