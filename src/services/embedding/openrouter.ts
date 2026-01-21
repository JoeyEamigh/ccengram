import { fetchWithRetry } from '../../utils/fetch-resilient.js';
import { log } from '../../utils/log.js';
import type { EmbeddingProvider, OpenRouterConfig } from './types.js';

const OPENROUTER_TIMEOUT = 30000;

const MODEL_DIMENSIONS: Record<string, number> = {
  'openai/text-embedding-3-small': 1536,
  'openai/text-embedding-3-large': 3072,
  'openai/text-embedding-ada-002': 1536,
};

type OpenRouterEmbeddingData = {
  embedding: number[];
  index: number;
};

type OpenRouterEmbeddingResponse = {
  data: OpenRouterEmbeddingData[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
};

export class OpenRouterProvider implements EmbeddingProvider {
  readonly name = 'openrouter';
  private apiKey: string;
  readonly model: string;
  readonly dimensions: number;

  constructor(config: OpenRouterConfig) {
    this.apiKey = config.apiKey ?? process.env['OPENROUTER_API_KEY'] ?? '';
    this.model = config.model;
    this.dimensions = MODEL_DIMENSIONS[this.model] ?? 1536;
  }

  async isAvailable(): Promise<boolean> {
    if (!this.apiKey) {
      log.warn('embedding', 'OpenRouter API key not configured');
      return false;
    }

    try {
      log.debug('embedding', 'Checking OpenRouter availability');
      const response = await fetchWithRetry(
        'https://openrouter.ai/api/v1/models',
        { headers: { Authorization: `Bearer ${this.apiKey}` } },
        { timeout: 5000, maxRetries: 1 },
      );
      if (response.ok) {
        log.info('embedding', 'OpenRouter provider ready', { model: this.model });
      }
      return response.ok;
    } catch (e) {
      const err = e as Error;
      log.debug('embedding', 'OpenRouter check failed', { error: err.message });
      return false;
    }
  }

  async embed(text: string): Promise<number[]> {
    const start = Date.now();
    const response = await fetchWithRetry(
      'https://openrouter.ai/api/v1/embeddings',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
          'HTTP-Referer': 'https://github.com/user/ccmemory',
          'X-Title': 'CCMemory',
        },
        body: JSON.stringify({
          model: this.model,
          input: text,
        }),
      },
      {
        timeout: OPENROUTER_TIMEOUT,
        onRetry: (attempt, error) => {
          log.warn('embedding', 'OpenRouter request retry', {
            attempt,
            error: error.message,
            model: this.model,
          });
        },
      },
    );

    if (!response.ok) {
      log.error('embedding', 'OpenRouter embed failed', { status: response.statusText });
      throw new Error(`OpenRouter embed failed: ${response.statusText}`);
    }

    const data = (await response.json()) as OpenRouterEmbeddingResponse;
    log.debug('embedding', 'OpenRouter embedded', { length: text.length, ms: Date.now() - start });

    const first = data.data[0];
    if (!first) {
      throw new Error('OpenRouter returned empty embedding response');
    }
    return first.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const start = Date.now();
    log.debug('embedding', 'OpenRouter batch embedding', { count: texts.length });

    try {
      const response = await fetchWithRetry(
        'https://openrouter.ai/api/v1/embeddings',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
            'HTTP-Referer': 'https://github.com/user/ccmemory',
            'X-Title': 'CCMemory',
          },
          body: JSON.stringify({
            model: this.model,
            input: texts,
          }),
        },
        {
          timeout: OPENROUTER_TIMEOUT,
          onRetry: (attempt, error) => {
            log.warn('embedding', 'OpenRouter batch request retry', {
              attempt,
              error: error.message,
              model: this.model,
            });
          },
        },
      );

      if (!response.ok) {
        throw new Error(`OpenRouter embed batch failed: ${response.statusText}`);
      }

      const data = (await response.json()) as OpenRouterEmbeddingResponse;
      log.info('embedding', 'OpenRouter batch complete', { count: texts.length, ms: Date.now() - start });

      return data.data.map(d => d.embedding);
    } catch (batchError) {
      log.warn('embedding', 'OpenRouter batch failed, falling back to individual embeds', {
        error: batchError instanceof Error ? batchError.message : String(batchError),
        count: texts.length,
      });

      const settled = await Promise.allSettled(texts.map(t => this.embed(t)));
      const results: number[][] = [];
      let failures = 0;

      for (let i = 0; i < settled.length; i++) {
        const result = settled[i];
        if (result && result.status === 'fulfilled') {
          results.push(result.value);
        } else if (result && result.status === 'rejected') {
          failures++;
          const reason = result.reason;
          log.warn('embedding', 'Individual embed failed', {
            index: i,
            error: reason instanceof Error ? reason.message : String(reason),
          });
          results.push([]);
        }
      }

      if (failures > 0) {
        log.warn('embedding', 'Fallback completed with failures', {
          total: texts.length,
          failures,
          succeeded: texts.length - failures,
          ms: Date.now() - start,
        });
      } else {
        log.info('embedding', 'Fallback complete', { count: texts.length, ms: Date.now() - start });
      }

      return results;
    }
  }
}
