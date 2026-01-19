import { describe, expect, test } from "bun:test";
import { chunkText, estimateTokens } from "../chunk.js";

describe("Text Chunking", () => {
  test("returns single chunk for small text", () => {
    const text = "Short text that fits in one chunk.";
    const chunks = chunkText(text);
    expect(chunks.length).toBe(1);
    expect(chunks[0]?.content).toBe(text);
  });

  test("splits long text into multiple chunks", () => {
    const text = "A".repeat(5000);
    const chunks = chunkText(text);
    expect(chunks.length).toBeGreaterThan(1);
  });

  test("chunks have overlap", () => {
    const text =
      "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.".repeat(
        50
      );
    const chunks = chunkText(text, { targetTokens: 100, overlap: 0.2 });

    expect(chunks.length).toBeGreaterThan(1);

    for (let i = 0; i < chunks.length - 1; i++) {
      const currentChunk = chunks[i];
      const nextChunk = chunks[i + 1];
      if (!currentChunk || !nextChunk) continue;

      const endOfCurrent = currentChunk.content.slice(-50);
      const startOfNext = nextChunk.content.slice(0, 100);

      const hasOverlap =
        startOfNext.includes(endOfCurrent.slice(-20)) ||
        endOfCurrent.includes(startOfNext.slice(0, 20));
      expect(hasOverlap).toBe(true);
    }
  });

  test("respects minimum chunk size", () => {
    const text = "Word. ".repeat(100);
    const chunks = chunkText(text, { minChunkSize: 50 });
    expect(chunks.every((c) => c.content.length >= 50)).toBe(true);
  });

  test("tracks offsets correctly", () => {
    const text = "First part. Second part. Third part.";
    const chunks = chunkText(text, { targetTokens: 5 });

    for (const chunk of chunks) {
      expect(chunk.startOffset).toBeGreaterThanOrEqual(0);
      expect(chunk.endOffset).toBeLessThanOrEqual(text.length + 50);
      expect(chunk.endOffset).toBeGreaterThan(chunk.startOffset);
    }
  });

  test("estimates tokens correctly", () => {
    const text = "Word ".repeat(100);
    const chunks = chunkText(text);
    const totalTokens = chunks.reduce((sum, c) => sum + c.tokensEstimate, 0);
    expect(totalTokens).toBeGreaterThanOrEqual(80);
    expect(totalTokens).toBeLessThanOrEqual(200);
  });

  test("handles paragraph breaks", () => {
    const text =
      "First paragraph with some text.\n\nSecond paragraph with more text.\n\nThird paragraph.";
    const chunks = chunkText(text);
    expect(chunks.length).toBeGreaterThanOrEqual(1);
    expect(chunks[0]?.content).toContain("paragraph");
  });

  test("handles empty text", () => {
    const chunks = chunkText("");
    expect(chunks.length).toBe(1);
    expect(chunks[0]?.content).toBe("");
    expect(chunks[0]?.tokensEstimate).toBe(0);
  });

  test("handles text with no sentence boundaries", () => {
    const text = "no punctuation here just words without any periods or stops";
    const chunks = chunkText(text);
    expect(chunks.length).toBe(1);
    expect(chunks[0]?.content).toBe(text);
  });

  test("respects custom target tokens", () => {
    const text = "Sentence. ".repeat(200);
    const smallChunks = chunkText(text, { targetTokens: 50 });
    const largeChunks = chunkText(text, { targetTokens: 500 });

    expect(smallChunks.length).toBeGreaterThan(largeChunks.length);
  });

  test("overlap ratio affects chunk boundaries", () => {
    const text =
      "The quick brown fox jumps over the lazy dog. ".repeat(100);
    const noOverlap = chunkText(text, { targetTokens: 100, overlap: 0 });
    const highOverlap = chunkText(text, { targetTokens: 100, overlap: 0.3 });

    expect(noOverlap.length).not.toBe(highOverlap.length);
  });
});

describe("Token Estimation", () => {
  test("estimates approximately 4 chars per token", () => {
    const text = "word";
    expect(estimateTokens(text)).toBe(1);
  });

  test("handles longer text", () => {
    const text = "a".repeat(100);
    expect(estimateTokens(text)).toBe(25);
  });

  test("handles empty string", () => {
    expect(estimateTokens("")).toBe(0);
  });
});
