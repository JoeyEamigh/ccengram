export type ChunkOptions = {
  targetTokens?: number;
  overlap?: number;
  minChunkSize?: number;
};

export type Chunk = {
  content: string;
  startOffset: number;
  endOffset: number;
  tokensEstimate: number;
};

const CHARS_PER_TOKEN = 4;

function splitLongSegment(
  segment: string,
  targetChars: number,
  overlapChars: number,
  baseOffset: number
): Chunk[] {
  const chunks: Chunk[] = [];
  let pos = 0;

  while (pos < segment.length) {
    const end = Math.min(pos + targetChars, segment.length);
    const content = segment.slice(pos, end);

    chunks.push({
      content,
      startOffset: baseOffset + pos,
      endOffset: baseOffset + end,
      tokensEstimate: Math.ceil(content.length / CHARS_PER_TOKEN),
    });

    if (end >= segment.length) break;

    pos = end - overlapChars;
    if (pos <= chunks[chunks.length - 1]!.startOffset - baseOffset) {
      pos = end;
    }
  }

  return chunks;
}

export function chunkText(text: string, options: ChunkOptions = {}): Chunk[] {
  const { targetTokens = 768, overlap = 0.1, minChunkSize = 100 } = options;

  const totalTokens = Math.ceil(text.length / CHARS_PER_TOKEN);

  if (totalTokens <= targetTokens) {
    return [
      {
        content: text,
        startOffset: 0,
        endOffset: text.length,
        tokensEstimate: totalTokens,
      },
    ];
  }

  const targetChars = targetTokens * CHARS_PER_TOKEN;
  const overlapChars = Math.floor(targetChars * overlap);

  const paragraphs = text.split(/\n\n+/);
  const chunks: Chunk[] = [];

  let currentChunk = "";
  let chunkStart = 0;
  let currentPos = 0;

  for (let pIdx = 0; pIdx < paragraphs.length; pIdx++) {
    const para = paragraphs[pIdx];
    if (para === undefined) continue;

    const sentences = para.split(/(?<=[.!?])\s+/);

    for (const sentence of sentences) {
      if (sentence.length > targetChars) {
        if (currentChunk.length >= minChunkSize) {
          chunks.push({
            content: currentChunk,
            startOffset: chunkStart,
            endOffset: chunkStart + currentChunk.length,
            tokensEstimate: Math.ceil(currentChunk.length / CHARS_PER_TOKEN),
          });
          currentChunk = "";
        }

        const longChunks = splitLongSegment(
          sentence,
          targetChars,
          overlapChars,
          currentPos
        );
        chunks.push(...longChunks);
        currentPos += sentence.length + 1;
        chunkStart = currentPos;
        continue;
      }

      const separator = currentChunk ? " " : "";
      const potentialChunk = currentChunk + separator + sentence;

      if (
        potentialChunk.length > targetChars &&
        currentChunk.length >= minChunkSize
      ) {
        chunks.push({
          content: currentChunk,
          startOffset: chunkStart,
          endOffset: chunkStart + currentChunk.length,
          tokensEstimate: Math.ceil(currentChunk.length / CHARS_PER_TOKEN),
        });

        const overlapText = currentChunk.slice(-overlapChars);
        currentChunk = overlapText + " " + sentence;
        chunkStart = currentPos - overlapChars;
      } else {
        currentChunk = potentialChunk;
      }

      currentPos += sentence.length + 1;
    }

    if (pIdx < paragraphs.length - 1) {
      currentPos += 1;
    }
  }

  if (currentChunk.length > 0) {
    chunks.push({
      content: currentChunk,
      startOffset: chunkStart,
      endOffset: chunkStart + currentChunk.length,
      tokensEstimate: Math.ceil(currentChunk.length / CHARS_PER_TOKEN),
    });
  }

  return chunks;
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}
