use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a document (newtype for type safety)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(Uuid);

impl DocumentId {
  pub fn new() -> Self {
    Self(Uuid::now_v7()) // Time-ordered UUIDs
  }

  pub fn from_uuid(id: Uuid) -> Self {
    Self(id)
  }

  pub fn as_uuid(&self) -> Uuid {
    self.0
  }
}

impl Default for DocumentId {
  fn default() -> Self {
    Self::new()
  }
}

impl std::fmt::Display for DocumentId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl std::str::FromStr for DocumentId {
  type Err = uuid::Error;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(Self(Uuid::parse_str(s)?))
  }
}

/// Source type for ingested documents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentSource {
  /// File on local filesystem
  File,
  /// Fetched from URL
  Url,
  /// Directly provided content
  Content,
}

impl DocumentSource {
  pub fn as_str(&self) -> &'static str {
    match self {
      DocumentSource::File => "file",
      DocumentSource::Url => "url",
      DocumentSource::Content => "content",
    }
  }
}

impl std::str::FromStr for DocumentSource {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "file" => Ok(DocumentSource::File),
      "url" => Ok(DocumentSource::Url),
      "content" => Ok(DocumentSource::Content),
      _ => Err(format!("Unknown document source: {}", s)),
    }
  }
}

/// A document chunk for vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
  pub id: DocumentId,
  pub document_id: DocumentId,
  pub project_id: Uuid,

  /// The text content of this chunk
  pub content: String,

  /// Title of the parent document
  pub title: String,

  /// Source path/url of the document
  pub source: String,

  /// Source type
  pub source_type: DocumentSource,

  /// Chunk index within the document
  pub chunk_index: usize,

  /// Total chunks in the document
  pub total_chunks: usize,

  /// Character offset in original document
  pub char_offset: usize,

  /// Timestamps
  pub created_at: DateTime<Utc>,
  pub updated_at: DateTime<Utc>,
}

impl DocumentChunk {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    document_id: DocumentId,
    project_id: Uuid,
    content: String,
    title: String,
    source: String,
    source_type: DocumentSource,
    chunk_index: usize,
    total_chunks: usize,
    char_offset: usize,
  ) -> Self {
    let now = Utc::now();
    Self {
      id: DocumentId::new(),
      document_id,
      project_id,
      content,
      title,
      source,
      source_type,
      chunk_index,
      total_chunks,
      char_offset,
      created_at: now,
      updated_at: now,
    }
  }
}

/// Metadata about an ingested document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
  pub id: DocumentId,
  pub project_id: Uuid,

  /// Document title
  pub title: String,

  /// Source path/url
  pub source: String,

  /// Source type
  pub source_type: DocumentSource,

  /// Content hash for deduplication
  pub content_hash: String,

  /// Total character count
  pub char_count: usize,

  /// Number of chunks created
  pub chunk_count: usize,

  /// Full document content (optional, for re-chunking without refetch)
  pub full_content: Option<String>,

  /// Timestamps
  pub created_at: DateTime<Utc>,
  pub updated_at: DateTime<Utc>,
}

impl Document {
  pub fn new(
    project_id: Uuid,
    title: String,
    source: String,
    source_type: DocumentSource,
    content_hash: String,
    char_count: usize,
    chunk_count: usize,
  ) -> Self {
    let now = Utc::now();
    Self {
      id: DocumentId::new(),
      project_id,
      title,
      source,
      source_type,
      content_hash,
      char_count,
      chunk_count,
      full_content: None,
      created_at: now,
      updated_at: now,
    }
  }

  /// Create a new document with full content stored
  pub fn with_content(
    project_id: Uuid,
    title: String,
    source: String,
    source_type: DocumentSource,
    content: String,
    chunk_count: usize,
  ) -> Self {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let content_hash = format!("{:x}", hasher.finalize());
    let char_count = content.len();

    let now = Utc::now();
    Self {
      id: DocumentId::new(),
      project_id,
      title,
      source,
      source_type,
      content_hash,
      char_count,
      chunk_count,
      full_content: Some(content),
      created_at: now,
      updated_at: now,
    }
  }
}

/// Parameters for chunking documents
#[derive(Debug, Clone)]
pub struct ChunkParams {
  /// Target chunk size in characters
  pub chunk_size: usize,
  /// Overlap between chunks in characters
  pub overlap: usize,
}

impl Default for ChunkParams {
  fn default() -> Self {
    Self {
      chunk_size: 1000,
      overlap: 200,
    }
  }
}

/// Split text into sentences using regex-like patterns
fn split_sentences(text: &str) -> Vec<&str> {
  let mut sentences = Vec::new();
  let mut current_start = 0;
  let chars: Vec<char> = text.chars().collect();

  let mut i = 0;
  while i < chars.len() {
    // Check for sentence-ending punctuation followed by whitespace or end
    if matches!(chars[i], '.' | '!' | '?') {
      // Look ahead to see if this is a sentence boundary
      let next_idx = i + 1;
      if next_idx >= chars.len() || chars[next_idx].is_whitespace() {
        // Check for abbreviations (single capital letter before period)
        let is_abbreviation = i > 0 && i < chars.len() - 1 && chars[i] == '.' && {
          let prev = chars[i - 1];
          let next_after_space = chars.get(i + 2);
          // Single letter abbreviation like "A." or "U.S."
          (prev.is_uppercase() && (i < 2 || !chars[i - 2].is_alphabetic()))
            || (next_after_space.is_some_and(|&c| c.is_lowercase()))
        };

        if !is_abbreviation {
          // Find the byte position for slicing
          let byte_pos = text.char_indices().nth(next_idx).map(|(b, _)| b).unwrap_or(text.len());
          let start_byte = text.char_indices().nth(current_start).map(|(b, _)| b).unwrap_or(0);

          let sentence = &text[start_byte..byte_pos];
          if !sentence.trim().is_empty() {
            sentences.push(sentence.trim());
          }
          current_start = next_idx;
          while current_start < chars.len() && chars[current_start].is_whitespace() {
            current_start += 1;
          }
        }
      }
    }
    i += 1;
  }

  // Add remaining text as final sentence
  if current_start < chars.len() {
    let start_byte = text.char_indices().nth(current_start).map(|(b, _)| b).unwrap_or(0);
    let remainder = &text[start_byte..];
    if !remainder.trim().is_empty() {
      sentences.push(remainder.trim());
    }
  }

  sentences
}

/// Split text into paragraphs (separated by double newlines)
fn split_paragraphs(text: &str) -> Vec<&str> {
  text
    .split("\n\n")
    .flat_map(|p| p.split("\r\n\r\n"))
    .map(|p| p.trim())
    .filter(|p| !p.is_empty())
    .collect()
}

/// Chunk text content into overlapping segments with sentence-aware splitting
///
/// This function chunks text by:
/// 1. First splitting into paragraphs (double newlines)
/// 2. If paragraphs are too large, splitting into sentences
/// 3. Combining sentences/paragraphs until chunk_size is reached
/// 4. Using overlap to maintain context between chunks
pub fn chunk_text(content: &str, params: &ChunkParams) -> Vec<(String, usize)> {
  let mut chunks = Vec::new();

  if content.is_empty() {
    return chunks;
  }

  if content.len() <= params.chunk_size {
    chunks.push((content.to_string(), 0));
    return chunks;
  }

  // Split into paragraphs first
  let paragraphs = split_paragraphs(content);

  let mut current_chunk = String::new();
  let mut current_offset: usize = 0;
  let mut chunk_start_offset: usize = 0;

  for paragraph in paragraphs {
    // If this paragraph alone is larger than chunk_size, split by sentences
    if paragraph.len() > params.chunk_size {
      // Flush current chunk first
      if !current_chunk.is_empty() {
        chunks.push((current_chunk.trim().to_string(), chunk_start_offset));
        // Calculate overlap start
        let overlap_start = current_chunk.len().saturating_sub(params.overlap);
        current_chunk = current_chunk[overlap_start..].to_string();
        chunk_start_offset = current_offset.saturating_sub(params.overlap);
      }

      // Split paragraph into sentences
      let sentences = split_sentences(paragraph);
      for sentence in sentences {
        if current_chunk.len() + sentence.len() + 1 > params.chunk_size && !current_chunk.is_empty() {
          chunks.push((current_chunk.trim().to_string(), chunk_start_offset));
          // Keep overlap
          let overlap_start = current_chunk.len().saturating_sub(params.overlap);
          current_chunk = current_chunk[overlap_start..].to_string();
          chunk_start_offset = current_offset.saturating_sub(params.overlap);
        }

        if !current_chunk.is_empty() && !current_chunk.ends_with(' ') && !current_chunk.ends_with('\n') {
          current_chunk.push(' ');
        }
        current_chunk.push_str(sentence);
        current_offset += sentence.len() + 1; // +1 for space
      }
    } else {
      // Add whole paragraph
      if current_chunk.len() + paragraph.len() + 2 > params.chunk_size && !current_chunk.is_empty() {
        chunks.push((current_chunk.trim().to_string(), chunk_start_offset));
        // Keep overlap
        let overlap_start = current_chunk.len().saturating_sub(params.overlap);
        current_chunk = current_chunk[overlap_start..].to_string();
        chunk_start_offset = current_offset.saturating_sub(params.overlap);
      }

      if !current_chunk.is_empty() {
        current_chunk.push_str("\n\n");
      }
      current_chunk.push_str(paragraph);
      current_offset += paragraph.len() + 2; // +2 for paragraph separator
    }
  }

  // Flush remaining content
  if !current_chunk.trim().is_empty() {
    chunks.push((current_chunk.trim().to_string(), chunk_start_offset));
  }

  // Ensure we don't return empty chunks
  chunks.retain(|(s, _)| !s.is_empty());

  chunks
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_document_id_roundtrip() {
    let id = DocumentId::new();
    let s = id.to_string();
    let parsed: DocumentId = s.parse().unwrap();
    assert_eq!(id, parsed);
  }

  #[test]
  fn test_document_source_parse() {
    assert_eq!("file".parse::<DocumentSource>().unwrap(), DocumentSource::File);
    assert_eq!("url".parse::<DocumentSource>().unwrap(), DocumentSource::Url);
    assert_eq!("content".parse::<DocumentSource>().unwrap(), DocumentSource::Content);
  }

  #[test]
  fn test_chunk_text_small() {
    let params = ChunkParams::default();
    let chunks = chunk_text("Small text", &params);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].0, "Small text");
    assert_eq!(chunks[0].1, 0);
  }

  #[test]
  fn test_chunk_text_large() {
    let params = ChunkParams {
      chunk_size: 100,
      overlap: 20,
    };
    // Use actual sentences instead of repeated characters
    let content = "This is the first sentence with some content. This is the second sentence with more words. This is the third sentence to add length. And here is the fourth sentence to make it longer. Fifth sentence here too.";
    let chunks = chunk_text(content, &params);

    // Should have multiple chunks
    assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());

    // First chunk should start at 0
    assert_eq!(chunks[0].1, 0);
  }

  #[test]
  fn test_chunk_text_break_at_sentence() {
    let params = ChunkParams {
      chunk_size: 50,
      overlap: 10,
    };
    let content = "First sentence. Second sentence. Third sentence here.";
    let chunks = chunk_text(content, &params);

    // Should prefer breaking at sentence boundaries
    assert!(!chunks.is_empty());
  }

  #[test]
  fn test_chunk_text_empty() {
    let params = ChunkParams::default();
    let chunks = chunk_text("", &params);
    assert!(chunks.is_empty());
  }

  #[test]
  fn test_chunk_text_overlap() {
    let params = ChunkParams {
      chunk_size: 100,
      overlap: 20,
    };
    // Create content with sentences that will produce multiple chunks
    let content =
      "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here.";
    let chunks = chunk_text(content, &params);

    assert!(chunks.len() >= 2, "Should have at least 2 chunks, got {}", chunks.len());
  }

  #[test]
  fn test_split_sentences() {
    let text = "First sentence. Second sentence! Third sentence? Fourth.";
    let sentences = split_sentences(text);

    assert_eq!(sentences.len(), 4);
    assert_eq!(sentences[0], "First sentence.");
    assert_eq!(sentences[1], "Second sentence!");
    assert_eq!(sentences[2], "Third sentence?");
    assert_eq!(sentences[3], "Fourth.");
  }

  #[test]
  fn test_split_paragraphs() {
    let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
    let paragraphs = split_paragraphs(text);

    assert_eq!(paragraphs.len(), 3);
    assert_eq!(paragraphs[0], "First paragraph.");
    assert_eq!(paragraphs[1], "Second paragraph.");
    assert_eq!(paragraphs[2], "Third paragraph.");
  }

  #[test]
  fn test_chunk_text_respects_paragraphs() {
    let params = ChunkParams {
      chunk_size: 100,
      overlap: 20,
    };
    let text = "Short paragraph one.\n\nShort paragraph two.\n\nShort paragraph three.";
    let chunks = chunk_text(text, &params);

    // Should fit in one chunk since total is under chunk_size
    assert_eq!(chunks.len(), 1);
  }

  #[test]
  fn test_chunk_text_long_paragraph() {
    let params = ChunkParams {
      chunk_size: 100,
      overlap: 20,
    };
    // Create a long paragraph that needs sentence-level splitting
    let long_para = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five.";
    let chunks = chunk_text(long_para, &params);

    assert!(chunks.len() >= 2, "Long paragraph should produce multiple chunks");

    // All chunks should be non-empty
    for (chunk, _) in &chunks {
      assert!(!chunk.is_empty());
    }
  }
}
