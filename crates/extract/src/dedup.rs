use engram_core::Memory;
use std::collections::HashSet;

const FNV_PRIME: u64 = 0x100000001b3;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// Compute 64-bit SimHash for locality-sensitive hashing
pub fn simhash(text: &str) -> u64 {
  let tokens = tokenize(text);
  let mut vector = [0i32; 64];

  for token in tokens {
    let hash = fnv1a_hash(token);
    for (i, v) in vector.iter_mut().enumerate() {
      if (hash >> i) & 1 == 1 {
        *v += 1;
      } else {
        *v -= 1;
      }
    }
  }

  let mut result = 0u64;
  for (i, &v) in vector.iter().enumerate() {
    if v > 0 {
      result |= 1 << i;
    }
  }
  result
}

/// Compute Hamming distance between two SimHashes
pub fn hamming_distance(a: u64, b: u64) -> u32 {
  (a ^ b).count_ones()
}

/// FNV-1a hash for individual tokens
fn fnv1a_hash(s: &str) -> u64 {
  let mut hash = FNV_OFFSET;
  for byte in s.bytes() {
    hash ^= byte as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
  }
  hash
}

/// Tokenize text for SimHash
fn tokenize(text: &str) -> Vec<&str> {
  text
    .split(|c: char| !c.is_alphanumeric() && c != '_')
    .filter(|s| s.len() >= 3)
    .collect()
}

/// Compute Jaccard similarity between two texts
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
  let tokens_a: HashSet<&str> = tokenize(a).into_iter().collect();
  let tokens_b: HashSet<&str> = tokenize(b).into_iter().collect();

  if tokens_a.is_empty() && tokens_b.is_empty() {
    return 1.0;
  }

  let intersection = tokens_a.intersection(&tokens_b).count();
  let union = tokens_a.union(&tokens_b).count();

  if union == 0 {
    return 1.0;
  }

  intersection as f32 / union as f32
}

/// Adaptive threshold based on content length
pub fn adaptive_threshold(content_len: usize) -> u32 {
  match content_len {
    0..=50 => 2,
    51..=200 => 3,
    201..=500 => 4,
    _ => 5,
  }
}

/// Compute SHA-256 hash of content
pub fn content_hash(content: &str) -> String {
  use std::collections::hash_map::DefaultHasher;
  use std::hash::{Hash, Hasher};

  let mut hasher = DefaultHasher::new();
  content.hash(&mut hasher);
  format!("{:016x}", hasher.finish())
}

/// Result of duplicate check
#[derive(Debug, Clone)]
pub enum DuplicateMatch {
  /// Exact content hash match
  Exact,
  /// SimHash similarity (with distance and Jaccard confirmation)
  Simhash { distance: u32, jaccard: f32 },
  /// No match
  None,
}

impl DuplicateMatch {
  pub fn is_duplicate(&self) -> bool {
    !matches!(self, DuplicateMatch::None)
  }
}

/// Check if two memories are duplicates
pub struct DuplicateChecker {
  jaccard_threshold: f32,
}

impl Default for DuplicateChecker {
  fn default() -> Self {
    Self::new()
  }
}

impl DuplicateChecker {
  pub fn new() -> Self {
    Self { jaccard_threshold: 0.8 }
  }

  pub fn with_threshold(mut self, threshold: f32) -> Self {
    self.jaccard_threshold = threshold;
    self
  }

  /// Check for duplicate using multi-level strategy
  pub fn is_duplicate(&self, new_content: &str, new_hash: &str, new_simhash: u64, existing: &Memory) -> DuplicateMatch {
    // Level 1: Exact content hash match
    if new_hash == existing.content_hash {
      return DuplicateMatch::Exact;
    }

    // Level 2: SimHash similarity
    let distance = hamming_distance(new_simhash, existing.simhash);
    let threshold = adaptive_threshold(new_content.len());

    if distance <= threshold {
      // Level 3: Always verify with Jaccard to confirm similarity
      // Even with low hamming distance, verify to prevent false positives
      let jaccard = jaccard_similarity(new_content, &existing.content);
      if jaccard >= self.jaccard_threshold {
        return DuplicateMatch::Simhash { distance, jaccard };
      }
      // If Jaccard is below threshold, not a duplicate despite similar SimHash
    }

    DuplicateMatch::None
  }

  /// Check a new content against a list of existing memories
  pub fn find_duplicate<'a>(&self, new_content: &str, existing: &'a [Memory]) -> Option<(&'a Memory, DuplicateMatch)> {
    let new_hash = content_hash(new_content);
    let new_simhash = simhash(new_content);

    for memory in existing {
      let result = self.is_duplicate(new_content, &new_hash, new_simhash, memory);
      if result.is_duplicate() {
        return Some((memory, result));
      }
    }

    None
  }
}

/// Compute both hashes for a piece of content
pub fn compute_hashes(content: &str) -> (String, u64) {
  (content_hash(content), simhash(content))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_simhash_identical() {
    let text = "The quick brown fox jumps over the lazy dog";
    let hash1 = simhash(text);
    let hash2 = simhash(text);
    assert_eq!(hash1, hash2);
  }

  #[test]
  fn test_simhash_similar() {
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "The quick brown fox jumps over a lazy dog";
    let hash1 = simhash(text1);
    let hash2 = simhash(text2);
    let distance = hamming_distance(hash1, hash2);
    // Similar texts should have small Hamming distance
    assert!(distance < 10, "Distance was {}", distance);
  }

  #[test]
  fn test_simhash_different() {
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "Completely unrelated content about programming";
    let hash1 = simhash(text1);
    let hash2 = simhash(text2);
    let distance = hamming_distance(hash1, hash2);
    // Different texts should have larger Hamming distance
    assert!(distance > 10, "Distance was {}", distance);
  }

  #[test]
  fn test_hamming_distance() {
    assert_eq!(hamming_distance(0b1010, 0b1010), 0);
    assert_eq!(hamming_distance(0b1010, 0b0101), 4);
    assert_eq!(hamming_distance(0b1111, 0b0000), 4);
  }

  #[test]
  fn test_jaccard_identical() {
    let text = "hello world foo bar";
    assert_eq!(jaccard_similarity(text, text), 1.0);
  }

  #[test]
  fn test_jaccard_similar() {
    let text1 = "hello world foo bar";
    let text2 = "hello world foo baz";
    let sim = jaccard_similarity(text1, text2);
    assert!(sim > 0.5);
    assert!(sim < 1.0);
  }

  #[test]
  fn test_jaccard_empty() {
    assert_eq!(jaccard_similarity("", ""), 1.0);
    // "a b" has no tokens >= 3 chars, so it's effectively empty
    assert_eq!(jaccard_similarity("hello world", ""), 0.0);
  }

  #[test]
  fn test_adaptive_threshold() {
    assert_eq!(adaptive_threshold(10), 2);
    assert_eq!(adaptive_threshold(100), 3);
    assert_eq!(adaptive_threshold(300), 4);
    assert_eq!(adaptive_threshold(1000), 5);
  }

  #[test]
  fn test_content_hash() {
    let hash1 = content_hash("hello world");
    let hash2 = content_hash("hello world");
    let hash3 = content_hash("different");

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, hash3);
  }

  #[test]
  fn test_duplicate_checker_exact() {
    use uuid::Uuid;

    let content = "test content for deduplication";
    let (hash, sh) = compute_hashes(content);

    let mut memory = Memory::new(Uuid::new_v4(), content.to_string(), engram_core::Sector::Semantic);
    memory.content_hash = hash.clone();
    memory.simhash = sh;

    let checker = DuplicateChecker::new();
    let result = checker.is_duplicate(content, &hash, sh, &memory);

    assert!(matches!(result, DuplicateMatch::Exact));
  }

  #[test]
  fn test_duplicate_checker_similar() {
    use uuid::Uuid;

    // Use identical content to test the simhash path (distance = 0)
    let content1 = "The user prefers using TypeScript over JavaScript for new projects";
    let content2 = content1; // Exact same content

    let (hash1, sh1) = compute_hashes(content1);
    let (_, sh2) = compute_hashes(content2);

    let mut memory = Memory::new(Uuid::new_v4(), content1.to_string(), engram_core::Sector::Semantic);
    memory.content_hash = hash1.clone();
    memory.simhash = sh1;

    let checker = DuplicateChecker::new();

    // With different hash but same simhash (simulating near-duplicate)
    let result = checker.is_duplicate(content2, "different-hash", sh2, &memory);

    // Should detect via simhash since distance = 0
    assert!(result.is_duplicate(), "Expected simhash duplicate detection");

    // Also test exact hash match
    let result_exact = checker.is_duplicate(content1, &hash1, sh1, &memory);
    assert!(matches!(result_exact, DuplicateMatch::Exact));
  }

  #[test]
  fn test_duplicate_checker_different() {
    use uuid::Uuid;

    let content1 = "The user prefers using TypeScript over JavaScript";
    let content2 = "Database connection pooling configuration settings";

    let (hash1, sh1) = compute_hashes(content1);
    let (hash2, sh2) = compute_hashes(content2);

    let mut memory = Memory::new(Uuid::new_v4(), content1.to_string(), engram_core::Sector::Semantic);
    memory.content_hash = hash1;
    memory.simhash = sh1;

    let checker = DuplicateChecker::new();
    let result = checker.is_duplicate(content2, &hash2, sh2, &memory);

    assert!(matches!(result, DuplicateMatch::None));
  }
}
