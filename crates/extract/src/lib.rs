pub mod classifier;
pub mod decay;
pub mod dedup;

pub use classifier::{classify_sector, extract_concepts, extract_files, sector_scores};
pub use decay::{
  DecayConfig, DecayResult, DecayStats, apply_decay, apply_decay_batch, days_until_salience, predict_salience,
};
pub use dedup::{
  DuplicateChecker, DuplicateMatch, adaptive_threshold, compute_hashes, content_hash, hamming_distance,
  jaccard_similarity, simhash,
};
