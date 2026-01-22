use chrono::{DateTime, Utc};
use engram_core::{Memory, Sector};

/// Configuration for decay processing
#[derive(Debug, Clone)]
pub struct DecayConfig {
  /// Minimum salience threshold (memories below this are candidates for archival)
  pub min_salience: f32,
  /// Salience threshold below which memories should be archived
  pub archive_threshold: f32,
  /// Maximum days without access before forced decay consideration
  pub max_idle_days: i64,
}

impl Default for DecayConfig {
  fn default() -> Self {
    Self {
      min_salience: 0.05,
      archive_threshold: 0.1,
      max_idle_days: 90,
    }
  }
}

/// Result of applying decay to a memory
#[derive(Debug, Clone)]
pub struct DecayResult {
  pub memory_id: engram_core::MemoryId,
  pub previous_salience: f32,
  pub new_salience: f32,
  pub days_since_access: f32,
  pub should_archive: bool,
}

/// Apply decay to a single memory
pub fn apply_decay(memory: &mut Memory, now: DateTime<Utc>, config: &DecayConfig) -> DecayResult {
  let previous_salience = memory.salience;

  // Calculate days since last access
  let days_since_access = (now - memory.last_accessed).num_days() as f32;

  // Apply the memory's built-in decay
  memory.apply_decay(now);

  // Check if should be archived
  let should_archive = memory.salience < config.archive_threshold || days_since_access > config.max_idle_days as f32;

  DecayResult {
    memory_id: memory.id,
    previous_salience,
    new_salience: memory.salience,
    days_since_access,
    should_archive,
  }
}

/// Apply decay to a batch of memories
pub fn apply_decay_batch(memories: &mut [Memory], now: DateTime<Utc>, config: &DecayConfig) -> Vec<DecayResult> {
  memories.iter_mut().map(|m| apply_decay(m, now, config)).collect()
}

/// Calculate expected salience after a given number of days
pub fn predict_salience(current_salience: f32, importance: f32, sector: Sector, access_count: u32, days: f32) -> f32 {
  let effective_rate = sector.decay_rate() / (importance + 0.1);
  let decay_factor = (-effective_rate * days).exp();

  // Access protection
  let access_protection = (1.0 + access_count as f32).ln() * 0.02;
  let access_protection = access_protection.min(0.1);

  (current_salience * decay_factor + access_protection).clamp(0.05, 1.0)
}

/// Estimate days until memory reaches a target salience
pub fn days_until_salience(
  current_salience: f32,
  target_salience: f32,
  importance: f32,
  sector: Sector,
  access_count: u32,
) -> Option<f32> {
  if current_salience <= target_salience {
    return Some(0.0);
  }

  // Access protection baseline
  let access_protection = (1.0 + access_count as f32).ln() * 0.02;
  let access_protection = access_protection.min(0.1);

  // If target is below the floor due to access protection, it may never be reached
  if target_salience < access_protection + 0.05 {
    return None;
  }

  let effective_rate = sector.decay_rate() / (importance + 0.1);

  // Solve: target = current * exp(-rate * days) + protection
  // target - protection = current * exp(-rate * days)
  // ln((target - protection) / current) = -rate * days
  // days = -ln((target - protection) / current) / rate

  let adjusted_target = target_salience - access_protection;
  if adjusted_target <= 0.0 || adjusted_target >= current_salience {
    return None;
  }

  let days = -(adjusted_target / current_salience).ln() / effective_rate;
  Some(days.max(0.0))
}

/// Batch statistics for decay processing
#[derive(Debug, Default)]
pub struct DecayStats {
  pub total_processed: usize,
  pub decayed_count: usize,
  pub archive_candidates: usize,
  pub average_salience_drop: f32,
}

impl DecayStats {
  pub fn from_results(results: &[DecayResult]) -> Self {
    if results.is_empty() {
      return Self::default();
    }

    let total_processed = results.len();
    let decayed_count = results.iter().filter(|r| r.new_salience < r.previous_salience).count();
    let archive_candidates = results.iter().filter(|r| r.should_archive).count();
    let average_salience_drop = results
      .iter()
      .map(|r| r.previous_salience - r.new_salience)
      .sum::<f32>()
      / total_processed as f32;

    Self {
      total_processed,
      decayed_count,
      archive_candidates,
      average_salience_drop,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use uuid::Uuid;

  #[test]
  fn test_apply_decay() {
    let mut memory = Memory::new(Uuid::new_v4(), "test".into(), Sector::Episodic);
    memory.salience = 1.0;
    memory.importance = 0.5;

    let config = DecayConfig::default();
    let future = Utc::now() + chrono::Duration::days(30);

    let result = apply_decay(&mut memory, future, &config);

    assert!(result.new_salience < result.previous_salience);
    assert_eq!(result.previous_salience, 1.0);
  }

  #[test]
  fn test_decay_varies_by_sector() {
    let config = DecayConfig::default();
    let future = Utc::now() + chrono::Duration::days(30);

    // Episodic should decay faster than Emotional
    let mut episodic = Memory::new(Uuid::new_v4(), "test".into(), Sector::Episodic);
    episodic.salience = 1.0;
    episodic.importance = 0.5;

    let mut emotional = Memory::new(Uuid::new_v4(), "test".into(), Sector::Emotional);
    emotional.salience = 1.0;
    emotional.importance = 0.5;

    apply_decay(&mut episodic, future, &config);
    apply_decay(&mut emotional, future, &config);

    assert!(
      episodic.salience < emotional.salience,
      "Episodic {} should be less than Emotional {}",
      episodic.salience,
      emotional.salience
    );
  }

  #[test]
  fn test_importance_slows_decay() {
    let config = DecayConfig::default();
    let future = Utc::now() + chrono::Duration::days(30);

    let mut low_importance = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    low_importance.salience = 1.0;
    low_importance.importance = 0.2;

    let mut high_importance = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    high_importance.salience = 1.0;
    high_importance.importance = 0.9;

    apply_decay(&mut low_importance, future, &config);
    apply_decay(&mut high_importance, future, &config);

    assert!(
      low_importance.salience < high_importance.salience,
      "Low importance {} should decay more than high importance {}",
      low_importance.salience,
      high_importance.salience
    );
  }

  #[test]
  fn test_access_count_provides_protection() {
    let config = DecayConfig::default();
    let future = Utc::now() + chrono::Duration::days(60);

    let mut rarely_accessed = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    rarely_accessed.salience = 1.0;
    rarely_accessed.importance = 0.5;
    rarely_accessed.access_count = 0;

    let mut frequently_accessed = Memory::new(Uuid::new_v4(), "test".into(), Sector::Semantic);
    frequently_accessed.salience = 1.0;
    frequently_accessed.importance = 0.5;
    frequently_accessed.access_count = 100;

    apply_decay(&mut rarely_accessed, future, &config);
    apply_decay(&mut frequently_accessed, future, &config);

    assert!(
      rarely_accessed.salience < frequently_accessed.salience,
      "Rarely accessed {} should decay more than frequently accessed {}",
      rarely_accessed.salience,
      frequently_accessed.salience
    );
  }

  #[test]
  fn test_predict_salience() {
    let predicted = predict_salience(1.0, 0.5, Sector::Episodic, 0, 30.0);
    assert!(predicted < 1.0);
    assert!(predicted > 0.05);
  }

  #[test]
  fn test_days_until_salience() {
    let days = days_until_salience(1.0, 0.5, 0.5, Sector::Episodic, 0);
    assert!(days.is_some());
    assert!(days.unwrap() > 0.0);
  }

  #[test]
  fn test_decay_stats() {
    let results = vec![
      DecayResult {
        memory_id: engram_core::MemoryId::new(),
        previous_salience: 1.0,
        new_salience: 0.8,
        days_since_access: 10.0,
        should_archive: false,
      },
      DecayResult {
        memory_id: engram_core::MemoryId::new(),
        previous_salience: 0.5,
        new_salience: 0.05,
        days_since_access: 100.0,
        should_archive: true,
      },
    ];

    let stats = DecayStats::from_results(&results);

    assert_eq!(stats.total_processed, 2);
    assert_eq!(stats.decayed_count, 2);
    assert_eq!(stats.archive_candidates, 1);
  }
}
