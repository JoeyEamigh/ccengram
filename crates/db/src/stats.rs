// Comprehensive statistics for the database

use crate::connection::{ProjectDb, Result};
use engram_core::{Sector, Tier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics for a project's memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
  pub total: usize,
  pub by_sector: HashMap<String, usize>,
  pub by_tier: HashMap<String, usize>,
  pub by_salience: SalienceDistribution,
  pub superseded_count: usize,
}

/// Distribution of salience scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceDistribution {
  pub high: usize,     // >= 0.7
  pub medium: usize,   // >= 0.4 and < 0.7
  pub low: usize,      // >= 0.2 and < 0.4
  pub very_low: usize, // < 0.2
}

/// Statistics for code indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStats {
  pub total_chunks: usize,
  pub total_files: usize,
  pub by_language: HashMap<String, usize>,
  pub recent_indexed: Vec<RecentIndexActivity>,
}

/// Recent index activity record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentIndexActivity {
  pub file_path: String,
  pub language: String,
  pub chunks: usize,
  pub indexed_at: String,
}

/// Combined project statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStats {
  pub memories: MemoryStats,
  pub code: CodeStats,
  pub documents: DocumentStats,
  pub entities: EntityStats,
}

/// Document statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStats {
  pub total: usize,
  pub total_chunks: usize,
}

/// Entity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityStats {
  pub total: usize,
  pub by_type: HashMap<String, usize>,
}

impl ProjectDb {
  /// Get comprehensive memory statistics
  pub async fn get_memory_stats(&self) -> Result<MemoryStats> {
    let memories = self.list_memories(None, None).await?;

    let mut by_sector: HashMap<String, usize> = HashMap::new();
    let mut by_tier: HashMap<String, usize> = HashMap::new();
    let mut high = 0;
    let mut medium = 0;
    let mut low = 0;
    let mut very_low = 0;
    let mut superseded_count = 0;

    for m in &memories {
      // Count by sector
      *by_sector.entry(m.sector.as_str().to_string()).or_insert(0) += 1;

      // Count by tier
      *by_tier.entry(m.tier.as_str().to_string()).or_insert(0) += 1;

      // Salience distribution
      if m.salience >= 0.7 {
        high += 1;
      } else if m.salience >= 0.4 {
        medium += 1;
      } else if m.salience >= 0.2 {
        low += 1;
      } else {
        very_low += 1;
      }

      // Superseded count
      if m.superseded_by.is_some() {
        superseded_count += 1;
      }
    }

    // Ensure all sectors and tiers appear in the stats
    for sector in [
      Sector::Semantic,
      Sector::Episodic,
      Sector::Procedural,
      Sector::Emotional,
      Sector::Reflective,
    ] {
      by_sector.entry(sector.as_str().to_string()).or_insert(0);
    }
    for tier in [Tier::Session, Tier::Project] {
      by_tier.entry(tier.as_str().to_string()).or_insert(0);
    }

    Ok(MemoryStats {
      total: memories.len(),
      by_sector,
      by_tier,
      by_salience: SalienceDistribution {
        high,
        medium,
        low,
        very_low,
      },
      superseded_count,
    })
  }

  /// Get comprehensive code statistics
  pub async fn get_code_stats(&self) -> Result<CodeStats> {
    let chunks = self.list_code_chunks(None, None).await?;

    // Count by language
    let mut by_language: HashMap<String, usize> = HashMap::new();
    let mut files_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for chunk in &chunks {
      let lang_str = format!("{:?}", chunk.language).to_lowercase();
      *by_language.entry(lang_str).or_insert(0) += 1;
      files_seen.insert(chunk.file_path.clone());
    }

    // Get recent indexed files (last 10 unique files by indexed_at)
    let mut file_info: HashMap<String, (String, usize, chrono::DateTime<chrono::Utc>)> = HashMap::new();
    for chunk in &chunks {
      let lang_str = format!("{:?}", chunk.language).to_lowercase();
      file_info
        .entry(chunk.file_path.clone())
        .and_modify(|(_, count, ts)| {
          *count += 1;
          if chunk.indexed_at > *ts {
            *ts = chunk.indexed_at;
          }
        })
        .or_insert((lang_str, 1, chunk.indexed_at));
    }

    let mut recent: Vec<_> = file_info.into_iter().collect();
    recent.sort_by(|a, b| b.1.2.cmp(&a.1.2));

    let recent_indexed: Vec<RecentIndexActivity> = recent
      .into_iter()
      .take(10)
      .map(|(path, (lang, chunks, ts))| RecentIndexActivity {
        file_path: path,
        language: lang,
        chunks,
        indexed_at: ts.to_rfc3339(),
      })
      .collect();

    Ok(CodeStats {
      total_chunks: chunks.len(),
      total_files: files_seen.len(),
      by_language,
      recent_indexed,
    })
  }

  /// Get document statistics
  pub async fn get_document_stats(&self) -> Result<DocumentStats> {
    let docs = self.list_document_metadata(self.project_id.as_str()).await?;
    let chunks = self.list_document_chunks(None, None).await?;

    Ok(DocumentStats {
      total: docs.len(),
      total_chunks: chunks.len(),
    })
  }

  /// Get entity statistics
  pub async fn get_entity_stats(&self) -> Result<EntityStats> {
    let entities = self.list_entities(None).await?;

    let mut by_type: HashMap<String, usize> = HashMap::new();
    for entity in &entities {
      *by_type
        .entry(format!("{:?}", entity.entity_type).to_lowercase())
        .or_insert(0) += 1;
    }

    Ok(EntityStats {
      total: entities.len(),
      by_type,
    })
  }

  /// Get all project statistics
  pub async fn get_project_stats(&self) -> Result<ProjectStats> {
    let memories = self.get_memory_stats().await?;
    let code = self.get_code_stats().await?;
    let documents = self.get_document_stats().await?;
    let entities = self.get_entity_stats().await?;

    Ok(ProjectStats {
      memories,
      code,
      documents,
      entities,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use engram_core::{Memory, ProjectId, Sector, Tier};
  use std::path::Path;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_memory_stats() {
    let data_dir = TempDir::new().unwrap();
    let db_path = data_dir.path().join("test.lancedb");
    let project_id = ProjectId::from_path(Path::new("/test"));
    let db = ProjectDb::open_at_path(project_id.clone(), db_path, 4096)
      .await
      .unwrap();

    // Create test memories with different properties
    let memories = vec![
      (Sector::Semantic, Tier::Project, 0.8),
      (Sector::Semantic, Tier::Session, 0.5),
      (Sector::Episodic, Tier::Session, 0.3),
      (Sector::Procedural, Tier::Project, 0.1),
    ];

    for (sector, tier, salience) in memories {
      let mut memory = Memory::new(
        uuid::Uuid::new_v4(),
        format!("Test memory with salience {}", salience),
        sector,
      );
      memory.tier = tier;
      memory.salience = salience;
      db.add_memory(&memory, None).await.unwrap();
    }

    let stats = db.get_memory_stats().await.unwrap();
    assert_eq!(stats.total, 4);
    assert_eq!(stats.by_sector.get("semantic"), Some(&2));
    assert_eq!(stats.by_sector.get("episodic"), Some(&1));
    assert_eq!(stats.by_sector.get("procedural"), Some(&1));
    assert_eq!(stats.by_tier.get("session"), Some(&2));
    assert_eq!(stats.by_tier.get("project"), Some(&2));
    assert_eq!(stats.by_salience.high, 1);
    assert_eq!(stats.by_salience.medium, 1);
    assert_eq!(stats.by_salience.low, 1);
    assert_eq!(stats.by_salience.very_low, 1);
  }

  #[tokio::test]
  async fn test_code_stats() {
    use engram_core::{ChunkType, CodeChunk, Language};

    let data_dir = TempDir::new().unwrap();
    let db_path = data_dir.path().join("test.lancedb");
    let project_id = ProjectId::from_path(Path::new("/test"));
    let db = ProjectDb::open_at_path(project_id.clone(), db_path, 4096)
      .await
      .unwrap();

    // Create test chunks
    let chunks_data = vec![
      ("src/main.rs", Language::Rust),
      ("src/main.rs", Language::Rust),
      ("src/lib.rs", Language::Rust),
      ("src/utils.ts", Language::TypeScript),
    ];

    for (path, lang) in chunks_data {
      let chunk = CodeChunk {
        id: uuid::Uuid::new_v4(),
        file_path: path.to_string(),
        language: lang,
        content: "test content".to_string(),
        start_line: 0,
        end_line: 10,
        chunk_type: ChunkType::Function,
        symbols: vec![],
        imports: vec![],
        calls: vec![],
        file_hash: "abc123".to_string(),
        indexed_at: chrono::Utc::now(),
        tokens_estimate: 10,
        definition_kind: None,
        definition_name: None,
        visibility: None,
        signature: None,
        docstring: None,
        parent_definition: None,
        embedding_text: None,
        content_hash: None,
        caller_count: 0,
        callee_count: 0,
      };
      db.add_code_chunk(&chunk, None).await.unwrap();
    }

    let stats = db.get_code_stats().await.unwrap();
    assert_eq!(stats.total_chunks, 4);
    assert_eq!(stats.total_files, 3);
    assert_eq!(stats.by_language.get("rust"), Some(&3));
    assert_eq!(stats.by_language.get("typescript"), Some(&1));
  }
}
