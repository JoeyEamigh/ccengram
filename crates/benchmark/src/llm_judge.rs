//! LLM-as-judge for comprehension testing.
//!
//! Evaluates whether exploration results enable understanding of the codebase.
//! Uses the existing `llm` crate (Claude CLI) to:
//! 1. Generate answers to comprehension questions based on exploration results
//! 2. Evaluate answers against expected concepts
//! 3. Score overall comprehension

use crate::scenarios::{ComprehensionQuestion, LlmJudgeConfig, ScenarioResult};
use llm::{InferenceRequest, Model};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from LLM judge evaluation.
#[derive(Debug, Error)]
pub enum JudgeError {
  #[error("LLM error: {0}")]
  Llm(String),
  #[error("Parse error: {0}")]
  Parse(String),
  #[error("Configuration error: {0}")]
  Config(String),
}

impl From<llm::LlmError> for JudgeError {
  fn from(err: llm::LlmError) -> Self {
    JudgeError::Llm(err.to_string())
  }
}

/// Result of evaluating a single comprehension question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
  /// The question that was asked
  pub question: String,
  /// The generated answer (from LLM)
  pub generated_answer: String,
  /// Score for this question (0.0-1.0)
  pub score: f64,
  /// Expected concepts that were found in the answer
  pub concepts_found: Vec<String>,
  /// Expected concepts that were missing
  pub concepts_missing: Vec<String>,
  /// Wrong concepts that appeared (indicates misunderstanding)
  pub wrong_concepts_found: Vec<String>,
  /// Explanation of the score
  pub explanation: String,
}

/// Overall comprehension evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensionResult {
  /// Per-question results
  pub questions: Vec<QuestionResult>,
  /// Overall weighted score (0.0-1.0)
  pub overall_score: f64,
  /// Whether the scenario passed comprehension criteria
  pub passed: bool,
  /// Summary of comprehension evaluation
  pub summary: String,
}

impl Default for ComprehensionResult {
  fn default() -> Self {
    Self {
      questions: Vec::new(),
      overall_score: 1.0,
      passed: true,
      summary: "No comprehension questions defined".to_string(),
    }
  }
}

/// Configuration for the LLM judge.
#[derive(Debug, Clone)]
pub struct JudgeConfiguration {
  /// Model to use for evaluation
  pub model: Model,
  /// Timeout for LLM calls in seconds
  pub timeout_secs: u64,
}

impl Default for JudgeConfiguration {
  fn default() -> Self {
    Self {
      model: Model::Haiku, // Use Haiku for speed/cost efficiency
      timeout_secs: 60,
    }
  }
}

/// LLM judge for comprehension evaluation.
pub struct LlmJudge {
  config: JudgeConfiguration,
}

impl LlmJudge {
  /// Create a new LLM judge with default configuration.
  pub fn new() -> Self {
    Self::with_config(JudgeConfiguration::default())
  }

  /// Create a new LLM judge with custom configuration.
  pub fn with_config(config: JudgeConfiguration) -> Self {
    Self { config }
  }

  /// Create a new LLM judge with a specific model.
  pub fn with_model(model: Model) -> Self {
    Self::with_config(JudgeConfiguration {
      model,
      ..Default::default()
    })
  }

  /// Check if the judge is properly configured.
  ///
  /// Returns true if the `claude` CLI is available.
  pub fn is_configured(&self) -> bool {
    // Check if claude CLI is available
    let which_cmd = if cfg!(windows) { "where" } else { "which" };
    std::process::Command::new(which_cmd)
      .arg("claude")
      .output()
      .map(|o| o.status.success())
      .unwrap_or(false)
  }

  /// Evaluate comprehension for a scenario result.
  pub async fn evaluate(
    &self,
    scenario_result: &ScenarioResult,
    judge_config: &LlmJudgeConfig,
  ) -> Result<ComprehensionResult, JudgeError> {
    if judge_config.comprehension_questions.is_empty() {
      return Ok(ComprehensionResult::default());
    }

    if !self.is_configured() {
      return Err(JudgeError::Config(
        "Claude CLI not found. Ensure 'claude' is in your PATH.".into(),
      ));
    }

    // Build context from exploration results
    let exploration_context = self.build_context(scenario_result);

    // Evaluate each question
    let mut question_results = Vec::new();
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for question in &judge_config.comprehension_questions {
      let result = self.evaluate_question(&exploration_context, question).await?;
      weighted_sum += result.score * question.weight;
      total_weight += question.weight;
      question_results.push(result);
    }

    let overall_score = if total_weight > 0.0 {
      weighted_sum / total_weight
    } else {
      1.0
    };

    let passed = judge_config
      .min_comprehension_score
      .is_none_or(|min| overall_score >= min);

    let summary = self.generate_summary(&question_results, overall_score, passed);

    Ok(ComprehensionResult {
      questions: question_results,
      overall_score,
      passed,
      summary,
    })
  }

  /// Build exploration context from scenario results.
  fn build_context(&self, result: &ScenarioResult) -> String {
    let mut context = String::new();

    context.push_str("# Exploration Results\n\n");

    // Add discovered files
    context.push_str("## Files Discovered\n");
    for step in &result.steps {
      for file in &step.files_found {
        context.push_str(&format!("- {}\n", file));
      }
    }

    // Add discovered symbols
    context.push_str("\n## Symbols Discovered\n");
    for step in &result.steps {
      for symbol in &step.symbols_found {
        context.push_str(&format!("- {}\n", symbol));
      }
    }

    // Add query progression
    context.push_str("\n## Exploration Steps\n");
    for (i, step) in result.steps.iter().enumerate() {
      context.push_str(&format!(
        "{}. Query: \"{}\"\n   Found {} results, {} files, {} symbols\n",
        i + 1,
        step.query,
        step.result_count,
        step.files_found.len(),
        step.symbols_found.len()
      ));
    }

    // Add accuracy metrics summary
    context.push_str(&format!(
      "\n## Metrics\n- File recall: {:.0}%\n- Symbol recall: {:.0}%\n- Noise ratio: {:.0}%\n",
      result.accuracy.file_recall * 100.0,
      result.accuracy.symbol_recall * 100.0,
      result.accuracy.noise_ratio * 100.0
    ));

    context
  }

  /// Evaluate a single comprehension question.
  async fn evaluate_question(
    &self,
    context: &str,
    question: &ComprehensionQuestion,
  ) -> Result<QuestionResult, JudgeError> {
    // Generate answer based on exploration context
    let answer = self.generate_answer(context, &question.question).await?;

    // Evaluate the answer
    let evaluation = self.score_answer(&answer, question)?;

    Ok(evaluation)
  }

  /// Generate an answer using the LLM.
  async fn generate_answer(&self, context: &str, question: &str) -> Result<String, JudgeError> {
    let system_prompt = "You are an expert software architect analyzing code exploration results. \
      Provide clear, concise answers based only on the information discovered during exploration. \
      If the exploration didn't reveal enough information, say so.";

    let prompt = format!(
      "Based on the following exploration results from a codebase, answer this question.\n\n\
      {}\n\n\
      Question: {}",
      context, question
    );

    let request = InferenceRequest::new(prompt)
      .with_system_prompt(system_prompt)
      .with_model(self.config.model)
      .with_timeout(self.config.timeout_secs);

    let response = llm::infer(request).await?;

    Ok(response.text)
  }

  /// Score an answer against expected concepts.
  fn score_answer(&self, answer: &str, question: &ComprehensionQuestion) -> Result<QuestionResult, JudgeError> {
    let answer_lower = answer.to_lowercase();

    // Check for expected concepts
    let mut concepts_found = Vec::new();
    let mut concepts_missing = Vec::new();

    for concept in &question.expected_concepts {
      if answer_lower.contains(&concept.to_lowercase()) {
        concepts_found.push(concept.clone());
      } else {
        concepts_missing.push(concept.clone());
      }
    }

    // Check for wrong concepts
    let mut wrong_concepts_found = Vec::new();
    for wrong in &question.wrong_concepts {
      if answer_lower.contains(&wrong.to_lowercase()) {
        wrong_concepts_found.push(wrong.clone());
      }
    }

    // Calculate score
    let expected_count = question.expected_concepts.len();
    let found_count = concepts_found.len();
    let wrong_count = wrong_concepts_found.len();

    let base_score = if expected_count > 0 {
      found_count as f64 / expected_count as f64
    } else {
      1.0
    };

    // Penalize for wrong concepts
    let penalty = (wrong_count as f64 * 0.2).min(0.5);
    let score = (base_score - penalty).max(0.0);

    let explanation = if wrong_count > 0 {
      format!(
        "Found {}/{} expected concepts, but {} incorrect concepts detected",
        found_count, expected_count, wrong_count
      )
    } else if expected_count == 0 {
      "No specific concepts to verify".to_string()
    } else {
      format!("Found {}/{} expected concepts", found_count, expected_count)
    };

    Ok(QuestionResult {
      question: question.question.clone(),
      generated_answer: answer.to_string(),
      score,
      concepts_found,
      concepts_missing,
      wrong_concepts_found,
      explanation,
    })
  }

  /// Generate a summary of the comprehension evaluation.
  fn generate_summary(&self, questions: &[QuestionResult], overall_score: f64, passed: bool) -> String {
    let total = questions.len();
    let high_scores = questions.iter().filter(|q| q.score >= 0.7).count();
    let _low_scores = questions.iter().filter(|q| q.score < 0.3).count();

    format!(
      "Comprehension: {:.0}% ({}/{} questions scored well). {}",
      overall_score * 100.0,
      high_scores,
      total,
      if passed { "PASSED" } else { "FAILED" }
    )
  }
}

impl Default for LlmJudge {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_score_answer_all_concepts_found() {
    let judge = LlmJudge::new();
    let question = ComprehensionQuestion {
      question: "What is the main component?".to_string(),
      expected_concepts: vec!["App".to_string(), "Workspace".to_string()],
      wrong_concepts: vec![],
      weight: 1.0,
    };

    let result = judge
      .score_answer("The main component is the App which contains a Workspace.", &question)
      .unwrap();

    assert!((result.score - 1.0).abs() < f64::EPSILON);
    assert_eq!(result.concepts_found.len(), 2);
    assert!(result.concepts_missing.is_empty());
  }

  #[test]
  fn test_score_answer_partial_concepts() {
    let judge = LlmJudge::new();
    let question = ComprehensionQuestion {
      question: "What are the core types?".to_string(),
      expected_concepts: vec!["Model".to_string(), "View".to_string(), "Controller".to_string()],
      wrong_concepts: vec![],
      weight: 1.0,
    };

    let result = judge
      .score_answer("The core types are Model and View.", &question)
      .unwrap();

    assert!((result.score - 2.0 / 3.0).abs() < 0.01);
    assert_eq!(result.concepts_found.len(), 2);
    assert_eq!(result.concepts_missing.len(), 1);
  }

  #[test]
  fn test_score_answer_with_wrong_concepts() {
    let judge = LlmJudge::new();
    let question = ComprehensionQuestion {
      question: "What pattern is used?".to_string(),
      expected_concepts: vec!["MVC".to_string()],
      wrong_concepts: vec!["Singleton".to_string()],
      weight: 1.0,
    };

    let result = judge
      .score_answer("This uses MVC with a Singleton pattern.", &question)
      .unwrap();

    // Found the expected concept but also found a wrong one
    assert!(result.score < 1.0);
    assert!(!result.wrong_concepts_found.is_empty());
  }

  #[test]
  fn test_is_configured_checks_claude_cli() {
    let judge = LlmJudge::new();
    // This test will pass if claude CLI is available, skip if not
    let _ = judge.is_configured();
  }

  #[test]
  fn test_default_config_uses_haiku() {
    let config = JudgeConfiguration::default();
    assert_eq!(config.model, Model::Haiku);
    assert_eq!(config.timeout_secs, 60);
  }

  #[test]
  fn test_with_model() {
    let judge = LlmJudge::with_model(Model::Sonnet);
    assert_eq!(judge.config.model, Model::Sonnet);
  }
}
