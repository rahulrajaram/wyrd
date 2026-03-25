use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde_json::Value;

use crate::embeddings;
use crate::yore_json::{
    DuplicatePair, QueryEnvelope, QueryResult, VocabularyResult, VocabularyTerm,
};

pub const DEFAULT_DOC_MAX_CHARS: usize = 4_000;
pub const DEFAULT_RERANK_SEMANTIC_WEIGHT: f64 = 0.65;
pub const DEFAULT_REFINE_SEMANTIC_WEIGHT: f64 = 0.55;

pub trait Embedder {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>>;
}

#[derive(Default)]
pub struct OrtEmbedder {
    cache: HashMap<String, Vec<f32>>,
}

impl OrtEmbedder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Embedder for OrtEmbedder {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        let embedding = embeddings::embed(text)?.ok_or_else(|| anyhow!(missing_model_message()))?;
        self.cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }
}

#[derive(Debug, Clone)]
pub struct RerankOptions {
    pub root: PathBuf,
    pub semantic_weight: f64,
    pub limit: Option<usize>,
    pub max_chars: usize,
}

impl Default for RerankOptions {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            semantic_weight: DEFAULT_RERANK_SEMANTIC_WEIGHT,
            limit: None,
            max_chars: DEFAULT_DOC_MAX_CHARS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterOptions {
    pub threshold: f64,
    pub limit: Option<usize>,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            threshold: embeddings::embedding_threshold(),
            limit: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefineOptions {
    pub root: PathBuf,
    pub threshold: f64,
    pub semantic_weight: f64,
    pub max_chars: usize,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            threshold: embeddings::embedding_threshold(),
            semantic_weight: DEFAULT_REFINE_SEMANTIC_WEIGHT,
            max_chars: DEFAULT_DOC_MAX_CHARS,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankOutput {
    pub command: &'static str,
    pub query: String,
    pub total_results: usize,
    pub lexical_weight: f64,
    pub semantic_weight: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    pub results: Vec<RerankedResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankedResult {
    pub path: String,
    pub original_rank: usize,
    pub reranked_rank: usize,
    pub original_score: f64,
    pub lexical_score: f64,
    pub semantic_score: f64,
    pub combined_score: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub doc_terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterOutput {
    pub command: &'static str,
    pub threshold: f64,
    pub total_terms: usize,
    pub total_clusters: usize,
    pub clusters: Vec<TermCluster>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TermCluster {
    pub label: String,
    pub size: usize,
    pub members: Vec<ClusterMember>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterMember {
    pub term: String,
    pub score: f64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RefineOutput {
    pub command: &'static str,
    pub total_pairs: usize,
    pub retained_pairs: usize,
    pub threshold: f64,
    pub lexical_weight: f64,
    pub semantic_weight: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    pub pairs: Vec<RefinedDuplicatePair>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RefinedDuplicatePair {
    pub file1: String,
    pub file2: String,
    pub lexical_score: f64,
    pub semantic_score: f64,
    pub refined_score: f64,
    pub jaccard: f64,
    pub simhash: f64,
    pub minhash: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbedOutput {
    pub command: &'static str,
    pub total_texts: usize,
    pub embedding_dim: usize,
    pub embeddings: Vec<EmbeddedText>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddedText {
    pub text: String,
    pub embedding: Vec<f32>,
}

pub fn missing_model_message() -> &'static str {
    "embeddings are unavailable. Set WYRD_EMBEDDING_MODEL to the ONNX file and optionally WYRD_EMBEDDING_TOKENIZER to tokenizer.json."
}

pub fn rerank_query_results<E: Embedder>(
    payload: QueryEnvelope,
    query: &str,
    options: &RerankOptions,
    embedder: &mut E,
) -> Result<RerankOutput> {
    validate_weight(options.semantic_weight)?;
    let lexical_weight = 1.0 - options.semantic_weight;
    let query_embedding = embedder.embed(query)?;
    let lexical_scores = normalized_scores(
        &payload
            .results
            .iter()
            .map(|result| result.score)
            .collect::<Vec<_>>(),
    );

    let mut warnings = Vec::new();
    let mut reranked = Vec::with_capacity(payload.results.len());

    for (index, result) in payload.results.iter().enumerate() {
        let lexical_score = lexical_scores[index];
        match embed_query_document(
            result,
            &options.root,
            options.max_chars,
            embedder,
            &query_embedding,
        ) {
            Ok(semantic_score) => {
                let semantic_norm = cosine_to_score(semantic_score);
                let combined_score =
                    lexical_weight * lexical_score + options.semantic_weight * semantic_norm;
                reranked.push(RerankedResult {
                    path: result.path.clone(),
                    original_rank: index + 1,
                    reranked_rank: 0,
                    original_score: result.score,
                    lexical_score,
                    semantic_score,
                    combined_score,
                    doc_terms: result.doc_terms.clone(),
                });
            }
            Err(error) => {
                warnings.push(format!("{}: {}", result.path, error));
                reranked.push(RerankedResult {
                    path: result.path.clone(),
                    original_rank: index + 1,
                    reranked_rank: 0,
                    original_score: result.score,
                    lexical_score,
                    semantic_score: 0.0,
                    combined_score: lexical_score,
                    doc_terms: result.doc_terms.clone(),
                });
            }
        }
    }

    reranked.sort_by(|left, right| {
        right
            .combined_score
            .partial_cmp(&left.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.original_rank.cmp(&right.original_rank))
    });

    if let Some(limit) = options.limit {
        reranked.truncate(limit);
    }

    for (index, result) in reranked.iter_mut().enumerate() {
        result.reranked_rank = index + 1;
    }

    Ok(RerankOutput {
        command: "rerank",
        query: query.to_string(),
        total_results: reranked.len(),
        lexical_weight,
        semantic_weight: options.semantic_weight,
        diagnostics: payload.diagnostics,
        warnings,
        results: reranked,
    })
}

pub fn cluster_vocabulary<E: Embedder>(
    payload: VocabularyResult,
    options: &ClusterOptions,
    embedder: &mut E,
) -> Result<ClusterOutput> {
    let terms: Vec<VocabularyTerm> = match options.limit {
        Some(limit) => payload.terms.into_iter().take(limit).collect(),
        None => payload.terms,
    };

    if terms.is_empty() {
        return Ok(ClusterOutput {
            command: "cluster",
            threshold: options.threshold,
            total_terms: 0,
            total_clusters: 0,
            clusters: Vec::new(),
        });
    }

    let mut embeddings_by_index = Vec::with_capacity(terms.len());
    for term in &terms {
        embeddings_by_index.push(embedder.embed(&term.term)?);
    }

    let mut visited = vec![false; terms.len()];
    let mut components = Vec::new();

    for start in 0..terms.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        visited[start] = true;

        while let Some(index) = stack.pop() {
            component.push(index);
            for neighbor in 0..terms.len() {
                if visited[neighbor] || index == neighbor {
                    continue;
                }
                let similarity = embeddings::cosine_similarity(
                    &embeddings_by_index[index],
                    &embeddings_by_index[neighbor],
                );
                if similarity >= options.threshold {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        component.sort_by(|left, right| compare_terms(&terms[*left], &terms[*right]));
        components.push(component);
    }

    components.sort_by(|left, right| {
        let left_best = &terms[left[0]];
        let right_best = &terms[right[0]];
        compare_terms(left_best, right_best)
    });

    let clusters = components
        .into_iter()
        .map(|component| {
            let members: Vec<ClusterMember> = component
                .iter()
                .map(|&index| ClusterMember {
                    term: terms[index].term.clone(),
                    score: terms[index].score,
                    count: terms[index].count,
                })
                .collect();
            TermCluster {
                label: members[0].term.clone(),
                size: members.len(),
                members,
            }
        })
        .collect::<Vec<_>>();

    Ok(ClusterOutput {
        command: "cluster",
        threshold: options.threshold,
        total_terms: terms.len(),
        total_clusters: clusters.len(),
        clusters,
    })
}

pub fn refine_duplicates<E: Embedder>(
    pairs: Vec<DuplicatePair>,
    options: &RefineOptions,
    embedder: &mut E,
) -> Result<RefineOutput> {
    validate_weight(options.semantic_weight)?;
    let lexical_weight = 1.0 - options.semantic_weight;
    let mut warnings = Vec::new();
    let mut resolved_pairs = Vec::new();
    let total_pairs = pairs.len();

    for pair in pairs {
        match embed_duplicate_pair(&pair, &options.root, options.max_chars, embedder) {
            Ok(semantic_score) => {
                let refined_score = lexical_weight * pair.combined
                    + options.semantic_weight * cosine_to_score(semantic_score);
                if refined_score >= options.threshold {
                    resolved_pairs.push(RefinedDuplicatePair {
                        file1: pair.file1,
                        file2: pair.file2,
                        lexical_score: pair.combined,
                        semantic_score,
                        refined_score,
                        jaccard: pair.jaccard,
                        simhash: pair.simhash,
                        minhash: pair.minhash,
                    });
                }
            }
            Err(error) => {
                warnings.push(format!("{} <-> {}: {}", pair.file1, pair.file2, error));
                if pair.combined >= options.threshold {
                    resolved_pairs.push(RefinedDuplicatePair {
                        file1: pair.file1,
                        file2: pair.file2,
                        lexical_score: pair.combined,
                        semantic_score: 0.0,
                        refined_score: pair.combined,
                        jaccard: pair.jaccard,
                        simhash: pair.simhash,
                        minhash: pair.minhash,
                    });
                }
            }
        }
    }

    resolved_pairs.sort_by(|left, right| {
        right
            .refined_score
            .partial_cmp(&left.refined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.file1.cmp(&right.file1))
            .then_with(|| left.file2.cmp(&right.file2))
    });

    Ok(RefineOutput {
        command: "refine",
        total_pairs,
        retained_pairs: resolved_pairs.len(),
        threshold: options.threshold,
        lexical_weight,
        semantic_weight: options.semantic_weight,
        warnings,
        pairs: resolved_pairs,
    })
}

pub fn embed_texts<E: Embedder>(texts: &[String], embedder: &mut E) -> Result<EmbedOutput> {
    let mut embeddings_out = Vec::with_capacity(texts.len());
    for text in texts {
        embeddings_out.push(EmbeddedText {
            text: text.clone(),
            embedding: embedder.embed(text)?,
        });
    }
    let embedding_dim = embeddings_out
        .first()
        .map(|record| record.embedding.len())
        .unwrap_or(0);
    Ok(EmbedOutput {
        command: "embed",
        total_texts: embeddings_out.len(),
        embedding_dim,
        embeddings: embeddings_out,
    })
}

fn compare_terms(left: &VocabularyTerm, right: &VocabularyTerm) -> std::cmp::Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| right.count.cmp(&left.count))
        .then_with(|| left.term.cmp(&right.term))
}

fn normalized_scores(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }

    let min = scores
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let max = scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));

    if (max - min).abs() < f64::EPSILON {
        return vec![1.0; scores.len()];
    }

    scores
        .iter()
        .map(|score| (score - min) / (max - min))
        .collect()
}

fn embed_query_document<E: Embedder>(
    result: &QueryResult,
    root: &Path,
    max_chars: usize,
    embedder: &mut E,
    query_embedding: &[f32],
) -> Result<f64> {
    let content = fs::read_to_string(resolve_path(root, &result.path))
        .with_context(|| format!("failed to read {}", result.path))?;
    let excerpt = normalize_excerpt(&content, max_chars);
    let document_embedding = embedder.embed(&excerpt)?;
    Ok(embeddings::cosine_similarity(
        query_embedding,
        &document_embedding,
    ))
}

fn embed_duplicate_pair<E: Embedder>(
    pair: &DuplicatePair,
    root: &Path,
    max_chars: usize,
    embedder: &mut E,
) -> Result<f64> {
    let left = fs::read_to_string(resolve_path(root, &pair.file1))
        .with_context(|| format!("failed to read {}", pair.file1))?;
    let right = fs::read_to_string(resolve_path(root, &pair.file2))
        .with_context(|| format!("failed to read {}", pair.file2))?;
    let left_embedding = embedder.embed(&normalize_excerpt(&left, max_chars))?;
    let right_embedding = embedder.embed(&normalize_excerpt(&right, max_chars))?;
    Ok(embeddings::cosine_similarity(
        &left_embedding,
        &right_embedding,
    ))
}

fn resolve_path(root: &Path, value: &str) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn normalize_excerpt(text: &str, max_chars: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.chars().take(max_chars).collect()
}

fn cosine_to_score(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn validate_weight(weight: f64) -> Result<()> {
    if (0.0..=1.0).contains(&weight) {
        Ok(())
    } else {
        Err(anyhow!("semantic weight must be between 0.0 and 1.0"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yore_json::{
        DuplicatePair, QueryEnvelope, QueryResult, VocabularyResult, VocabularyTerm,
    };
    use serde_json::Map;
    use tempfile::tempdir;

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
            let lowered = text.to_lowercase();
            if lowered.contains("auth") || lowered.contains("login") || lowered.contains("signin") {
                Ok(vec![1.0, 0.0, 0.0])
            } else if lowered.contains("billing") || lowered.contains("invoice") {
                Ok(vec![0.0, 1.0, 0.0])
            } else {
                Ok(vec![0.0, 0.0, 1.0])
            }
        }
    }

    #[test]
    fn rerank_prefers_semantic_match() {
        let dir = tempdir().expect("tempdir");
        fs::create_dir_all(dir.path().join("docs")).expect("mkdir");
        fs::write(
            dir.path().join("docs/auth.md"),
            "Authentication and login guide",
        )
        .expect("write");
        fs::write(
            dir.path().join("docs/billing.md"),
            "Invoices and billing walkthrough",
        )
        .expect("write");

        let payload = QueryEnvelope {
            results: vec![
                QueryResult {
                    path: "docs/billing.md".into(),
                    score: 10.0,
                    doc_terms: Vec::new(),
                    extra: Map::new(),
                },
                QueryResult {
                    path: "docs/auth.md".into(),
                    score: 5.0,
                    doc_terms: Vec::new(),
                    extra: Map::new(),
                },
            ],
            diagnostics: None,
            extra: Map::new(),
        };

        let output = rerank_query_results(
            payload,
            "auth",
            &RerankOptions {
                root: dir.path().to_path_buf(),
                ..RerankOptions::default()
            },
            &mut MockEmbedder,
        )
        .expect("rerank");

        assert_eq!(output.results[0].path, "docs/auth.md");
    }

    #[test]
    fn cluster_groups_semantic_neighbors() {
        let payload = VocabularyResult {
            format: "json".into(),
            limit: 3,
            total: 3,
            terms: vec![
                VocabularyTerm {
                    term: "login".into(),
                    score: 5.0,
                    count: 10,
                    extra: Map::new(),
                },
                VocabularyTerm {
                    term: "signin".into(),
                    score: 4.0,
                    count: 8,
                    extra: Map::new(),
                },
                VocabularyTerm {
                    term: "invoice".into(),
                    score: 3.0,
                    count: 6,
                    extra: Map::new(),
                },
            ],
            stopwords: None,
            used_default_stopwords: true,
            auto_common_terms: None,
            include_stemming: false,
            extra: Map::new(),
        };

        let output = cluster_vocabulary(
            payload,
            &ClusterOptions {
                threshold: 0.8,
                limit: None,
            },
            &mut MockEmbedder,
        )
        .expect("cluster");

        assert_eq!(output.total_clusters, 2);
        assert_eq!(output.clusters[0].members.len(), 2);
        assert_eq!(output.clusters[0].members[0].term, "login");
    }

    #[test]
    fn refine_filters_low_semantic_pairs() {
        let dir = tempdir().expect("tempdir");
        fs::create_dir_all(dir.path().join("docs")).expect("mkdir");
        fs::write(
            dir.path().join("docs/auth.md"),
            "Authentication and login guide",
        )
        .expect("write");
        fs::write(dir.path().join("docs/login.md"), "User login and auth flow").expect("write");
        fs::write(
            dir.path().join("docs/billing.md"),
            "Invoices and billing walkthrough",
        )
        .expect("write");

        let pairs = vec![
            DuplicatePair {
                file1: "docs/auth.md".into(),
                file2: "docs/login.md".into(),
                jaccard: 0.8,
                simhash: 0.8,
                minhash: 0.8,
                combined: 0.8,
                extra: Map::new(),
            },
            DuplicatePair {
                file1: "docs/auth.md".into(),
                file2: "docs/billing.md".into(),
                jaccard: 0.8,
                simhash: 0.8,
                minhash: 0.8,
                combined: 0.8,
                extra: Map::new(),
            },
        ];

        let output = refine_duplicates(
            pairs,
            &RefineOptions {
                root: dir.path().to_path_buf(),
                threshold: 0.75,
                semantic_weight: 0.55,
                max_chars: DEFAULT_DOC_MAX_CHARS,
            },
            &mut MockEmbedder,
        )
        .expect("refine");

        assert_eq!(output.pairs.len(), 1);
        assert_eq!(output.pairs[0].file2, "docs/login.md");
    }

    #[test]
    fn embed_returns_all_requested_texts() {
        let output =
            embed_texts(&["auth".into(), "billing".into()], &mut MockEmbedder).expect("embed");
        assert_eq!(output.total_texts, 2);
        assert_eq!(output.embedding_dim, 3);
    }

    #[test]
    fn normalized_scores_handle_equal_values() {
        assert_eq!(normalized_scores(&[2.0, 2.0]), vec![1.0, 1.0]);
    }

    #[test]
    fn cosine_score_clamps() {
        assert_eq!(cosine_to_score(1.0), 1.0);
        assert_eq!(cosine_to_score(-1.0), 0.0);
        assert_eq!(cosine_to_score(0.0), 0.0);
    }

    #[test]
    fn refine_total_pairs_counts_pairs_and_warnings() {
        let output = RefineOutput {
            command: "refine",
            total_pairs: 2,
            retained_pairs: 1,
            threshold: 0.7,
            lexical_weight: 0.45,
            semantic_weight: 0.55,
            warnings: vec!["warn".into()],
            pairs: vec![RefinedDuplicatePair {
                file1: "a".into(),
                file2: "b".into(),
                lexical_score: 0.8,
                semantic_score: 0.9,
                refined_score: 0.85,
                jaccard: 0.8,
                simhash: 0.8,
                minhash: 0.8,
            }],
        };
        assert_eq!(output.total_pairs, 2);
    }
}
