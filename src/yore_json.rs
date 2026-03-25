use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryResult {
    pub path: String,
    pub score: f64,
    #[serde(default)]
    pub doc_terms: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryEnvelope {
    pub results: Vec<QueryResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl QueryEnvelope {
    pub fn original_query(&self) -> Option<&str> {
        self.extra.get("query").and_then(Value::as_str).or_else(|| {
            self.results
                .iter()
                .find_map(|result| result.extra.get("query").and_then(Value::as_str))
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct QueryError {
    #[allow(dead_code)]
    query: Option<String>,
    error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VocabularyTerm {
    pub term: String,
    pub score: f64,
    pub count: usize,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VocabularyResult {
    pub format: String,
    pub limit: usize,
    pub total: usize,
    pub terms: Vec<VocabularyTerm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopwords: Option<String>,
    #[serde(default)]
    pub used_default_stopwords: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_common_terms: Option<usize>,
    #[serde(default)]
    pub include_stemming: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DuplicatePair {
    pub file1: String,
    pub file2: String,
    pub jaccard: f64,
    pub simhash: f64,
    pub minhash: f64,
    pub combined: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

pub fn parse_query_payload(input: &str) -> Result<QueryEnvelope> {
    let value: Value = serde_json::from_str(input).context("failed to parse query JSON")?;
    if let Ok(results) = serde_json::from_value::<Vec<QueryResult>>(value.clone()) {
        return Ok(QueryEnvelope {
            results,
            diagnostics: None,
            extra: Map::new(),
        });
    }

    if let Ok(wrapper) = serde_json::from_value::<QueryEnvelope>(value.clone()) {
        return Ok(wrapper);
    }

    if let Ok(error) = serde_json::from_value::<QueryError>(value) {
        bail!("yore query returned an error payload: {}", error.error);
    }

    Err(anyhow!(
        "stdin was valid JSON, but not a recognized yore query payload"
    ))
}

pub fn parse_vocabulary_payload(input: &str) -> Result<VocabularyResult> {
    serde_json::from_str(input)
        .context("failed to parse vocabulary JSON")
        .map_err(|error| anyhow!("stdin was not a recognized yore vocabulary payload: {error}"))
}

pub fn parse_duplicate_pairs(input: &str) -> Result<Vec<DuplicatePair>> {
    serde_json::from_str(input)
        .context("failed to parse dupes JSON")
        .map_err(|error| anyhow!("stdin was not a recognized yore dupes payload: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_query_array_payload() {
        let payload = r#"[{"path":"docs/auth.md","score":1.2,"query":"auth"}]"#;
        let parsed = parse_query_payload(payload).expect("should parse");
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.results[0].path, "docs/auth.md");
        assert!(parsed.diagnostics.is_none());
        assert_eq!(parsed.original_query(), Some("auth"));
    }

    #[test]
    fn parses_query_wrapper_payload() {
        let payload = r#"{"query":"auth","results":[{"path":"docs/auth.md","score":1.2}],"diagnostics":{"tokens":["auth"]}}"#;
        let parsed = parse_query_payload(payload).expect("should parse");
        assert_eq!(parsed.results.len(), 1);
        assert!(parsed.diagnostics.is_some());
        assert_eq!(parsed.original_query(), Some("auth"));
    }

    #[test]
    fn rejects_query_error_payload() {
        let payload = r#"{"query":"the and","error":"no_query_terms"}"#;
        let error = parse_query_payload(payload).expect_err("should fail");
        assert!(error.to_string().contains("no_query_terms"));
    }
}
