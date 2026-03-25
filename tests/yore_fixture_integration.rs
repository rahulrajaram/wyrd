use std::fs;

use anyhow::Result;
use tempfile::tempdir;
use wyrd::ops::{
    ClusterOptions, Embedder, RefineOptions, RerankOptions, cluster_vocabulary, refine_duplicates,
    rerank_query_results,
};
use wyrd::yore_json::{parse_duplicate_pairs, parse_query_payload, parse_vocabulary_payload};

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
fn rerank_array_fixture_prefers_auth_doc() {
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

    let payload = parse_query_payload(include_str!("fixtures/query_results.json")).expect("parse");
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
fn rerank_wrapped_fixture_preserves_diagnostics() {
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

    let payload = parse_query_payload(include_str!("fixtures/query_wrapped.json")).expect("parse");
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

    assert!(output.diagnostics.is_some());
    assert_eq!(output.results[0].path, "docs/auth.md");
}

#[test]
fn cluster_vocabulary_fixture_groups_login_and_signin() {
    let payload =
        parse_vocabulary_payload(include_str!("fixtures/vocabulary.json")).expect("parse");
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
    assert_eq!(output.clusters[0].members[1].term, "signin");
}

#[test]
fn refine_dupes_fixture_drops_billing_pair() {
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

    let pairs = parse_duplicate_pairs(include_str!("fixtures/dupes.json")).expect("parse");
    let output = refine_duplicates(
        pairs,
        &RefineOptions {
            root: dir.path().to_path_buf(),
            threshold: 0.75,
            semantic_weight: 0.55,
            ..RefineOptions::default()
        },
        &mut MockEmbedder,
    )
    .expect("refine");

    assert_eq!(output.total_pairs, 2);
    assert_eq!(output.retained_pairs, 1);
    assert_eq!(output.pairs[0].file2, "docs/login.md");
}
