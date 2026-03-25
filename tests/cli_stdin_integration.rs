use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;
use tempfile::tempdir;

fn run_wyrd(args: &[&str], stdin: &str) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_wyrd"))
        .args(args)
        .env("WYRD_TEST_EMBEDDER", "keyword-fixture")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn wyrd");

    child
        .stdin
        .as_mut()
        .expect("stdin pipe")
        .write_all(stdin.as_bytes())
        .expect("write stdin");

    let output = child.wait_with_output().expect("wait for wyrd");
    assert!(
        output.status.success(),
        "wyrd exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).expect("parse command output as JSON")
}

fn run_wyrd_output(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_wyrd"))
        .args(args)
        .env("WYRD_TEST_EMBEDDER", "keyword-fixture")
        .output()
        .expect("spawn wyrd")
}

fn write_doc(path: &Path, body: &str) {
    fs::write(path, body).expect("write fixture document");
}

#[test]
fn rerank_reads_query_results_from_stdin() {
    let dir = tempdir().expect("tempdir");
    fs::create_dir_all(dir.path().join("docs")).expect("mkdir docs");
    write_doc(
        &dir.path().join("docs/auth.md"),
        "Authentication and login guide",
    );
    write_doc(
        &dir.path().join("docs/billing.md"),
        "Invoices and billing walkthrough",
    );

    let root = dir.path().display().to_string();
    let output = run_wyrd(
        &["rerank", "--query", "auth", "--root", &root],
        include_str!("fixtures/query_results.json"),
    );

    assert_eq!(output["command"], "rerank");
    assert_eq!(output["results"][0]["path"], "docs/auth.md");
    assert_eq!(output["total_results"], 2);
}

#[test]
fn rerank_help_documents_query_contract() {
    let output = run_wyrd_output(&["rerank", "--help"]);
    assert!(
        output.status.success(),
        "wyrd help exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("yore query --query"));
    assert!(stdout.contains("omits it"));
}

#[test]
fn cluster_reads_vocabulary_from_stdin() {
    let output = run_wyrd(
        &["cluster", "--threshold", "0.8"],
        include_str!("fixtures/vocabulary.json"),
    );

    assert_eq!(output["command"], "cluster");
    assert_eq!(output["total_clusters"], 2);
    assert_eq!(output["clusters"][0]["members"][0]["term"], "login");
    assert_eq!(output["clusters"][0]["members"][1]["term"], "signin");
}

#[test]
fn refine_reads_duplicate_pairs_from_stdin() {
    let dir = tempdir().expect("tempdir");
    fs::create_dir_all(dir.path().join("docs")).expect("mkdir docs");
    write_doc(
        &dir.path().join("docs/auth.md"),
        "Authentication and login guide",
    );
    write_doc(
        &dir.path().join("docs/login.md"),
        "User login and auth flow",
    );
    write_doc(
        &dir.path().join("docs/billing.md"),
        "Invoices and billing walkthrough",
    );

    let root = dir.path().display().to_string();
    let output = run_wyrd(
        &["refine", "--root", &root, "--threshold", "0.75"],
        include_str!("fixtures/dupes.json"),
    );

    assert_eq!(output["command"], "refine");
    assert_eq!(output["retained_pairs"], 1);
    assert_eq!(output["pairs"][0]["file2"], "docs/login.md");
}

#[test]
fn embed_reads_line_fixture_from_stdin() {
    let output = run_wyrd(
        &["embed", "--lines"],
        include_str!("fixtures/embed_lines.txt"),
    );

    assert_eq!(output["command"], "embed");
    assert_eq!(output["total_texts"], 2);
    assert_eq!(output["embedding_dim"], 3);
    assert_eq!(output["embeddings"][0]["text"], "auth");
    assert_eq!(output["embeddings"][1]["text"], "billing");
}
