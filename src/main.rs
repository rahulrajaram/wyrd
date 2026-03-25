use std::env;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use wyrd::embeddings;
use wyrd::ops::{
    ClusterOptions, DEFAULT_DOC_MAX_CHARS, DEFAULT_REFINE_SEMANTIC_WEIGHT,
    DEFAULT_RERANK_SEMANTIC_WEIGHT, Embedder, OrtEmbedder, RefineOptions, RerankOptions,
    cluster_vocabulary, embed_texts, refine_duplicates, rerank_query_results,
};
use wyrd::yore_json::{parse_duplicate_pairs, parse_query_payload, parse_vocabulary_payload};

#[derive(Parser, Debug)]
#[command(
    name = "wyrd",
    version,
    about = "Semantic companion CLI for yore JSON pipelines"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Semantic reranking for `yore query --json` output.
    Rerank {
        /// Original query text. Pass the same string used with `yore query --query`
        /// because current yore JSON output omits it.
        #[arg(long)]
        query: String,

        /// Root directory used to resolve relative yore result paths.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Weight for semantic similarity when combining with lexical score.
        #[arg(long, default_value_t = DEFAULT_RERANK_SEMANTIC_WEIGHT)]
        semantic_weight: f64,

        /// Trim output to the top N reranked results.
        #[arg(long)]
        limit: Option<usize>,

        /// Maximum normalized characters to embed per file.
        #[arg(long, default_value_t = DEFAULT_DOC_MAX_CHARS)]
        max_chars: usize,
    },

    /// Semantic clustering for `yore vocabulary --format json` output.
    Cluster {
        /// Cosine similarity threshold for graph clustering.
        #[arg(long)]
        threshold: Option<f64>,

        /// Only cluster the top N vocabulary terms.
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Semantic refinement for `yore dupes --json` output.
    Refine {
        /// Root directory used to resolve relative file paths from yore dupes output.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Minimum refined score to retain a pair.
        #[arg(long)]
        threshold: Option<f64>,

        /// Weight for semantic similarity when combining with yore's lexical duplicate score.
        #[arg(long, default_value_t = DEFAULT_REFINE_SEMANTIC_WEIGHT)]
        semantic_weight: f64,

        /// Maximum normalized characters to embed per file.
        #[arg(long, default_value_t = DEFAULT_DOC_MAX_CHARS)]
        max_chars: usize,
    },

    /// Emit raw embeddings for text provided as args or via stdin.
    Embed {
        /// Embed each non-empty stdin line separately instead of the whole stdin payload.
        #[arg(long)]
        lines: bool,

        /// Text values to embed. If omitted, stdin is used.
        texts: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut embedder = AppEmbedder::from_env()?;

    match cli.command {
        Commands::Rerank {
            query,
            root,
            semantic_weight,
            limit,
            max_chars,
        } => {
            let stdin = read_required_stdin()?;
            let payload = parse_query_payload(&stdin)?;
            let output = rerank_query_results(
                payload,
                &query,
                &RerankOptions {
                    root,
                    semantic_weight,
                    limit,
                    max_chars,
                },
                &mut embedder,
            )?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Commands::Cluster { threshold, limit } => {
            let stdin = read_required_stdin()?;
            let payload = parse_vocabulary_payload(&stdin)?;
            let output = cluster_vocabulary(
                payload,
                &ClusterOptions {
                    threshold: threshold.unwrap_or_else(embeddings::embedding_threshold),
                    limit,
                },
                &mut embedder,
            )?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Commands::Refine {
            root,
            threshold,
            semantic_weight,
            max_chars,
        } => {
            let stdin = read_required_stdin()?;
            let pairs = parse_duplicate_pairs(&stdin)?;
            let output = refine_duplicates(
                pairs,
                &RefineOptions {
                    root,
                    threshold: threshold.unwrap_or_else(embeddings::embedding_threshold),
                    semantic_weight,
                    max_chars,
                },
                &mut embedder,
            )?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Commands::Embed { lines, texts } => {
            let texts = collect_embed_texts(texts, lines)?;
            let output = embed_texts(&texts, &mut embedder)?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}

enum AppEmbedder {
    Ort(OrtEmbedder),
    KeywordFixture(KeywordFixtureEmbedder),
}

impl AppEmbedder {
    fn from_env() -> Result<Self> {
        match env::var("WYRD_TEST_EMBEDDER").ok().as_deref() {
            // Allows subprocess integration tests to cover CLI behavior without local ONNX assets.
            Some("keyword-fixture") => Ok(Self::KeywordFixture(KeywordFixtureEmbedder)),
            Some(other) => bail!("unsupported WYRD_TEST_EMBEDDER value: {other}"),
            None => Ok(Self::Ort(OrtEmbedder::new())),
        }
    }
}

impl Embedder for AppEmbedder {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Ort(embedder) => embedder.embed(text),
            Self::KeywordFixture(embedder) => embedder.embed(text),
        }
    }
}

struct KeywordFixtureEmbedder;

impl Embedder for KeywordFixtureEmbedder {
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

fn read_required_stdin() -> Result<String> {
    if io::stdin().is_terminal() {
        bail!("this subcommand expects JSON on stdin");
    }
    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read stdin")?;
    if buffer.trim().is_empty() {
        bail!("stdin was empty");
    }
    Ok(buffer)
}

fn collect_embed_texts(texts: Vec<String>, lines: bool) -> Result<Vec<String>> {
    if !texts.is_empty() {
        return Ok(texts);
    }

    if io::stdin().is_terminal() {
        bail!(
            "embed expects text arguments or piped stdin. {}",
            wyrd::ops::missing_model_message()
        );
    }

    let stdin = read_required_stdin()?;
    let values = if lines {
        stdin
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    } else {
        let trimmed = stdin.trim();
        if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed.to_string()]
        }
    };

    if values.is_empty() {
        bail!("no text was provided to embed");
    }

    Ok(values)
}
