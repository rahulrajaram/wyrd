//! Optional ONNX-based embedding support for semantic similarity.
//!
//! Provides local inference using all-MiniLM-L6-v2 (384-dim, ~23MB INT8 ONNX)
//! for semantic reranking, clustering, and duplicate refinement.
//!
//! # Environment Variables
//!
//! - `WYRD_EMBEDDING_MODEL`: Path to `.onnx` model file (required to enable)
//! - `WYRD_EMBEDDING_TOKENIZER`: Path to `tokenizer.json` (defaults to model dir)
//! - `WYRD_EMBEDDING_THRESHOLD`: Default cosine similarity threshold (default: 0.8)

use std::env;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use ndarray::Array2;
use ort::Session;
use tokenizers::Tokenizer;

/// Dimensionality of the all-MiniLM-L6-v2 embedding space.
pub const EMBEDDING_DIM: usize = 384;

/// Default cosine similarity threshold for semantic operations.
pub const DEFAULT_EMBEDDING_THRESHOLD: f64 = 0.8;

/// Errors that can occur during embedding operations.
#[derive(Debug)]
pub enum EmbeddingError {
    ModelLoad(String),
    TokenizerLoad(String),
    Inference(String),
    InvalidData(String),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingError::ModelLoad(msg) => write!(f, "model load error: {msg}"),
            EmbeddingError::TokenizerLoad(msg) => write!(f, "tokenizer load error: {msg}"),
            EmbeddingError::Inference(msg) => write!(f, "inference error: {msg}"),
            EmbeddingError::InvalidData(msg) => write!(f, "invalid embedding data: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

static MODEL: OnceLock<Option<Session>> = OnceLock::new();
static TOKENIZER: OnceLock<Option<Tokenizer>> = OnceLock::new();

fn get_model() -> Option<&'static Session> {
    let session = MODEL.get_or_init(|| {
        let model_path = env::var("WYRD_EMBEDDING_MODEL").ok()?;
        let path = Path::new(&model_path);
        if !path.exists() {
            tracing::warn!("WYRD_EMBEDDING_MODEL path does not exist: {}", model_path);
            return None;
        }
        match Session::builder()
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.commit_from_file(path))
        {
            Ok(session) => {
                tracing::info!("Loaded ONNX embedding model from {}", model_path);
                Some(session)
            }
            Err(error) => {
                tracing::warn!("Failed to load ONNX model: {}", error);
                None
            }
        }
    });
    session.as_ref()
}

fn get_tokenizer() -> Option<&'static Tokenizer> {
    let tokenizer = TOKENIZER.get_or_init(|| {
        let tokenizer_path = if let Ok(path) = env::var("WYRD_EMBEDDING_TOKENIZER") {
            path
        } else if let Ok(model_path) = env::var("WYRD_EMBEDDING_MODEL") {
            let dir = Path::new(&model_path).parent().unwrap_or(Path::new("."));
            dir.join("tokenizer.json").to_string_lossy().to_string()
        } else {
            return None;
        };

        let path = Path::new(&tokenizer_path);
        if !path.exists() {
            tracing::warn!(
                "Tokenizer file not found at {}, embeddings disabled",
                tokenizer_path
            );
            return None;
        }

        match Tokenizer::from_file(path) {
            Ok(tokenizer) => {
                tracing::info!("Loaded tokenizer from {}", tokenizer_path);
                Some(tokenizer)
            }
            Err(error) => {
                tracing::warn!("Failed to load tokenizer: {}", error);
                None
            }
        }
    });
    tokenizer.as_ref()
}

/// Returns true if `WYRD_EMBEDDING_MODEL` is set.
pub fn embeddings_enabled() -> bool {
    env::var("WYRD_EMBEDDING_MODEL").is_ok()
}

/// Returns the configured embedding threshold, or [`DEFAULT_EMBEDDING_THRESHOLD`].
pub fn embedding_threshold() -> f64 {
    env::var("WYRD_EMBEDDING_THRESHOLD")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_EMBEDDING_THRESHOLD)
}

/// Compute an embedding for the given text.
///
/// Returns `Ok(None)` if embeddings are not configured.
pub fn embed(text: &str) -> Result<Option<Vec<f32>>, EmbeddingError> {
    let session = match get_model() {
        Some(session) => session,
        None => return Ok(None),
    };
    let tokenizer = match get_tokenizer() {
        Some(tokenizer) => tokenizer,
        None => return Ok(None),
    };

    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| EmbeddingError::Inference(format!("tokenization failed: {error}")))?;

    let ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let type_ids = encoding.get_type_ids();
    let seq_len = ids.len();

    let input_ids = Array2::from_shape_vec(
        (1, seq_len),
        ids.iter().map(|&value| value as i64).collect(),
    )
    .map_err(|error| EmbeddingError::Inference(format!("input_ids shape error: {error}")))?;
    let attention = Array2::from_shape_vec(
        (1, seq_len),
        attention_mask.iter().map(|&value| value as i64).collect(),
    )
    .map_err(|error| EmbeddingError::Inference(format!("attention_mask shape error: {error}")))?;
    let token_types = Array2::from_shape_vec(
        (1, seq_len),
        type_ids.iter().map(|&value| value as i64).collect(),
    )
    .map_err(|error| EmbeddingError::Inference(format!("token_type_ids shape error: {error}")))?;

    let inputs = ort::inputs![
        "input_ids" => input_ids.view(),
        "attention_mask" => attention.view(),
        "token_type_ids" => token_types.view(),
    ]
    .map_err(|error| EmbeddingError::Inference(format!("input construction failed: {error}")))?;

    let outputs = session
        .run(inputs)
        .map_err(|error| EmbeddingError::Inference(format!("model run failed: {error}")))?;

    let output_tensor = outputs
        .get("last_hidden_state")
        .or_else(|| outputs.values().next())
        .ok_or_else(|| EmbeddingError::Inference("no output tensor found".to_string()))?;

    let array = output_tensor
        .try_extract_tensor::<f32>()
        .map_err(|error| EmbeddingError::Inference(format!("tensor extraction failed: {error}")))?;

    let hidden = array.view();
    let mut pooled = vec![0.0f32; EMBEDDING_DIM];

    let mask_sum: f32 = attention_mask.iter().map(|&value| value as f32).sum();
    if mask_sum == 0.0 {
        return Ok(Some(pooled));
    }

    for (token_idx, &mask_value) in attention_mask.iter().enumerate() {
        if mask_value == 0 {
            continue;
        }
        for dim in 0..EMBEDDING_DIM {
            pooled[dim] += hidden[[0, token_idx, dim]];
        }
    }

    for value in &mut pooled {
        *value /= mask_sum;
    }

    l2_normalize(&mut pooled);
    Ok(Some(pooled))
}

/// Cosine similarity between two embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "embedding dimensions must match");
    a.iter()
        .zip(b.iter())
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum()
}

/// Encode an embedding as little-endian f32 bytes.
pub fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decode an embedding from little-endian f32 bytes.
pub fn decode_embedding(bytes: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
    if bytes.len() % 4 != 0 {
        return Err(EmbeddingError::InvalidData(format!(
            "embedding blob length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(buf));
    }
    Ok(values)
}

fn l2_normalize(values: &mut [f32]) {
    let norm: f32 = values
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let mut vector = vec![1.0f32; EMBEDDING_DIM];
        l2_normalize(&mut vector);
        let similarity = cosine_similarity(&vector, &vector);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let mut left = vec![0.0f32; EMBEDDING_DIM];
        let mut right = vec![0.0f32; EMBEDDING_DIM];
        left[0] = 1.0;
        right[1] = 1.0;
        let similarity = cosine_similarity(&left, &right);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let original: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|index| index as f32 * 0.01)
            .collect();
        let encoded = encode_embedding(&original);
        let decoded = decode_embedding(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_embedding_rejects_bad_length() {
        let bad = vec![0u8; 5];
        assert!(decode_embedding(&bad).is_err());
    }
}
