# wyrd

`wyrd` is optional semantic tooling that pairs well with [`yore`](../yore) when
embedding-driven workflows are useful. `yore` stands on its own for
deterministic, lexical search, and `wyrd` is available when you want semantic
reranking, clustering, refinement, or raw embeddings on top of `yore` output or
other pipe-friendly inputs.

## Why it exists

- `yore` stays deterministic: BM25, hashes, and link analysis remain the core.
- `wyrd` adds semantic workflows only when they help; it is not a required
  sidecar for everyday `yore` usage.
- The split keeps model-backed behavior opt-in while still making `yore` and
  `wyrd` easy to combine in Unix-pipe workflows.

## Install

```bash
cargo install --path .
```

## Model setup

`wyrd` expects a local ONNX export of `all-MiniLM-L6-v2` plus a matching `tokenizer.json`.

Set:

```bash
export WYRD_EMBEDDING_MODEL=/path/to/all-MiniLM-L6-v2.onnx
export WYRD_EMBEDDING_TOKENIZER=/path/to/tokenizer.json
```

Optional:

```bash
export WYRD_EMBEDDING_THRESHOLD=0.8
```

Notes:

- The ONNX model is not vendored in this repo.
- `ort` and `ort-sys` are pinned to `2.0.0-rc.6`.
- This repo currently targets `rustc 1.86.x`; newer `ort` release candidates may require `rustc 1.88+`.

## Copy-paste pipelines

### `rerank`

Semantic reranking for `yore query --json` output.

```bash
yore query auth --json | wyrd rerank --query auth
yore query "token refresh" --json | wyrd rerank --query "token refresh" --limit 5
yore query "password reset" --json | wyrd rerank --query "password reset" --root "$PWD"
```

Contract:

- Pass the exact same query string to `yore query` and `wyrd rerank --query`.

```bash
query="token refresh"
yore query "$query" --json | wyrd rerank --query "$query" --limit 5
```

Why `--query` is required:

- Current `yore query --json` output includes scores and paths, but not the original query text.
- Until `yore` includes that field, `wyrd rerank --query` is the supported interface.

### `cluster`

Semantic grouping for `yore vocabulary --format json`.

```bash
yore vocabulary --index .yore --format json --limit 200 | wyrd cluster
yore vocabulary --index .yore --format json --limit 100 | wyrd cluster --threshold 0.82
yore vocabulary --index .yore --format json | wyrd cluster --limit 50
```

### `refine`

Semantic refinement for `yore dupes --json`.

```bash
yore dupes --index .yore --json | wyrd refine
yore dupes --index .yore --json | wyrd refine --threshold 0.78
yore dupes --index .yore --json | wyrd refine --root "$PWD" --semantic-weight 0.6
```

### `embed`

Raw text to embedding JSON.

```bash
printf 'authentication flow\n' | wyrd embed
printf 'auth\nbilling\n' | wyrd embed --lines
printf 'authentication flow\nbilling webhook\npassword reset\n' | wyrd embed --lines
```

## Path resolution

- `yore` JSON often contains relative paths.
- `wyrd` resolves them relative to the current working directory by default.
- Use `--root` if you want to point resolution somewhere else.

## Development

```bash
cargo check
cargo test
```
