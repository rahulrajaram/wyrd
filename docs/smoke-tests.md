# Smoke Tests

## 2026-03-24: `yore query --json | wyrd rerank`

Smoke command:

```bash
tmpdir=$(mktemp -d /tmp/wyrd-smoke-XXXXXX)
yore build /home/rahul/Documents/yore --output "$tmpdir"/index --types md,txt,rst
export WYRD_EMBEDDING_MODEL=/home/rahul/Documents/haake/models/all-MiniLM-L6-v2/model.onnx
export WYRD_EMBEDDING_TOKENIZER=/home/rahul/Documents/haake/models/all-MiniLM-L6-v2/tokenizer.json
yore query --query "index path" --json --doc-terms 5 --explain --index "$tmpdir"/index \
  | cargo run --quiet -- rerank --query "index path" --limit 3
```

Result:

- PASS. `wyrd rerank` consumed live `yore query --json` output and returned three reranked results with no warnings or stderr noise.
- The top three results stayed `/home/rahul/Documents/yore/README.md`, `/home/rahul/Documents/yore/docs/YORE_MAINTENANCE_WORKFLOWS.md`, and `/home/rahul/Documents/yore/CHANGELOG.md`, so this smoke test validated the live pipe more than it changed the ordering.

Quirks:

- This historical smoke predates the embedded-query handoff, so `wyrd rerank --query` had to repeat `"index path"`.
- Current `wyrd` also tolerates rootless absolute-style paths from `yore` JSON when the indexed source path was absolute.

## 2026-03-25: `yore query --json | wyrd rerank` without `--query`

Smoke command:

```bash
tmpdir=$(mktemp -d /tmp/wyrd-smoke-XXXXXX)
yore build /home/rahul/Documents/yore --output "$tmpdir"/index --types md,txt,rst
export WYRD_EMBEDDING_MODEL=/home/rahul/Documents/haake/models/all-MiniLM-L6-v2/model.onnx
export WYRD_EMBEDDING_TOKENIZER=/home/rahul/Documents/haake/models/all-MiniLM-L6-v2/tokenizer.json
yore query --query "index path" --json --doc-terms 5 --explain --index "$tmpdir"/index \
  | cargo run --quiet -- rerank --limit 3
```

Result:

- PASS. `wyrd rerank` inferred the original query from live `yore query --json` output and returned three reranked results.
- The same three `yore` documents stayed at the top, so the run validated the cross-repo query handoff more than ranking changes.

Quirks:

- `yore query --json --explain` now carries the original query at the wrapper level and per-result, keeping older result-array consumers usable.
- Current `yore` builds may emit rootless absolute-style paths like `home/rahul/...`; `wyrd` now resolves those without extra flags.
