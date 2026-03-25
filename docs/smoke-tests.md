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

- `yore query --json` still omits the original query text at the top level, so `wyrd rerank --query` had to repeat `"index path"`.
- Building the `yore` index from an absolute source path produced absolute `path` values in the JSON response; `wyrd` handled them without extra flags.
