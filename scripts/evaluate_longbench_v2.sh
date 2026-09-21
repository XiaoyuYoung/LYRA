#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a local LYRA checkpoint directory}"
: "${LONGBENCH_V2:?Set LONGBENCH_V2 to the official local data.json file}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/evaluation}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-16384}"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --longbench-v2 "$LONGBENCH_V2"
  --benchmark longbench-v2
  --output-dir "$OUTPUT_DIR"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
)

if [[ -n "${BASE_MODEL:-}" ]]; then
  ARGS+=(--base-model "$BASE_MODEL")
fi

python -m lyra.eval_longbench "${ARGS[@]}" "$@"
