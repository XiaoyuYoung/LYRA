#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a local Qwen3-8B directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a local JSON or JSONL training shard}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/lyra-qwen3-8b}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m lyra.train \
  --model "$MODEL_DIR" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "$MAX_LENGTH" \
  --strategy last_block \
  --tvmf-layers last \
  --gradient-checkpointing \
  "$@"
