#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
uv sync

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$OUTPUT_DIR" outputs/mbpp_lora_sft

uv run python -m src.train_mbpp_sft \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --work-dir outputs/mbpp_lora_sft \
  --seed 13 \
  --instruction-limit 2500 \
  --max-steps 260 \
  --learning-rate 2e-4
