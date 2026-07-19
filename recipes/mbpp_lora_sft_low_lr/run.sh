#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
uv sync

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export HF_MODULES_CACHE="$HF_HOME/modules"
export XDG_CACHE_HOME="$PWD/.cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$OUTPUT_DIR" outputs/mbpp_lora_sft_low_lr

uv run python -m src.train_mbpp_sft \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --work-dir outputs/mbpp_lora_sft_low_lr \
  --seed 13 \
  --instruction-limit 2500 \
  --max-steps 180 \
  --learning-rate 1e-4
