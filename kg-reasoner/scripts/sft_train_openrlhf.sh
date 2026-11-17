#!/usr/bin/env bash
# SFT warm-up for large backbones (LLaMA-3.1-8B-Instruct / Qwen3-30B-A3B) with OpenRLHF + DeepSpeed.
# Data: scripts/build_warmup_data.py output (fields: prompt, response).
# Note: unlike scripts/sft_warmup.py, the OpenRLHF SFT trainer also trains on the <triples> tokens.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=${ROOT}:${ROOT}/OpenRLHF-RAG:${PYTHONPATH:-}
BASE_MODEL=${BASE_MODEL:?set BASE_MODEL, e.g. meta-llama/Llama-3.1-8B-Instruct}
deepspeed --module openrlhf.cli.train_sft \
  --pretrain "${BASE_MODEL}" \
  --dataset "${WARMUP_DATA:-${ROOT}/data/sft/warmup.jsonl}" --input_key prompt --output_key response \
  --max_len 4096 --max_epochs 2 --learning_rate 1e-5 \
  --train_batch_size 64 --micro_train_batch_size 2 \
  --zero_stage 3 --bf16 --flash_attn --gradient_checkpointing \
  --save_path "${SAVE_DIR:-${ROOT}/ckpt/sft}"
