#!/usr/bin/env bash
# GRPO training with OpenRLHF + Ray + vLLM (paper Sec. 4.3, Appendix A).
# Hyper-parameters from Appendix A: 16 rollouts per query, batch size 16, lr 1e-6, rollout temperature 1,
# PPO clip ratio 0.2, KL coefficient 1e-5, 2 epochs; hard-case resampling from the 2nd epoch on.
#
# Before running (separate terminals / tmux panes):
#   bash scripts/ray_start.sh
#   python -m kg_reasoner.kg.server --kg_backend sparql --sparql_endpoint http://localhost:8890/sparql \
#          --kg_ranker gnn --gnn_ckpt ckpt/gnn.pt --port 5003
#   python reward-remote/reward_server.py --data_path data/rl/train.jsonl --port 1278 \
#          --answer_judge hf --judge_model meta-llama/Llama-3.2-3B-Instruct
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=${ROOT}:${ROOT}/OpenRLHF-RAG:${PYTHONPATH:-}
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

SFT_MODEL=${SFT_MODEL:?set SFT_MODEL to the SFT warm-up checkpoint}   # scripts/sft_warmup.py / sft_train_openrlhf.sh
TRAIN_DATA=${TRAIN_DATA:-${ROOT}/data/rl/train.jsonl}                 # scripts/build_rl_data.py (2,000 samples)
HARD_DATA=${HARD_DATA:-${ROOT}/data/rl/hard.jsonl}                    # scripts/select_hard_cases.py
SAVE_DIR=${SAVE_DIR:-${ROOT}/ckpt/grpo}
KG_SERVER=${KG_SERVER:-http://127.0.0.1:5003}
RM_URL=${RM_URL:-http://127.0.0.1:1278/get_reward}
WANDB_ARGS=${WANDB_TOKEN:+--use_wandb ${WANDB_TOKEN} --wandb_run_name kg-reasoner}

N_SAMPLES=16          # rollouts per query
ROLLOUT_BS=16         # queries per rollout batch ("batch size of 16")
TRAIN_BS=$((N_SAMPLES * ROLLOUT_BS))
LR=1e-6
KL=1e-5
TEMP=1.0
CLIP=0.2

run_epoch () {   # $1 = policy init, $2 = prompt data, $3 = save path
  ray job submit --address="http://127.0.0.1:8267" \
    --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"${PYTHONPATH}\"}}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 --ref_num_gpus_per_node 4 \
    --actor_num_nodes 1 --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 --vllm_tensor_parallel_size 2 \
    --colocate_actor_ref \
    --pretrain "$1" \
    --prompt_data "$2" --input_key question \
    --remote_rm_url "${RM_URL}" \
    --save_path "$3" --ckpt_path "$3/ckpt" --save_steps 20 --max_ckpt_num 3 \
    --advantage_estimator group_norm \
    --use_kl_loss --use_kl_estimator_k3 --init_kl_coef ${KL} \
    --eps_clip ${CLIP} \
    --n_samples_per_prompt ${N_SAMPLES} \
    --rollout_batch_size ${ROLLOUT_BS} --micro_rollout_batch_size 4 \
    --train_batch_size ${TRAIN_BS} --micro_train_batch_size 1 \
    --num_episodes 1 --max_epochs 1 \
    --actor_learning_rate ${LR} --lr_warmup_ratio 0.0 \
    --temperature ${TEMP} \
    --prompt_max_len 1024 --generate_max_len 8192 \
    --max_search_turns 8 --max_tokens_per_turn 512 \
    --kg_backend remote --kg_server_url "${KG_SERVER}" --kg_topk 5 \
    --zero_stage 2 --bf16 --flash_attn --gradient_checkpointing --packing_samples \
    --vllm_sync_backend nccl --overlap_comm \
    ${WANDB_ARGS}
}

# Epoch 1: full RL set -> stable initial policy
run_epoch "${SFT_MODEL}" "${TRAIN_DATA}" "${SAVE_DIR}/epoch1"
# Epoch 2: hard multi-hop cases only (SFT-loss > epsilon, Eq. 10)
[ -f "${HARD_DATA}" ] || python "${ROOT}/scripts/select_hard_cases.py" --model "${SFT_MODEL}" \
    --train_file "${TRAIN_DATA}" --out "${HARD_DATA}"
run_epoch "${SAVE_DIR}/epoch1" "${HARD_DATA}" "${SAVE_DIR}/epoch2"
