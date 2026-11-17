#!/usr/bin/env bash
# Start a single-node Ray cluster with 8 GPUs (8x A100 80G in the paper).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
HEAD_NODE_IP=${HEAD_NODE_IP:-127.0.0.1}
ray start --head --node-ip-address "${HEAD_NODE_IP}" --num-gpus 8 --port 8266 --dashboard-port 8267
