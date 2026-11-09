#!/usr/bin/env bash
set -euo pipefail
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
torchrun \
  --nnodes="${SLURM_NNODES:?}" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="${SLURM_NODEID:?}" \
  --master_addr="${MASTER_ADDR:?}" \
  --master_port="${MASTER_PORT:-29500}" \
  train.py "$@"
