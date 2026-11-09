#!/usr/bin/env bash
set -euo pipefail
export RANK="${OMPI_COMM_WORLD_RANK:?}"
export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?}"
export LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:?}"
export MASTER_ADDR="${MASTER_ADDR:?}"
export MASTER_PORT="${MASTER_PORT:-29500}"
exec python train.py "$@"
