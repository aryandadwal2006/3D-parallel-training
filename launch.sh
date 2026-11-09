#!/usr/bin/env bash
set -euo pipefail
DP=${DP:-1}
PP=${PP:-2}
TP=${TP:-1}
NPROC=$((DP * PP * TP))
torchrun --standalone --nproc_per_node="$NPROC" train.py --dp "$DP" --pp "$PP" --tp "$TP" "$@"
