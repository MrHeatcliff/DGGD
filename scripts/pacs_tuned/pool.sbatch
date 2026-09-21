#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
exec .venv/bin/python -u scripts/pacs_tuned/pool.py "$1" --gpus "$2"
