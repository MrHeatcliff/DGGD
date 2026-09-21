#!/bin/bash
set -euo pipefail
cd /home/vn-user0101/Dat/DomainBed
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -u scripts/pacs/pool.py "$1" --gpus "$2"
