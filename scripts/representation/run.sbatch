#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
out="$1"
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} -eq 4 ]]
backbones=(dinov2 resnet18 resnet50 resnet50_augmix)
pids=()
for i in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${devices[$i]}"
    .venv/bin/python -u scripts/representation/analyze.py --backbone "${backbones[$i]}" --output "$out/smoke" --smoke
  ) > "$out/smoke-${backbones[$i]}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" -ne 0 ]]; then touch "$out/SMOKE_FAILED"; exit 1; fi
touch "$out/SMOKE_PASSED"
pids=()
for i in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${devices[$i]}"
    .venv/bin/python -u scripts/representation/analyze.py --backbone "${backbones[$i]}" --output "$out/full"
  ) > "$out/full-${backbones[$i]}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" -ne 0 ]]; then touch "$out/FAILED"; exit 1; fi
.venv/bin/python scripts/representation/collect.py "$out/full" > "$out/collection.log"
