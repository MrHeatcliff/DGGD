#!/bin/bash
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$workspace"
export HF_HOME="$workspace/.cache/hf-publish"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec .venv/bin/python -u scripts/publish_checkpoints.py --repo datTrantien17/DGGD-checkpoints
