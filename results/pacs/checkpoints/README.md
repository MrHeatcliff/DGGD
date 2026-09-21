# Checkpoint export — upload blocked by token permissions

Prepared destination: https://huggingface.co/lucaznguyenofficial/DGGD-checkpoints

**The checkpoints have not been uploaded.** The current Hugging Face token authenticates as `lucaznguyenofficial`, but repository creation returned HTTP 403 (no write/create permission). The destination repo is not confirmed to exist. No `verified.json` is present.

The local export contains **360 completed final checkpoints, 36.67 GiB**. [manifest.json](manifest.json) lists every file, SHA256, size, backbone/algorithm, target domain, seed, checkpoint step, validation score and target accuracy. It includes 48 tuned ERM checkpoints and 312 checkpoints for 26 other algorithms. Intermediate tuning, incomplete and smoke checkpoints are excluded.

## Resume publication

In the experiment workspace, authenticate interactively with an appropriately scoped write token (do not put the token into source code or chat):

```bash
.venv/bin/hf auth login --force --no-add-to-git-credential
.venv/bin/python scripts/publish_checkpoints.py --repo lucaznguyenofficial/DGGD-checkpoints
```

The publisher stages files using hard links, creates a public model repository, uploads resumably with four workers, then checks all 360 remote sizes and LFS SHA256 values. It writes `verified.json` only after verification succeeds. If a different account is used, change the repo namespace accordingly. Checkpoints remain safely stored locally while access is fixed.
