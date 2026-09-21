# Public checkpoint export

Repository: https://huggingface.co/datTrantien17/DGGD-checkpoints

Upload is in progress. Expected: 360 completed final checkpoints, 36.67 GiB. `manifest.json` lists sizes and SHA256 hashes. Upload is fully verified only when `verified.json` exists.

Includes 48 tuned ERM checkpoints and 312 final checkpoints for 26 other algorithms. Intermediate Optuna, smoke and incomplete checkpoints are excluded.

Authentication is isolated to this workspace using `.cache/hf-publish` (ignored by Git, directory mode 0700, token mode 0600). Global HF credentials and Git credentials are not modified. Launch or resume:

```bash
bash scripts/publish_checkpoints_local.sh
```

The wrapper sets repository-local HF_HOME and clears inherited token environment overrides. Never put tokens in code, commits or logs. A dedicated cache separates configuration but is not a security boundary against processes running as the same operating-system user.
