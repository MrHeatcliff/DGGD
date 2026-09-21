# PACS experiment snapshot — 2026-09-21

## Results

- [Completed representation diagnostics](representation/README.md)
- [Checkpoint download and verification status](checkpoints/README.md)

- [ERM: all four backbones, all domains and three seeds](erm/README.md)
- [Selected ERM hyperparameters: 16 backbone/domain configurations](erm/hyperparameters.md)
- [Full selected configurations and confirmation scores](erm/selected_hparams.json)
- [640 Optuna trials, by backbone and held-out domain](erm/optuna_trials/)
- [Search and confirmation task/result audit](erm/search_and_confirmation.json)
- [ERM final runs CSV](erm/completed_runs.csv)
- [Other algorithms: 26 complete, 7 unfinished](algorithms/README.md)
- [Other algorithms final runs CSV](algorithms/completed_runs.csv)
- [Other algorithms task metadata, effective hyperparameters and results](algorithms/completed_runs.json)

ERM finished all 640 search trials (including pruning), 96 confirmation runs, and 48 final runs. The other campaign finished 312/396 final runs: 26 algorithms each have all four domains and three seeds. Transfer, CausIRL_CORAL, CausIRL_MMD, EQRM, RDM, ADRMX, and URM have no complete final runs in this snapshot. Their five-update smoke tests do not count as final results. The last allocation was cancelled with SIGTERM on September 21 at 09:49 UTC+7; the available log does not identify the initiator. This snapshot is not a live training-status page.

## Evaluation and selection

These are **best-source-validation-checkpoint results, not target-oracle results**. Each PACS domain is held out in turn; the other three are independently split into 80% training and 20% validation. Splits are fixed with `misc.seed_hash(0, env)`. Validation uses deterministic transforms. The selection metric gives equal weight to the three source-domain validation accuracies.

ERM runs 40 Optuna trials per backbone/target domain, seed 0. TPE is multivariate with constant liar and 10 startup trials. Median pruning uses 10 startup trials, 1500 warmup steps, 250-step intervals, and at least 5 comparable trials. Each trial has up to 5000 updates. The top 3 COMPLETE configurations are confirmed with seeds 1 and 2; the winner maximizes the average best source-validation score across seeds 0, 1, 2. Final training uses fresh seeds 100, 101, 102, 5000 updates and validation every 250 updates. The best checkpoint is reloaded and evaluated once on the full target domain. Strict improvements replace the checkpoint; ties keep the earlier checkpoint. Smoke-test target results never enter tuning or final reporting.

| Tuned hyperparameter | DINOv2 | ResNet18 / ResNet50 / ResNet50 AugMix |
| --- | --- | --- |
| Learning rate | log-uniform 1e-6–1e-4 | log-uniform 5e-6–3e-4 |
| Weight decay | log-uniform 1e-7–1e-2 | same |
| Batch per source domain | 16, 32, 64 | same |
| Feature dropout | 0, 0.1, 0.2, 0.3, 0.5 | same |
| Freeze BatchNorm statistics | not applicable | true / false |

ERM uses full-backbone fine-tuning, a linear classifier, Adam, cross-entropy, BF16, and no learning-rate scheduler. DINOv2 is ViT-B/14 with 3840-dimensional features (last four CLS tokens plus mean last-layer patch tokens); ResNets use last-layer pooled features (512 or 2048 dimensions). Batch is per source; effective batch is three times larger.

The 33-algorithm campaign uses ResNet50 `timm resnet50.ram_in1k`, **default algorithm-specific DomainBed hyperparameters, no Optuna**, and FP32 native algorithm updates. It uses the same source splits, source-based checkpoint selection, final seeds, and step budget. Fishr uses the documented analytic linear-head gradient compatibility implementation in `scripts/pacs_algorithms/compat.py` for this PyTorch/BackPACK environment. See [campaign notes](../../experiments/pacs_algorithms/README.md).

All reported ± values are sample standard deviations across the three training seeds, not confidence intervals. AVG is the equally weighted average of four domain accuracies per seed, followed by mean and SD across seeds. Seeds vary training randomness, not the source-data split.

**Comparison limits:** ERM is tuned while the other algorithms use defaults, and precision differs; this is not a matched tuning-budget algorithm comparison. DINOv2 and ResNet also differ in pretraining data/objective, architecture, and representation aggregation. Results do not establish a causal architecture advantage. This custom protocol should not be described as an identical official DomainBed benchmark protocol.

## Reproduction and artifacts

See [environment/data setup](../../DOMAIN_GEN_SETUP.md), [ERM protocol and launch notes](../../experiments/pacs_erm/README.md), and [algorithm campaign](../../experiments/pacs_algorithms/README.md). `requirements-domain-gen.lock.txt` records the Python package environment. The DINOv2 source cache used commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

Slurm launch scripts/configuration include cluster-specific partitions and workspace paths; adapt these for a different machine. Pre-cache model weights on a network-enabled node. `scripts/pacs/smoke_gpu.py` tests the real trainer offline on GPU, including checkpoint save/reload and all four target domains. New ERM pool workers require this smoke gate before starting a ResNet backbone. The concurrency benchmark selected 4 jobs/GPU with CUDA MPS for the separate two-GPU algorithm campaign; it is not a universal guarantee for arbitrary hardware.

Datasets, pretrained weights, trained checkpoints, environment directories, and live queue state are excluded from Git. The lightweight JSON/CSV artifacts preserve completed metrics and hyperparameters. To refresh this snapshot from local outputs:

```bash
.venv/bin/python scripts/export_pacs_results.py
```
