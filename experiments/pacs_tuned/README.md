# PACS: tuned CORAL, MMD, DANN, IRM across four backbones

Independent campaign: `outputs/pacs_methods_optuna`. Original ERM results and the default-hyperparameter 33-algorithm campaign are not modified.

## Resources and startup

At launch, Slurm reported 54 unallocated GPUs (trial 8, junior 8, general 10, collaborative 8, container 20). This does not mean the account can allocate them all. Container requests were held by QOSMaxNodePerUserLimit; junior rejected 4 GPUs with QOSMaxGRESPerUser. The campaign started with 4 H200 GPUs (allocation 5794 on gpu09 / gpu_collaborative), then expanded to **8 GPUs** after scheduler feasibility checks: 4 collaborative, 2 junior, and 2 general pinned to gpu07 (an already-used node within the account node limit). No other user's jobs are cancelled. One native training process per GPU, no MPS. Controller can support 4 or 8 allocated GPUs with a matching pool configuration.

Every GPU must pass all 16 method/backbone smoke cases before claiming tuning tasks. The four-GPU pool covers all four held-out domains, giving **64 distinct method/backbone/domain cases**; the additional two-GPU pools repeat their smoke checks on their own nodes. Smoke uses batch 64 per source, FP32, 10 real updates, checkpoint save/reload, finite-state checks, no Internet connections. IRM post-anneal/reset path is exercised; DANN uses 8 discriminator steps per generator step, width 1024/depth 5, nonzero gradient penalty and runs long enough to exercise both optimizers. Smoke target metrics do not enter tuning. An error stops the campaign and records a preflight diagnostic.

## Search protocol

Four backbones: DINOv2 ViT-B/14, ResNet18, ResNet50, timm ResNet50 RAM/AugMix. Backbones run in that order, each with CORAL, MMD, DANN, IRM. Identifiers use `METHOD--backbone`.

Each method/backbone/held-out-domain has an independent Optuna study:

- CORAL, MMD, IRM: **60 trials per domain**.
- DANN: **80 trials per domain** for its larger search space.
- TPE: multivariate, constant liar, 15 startup trials; seeded reproducibly.
- Native FP32 `algorithm.update`, 5000 calls per trial, source validation every 250 calls.
- Median pruner: 10 startup trials, at least 5 comparable trials, warmup 1500. IRM cannot be pruned until at least 500 updates after its configured penalty anneal.
- Source data split 80/20 with fixed `misc.seed_hash(0, env)`, same as ERM. Source validation averages the three domains equally.
- Search seed 0; top 3 COMPLETE configurations confirmed with seeds 1 and 2. Select by average best source validation over seeds 0, 1, 2.
- Final seeds 100, 101, 102 for all four domains; 5000 updates, reload the source-selected best checkpoint, evaluate the entire held-out domain once. No target-based tuning or checkpoint choice.
- Non-finite search trials are explicitly recorded as failed Optuna trials; infrastructure failures retry up to 3 times and then stop for inspection. Confirmation/final numerical failures are not silently discarded.

Total planned work: **4160 search trials, 384 confirmation runs, 192 final runs** (4736 tasks before retries). Pruning reduces actual update counts. This is a substantial finite search budget, not a promise of global-optimum hyperparameters.

## Search spaces

| Parameter | Search |
| --- | --- |
| Backbone learning rate | DINOv2 log 1e-6–1e-4; ResNets log 5e-6–3e-4 |
| Weight decay | log 1e-7–1e-2 |
| Batch/source | 16, 32, 64 |
| Feature dropout | 0, 0.1, 0.2, 0.3, 0.5 |
| ResNet freeze_bn | true / false |
| CORAL/MMD mmd_gamma | log 1e-3–1e2 |
| IRM irm_lambda | log 1e-1–1e5 |
| IRM penalty anneal | 0, 100, 500, 1000, 1500, 2500 updates |
| DANN lr_g, weight_decay_g | backbone LR and weight decay above |
| DANN lr_d | log 1e-6–1e-3 |
| DANN adversarial lambda | log 1e-2–1e2 |
| DANN discriminator weight decay | log 1e-7–1e-2 |
| DANN gradient penalty | 0, 0.01, 0.1, 1, 10 |
| DANN discriminator steps / generator step | 1, 2, 4, 8 |
| DANN beta1 | 0, 0.5 |
| DANN discriminator width/depth/dropout | 128/256/512/1024; 3/4/5; 0/0.1/0.5 |

`mmd_gamma` is the native algorithm's penalty coefficient, not a newly tuned Gaussian bandwidth. DINOv2 uses the same 3840-dimensional feature aggregation as ERM. All backbone parameters are trainable; freeze_bn controls ResNet BatchNorm statistics.

## Reporting and comparison limits

Each pair is automatically collected after all 12 final runs finish. Reports include per-domain mean ± sample SD and equal-domain AVG across all three seeds, raw runs, selected hparams/confirmation scores, and Optuna history. Completed pair reports appear in `outputs/pacs_methods_optuna/reports/`, then all pairs are collected at campaign completion.

ERM was BF16 with 40 trials/domain, while this campaign uses FP32 and 60/80 trials/domain. DANN's 5000 calls contain discriminator-only calls depending on the tuned ratio; they are not 5000 generator updates. These differences must accompany comparisons; the campaign is not a perfectly matched-compute benchmark.

Launch/resume (with cached weights):

```bash
.venv/bin/python -u scripts/pacs_tuned/controller.py
```

The controller owns an exclusive lock and persisted Optuna state. STOP halts the campaign and releases only its own allocations. The scheduler can replace time-limited pools; completed tasks are preserved, interrupted ones restart from their original seed and archive previous artifacts.
