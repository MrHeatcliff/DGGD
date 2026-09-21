# PACS: ERM with backbone-specific Optuna tuning

This experiment uses **at most 8 GPUs simultaneously**, as requested. It runs
DINOv2 ViT-B/14 first, then ResNet18, ResNet50 (torchvision ImageNet pretrained),
and ResNet50 AugMix (`resnet50.ram_in1k`). Each backbone completes search,
confirmation and final training before the next backbone starts.

## Protocol

- Four separate leave-one-domain-out studies per backbone, in DomainBed order:
  art_painting, cartoon, photo, sketch. Train on the other three domains.
- Within each source domain, fixed 80% training / 20% validation split using
  DomainBed `misc.seed_hash(0, env)`. Validation has deterministic transforms;
  only training images use augmentation. Splits are the same across candidates
  and training seeds. Seeds therefore measure training variability, not split
  variability.
- **40 Optuna TPE trials per backbone/domain**, including a baseline. Multivariate
  TPE, constant liar, ten startup trials. Search objective is the highest
  unweighted mean accuracy over the three source validation domains.
- Each trial has a budget of **5,000 optimizer updates**, validation every 250.
  Median pruning starts after 1,500 updates, ten complete startup trials and at
  least five comparable trials. Pruned trials count toward the 40-trial budget.
- Retrain the top three COMPLETE candidates with seeds 1 and 2, without pruning.
  Select by mean source validation score over search seed 0 plus these two seeds.
- Train the winner afresh with seeds **100, 101, 102**, 5,000 updates each.
  Save the best source-validation checkpoint, reload it, and evaluate the entire
  held-out target domain once. Target scores never enter Optuna or selection.
- Final collection requires every domain/seed. Report per-domain mean ± sample
  standard deviation, and the equal-domain mean per seed followed by mean ± SD.
- Maximum planned workload: 640 search trials, 96 confirmation runs, 48 final
  runs. Search pruning can reduce compute; infrastructure retries are separate.
  This is a finite, substantial search budget, not a guarantee of a global optimum.

All models use the existing `domainbed.algorithms.ERM` architecture, Adam optimizer
and cross-entropy, pretrained backbones and a linear classifier. The trial runner
uses BF16 autocast on H200, no learning-rate scheduler and no gradient clipping.
Batch size is **per source domain** (effective batch is three times this value).
DINOv2 uses the repository's 3,840-dimensional representation (last four CLS tokens
plus pooled last-layer patch features) and full-backbone fine-tuning.

| Hyperparameter | DINOv2 | ResNet variants |
| --- | --- | --- |
| Learning rate | log-uniform 1e-6 to 1e-4 | log-uniform 5e-6 to 3e-4 |
| Adam weight decay | log-uniform 1e-7 to 1e-2 | same |
| Batch size/domain | 16, 32, 64 | same |
| Feature dropout | 0, 0.1, 0.2, 0.3, 0.5 | same |
| Freeze BatchNorm running statistics | not applicable | true / false |

## Execution and monitoring

Main directory: `outputs/pacs_erm_optuna/`.

```bash
cat outputs/pacs_erm_optuna/status.json
squeue -u "$USER"
tail -f outputs/pacs_erm_optuna/controller.log
```

A lightweight controller runs in tmux session `pacs-erm-optuna`. Only this process
owns Optuna study state; GPU workers do not concurrently open a SQLite database or
Optuna journal on the shared filesystem. Study and sampler states are persisted
atomically in `state.pkl`, and completed study trials export to CSV.
Two Slurm allocations request four GPUs each on `gpu_collaborative` and
`gpu_container`. Every allocation launches one isolated process per assigned GPU.
The total requested by this experiment never exceeds eight; existing unrelated
jobs are unaffected. If Slurm cannot schedule an allocation it stays pending.

Workers claim JSON tasks through atomic file rename. The controller retries failed
trials at most three attempts, and can replace expired allocations. A fatal error
writes `FAILED.txt` and stops this experiment's allocations. `STOP` is the explicit
stop switch; use it before cancelling jobs to prevent automatic resubmission:

```bash
touch outputs/pacs_erm_optuna/STOP
```

Resume after inspecting/fixing an error or removing a deliberate stop marker:

```bash
rm -f outputs/pacs_erm_optuna/STOP
.venv/bin/python -u scripts/pacs/controller.py
```

A controller lock prevents multiple concurrent controllers. It retains completed
work. Interrupted individual training tasks restart from their original seed;
optimizer state is not resumed mid-trial.

## Outputs

- `runs/<task>/metrics.jsonl`: training loss and source validation curves.
- `runs/<task>/best.pt`: best source-selected model, hparams and task metadata.
- `runs/<task>/result.json`: completion/pruning status; target accuracy only for
  final tasks. The earlier preflight also evaluated target for pipeline checking,
  independently of the search; its outputs are never consumed by the controller.
- `reports/selections.json`: candidate confirmation scores and winning hparams.
- `studies/<backbone>_env<N>.csv`: all Optuna trials for a completed backbone.
- `reports/<backbone>.md`: collected results after that backbone finishes.
- `reports/all_backbones.md`, `.json`, `_runs.csv`: automatic final collection.
- `COMPLETE`: present only when all four backbone reports are complete.
- `provenance/`: source revisions, experiment code snapshot, environment lock.

DINOv2 source is pinned locally under `.cache/dinov2` to the revision in
`provenance/dinov2_commit.txt`. Its checkpoint and the ResNet checkpoints are cached.
The initial physical GPU snapshot is `gpu_inventory_initial.json`: 50 free healthy
partition GPUs (8 trial, 8 junior, 8 general, 12 collaborative, 14 container).
The subsequently requested cap of eight overrides the original half-cluster plan.

### Recovery on 2026-09-20

ResNet18 could not download `resnet18-f37072fd.pth` because compute-node DNS was unavailable. Downloaded the original torchvision V1 weights into the shared default Torch cache from the login node and verified the SHA256 prefix `f37072fd`. ResNet18, ResNet50, and ResNet50 AugMix each passed a CPU forward/backward/optimizer update with socket connections blocked and `HF_HUB_OFFLINE=1`; all required weights are cached. The resumed controller exports `HF_HUB_OFFLINE=1` to its Slurm workers.

Archived the eight failed ResNet18 task directories, queue claims, STOP/FAILED markers and the original state under `outputs/pacs_erm_optuna/recovery/20260920_222619/`. Reset only those infrastructure-failed tasks to WAITING with a fresh retry budget. DINOv2 results, Optuna trial identities, hyperparameters, seeds and the 8-GPU limit were preserved. Resubmitted pools 5687 and 5688; scheduler quotas may keep a pool pending.

### Mandatory GPU smoke check

`scripts/pacs/smoke_gpu.py` exercises the production trainer with real PACS data, 5 optimizer updates, BF16, batch size 64 per source, 2 data-loader workers, all four held-out domains, dropout 0.5, and both frozen/unfrozen BatchNorm. It validates source evaluation, checkpoint save/reload, and full held-out evaluation. Internet socket connections are forbidden, while local Unix sockets remain available for PyTorch data-loader IPC. Smoke results are isolated from Optuna and final reporting.

New pool worker processes run this check before their first trial of each ResNet backbone on each allocated GPU. A nonzero exit writes `PREFLIGHT_FAILED.json` and STOP before the trial can start. The existing allocation is checked separately via an overlapping Slurm step within its existing GPU allocation. A passing smoke test checks startup and short training; it does not guarantee immunity from later hardware, scheduler, or numerical failures.
