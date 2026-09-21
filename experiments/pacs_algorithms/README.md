# PACS: 33 algorithms other than ERM

This additional campaign uses **two GPUs**, independently of the eight-GPU ERM
Optuna campaign. It covers every name in `domainbed.algorithms.ALGORITHMS` except
`ERM`: 33 algorithms × 4 held-out domains × 3 seeds = **396 final runs**.

Assumptions for the unspecified backbone/search settings: use the checkout's
**ResNet50 AugMix** backbone and **algorithm-specific default hyperparameters**.
There is no Optuna search in this campaign. The ERM campaign remains separately
hyperparameter-tuned, so comparisons must retain that distinction.

## Training protocol

- Seeds 100, 101, 102, matching the independent final ERM training seeds.
- PACS domain order: art_painting, cartoon, photo, sketch.
- Hold out one entire target domain; use an 80/20 split of each source domain,
  with `misc.seed_hash(0, env)` as in the ERM campaign. Same split across seeds.
- Source training uses the repository's augmentation. Validation uses deterministic
  resize/normalization and never training augmentation.
- Train **5,000 updates**, evaluate source validation every **250** updates.
- Instantiate each native algorithm and call **its own `update(minibatches, None)`**.
  Preserve each algorithm's optimizer, penalties, warmups, batch size and defaults.
- Use FP32 (including evaluation), allowing higher-order-gradient algorithms to
  execute their native paths. The separate ERM tuning campaign uses BF16.
- Choose checkpoint by equally weighted mean source-domain validation accuracy.
  Reload that checkpoint and evaluate the full held-out target domain once.
- Report each domain's mean ± sample SD across three seeds, plus the equal-domain
  average per seed and its mean ± SD.

Before any final run, all 33 algorithms undergo five-step smoke tests on real PACS
images. The smoke tests use default batch sizes and exercise post-warmup paths by
advancing counters at step four; these artificial counters are **only** used in
smoke tests. Source evaluation uses 32 validation samples per source domain for
smoke tests. Full training always uses normal counters and the full source splits.
No target-domain images are evaluated by this smoke test.

## Operation

```bash
cat outputs/pacs_algorithms/status.json
tail -f outputs/pacs_algorithms/controller.log
squeue -u "$USER"
```

The controller runs in tmux `pacs-33-algorithms`. Its two-GPU Slurm allocation is on
`gpu_collaborative`, and can be replaced when its time limit expires. Multiple isolated training processes may share each GPU via the measured MPS configuration. No other campaign's jobs are cancelled or altered.

A failed smoke test blocks the final-training stage, records `NEEDS_ATTENTION.json`,
and allows the remaining smoke tests to finish. After fixing an issue:

```bash
touch outputs/pacs_algorithms/RETRY_FAILED
```

Final tasks retry up to three attempts, starting from their original seed.
Persistent failures are retained and prevent publication of complete final results.

To stop this campaign (and prevent job replacement):

```bash
touch outputs/pacs_algorithms/STOP
```

The native experiment can be resumed with the same command after removing `STOP`:

```bash
.venv/bin/python -u scripts/pacs_algorithms/controller.py
```

## Artifacts

Under `outputs/pacs_algorithms/`:

- `config.json`: explicit algorithm list, seeds and budget.
- `state.json`, `status.json`: durable queue and current status.
- `runs/<task>/hparams.json`: actual algorithm-specific hyperparameters.
- `runs/<task>/metrics.jsonl`: native training metrics and source validation.
- `runs/<task>/best.pt`: best source-selected checkpoint.
- `runs/<task>/result.json`: completed-run metadata and final target accuracy.
- `reports/all_algorithms.md`, `.json`, `_runs.csv`: automatic final collection.
- `COMPLETE`: only created after all 396 final runs and collection succeed.

## Fishr compatibility

BackPACK 1.7.1 with this PyTorch/runtime did not populate `grad_batch` for Fishr's
linear head. The isolated `scripts/pacs_algorithms/compat.py` adapter replaces
only that extraction with exact per-example softmax-cross-entropy gradients:
`dW_i = (softmax(logits_i) - one_hot(y_i)) outer features_i`,
`db_i = softmax(logits_i) - one_hot(y_i)`.
The native Fishr variance, EMA, penalty schedule and optimizer remain unchanged.
Both per-example gradients and higher-order derivatives of the variance penalty
were checked against a float64 autograd reference (`preflight/fishr_gradient_check.log`).
This adapter does not modify the core algorithm module or the running ERM campaign.

## Multi-process GPU optimization

The campaign was temporarily paused for an H200 benchmark on the same compute
node. Completed final runs were retained. Two interrupted runs restart from their
original seeds; their first-attempt checkpoints and logs are archived.

Measured data and methodology are under `outputs/pacs_gpu_benchmark/`. Compare
aggregate optimizer updates per second, including one full source-validation pass
and checkpoint save per 250 measured updates, not just memory capacity or CUDA-core
counts. Model/data initialization and 25 warmup updates are excluded by a shared
start barrier. ERM++, IGA and RDM represent ordinary, higher-order-gradient and
large-minibatch workloads. Each GPU has 132 Hopper SMs and about 140 GiB of VRAM.

The runtime supports configurable `jobs_per_gpu`, isolated MPS daemons for each
allocated GPU, a per-algorithm concurrency cap, and conservative VRAM admission.
The reservation estimate is `1.25 × measured smoke peak + 1.5 GiB` per job, bounded
by an explicit per-GPU memory budget. This allows multiple independent trials on
the **same two physical GPUs**, without changing batch size or algorithm math.

The optimized runtime uses one Torch CPU thread and one DataLoader worker per
source domain per training process. These runtime settings are recorded in each
task JSON. The original experiment code snapshot is retained in
`provenance/initial_experiment_code.tar.gz`; subsequent snapshots capture the
optimized scheduler.

Selected runtime (2026-09-19): **4 concurrent jobs/GPU, MPS enabled**, on the same
two physical GPUs; 32 CPU cores allocated in total, 118 GiB admission budget/GPU.
Short benchmark throughput gains relative to one non-MPS job/GPU were 1.82× for
ERM++, 1.61× for IGA, and 2.55× for RDM. Eight ERM++ jobs/GPU were only 3.5% faster
than four, so four was selected. The four-job non-MPS IGA branch was intentionally
stopped after two-job non-MPS IGA underperformed the single-job baseline.

Benchmark tables: `outputs/pacs_gpu_benchmark/report.md` and `all_results.json`.
These gains apply to the measured representative workloads, not a promised uniform
speedup for all 33 algorithms. First production allocation with this setting: 5512.

### Resume 2026-09-20

Resumed after cancelled allocations 5512/5528/5529 using `gpu_junior`, a 24-hour pool walltime, and exactly 2 GPUs. Allocation 5697 started on gpu03. Partition and pool walltime are now configuration fields, so automatic replacement stays within the selected partition's time limit. Existing 41 final results remain intact; interrupted runs restart with their original task/seed and archived incomplete artifacts. Concurrency remains 4 jobs/GPU with MPS and the existing memory admission budget. State/config backup is in `outputs/pacs_algorithms/recovery/20260920_224507/`.
