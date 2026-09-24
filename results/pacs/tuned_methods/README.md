# Tuned DG methods — snapshot 2026-09-24

Completed: CORAL–DINOv2 and MMD–DINOv2, each with 4 targets × 3 final seeds, after 60 Optuna trials/domain and top-3 confirmation on two additional seeds. Best checkpoints selected by source validation, not target oracle. Native FP32 updates; compare ERM BF16 results with this precision and search-budget difference in mind.

- [CORAL final results](CORAL--dinov2.md) · [raw runs](CORAL--dinov2_runs.csv)
- [MMD final results](MMD--dinov2.md) · [raw runs](MMD--dinov2_runs.csv)
- [Selected hyperparameters and confirmation scores](selections.json)
- [Visualizations](../visualizations/README.md)
- [Full protocol](../../../experiments/pacs_tuned/README.md)

DANN–DINOv2 is in search; IRM–DINOv2 and the ResNet variants have no final tuned results in this snapshot. Their absence is not an algorithm failure or a zero accuracy.
