# PACS — tuned CORAL / MMD / DANN / IRM

Target accuracy (%), mean ± sample standard deviation over three independent training seeds.
Overall: equally weighted mean of the four held-out domains for each seed.
Hyperparameters and checkpoints were selected exclusively using source-domain validation.

Selection: best source-validation checkpoint, not target-oracle selection and not necessarily the last checkpoint.
Each final run trains for 5000 updates, reloads its best source-validation checkpoint, and evaluates the full target domain once.

| Backbone | Art painting | Cartoon | Photo | Sketch | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CORAL--dinov2 | 96.55 ± 1.45 | 94.23 ± 0.95 | 99.76 ± 0.21 | 88.32 ± 1.35 | 94.71 ± 0.51 |
