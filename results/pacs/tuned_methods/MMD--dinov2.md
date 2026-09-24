# PACS — tuned CORAL / MMD / DANN / IRM

Target accuracy (%), mean ± sample standard deviation over three independent training seeds.
Overall: equally weighted mean of the four held-out domains for each seed.
Hyperparameters and checkpoints were selected exclusively using source-domain validation.

Selection: best source-validation checkpoint, not target-oracle selection and not necessarily the last checkpoint.
Each final run trains for 5000 updates, reloads its best source-validation checkpoint, and evaluates the full target domain once.

| Backbone | Art painting | Cartoon | Photo | Sketch | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMD--dinov2 | 98.03 ± 0.33 | 94.75 ± 0.79 | 99.96 ± 0.03 | 87.99 ± 2.11 | 95.18 ± 0.55 |
