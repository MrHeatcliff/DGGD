# PACS erm

Target accuracy (%), mean ± sample SD over seeds 100, 101, 102. AVG uses equal domain weights per seed. Best checkpoint selected on source validation, not target oracle.

| Model | Art Painting | Cartoon | Photo | Sketch | AVG |
| --- | ---: | ---: | ---: | ---: | ---: |
| dinov2 | 98.34 ± 0.34 | 95.42 ± 0.20 | 99.92 ± 0.09 | 88.05 ± 2.85 | 95.43 ± 0.81 |
| resnet18 | 79.52 ± 0.88 | 74.62 ± 0.42 | 95.81 ± 0.30 | 72.97 ± 2.58 | 80.73 ± 0.80 |
| resnet50 | 85.01 ± 1.30 | 76.19 ± 0.36 | 96.35 ± 0.37 | 75.92 ± 1.09 | 83.37 ± 0.23 |
| resnet50_augmix | 86.80 ± 4.14 | 78.77 ± 1.53 | 98.40 ± 0.12 | 78.56 ± 1.63 | 85.63 ± 1.26 |

Completed final runs: 48/48.

Incomplete algorithms/backbones are not included in the final comparison table:
- None.
