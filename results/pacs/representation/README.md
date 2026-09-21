# PACS representation diagnostics

Post-hoc analysis; fixed ERM checkpoints, no target-based model selection. Ridge probes choose regularization using source validation only. Domain-probe labels from target are used solely for diagnosis. Pretrained has one deterministic probe per split; finetuned has three training seeds.

| Backbone | Phase | Full probe target AVG (%) | PCA256 probe AVG (%) | Cross-domain 1NN (%) | Class/domain distance ratio | Domain probe (%) | CKA vs pretrained |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dinov2 | pretrained | 95.4306 | 96.0333 | 89.0515 | 1.2865 | 99.1793 | 1.0000 |
| dinov2 | finetuned | 95.6791 | 95.6781 | 95.2640 | 0.2898 | 98.2564 | 0.6331 |
| resnet18 | pretrained | 65.6541 | 66.4763 | 54.5603 | 1.6766 | 94.6248 | 1.0000 |
| resnet18 | finetuned | 81.3667 | 81.2917 | 79.3371 | 0.1186 | 89.3128 | 0.3069 |
| resnet50 | pretrained | 68.1886 | 68.3185 | 53.8725 | 1.6824 | 96.0588 | 1.0000 |
| resnet50 | finetuned | 83.9953 | 83.8286 | 83.0844 | 0.0783 | 90.9903 | 0.2805 |
| resnet50_augmix | pretrained | 70.0122 | 70.7827 | 58.4394 | 1.4265 | 96.7893 | 1.0000 |
| resnet50_augmix | finetuned | 86.2656 | 86.1336 | 85.1899 | 0.0976 | 91.7839 | 0.3388 |

Lower distance ratio suggests closer same-class cross-domain centroids relative to different classes. Lower domain accuracy is not sufficient evidence of good representations. CKA measures change, not quality. Ablation heads are refitted on fixed features; this does not retrain each feature variant end-to-end. Target labels must not be fed back into checkpoint selection.

Detailed results, domain/seed values and DINOv2 CLS/patch ablations: all_results.json.
