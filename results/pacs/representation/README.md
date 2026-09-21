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

## Interpretation

- Under this source-selected ridge-probe protocol, pretrained DINOv2 reaches 95.43% average target accuracy, versus 65.65–70.01% for pretrained ResNets. After fine-tuning and refitting a diagnostic head, DINOv2 reaches 95.68% and ResNets 81.37–86.27%. This supports a large pre-existing representation advantage; ridge-probe scores are not the original ERM head results.
- Matching feature dimension at PCA-256 does not remove the gap. Dimensionality alone therefore does not explain it under this protocol.
- DINOv2's domain probe accuracy remains very high (98.26% after fine-tuning), higher than the ResNets. The measurements do **not** support the simple claim that DINOv2 succeeds by erasing domain information. Label information can transfer well while domain information remains decodable.
- The centroid distance ratio is smaller for fine-tuned ResNets than DINOv2, despite worse target classification. This aggregate geometry statistic alone does not rank representation quality reliably here.
- DINOv2's CKA to its pretrained representation is higher than ResNet's after fine-tuning. This indicates greater measured stability on the sampled images, not proof that learning rate or architecture caused the result.
- Architecture, pretraining dataset, objective, model size, and optimization are confounded. No causal attribution to any one factor is justified by these diagnostics alone. All target-label analyses are post-hoc and do not alter the fixed benchmark results.
