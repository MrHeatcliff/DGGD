# PACS representations and original-classifier decision slices

24 fixed-seed checkpoint views: ERM on four backbones, CORAL/MMD on DINOv2; four target domains each. Seed 100 was fixed before visualization, not selected for appearance or accuracy. DANN/IRM and their ResNet configurations do not yet have complete tuned results and are not presented as final.

## How to read these plots

- PCA is fitted only to original source-training features, independently for each checkpoint. Shared colors, balanced images and deterministic seeds make examples comparable; axes/rotations across models are not aligned.
- Top-left: class labels. Top-right: domains on exactly the same PCA coordinates. Source points come from source validation; target points from the held-out domain.
- Bottom-left: exact original linear classifier evaluated on z = source_mean + PC1*u + PC2*v. This is a 2-D affine slice, not a classifier trained on 2-D points and not the entire high-dimensional boundary. Projected observations usually have residual components outside this plane. Slice/full-head agreement and explained variance quantify this limitation.
- Bottom-right: full-feature true-class margin, using original head weights with FP32 arithmetic. Negative means the head predicts another class. It is not affected by the 2-D projection. Logit magnitudes across independently trained models are not calibrated.
- t-SNE uses L2-normalized balanced features and a PCA-50 preprocessing step. It uses target images only for post-hoc description; no tuning or model selection. Cluster size, spacing and rotation are not directly comparable between panels. No decision boundary is drawn in t-SNE coordinates.
- ERM reuses its BF16 feature caches; CORAL/MMD features are native FP32. Tiny differences from original ERM predictions can arise from applying the head in FP32. Original target accuracy and diagnostic-head accuracy are both in metrics.json.
- plane.npz contains the source-fitted mean/PCA basis and trained head parameters to reproduce each boundary slice. points.csv contains displayed sample identities, embeddings, predictions and margins; no source images are committed. PNG is for GitHub viewing, PDF for export.

## Important interpretation

The first two PCs maximize source feature variance, not classification fidelity. In particular, the Sketch DINOv2 slices agree with the full head on only about 35–40% of target images. These colorful regions must not be read as the actual predictions of the high-dimensional model on all projected points. Use the full-feature margin panel and original target accuracy for that purpose. The exact pairwise class boundary in feature space is `(w_i - w_j) @ z + (b_i - b_j) = 0`; the winning class must also beat every other class. The exported plane is one explicit intersection with this full classifier.

## Method comparison plates

![Three-seed target results](accuracy_comparison.png)

The examples below use seed 100; the accuracy chart above uses all three final seeds.

### Held-out art_painting

![DINOv2 method comparison: art_painting](comparison_art_painting.png)

[PDF](comparison_art_painting.pdf)

### Held-out cartoon

![DINOv2 method comparison: cartoon](comparison_cartoon.png)

[PDF](comparison_cartoon.pdf)

### Held-out photo

![DINOv2 method comparison: photo](comparison_photo.png)

[PDF](comparison_photo.pdf)

### Held-out sketch

![DINOv2 method comparison: sketch](comparison_sketch.png)

[PDF](comparison_sketch.pdf)

## Gallery

### ERM / dinov2 — photo

Original target accuracy (seed 100): 100.00%; slice/full-head agreement: 38.74%.

![PCA and classifier slice](ERM_dinov2_e2_s100/pca_boundary.png)

[t-SNE](ERM_dinov2_e2_s100/tsne.png) · [PDF](ERM_dinov2_e2_s100/pca_boundary.pdf) · [coordinates](ERM_dinov2_e2_s100/points.csv) · [metrics](ERM_dinov2_e2_s100/metrics.json)

### ERM / dinov2 — art_painting

Original target accuracy (seed 100): 98.73%; slice/full-head agreement: 46.14%.

![PCA and classifier slice](ERM_dinov2_e0_s100/pca_boundary.png)

[t-SNE](ERM_dinov2_e0_s100/tsne.png) · [PDF](ERM_dinov2_e0_s100/pca_boundary.pdf) · [coordinates](ERM_dinov2_e0_s100/points.csv) · [metrics](ERM_dinov2_e0_s100/metrics.json)

### ERM / dinov2 — sketch

Original target accuracy (seed 100): 90.74%; slice/full-head agreement: 39.73%.

![PCA and classifier slice](ERM_dinov2_e3_s100/pca_boundary.png)

[t-SNE](ERM_dinov2_e3_s100/tsne.png) · [PDF](ERM_dinov2_e3_s100/pca_boundary.pdf) · [coordinates](ERM_dinov2_e3_s100/points.csv) · [metrics](ERM_dinov2_e3_s100/metrics.json)

### ERM / dinov2 — cartoon

Original target accuracy (seed 100): 95.35%; slice/full-head agreement: 52.18%.

![PCA and classifier slice](ERM_dinov2_e1_s100/pca_boundary.png)

[t-SNE](ERM_dinov2_e1_s100/tsne.png) · [PDF](ERM_dinov2_e1_s100/pca_boundary.pdf) · [coordinates](ERM_dinov2_e1_s100/points.csv) · [metrics](ERM_dinov2_e1_s100/metrics.json)

### ERM / resnet18 — sketch

Original target accuracy (seed 100): 75.52%; slice/full-head agreement: 49.66%.

![PCA and classifier slice](ERM_resnet18_e3_s100/pca_boundary.png)

[t-SNE](ERM_resnet18_e3_s100/tsne.png) · [PDF](ERM_resnet18_e3_s100/pca_boundary.pdf) · [coordinates](ERM_resnet18_e3_s100/points.csv) · [metrics](ERM_resnet18_e3_s100/metrics.json)

### ERM / resnet18 — photo

Original target accuracy (seed 100): 95.81%; slice/full-head agreement: 41.14%.

![PCA and classifier slice](ERM_resnet18_e2_s100/pca_boundary.png)

[t-SNE](ERM_resnet18_e2_s100/tsne.png) · [PDF](ERM_resnet18_e2_s100/pca_boundary.pdf) · [coordinates](ERM_resnet18_e2_s100/points.csv) · [metrics](ERM_resnet18_e2_s100/metrics.json)

### ERM / resnet18 — cartoon

Original target accuracy (seed 100): 74.27%; slice/full-head agreement: 50.38%.

![PCA and classifier slice](ERM_resnet18_e1_s100/pca_boundary.png)

[t-SNE](ERM_resnet18_e1_s100/tsne.png) · [PDF](ERM_resnet18_e1_s100/pca_boundary.pdf) · [coordinates](ERM_resnet18_e1_s100/points.csv) · [metrics](ERM_resnet18_e1_s100/metrics.json)

### ERM / resnet18 — art_painting

Original target accuracy (seed 100): 80.13%; slice/full-head agreement: 47.61%.

![PCA and classifier slice](ERM_resnet18_e0_s100/pca_boundary.png)

[t-SNE](ERM_resnet18_e0_s100/tsne.png) · [PDF](ERM_resnet18_e0_s100/pca_boundary.pdf) · [coordinates](ERM_resnet18_e0_s100/points.csv) · [metrics](ERM_resnet18_e0_s100/metrics.json)

### ERM / resnet50 — sketch

Original target accuracy (seed 100): 74.83%; slice/full-head agreement: 20.54%.

![PCA and classifier slice](ERM_resnet50_e3_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_e3_s100/tsne.png) · [PDF](ERM_resnet50_e3_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_e3_s100/points.csv) · [metrics](ERM_resnet50_e3_s100/metrics.json)

### ERM / resnet50 — art_painting

Original target accuracy (seed 100): 86.38%; slice/full-head agreement: 47.61%.

![PCA and classifier slice](ERM_resnet50_e0_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_e0_s100/tsne.png) · [PDF](ERM_resnet50_e0_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_e0_s100/points.csv) · [metrics](ERM_resnet50_e0_s100/metrics.json)

### ERM / resnet50 — cartoon

Original target accuracy (seed 100): 76.54%; slice/full-head agreement: 45.22%.

![PCA and classifier slice](ERM_resnet50_e1_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_e1_s100/tsne.png) · [PDF](ERM_resnet50_e1_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_e1_s100/points.csv) · [metrics](ERM_resnet50_e1_s100/metrics.json)

### ERM / resnet50 — photo

Original target accuracy (seed 100): 96.77%; slice/full-head agreement: 40.36%.

![PCA and classifier slice](ERM_resnet50_e2_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_e2_s100/tsne.png) · [PDF](ERM_resnet50_e2_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_e2_s100/points.csv) · [metrics](ERM_resnet50_e2_s100/metrics.json)

### ERM / resnet50_augmix — art_painting

Original target accuracy (seed 100): 88.96%; slice/full-head agreement: 65.09%.

![PCA and classifier slice](ERM_resnet50_augmix_e0_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_augmix_e0_s100/tsne.png) · [PDF](ERM_resnet50_augmix_e0_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_augmix_e0_s100/points.csv) · [metrics](ERM_resnet50_augmix_e0_s100/metrics.json)

### ERM / resnet50_augmix — cartoon

Original target accuracy (seed 100): 77.47%; slice/full-head agreement: 63.48%.

![PCA and classifier slice](ERM_resnet50_augmix_e1_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_augmix_e1_s100/tsne.png) · [PDF](ERM_resnet50_augmix_e1_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_augmix_e1_s100/points.csv) · [metrics](ERM_resnet50_augmix_e1_s100/metrics.json)

### ERM / resnet50_augmix — sketch

Original target accuracy (seed 100): 76.69%; slice/full-head agreement: 21.56%.

![PCA and classifier slice](ERM_resnet50_augmix_e3_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_augmix_e3_s100/tsne.png) · [PDF](ERM_resnet50_augmix_e3_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_augmix_e3_s100/points.csv) · [metrics](ERM_resnet50_augmix_e3_s100/metrics.json)

### ERM / resnet50_augmix — photo

Original target accuracy (seed 100): 98.50%; slice/full-head agreement: 65.69%.

![PCA and classifier slice](ERM_resnet50_augmix_e2_s100/pca_boundary.png)

[t-SNE](ERM_resnet50_augmix_e2_s100/tsne.png) · [PDF](ERM_resnet50_augmix_e2_s100/pca_boundary.pdf) · [coordinates](ERM_resnet50_augmix_e2_s100/points.csv) · [metrics](ERM_resnet50_augmix_e2_s100/metrics.json)

### CORAL / dinov2 — sketch

Original target accuracy (seed 100): 89.18%; slice/full-head agreement: 38.51%.

![PCA and classifier slice](CORAL_dinov2_e3_s100/pca_boundary.png)

[t-SNE](CORAL_dinov2_e3_s100/tsne.png) · [PDF](CORAL_dinov2_e3_s100/pca_boundary.pdf) · [coordinates](CORAL_dinov2_e3_s100/points.csv) · [metrics](CORAL_dinov2_e3_s100/metrics.json)

### CORAL / dinov2 — cartoon

Original target accuracy (seed 100): 93.13%; slice/full-head agreement: 54.78%.

![PCA and classifier slice](CORAL_dinov2_e1_s100/pca_boundary.png)

[t-SNE](CORAL_dinov2_e1_s100/tsne.png) · [PDF](CORAL_dinov2_e1_s100/pca_boundary.pdf) · [coordinates](CORAL_dinov2_e1_s100/points.csv) · [metrics](CORAL_dinov2_e1_s100/metrics.json)

### CORAL / dinov2 — art_painting

Original target accuracy (seed 100): 94.87%; slice/full-head agreement: 45.70%.

![PCA and classifier slice](CORAL_dinov2_e0_s100/pca_boundary.png)

[t-SNE](CORAL_dinov2_e0_s100/tsne.png) · [PDF](CORAL_dinov2_e0_s100/pca_boundary.pdf) · [coordinates](CORAL_dinov2_e0_s100/points.csv) · [metrics](CORAL_dinov2_e0_s100/metrics.json)

### CORAL / dinov2 — photo

Original target accuracy (seed 100): 99.88%; slice/full-head agreement: 45.69%.

![PCA and classifier slice](CORAL_dinov2_e2_s100/pca_boundary.png)

[t-SNE](CORAL_dinov2_e2_s100/tsne.png) · [PDF](CORAL_dinov2_e2_s100/pca_boundary.pdf) · [coordinates](CORAL_dinov2_e2_s100/points.csv) · [metrics](CORAL_dinov2_e2_s100/metrics.json)

### MMD / dinov2 — cartoon

Original target accuracy (seed 100): 95.52%; slice/full-head agreement: 45.22%.

![PCA and classifier slice](MMD_dinov2_e1_s100/pca_boundary.png)

[t-SNE](MMD_dinov2_e1_s100/tsne.png) · [PDF](MMD_dinov2_e1_s100/pca_boundary.pdf) · [coordinates](MMD_dinov2_e1_s100/points.csv) · [metrics](MMD_dinov2_e1_s100/metrics.json)

### MMD / dinov2 — photo

Original target accuracy (seed 100): 99.94%; slice/full-head agreement: 58.32%.

![PCA and classifier slice](MMD_dinov2_e2_s100/pca_boundary.png)

[t-SNE](MMD_dinov2_e2_s100/tsne.png) · [PDF](MMD_dinov2_e2_s100/pca_boundary.pdf) · [coordinates](MMD_dinov2_e2_s100/points.csv) · [metrics](MMD_dinov2_e2_s100/metrics.json)

### MMD / dinov2 — sketch

Original target accuracy (seed 100): 86.08%; slice/full-head agreement: 35.20%.

![PCA and classifier slice](MMD_dinov2_e3_s100/pca_boundary.png)

[t-SNE](MMD_dinov2_e3_s100/tsne.png) · [PDF](MMD_dinov2_e3_s100/pca_boundary.pdf) · [coordinates](MMD_dinov2_e3_s100/points.csv) · [metrics](MMD_dinov2_e3_s100/metrics.json)

### MMD / dinov2 — art_painting

Original target accuracy (seed 100): 97.66%; slice/full-head agreement: 42.24%.

![PCA and classifier slice](MMD_dinov2_e0_s100/pca_boundary.png)

[t-SNE](MMD_dinov2_e0_s100/tsne.png) · [PDF](MMD_dinov2_e0_s100/pca_boundary.pdf) · [coordinates](MMD_dinov2_e0_s100/points.csv) · [metrics](MMD_dinov2_e0_s100/metrics.json)
