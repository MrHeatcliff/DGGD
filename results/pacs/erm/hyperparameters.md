# Selected ERM hyperparameters

One selected configuration per backbone and held-out domain. Selection uses only source validation; see selected_hparams.json for confirmation scores and alternative finalists.

| Backbone | Target | Trial | LR | Weight decay | Batch/source | Dropout | Freeze BN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dinov2 | art_painting | 28 | 3.8097811e-06 | 1.6982902e-05 | 64 | 0.2 | False |
| dinov2 | cartoon | 20 | 3.9905772e-06 | 2.8150966e-06 | 32 | 0.1 | False |
| dinov2 | photo | 32 | 2.1192016e-06 | 0.00018079227 | 32 | 0.1 | False |
| dinov2 | sketch | 2 | 3.2353394e-06 | 6.9662082e-05 | 32 | 0.5 | False |
| resnet18 | art_painting | 16 | 7.1286545e-05 | 1.3591177e-05 | 64 | 0.1 | False |
| resnet18 | cartoon | 26 | 1.4112144e-05 | 0.0072342825 | 32 | 0.0 | False |
| resnet18 | photo | 23 | 3.353196e-05 | 1.0417448e-05 | 32 | 0.2 | False |
| resnet18 | sketch | 35 | 2.9488961e-05 | 4.0049822e-05 | 16 | 0.3 | False |
| resnet50 | art_painting | 29 | 1.5968179e-05 | 0.0075222003 | 32 | 0.0 | False |
| resnet50 | cartoon | 11 | 1.4013033e-05 | 3.8917778e-07 | 64 | 0.3 | False |
| resnet50 | photo | 27 | 2.5876193e-05 | 0.00058565018 | 64 | 0.5 | True |
| resnet50 | sketch | 28 | 2.9129946e-05 | 0.0003682048 | 32 | 0.2 | False |
| resnet50_augmix | art_painting | 16 | 1.5227325e-05 | 1.8910924e-05 | 32 | 0.2 | False |
| resnet50_augmix | cartoon | 24 | 7.071059e-06 | 1.7198888e-07 | 64 | 0.2 | False |
| resnet50_augmix | photo | 37 | 1.9068903e-05 | 3.4062124e-06 | 64 | 0.5 | False |
| resnet50_augmix | sketch | 30 | 1.4990254e-05 | 2.2985541e-05 | 64 | 0.5 | True |
