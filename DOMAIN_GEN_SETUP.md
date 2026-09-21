# Domain generalization workspace

## Environment

```bash
cd /home/vn-user0101/Dat/DomainBed
source .venv/bin/activate
```

Python 3.10.21, PyTorch 2.5.1 + CUDA 12.4, torchvision 0.20.1,
NumPy 1.26.4; CUDA verified on NVIDIA H200 NVL.
The original `domainbed/requirements.txt` remains unchanged. This installation
uses newer PyTorch/BackPACK/gdown versions; exact installed packages are in
`requirements-domain-gen.lock.txt`.

To recreate:

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements-domain-gen.lock.txt
```

## Data

All datasets are under `domainbed/data/`:

| Dataset argument | Directory | Domains | Classes |
| --- | --- | --- | --- |
| PACS | PACS | 4 | 7 |
| VLCS | VLCS | 4 | 5 |
| TerraIncognita | terra_incognita | 4 | 10 |
| DomainNet | domain_net | 6 | 345 |

Sources used:

- PACS: https://huggingface.co/datasets/Azeez577/PACS (PACS.zip mirror;
  original Google Drive link in this checkout failed).
- VLCS: https://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file,
  linked by https://github.com/fmcarlucci/JigenDG. Its `full` split contains
  `train` + `crossval`; use `full` + `test` once each. Numeric class labels are
  retained, consistently across domains. Domains are named Caltech101,
  LabelMe, SUN09, VOC2007 to preserve DomainBed's C,L,S,V ordering.
- TerraIncognita: the original Google Cloud LILA archives used in
  `domainbed/scripts/download.py`. Keep locations 100,38,43,46 and the same
  ten categories as that script. Annotation indexing and streaming extraction
  produce the same subset without its quadratic annotation loop.
- DomainNet: https://ai.bu.edu/M3SDA/ is the original source;
  https://huggingface.co/yashkant/evalmerge/tree/main/domainnet supplies archive
  mirrors. Mirror SHA-256 checksums are verified. Remove files listed in
  `domainbed/misc/domain_net_duplicates.txt`, as the original downloader does.

Downloaded archives are retained in `domainbed/data/.archives/`.
Preparation code is `scripts/prepare_domain_gen.py`; it is restartable using
`.DATASET.prepared` markers. Logs are under `setup_logs/`.

## Training

From the repository root, after activating `.venv`:

```bash
OMP_NUM_THREADS=4 python -m domainbed.scripts.train \
  --data_dir domainbed/data \
  --dataset PACS --algorithm ERM --test_envs 0 \
  --hparams '{"batch_size": 32}' \
  --output_dir outputs/PACS_ERM_env0
```

Replace `PACS` with `VLCS`, `TerraIncognita`, or `DomainNet`.
This checkout defaults to the timm AugMix pretrained ResNet50
(`resnet50_augmix=true`, `freeze_bn=false`). Smoke tests use these defaults.
The torchvision and timm pretrained weights have been downloaded to the user's
model caches. Training results and large files are excluded from git.

## Smoke tests

```bash
python scripts/smoke_domain_gen.py \
  --dataset PACS --output_dir outputs/smoke_repeat/PACS
```

Run once per dataset. The runner calls the actual `domainbed.scripts.train`
entry point with ERM, pretrained ResNet50, batch size 2 per source domain,
3 optimizer steps and domain 0 held out. It inventories the complete dataset,
checks matching class indices across domains, then uses 40 reproducibly sampled
real images per domain (32 train / 8 validation). It checks finite losses and
gradients, changed classifier weights, evaluation metrics, and checkpoint reload.
Worker count is 0 only within the smoke runner; production defaults are unchanged.

`outputs/smoke/*/dataset_inventory.json` records full dataset counts;
`smoke_checks.json`, `results.jsonl`, `model.pkl`, and `done` record validation.
These are pipeline tests, not convergence or benchmark-accuracy measurements.

Additional validation: all images in PACS, VLCS and TerraIncognita were fully
decoded with Pillow without errors (`domainbed/data/image_validation_small.json`).
A standard CLI PACS run, with the full dataset and default multiprocessing loaders,
completed one optimizer step and evaluated every split; artifacts are in
`outputs/smoke/PACS_full_pipeline/`.
Source revisions and archive checksums are retained in `domainbed/data/sources.json`
and `domainbed/data/archive_sha256.json`.

## Verified dataset totals

| Dataset | Images | Smoke test |
| --- | ---: | --- |
| PACS | 9,991 | PASS: 3 optimizer steps |
| VLCS | 10,729 | PASS: 3 optimizer steps |
| TerraIncognita | 24,788 | PASS: 3 optimizer steps |
| DomainNet | 586,575 | PASS: 3 optimizer steps |

Machine-readable summary: `outputs/smoke/summary.json`.
DomainNet extraction verified ZIP CRCs and removed 9,431 listed duplicates.
Terra archive also matches the original Google Cloud Storage MD5.
