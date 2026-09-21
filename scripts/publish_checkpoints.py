"""Stage completed final PACS checkpoints, resumably upload, verify every SHA256."""
import argparse,hashlib,json,os,time
from pathlib import Path
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args()
 stage=ROOT/'outputs/hf_checkpoint_export';stage.mkdir(parents=True,exist_ok=True)
 records=[]
 for campaign,label in [('pacs_erm_optuna','erm'),('pacs_algorithms','algorithms')]:
  for f in sorted((ROOT/'outputs'/campaign/'runs').glob('*/result.json')):
   task=json.loads((f.parent/'task.json').read_text());result=json.loads(f.read_text())
   if task['phase']!='final' or result['state']!='COMPLETE':continue
   src=f.parent/'best.pt';assert src.exists()
   dest=stage/label/task['id']/'best.pt';dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():os.link(src,dest)
   h=hashlib.sha256()
   with src.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
   record={'campaign':label,'task_id':task['id'],'backbone':task['backbone'],'algorithm':task.get('algorithm','ERM'),'env':task['env'],'seed':task['seed'],'best_step':result['best_step'],'target_acc':result['target_acc'],'source_val':result['best_source_val'],'path':str(dest.relative_to(stage)),'bytes':src.stat().st_size,'sha256':h.hexdigest()}
   records.append(record)
   (dest.parent/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
 manifest={'repo_id':a.repo,'selection':'best source-validation checkpoint; final runs only; not target oracle','count':len(records),'total_bytes':sum(r['bytes'] for r in records),'checkpoints':records}
 (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 local=ROOT/'results/pacs/checkpoints';local.mkdir(parents=True,exist_ok=True)
 (local/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 card='''---
model_name: DGGD PACS final checkpoints
library_name: pytorch
tags:
- domain-generalization
- image-classification
- pacs
---
# DGGD PACS final checkpoints

Code, environment, protocol and results: https://github.com/MrHeatcliff/DGGD

360 completed final checkpoints: 48 tuned ERM runs (4 backbones × 4 domains × 3 seeds), plus 312 runs from 26 other algorithms (ResNet50 AugMix, default hyperparameters). All are selected on source-domain validation, not target oracle. No incomplete or smoke runs, intermediate Optuna trial weights, training data or optimizer-resume guarantee.

See `manifest.json` for paths, SHA256, sizes, source validation, target accuracy and selected update. Domain order: art_painting, cartoon, photo, sketch. Seeds: 100, 101, 102.

## Loading

Use the matching DGGD code and package environment; these are PyTorch state dictionaries, not Transformers models. The backbone initially requires cached pretrained weights and DINOv2 code where applicable. `best.pt` contains `model_dict`, `hparams`, and task metadata. Build `domainbed.algorithms.ERM` for ERM, or `scripts.pacs_algorithms.compat.algorithm_class(name)` for other algorithms, then load `model_dict` and call `eval()`. Fishr requires the compatibility class. Only load pickle checkpoints from a trusted source.

```python
from huggingface_hub import hf_hub_download
import torch
path = hf_hub_download(repo_id="REPO_ID", filename="PATH_FROM_MANIFEST")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
```

These checkpoints inherit the applicable terms of their original pretrained backbones and code; see DINOv2, torchvision and timm model documentation. This upload does not claim a new blanket license over third-party weights. Evaluation uses the custom protocol documented in the linked repository; ERM and other algorithms have different tuning budgets and numeric precision.
'''.replace('REPO_ID',a.repo)
 (stage/'README.md').write_text(card)
 print('STAGED',len(records),manifest['total_bytes'],flush=True)
 api=HfApi();print('AUTHENTICATED',api.whoami()['name'],flush=True)
 api.create_repo(a.repo,repo_type='model',private=False,exist_ok=True)
 assert not api.repo_info(a.repo).private,'Expected public repo'
 api.upload_large_folder(repo_id=a.repo,repo_type='model',folder_path=stage,allow_patterns=['*.pt','*/metadata.json','manifest.json','README.md'],num_workers=4,print_report_every=60)
 files={f.path:f for f in api.list_repo_tree(a.repo,recursive=True) if hasattr(f,'size')}
 for r in records:
  f=files[r['path']];assert f.size==r['bytes'],r['path']
  lfs=f.lfs;sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
  assert sha==r['sha256'],f"SHA mismatch {r['path']} {sha}"
 info=api.repo_info(a.repo)
 verified={'repo_id':a.repo,'revision':info.sha,'verified_checkpoints':len(records),'verified_at':time.time(),'total_bytes':manifest['total_bytes']}
 (local/'verified.json').write_text(json.dumps(verified,indent=2)+'\n')
 (stage/'VERIFIED').write_text(json.dumps(verified));print('VERIFIED',verified,flush=True)

if __name__=='__main__':main()
