"""Every method/backbone, maximum batch, post-anneal and discriminator paths."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.pacs_tuned.controller import base_hparams
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--env',type=int,required=True);a=p.parse_args()
out=Path(a.output)
for backbone in ['dinov2','resnet18','resnet50','resnet50_augmix']:
 for algorithm in ['CORAL','MMD','DANN','IRM']:
    pair=algorithm+'--'+backbone;dest=out/pair;dest.mkdir(parents=True,exist_ok=True)
    h=base_hparams(backbone);h.update(batch_size=64,lr=1e-5 if backbone=='dinov2' else 5e-5,weight_decay=1e-4,vit_dropout=.5,resnet_dropout=.5,freeze_bn=False)
    if algorithm=='DANN':h.update(lr_g=h['lr'],lr_d=5e-5,weight_decay_g=1e-4,weight_decay_d=1e-4,d_steps_per_g_step=8,grad_penalty=1.,mlp_width=1024,mlp_depth=5,mlp_dropout=.5)
    task={'id':'smoke_'+pair,'phase':'smoke','backbone':pair,'env':a.env,'seed':100,'steps':10,'eval_every':10,'workers':2,'hparams':h,'output':str(dest)}
    path=dest/'input.json';path.write_text(json.dumps(task))
    with (dest/'train.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts/pacs_tuned/train_trial.py'),str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((dest/'result.json').read_text());assert result['state']=='COMPLETE'
    print('PASS',pair,a.env,result['max_gpu_memory_gb'],flush=True)
(out/'PASSED').touch()
