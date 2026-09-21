"""Post-hoc PACS representation diagnostics; never modify ERM selections."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import ImageFolder
from domainbed import algorithms,datasets,hparams_registry
from domainbed.lib import misc
from scripts.pacs.controller import base_hparams


def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False));t.replace(p)


def offline():
    os.environ['HF_HUB_OFFLINE']='1'
    original=socket.socket.connect
    def connect(s,address):
        if s.family==socket.AF_UNIX:return original(s,address)
        raise RuntimeError('Offline representation analysis: network unavailable by design')
    socket.socket.connect=connect


def extract(backbone,checkpoint,path,smoke):
    if path.exists():return torch.load(path,weights_only=False,map_location='cpu')
    torch.manual_seed(20260921)
    h=hparams_registry.default_hparams('ERM','PACS');h.update(base_hparams(backbone))
    h['dinov2_repo']=str(ROOT/'.cache/dinov2')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint else None
    if ck:h.update(ck['hparams'])
    h['dinov2_repo']=str(ROOT/'.cache/dinov2')
    m=algorithms.ERM((3,224,224),7,3,h)
    if ck:m.load_state_dict(ck['model_dict'])
    m.cuda().eval()
    data=datasets.PACS(str(ROOT/'domainbed/data'),[0,1,2,3],h)
    z=[];ys=[];ds=[];vs=[];names=[];pred=[]
    with torch.inference_mode():
        for env,d in enumerate(data.datasets):
            val,_=misc.split_dataset(d,int(.2*len(d)),misc.seed_hash(0,env))
            vi=set(val.keys)
            inds=np.arange(len(d))
            if smoke:
                rng=np.random.default_rng(77+env);selected=[]
                for label in range(7):
                    for validation,count in [(True,8),(False,24)]:
                        cell=[i for i in inds if d.targets[i]==label and (int(i) in vi)==validation]
                        selected.extend(rng.choice(cell,min(count,len(cell)),replace=False).tolist())
                inds=np.array(sorted(selected))
            loader=DataLoader(Subset(d,inds.tolist()),batch_size=128,num_workers=2,pin_memory=True)
            for x,y in loader:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    zz=m.featurizer(x.cuda(non_blocking=True));pp=m.classifier(zz).argmax(1)
                z.append(zz.float().cpu());ys.append(y);pred.append(pp.cpu())
            ds.extend([env]*len(inds));vs.extend([int(i) in vi for i in inds]);names.extend([str(Path(d.samples[i][0]).relative_to(ROOT/'domainbed/data/PACS')) for i in inds])
    result={'z':torch.cat(z),'y':torch.cat(ys),'domain':torch.tensor(ds),'val':torch.tensor(vs),
            'pred':torch.cat(pred),'images':names,'checkpoint':str(checkpoint) if checkpoint else None}
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(path)
    del m,ck;torch.cuda.empty_cache()
    return result


def variants(z,b):
    yield 'full',z
    if b=='dinov2':
        yield 'last_cls',z[:,2304:3072]
        yield 'last_cls_patch',z[:,2304:3840]


def fit_probe(z,y,tr,va,te,domains,pca=None):
    # Normalize each vector, then center using only source training samples.
    z=F.normalize(z.cuda(),dim=1);y=y.cuda()
    x=z[tr];mu=x.mean(0);x=x-mu
    eig,q=torch.linalg.eigh(x.T@x)
    eig=eig.clamp_min(0)
    if pca:
        q=q[:,-min(pca,q.shape[1]):];eig=eig[-q.shape[1]:]
    rhs=q.T@x.T@F.one_hot(y[tr],7).float()
    projected=(z-mu)@q
    options=[]
    for alpha in [.001,.01,.1,1.,10.,100.]:
        prediction=(projected@(rhs/(eig[:,None]+alpha))).argmax(1).cpu()
        score=np.mean([float((prediction[va & (domains==e)]==y.cpu()[va & (domains==e)]).float().mean()) for e in range(4) if (va & (domains==e)).any()])
        options.append((score,alpha,prediction))
    score,alpha,p=max(options,key=lambda a:a[0])
    return {'source_val':score,'alpha':alpha,'target_acc':float((p[te]==y.cpu()[te]).float().mean()),'pca_dimensions':pca}


def balanced_indices(d,target):
    mask=d['val'] | (d['domain']==target)
    cells=[torch.where(mask & (d['domain']==e) & (d['y']==c))[0] for e in range(4) for c in range(7)]
    n=min(40,min(map(len,cells)));n-=n%2
    assert n>=2,'Insufficient samples per class/domain'
    rng=np.random.default_rng(20260921)
    return torch.cat([v[torch.tensor(rng.permutation(len(v))[:n])] for v in cells]),n


def geometry(d,pre,target):
    idx,n=balanced_indices(d,target)
    z=F.normalize(d['z'][idx].cuda(),dim=1);p=F.normalize(pre['z'][idx].cuda(),dim=1)
    y=d['y'][idx];dom=d['domain'][idx]
    centers=torch.stack([z[(dom==e)&(y==c)].mean(0) for e in range(4) for c in range(7)])
    centers=F.normalize(centers,dim=1);dist=(1-centers@centers.T).clamp_min(0)
    ec=torch.arange(4).repeat_interleave(7);cc=torch.arange(7).repeat(4)
    within=float(dist[(ec[:,None]!=ec[None,:])&(cc[:,None]==cc[None,:])].mean())
    between=float(dist[(ec[:,None]==ec[None,:])&(cc[:,None]!=cc[None,:])].mean())
    a=z-z.mean(0);b=p-p.mean(0);ka=a@a.T;kb=b@b.T
    cka=float((ka*kb).sum()/torch.sqrt((ka*ka).sum()*(kb*kb).sum()).clamp_min(1e-12))
    # Domain probe is a post-hoc diagnostic, balanced within each class/domain.
    train=(torch.arange(len(idx))%n)<n//2;test=~train
    mu=z[train].mean(0);x=z[train]-mu;xt=z[test]-mu
    # Dual ridge fixed regularization; no target-guided hyperparameter selection.
    coef=torch.linalg.solve(x@x.T+.1*torch.eye(len(x),device='cuda'),F.one_hot(dom[train].cuda(),4).float())
    dp=(xt@x.T@coef).argmax(1).cpu()
    query=torch.where(d['domain']==target)[0];ref=torch.where(d['val']&(d['domain']!=target))[0]
    refs=F.normalize(d['z'][ref].cuda(),dim=1);correct=0
    for chunk in query.split(256):
        nn=(F.normalize(d['z'][chunk].cuda(),dim=1)@refs.T).argmax(1).cpu()
        correct+=int((d['y'][ref[nn]]==d['y'][chunk]).sum())
    return {'same_class_cross_domain_cosine_distance':within,'different_class_within_domain_cosine_distance':between,
            'distance_ratio':within/max(between,1e-12),'linear_cka_vs_pretrained':cka,
            'balanced_domain_probe_acc':float((dp==dom[test]).float().mean()),'domain_chance':.25,
            'samples_per_class_domain':n,'cross_domain_1nn_acc':correct/len(query)}


def run(args):
    offline();torch.set_num_threads(4)
    assert torch.cuda.is_available()
    out=Path(args.output)/args.backbone;out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in csv.DictReader((ROOT/'outputs/pacs_erm_optuna/reports/all_backbones_runs.csv').open()) if r['backbone']==args.backbone]
    if args.smoke:rows=[r for r in rows if r['seed']=='100' and r['env']=='0']
    pre=extract(args.backbone,None,out/'features/pretrained.pt',args.smoke)
    for target in sorted({int(r['env']) for r in rows}):
        for phase,row in [('pretrained',None)]+[('finetuned',r) for r in rows if int(r['env'])==target]:
            seed=None if row is None else int(row['seed']);name=f'{phase}_e{target}_s{seed}'
            dest=out/(name+'.json')
            if dest.exists():continue
            checkpoint=None if row is None else Path(row['artifact_dir'])/'best.pt'
            d=pre if row is None else extract(args.backbone,checkpoint,out/f'features/{name}.pt',args.smoke)
            assert d['images']==pre['images']
            tr=(d['domain']!=target)&~d['val'];va=(d['domain']!=target)&d['val'];te=d['domain']==target
            r={'backbone':args.backbone,'phase':phase,'target':target,'seed':seed,'smoke':args.smoke,
               'probe_type':'L2-normalized ridge least-squares linear classifier','target_samples':int(te.sum()),
               'geometry':geometry(d,pre,target),'probes':{}}
            for variant,z in variants(d['z'],args.backbone):
                r['probes'][variant]=fit_probe(z,d['y'],tr,va,te,d['domain'])
            r['probes']['full_pca256']=fit_probe(d['z'],d['y'],tr,va,te,d['domain'],pca=256)
            if row:
                acc=float((d['pred'][te]==d['y'][te]).float().mean());r['checkpoint_target_acc']=acc
                r['reported_target_acc']=float(row['target_acc'])
                if not args.smoke:assert abs(acc-r['reported_target_acc'])<1e-6,'Checkpoint accuracy mismatch'
            write(dest,r);print('COMPLETE',args.backbone,name,flush=True)
    write(out/'COMPLETE.json',{'finished':time.time(),'smoke':args.smoke,'gpu':torch.cuda.get_device_name()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
