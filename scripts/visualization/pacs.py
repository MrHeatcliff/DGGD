"""PACS representation plots and exact linear-classifier slices (not 2-D retraining)."""
import argparse,csv,json,os,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from domainbed import algorithms,datasets
from domainbed.lib import misc

OUT=ROOT/'results/pacs/visualizations';CACHE=ROOT/'outputs/pacs_visualizations'
DOMAINS=['art_painting','cartoon','photo','sketch']
CLASSES=sorted(p.name for p in (ROOT/'domainbed/data/PACS/art_painting').iterdir() if p.is_dir())
COLORS=plt.get_cmap('tab10').colors[:7];CMAP=ListedColormap(COLORS)


def extract(row,method):
    backbone=row['backbone'];env=int(row['env']);seed=int(row['seed'])
    if method=='ERM':
        f=ROOT/f'outputs/pacs_representation_20260921/full/{backbone}/features/finetuned_e{env}_s{seed}.pt'
        d=torch.load(f,map_location='cpu',weights_only=False)
        ck=torch.load(Path(row['artifact_dir'])/'best.pt',map_location='cpu',weights_only=False)
        return d,ck
    path=CACHE/f'{method}_{backbone}_e{env}_s{seed}.pt'
    ck=torch.load(Path(row['artifact_dir'])/'best.pt',map_location='cpu',weights_only=False)
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False),ck
    h=dict(ck['hparams']);h['dinov2_repo']=str(ROOT/'.cache/dinov2')
    model=getattr(algorithms,method)((3,224,224),7,3,h).cuda().eval();model.load_state_dict(ck['model_dict'])
    data=datasets.PACS(str(ROOT/'domainbed/data'),[0,1,2,3],h)
    zs=[];ys=[];ds=[];vs=[];ps=[];images=[]
    with torch.inference_mode():
        for e,d in enumerate(data.datasets):
            val,_=misc.split_dataset(d,int(.2*len(d)),misc.seed_hash(0,e));v=set(val.keys)
            for x,y in DataLoader(d,batch_size=128,num_workers=2,pin_memory=True):
                z=model.featurizer(x.cuda(non_blocking=True));pred=model.classifier(z).argmax(1)
                zs.append(z.cpu());ys.append(y);ps.append(pred.cpu())
            ds.extend([e]*len(d));vs.extend([i in v for i in range(len(d))]);images.extend([str(Path(x).relative_to(ROOT/'domainbed/data/PACS')) for x,y in d.samples])
    d={'z':torch.cat(zs),'y':torch.cat(ys),'domain':torch.tensor(ds),'val':torch.tensor(vs),'pred':torch.cat(ps),'images':images}
    target=d['domain']==env
    assert abs(float((d['pred'][target]==d['y'][target]).float().mean())-float(row['target_acc']))<1e-6,'Original checkpoint accuracy mismatch'
    tmp=path.with_suffix('.tmp');torch.save(d,tmp);tmp.replace(path);del model;torch.cuda.empty_cache()
    return d,ck


def sample_indices(d,target):
    rng=np.random.default_rng(20260924);mask=d['val'].numpy() | (d['domain'].numpy()==target)
    groups=[np.flatnonzero(mask & (d['domain'].numpy()==e) & (d['y'].numpy()==c)) for e in range(4) for c in range(7)]
    n=min(40,min(map(len,groups)))
    return np.concatenate([rng.choice(g,n,replace=False) for g in groups]),n


def run_case(row,method):
    backbone=row['backbone'];env=int(row['env']);case=f'{method}_{backbone}_e{env}_s100'
    dest=OUT/case;dest.mkdir(exist_ok=True)
    if (dest/'metrics.json').exists():return json.loads((dest/'metrics.json').read_text())
    d,ck=extract(row,method);z=d['z'].numpy();y=d['y'].numpy();dom=d['domain'].numpy();val=d['val'].numpy()
    train=(dom!=env)&~val;target=dom==env;source_val=(dom!=env)&val
    w=ck['model_dict']['classifier.weight'].float().numpy();b=ck['model_dict']['classifier.bias'].float().numpy()
    pca=PCA(n_components=2,svd_solver='randomized',random_state=20260924).fit(z[train])
    xy=pca.transform(z);basis=pca.components_;mean=pca.mean_
    projected_w=w@basis.T;projected_b=w@mean+b
    full_logits=z@w.T+b;full_pred=full_logits.argmax(1)
    slice_pred=(xy@projected_w.T+projected_b).argmax(1)
    # Check the 2-D equation exactly equals applying the original head after inverse PCA.
    q=xy[:31];assert np.allclose(pca.inverse_transform(q)@w.T+b,q@projected_w.T+projected_b,atol=2e-4,rtol=2e-4)
    idx,n=sample_indices(d,env);shown=xy[idx]
    lo=np.min(shown,axis=0);hi=np.max(shown,axis=0);pad=(hi-lo)*.08;lo-=pad;hi+=pad
    gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],240),np.linspace(lo[1],hi[1],240))
    grid=np.c_[gx.ravel(),gy.ravel()];regions=(grid@projected_w.T+projected_b).argmax(1).reshape(gx.shape)
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    a=axes[0,0];a.scatter(*shown.T,c=y[idx],cmap=CMAP,vmin=0,vmax=6,s=9,alpha=.65,rasterized=True);a.set_title('Representation PCA: color = true class')
    a=axes[0,1]
    for e in range(4):
        pick=dom[idx]==e;a.scatter(*shown[pick].T,s=9,alpha=.6,label=DOMAINS[e])
    a.legend(fontsize=8);a.set_title('Same PCA coordinates: color = domain')
    a=axes[1,0];a.pcolormesh(gx,gy,regions,cmap=CMAP,vmin=0,vmax=6,shading='auto',alpha=.23,rasterized=True)
    if len(np.unique(regions))>1:a.contour(gx,gy,regions,levels=np.arange(6)+.5,colors='gray',linewidths=.4)
    ti=idx[dom[idx]==env];a.scatter(*xy[ti].T,c=y[ti],cmap=CMAP,vmin=0,vmax=6,s=13,edgecolors='white',linewidths=.2,rasterized=True)
    fidelity=float((slice_pred[target]==full_pred[target]).mean())
    a.set_title(f'Original classifier on PCA plane; target points\nSlice/full-head agreement: {fidelity:.1%} (not target accuracy)')
    for a in [axes[0,0],axes[0,1],axes[1,0]]:
        a.set_xlim(lo[0],hi[0]);a.set_ylim(lo[1],hi[1]);a.set_xlabel('Source-fitted PC1');a.set_ylabel('Source-fitted PC2')
    other=full_logits.copy();other[np.arange(len(y)),y]=-np.inf
    margin=full_logits[np.arange(len(y)),y]-other.max(1)
    a=axes[1,1];a.hist(margin[source_val],bins=45,density=True,alpha=.55,label='Source validation');a.hist(margin[target],bins=45,density=True,alpha=.55,label='Target');a.axvline(0,color='black',ls='--');a.legend();a.set_xlabel('True-class logit minus strongest alternative (full feature)');a.set_ylabel('Density');a.set_title('Actual full-feature classification margin\nNegative = error; positive = correct')
    handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[c],label=CLASSES[c],markersize=6) for c in range(7)]
    axes[0,0].legend(handles=handles,fontsize=7,ncol=2,loc='best')
    fig.suptitle(f'{method} / {backbone} | held-out {DOMAINS[env]} | fixed seed 100\nPCA explains {pca.explained_variance_ratio_.sum():.1%} of source feature variance; 2-D slice is not the full boundary',fontsize=12)
    fig.savefig(dest/'pca_boundary.png',dpi=170);fig.savefig(dest/'pca_boundary.pdf');plt.close(fig)
    # t-SNE is only a descriptive manifold view, never a space for a replacement classifier.
    norm=z[idx]/np.maximum(np.linalg.norm(z[idx],axis=1,keepdims=True),1e-12)
    latent=PCA(n_components=min(50,len(idx)-1,z.shape[1]),svd_solver='randomized',random_state=20260924).fit_transform(norm)
    embedding=TSNE(n_components=2,perplexity=min(30.,(len(idx)-1)/3),init='pca',learning_rate='auto',max_iter=1000,random_state=20260924,n_jobs=2).fit_transform(latent)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].scatter(*embedding.T,c=y[idx],cmap=CMAP,vmin=0,vmax=6,s=10,alpha=.8,rasterized=True);axes[0].legend(handles=handles,fontsize=7,ncol=2);axes[0].set_title('Color = class')
    for e in range(4):
        pick=dom[idx]==e;axes[1].scatter(*embedding[pick].T,s=10,alpha=.7,label=DOMAINS[e])
    axes[1].legend(fontsize=8);axes[1].set_title('Color = domain')
    for a in axes:a.set_xticks([]);a.set_yticks([])
    fig.suptitle(f'{method} / {backbone} | target {DOMAINS[env]} | t-SNE descriptive only\nSame balanced images; independent embedding per checkpoint; no boundary inferred here',fontsize=11)
    fig.savefig(dest/'tsne.png',dpi=170);plt.close(fig)
    with (dest/'points.csv').open('w') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['image','class','domain','pc1','pc2','tsne1','tsne2','full_head_prediction','slice_prediction','full_head_margin'])
        for j,i in enumerate(idx):writer.writerow([d['images'][i],CLASSES[y[i]],DOMAINS[dom[i]],*xy[i],*embedding[j],CLASSES[full_pred[i]],CLASSES[slice_pred[i]],margin[i]])
    r={'method':method,'backbone':backbone,'target':DOMAINS[env],'seed':100,'class_order':CLASSES,'checkpoint':str(Path(row['artifact_dir']).relative_to(ROOT))+'/best.pt','original_target_acc':float(row['target_acc']),'fp32_head_on_cached_feature_target_acc':float((full_pred[target]==y[target]).mean()),'pca_variance_explained':pca.explained_variance_ratio_.tolist(),'slice_full_head_target_agreement':fidelity,'slice_full_head_source_val_agreement':float((slice_pred[source_val]==full_pred[source_val]).mean()),'balanced_samples_per_class_domain':n,'target_count':int(target.sum()),'directory':case}
    np.savez_compressed(dest/'plane.npz',mean=mean,basis=basis,classifier_weight=w,classifier_bias=b,projected_weight=projected_w,projected_bias=projected_b)
    (dest/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print('COMPLETE',case,flush=True)
    return r


def main():
    os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(4)
    original=socket.socket.connect
    def offline(s,address):
        if s.family==socket.AF_UNIX:return original(s,address)
        raise RuntimeError('Offline extraction: cache weights first')
    socket.socket.connect=offline
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    records=[]
    for method in ['ERM','CORAL','MMD']:
        path=ROOT/('outputs/pacs_erm_optuna/reports/all_backbones_runs.csv' if method=='ERM' else f'outputs/pacs_methods_optuna/reports/{method}--dinov2_runs.csv')
        for row in csv.DictReader(path.open()):
            if row['seed']!='100':continue
            row['backbone']=row['backbone'].split('--')[-1]
            records.append(run_case(row,method))
    assert len(records)==24
    (OUT/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    lines=['# PACS representations and original-classifier decision slices','', '24 fixed-seed checkpoint views: ERM on four backbones, CORAL/MMD on DINOv2; four target domains each. Seed 100 was fixed before visualization, not selected for appearance or accuracy. DANN/IRM and their ResNet configurations do not yet have complete tuned results and are not presented as final.', '', '## How to read these plots','', '- PCA is fitted only to original source-training features, independently for each checkpoint. Shared colors, balanced images and deterministic seeds make examples comparable; axes/rotations across models are not aligned.','- Top-left: class labels. Top-right: domains on exactly the same PCA coordinates. Source points come from source validation; target points from the held-out domain.','- Bottom-left: exact original linear classifier evaluated on z = source_mean + PC1*u + PC2*v. This is a 2-D affine slice, not a classifier trained on 2-D points and not the entire high-dimensional boundary. Projected observations usually have residual components outside this plane. Slice/full-head agreement and explained variance quantify this limitation.','- Bottom-right: full-feature true-class margin, using original head weights with FP32 arithmetic. Negative means the head predicts another class. It is not affected by the 2-D projection. Logit magnitudes across independently trained models are not calibrated.','- t-SNE uses L2-normalized balanced features and a PCA-50 preprocessing step. It uses target images only for post-hoc description; no tuning or model selection. Cluster size, spacing and rotation are not directly comparable between panels. No decision boundary is drawn in t-SNE coordinates.','- ERM reuses its BF16 feature caches; CORAL/MMD features are native FP32. Tiny differences from original ERM predictions can arise from applying the head in FP32. Original target accuracy and diagnostic-head accuracy are both in metrics.json.','- plane.npz contains the source-fitted mean/PCA basis and trained head parameters to reproduce each boundary slice. points.csv contains displayed sample identities, embeddings, predictions and margins; no source images are committed. PNG is for GitHub viewing, PDF for export.','', '## Gallery','']
    for r in records:
        folder=r['directory'];lines += [f"### {r['method']} / {r['backbone']} — {r['target']}",'',f"Original target accuracy (seed 100): {r['original_target_acc']:.2%}; slice/full-head agreement: {r['slice_full_head_target_agreement']:.2%}.",'',f'![PCA and classifier slice]({folder}/pca_boundary.png)','',f'[t-SNE]({folder}/tsne.png) · [PDF]({folder}/pca_boundary.pdf) · [coordinates]({folder}/points.csv) · [metrics]({folder}/metrics.json)','']
    (OUT/'README.md').write_text('\n'.join(lines));(CACHE/'COMPLETE').touch()

if __name__=='__main__':main()
