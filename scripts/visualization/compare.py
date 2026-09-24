"""Build comparison plates and validate published point/plane artifacts."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/pacs/visualizations'
methods=['ERM','CORAL','MMD'];domains=['art_painting','cartoon','photo','sketch'];colors=plt.get_cmap('tab10').colors[:7];cmap=ListedColormap(colors)
for env,domain in enumerate(domains):
 fig,axes=plt.subplots(3,3,figsize=(13,12),constrained_layout=True)
 identities=[]
 for row,method in enumerate(methods):
  dest=out/f'{method}_dinov2_e{env}_s100';meta=json.loads((dest/'metrics.json').read_text());points=list(csv.DictReader((dest/'points.csv').open()));plane=np.load(dest/'plane.npz')
  identities.append([r['image'] for r in points]);classes=meta['class_order']
  xy=np.array([[float(r['pc1']),float(r['pc2'])] for r in points]);y=np.array([classes.index(r['class']) for r in points]);d=np.array([domains.index(r['domain']) for r in points])
  predicted=(xy@plane['projected_weight'].T+plane['projected_bias']).argmax(1)
  assert all(classes[p]==r['slice_prediction'] for p,r in zip(predicted,points))
  axes[row,0].scatter(*xy.T,c=y,cmap=cmap,vmin=0,vmax=6,s=6,alpha=.6)
  for e in range(4):
   mask=d==e;axes[row,1].scatter(*xy[mask].T,s=6,alpha=.6,label=domains[e])
  lo=xy.min(0);hi=xy.max(0);pad=(hi-lo)*.08;lo-=pad;hi+=pad
  xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],200),np.linspace(lo[1],hi[1],200));grid=np.c_[xx.ravel(),yy.ravel()]
  region=(grid@plane['projected_weight'].T+plane['projected_bias']).argmax(1).reshape(xx.shape)
  axes[row,2].pcolormesh(xx,yy,region,cmap=cmap,vmin=0,vmax=6,shading='auto',alpha=.25,rasterized=True)
  mask=d==env;axes[row,2].scatter(*xy[mask].T,c=y[mask],cmap=cmap,vmin=0,vmax=6,s=10,edgecolors='white',linewidths=.15)
  for a in axes[row]:a.set_xlim(lo[0],hi[0]);a.set_ylim(lo[1],hi[1]);a.set_xlabel('PC1 (independent source PCA)')
  axes[row,0].set_ylabel(method+'\nPC2');axes[row,0].set_title(f'Class structure | variance {sum(meta["pca_variance_explained"]):.1%}')
  axes[row,1].set_title('Domain structure')
  axes[row,2].set_title(f'Classifier slice | agreement {meta["slice_full_head_target_agreement"]:.1%}')
  if row==0:axes[row,1].legend(fontsize=7)
 assert identities[0]==identities[1]==identities[2],'Compared plots must use identical images'
 legend=[plt.Line2D([0],[0],marker='o',ls='',color=colors[i],label=c) for i,c in enumerate(classes)]
 axes[2,0].legend(handles=legend,ncol=2,fontsize=7)
 fig.suptitle(f'DINOv2 methods — held-out {domain}, fixed seed 100\nSame images/colors, independent PCA axes. Slice agreement is NOT target accuracy.',fontsize=13)
 fig.savefig(out/f'comparison_{domain}.png',dpi=160);fig.savefig(out/f'comparison_{domain}.pdf');plt.close(fig)
# Three-seed results, separate from the seed-100 projection diagnostics.
fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
for j,m in enumerate(methods):
 p=ROOT/('outputs/pacs_erm_optuna/reports/dinov2.json' if m=='ERM' else f'outputs/pacs_methods_optuna/reports/{m}--dinov2.json')
 r=json.loads(p.read_text())[0];vals=[r['domains'][d] for d in domains]+[r['overall']]
 ax.bar(np.arange(5)+(j-1)*.25,[v['mean']*100 for v in vals],width=.24,yerr=[v['std']*100 for v in vals],capsize=3,label=m)
ax.set_xticks(np.arange(5),domains+['AVG']);ax.set_ylim(0,105);ax.set_ylabel('Target accuracy (%)');ax.legend();ax.set_title('DINOv2: mean ± sample SD across 3 seeds\nSource-selected checkpoint; ERM BF16/40 trials, CORAL & MMD FP32/60 trials per domain')
fig.savefig(out/'accuracy_comparison.png',dpi=170);fig.savefig(out/'accuracy_comparison.pdf');plt.close(fig)
readme=out/'README.md';s=readme.read_text();section='''## Method comparison plates

![Three-seed target results](accuracy_comparison.png)

The examples below use seed 100; the accuracy chart above uses all three final seeds.

'''
for d in domains:section+=f'### Held-out {d}\n\n![DINOv2 method comparison: {d}](comparison_{d}.png)\n\n[PDF](comparison_{d}.pdf)\n\n'
s=s.replace('## Gallery',section+'## Gallery');readme.write_text(s)
print('PASS: same sample identities across methods; all four decision-plane exports reproduce stored slice predictions')
