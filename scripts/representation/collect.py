"""Collect only after all four representation workers finish."""
import csv,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);bs=['dinov2','resnet18','resnet50','resnet50_augmix'];rows=[]
for b in bs:
 assert (root/b/'COMPLETE.json').exists(),b
 rr=[json.loads(p.read_text()) for p in sorted((root/b).glob('*_e*_s*.json'))]
 assert len(rr)==16 and not any(r['smoke'] for r in rr),(b,len(rr))
 rows.extend(rr)
(root/'all_results.json').write_text(json.dumps(rows,indent=2))
lines=['# PACS representation diagnostics','', 'Post-hoc analysis; fixed ERM checkpoints, no target-based model selection. Ridge probes choose regularization using source validation only. Domain-probe labels from target are used solely for diagnosis. Pretrained has one deterministic probe per split; finetuned has three training seeds.','', '| Backbone | Phase | Full probe target AVG (%) | PCA256 probe AVG (%) | Cross-domain 1NN (%) | Class/domain distance ratio | Domain probe (%) | CKA vs pretrained |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for b in bs:
 for phase in ['pretrained','finetuned']:
  rr=[r for r in rows if r['backbone']==b and r['phase']==phase]
  mean=lambda fn:statistics.mean(fn(r) for r in rr)
  vals=[mean(lambda r:r['probes']['full']['target_acc'])*100,mean(lambda r:r['probes']['full_pca256']['target_acc'])*100,mean(lambda r:r['geometry']['cross_domain_1nn_acc'])*100,mean(lambda r:r['geometry']['distance_ratio']),mean(lambda r:r['geometry']['balanced_domain_probe_acc'])*100,mean(lambda r:r['geometry']['linear_cka_vs_pretrained'])]
  lines.append('| '+b+' | '+phase+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines+=['','Lower distance ratio suggests closer same-class cross-domain centroids relative to different classes. Lower domain accuracy is not sufficient evidence of good representations. CKA measures change, not quality. Ablation heads are refitted on fixed features; this does not retrain each feature variant end-to-end. Target labels must not be fed back into checkpoint selection.','', 'Detailed results, domain/seed values and DINOv2 CLS/patch ablations: all_results.json.']
(root/'report.md').write_text('\n'.join(lines)+'\n');(root/'COMPLETE').touch()
print('\n'.join(lines))
