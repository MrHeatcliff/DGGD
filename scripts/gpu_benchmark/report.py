"""Summarize measured throughput and choose the smallest near-peak concurrency."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'outputs/pacs_gpu_benchmark'
rows=json.loads((root/'all_results.json').read_text())
selection={}
lines=['# PACS concurrency benchmark','',
'One optimizer update consumes a minibatch from each of three source domains.',
'Each process warms up for 25 updates, synchronizes at a barrier, then measures 250 updates plus one full source validation and checkpoint write.',
'Native FP32 algorithms, 1 Torch thread and 1 DataLoader worker per source; H200, 132 SMs, approximately 140 GiB VRAM.',
'Warmup/penalty counters start past annealing thresholds for representative steady-state compute.',
'132 Hopper SMs × 128 FP32 lanes/SM correspond to 16,896 FP32 CUDA cores; this arithmetic is not a job-count limit.', '',
'| Algorithm | MPS | Jobs/GPU | Aggregate updates/s | Peak device MiB | Success |',
'| --- | --- | ---: | ---: | ---: | --- |']
for r in rows:
 skipped=root/f"{r['mode']}_{r['algorithm']}_{r['jobs_per_gpu']}"/'SKIPPED.json'
 if skipped.exists():r['skipped_reason']=json.loads(skipped.read_text())['reason']
 lines.append(f"| {r['algorithm']} | {r['mode']} | {r['jobs_per_gpu']} | {r.get('aggregate_updates_per_second',0):.2f} | {r.get('peak_gpu_memory_mib',0)} | {'SKIPPED' if r.get('skipped_reason') else r['success']} |")
for algorithm in sorted({r['algorithm'] for r in rows}):
 candidates=[r for r in rows if r['algorithm']==algorithm and r['mode']=='mps' and r['success']]
 peak=max(r['aggregate_updates_per_second'] for r in candidates)
 winner=min([r for r in candidates if r['aggregate_updates_per_second']>=.95*peak],key=lambda r:r['jobs_per_gpu'])
 baseline=next(r for r in rows if r['algorithm']==algorithm and r['mode']=='no_mps' and r['jobs_per_gpu']==1)
 selection[algorithm]={'jobs_per_gpu':winner['jobs_per_gpu'],'updates_per_second':winner['aggregate_updates_per_second'],
 'speedup_vs_single_no_mps':winner['aggregate_updates_per_second']/baseline['aggregate_updates_per_second']}
lines += ['', 'Choose the smallest MPS concurrency within 5% of measured peak throughput.',
 'These are short representative measurements, not a guarantee for every algorithm or full training run.',
 '', 'References: [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/index.html),',
 '[NVIDIA Hopper SM architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/),',
 '[NVIDIA H200 architecture overview](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf).']
(root/'report.md').write_text('\n'.join(lines)+'\n')
(root/'selection.json').write_text(json.dumps(selection,indent=2))
print(json.dumps(selection,indent=2))
