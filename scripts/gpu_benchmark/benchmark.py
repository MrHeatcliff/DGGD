"""Compare concurrent native PACS training processes, with/without isolated CUDA MPS."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/pacs_gpu_benchmark'


def write(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def run_mode(mode,gpu):
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    mps=mode=='mps'
    if mps:
        pipe=Path('/tmp')/f"pacs-mps-{os.environ['SLURM_JOB_ID']}-{uuid.uuid4().hex[:8]}"
        pipe.mkdir(mode=0o700)
        log=pipe/'logs';log.mkdir()
        environment.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(log))
        subprocess.run(['nvidia-cuda-mps-control','-d'],env=environment,check=True)
        time.sleep(2)
    summaries=[]
    try:
        for algorithm,concurrencies in [('ERMPlusPlus',[1,2,4,8]),('IGA',[1,2,4]),('RDM',[1,2,4])]:
            for concurrency in concurrencies:
                group=RUN/f'{mode}_{algorithm}_{concurrency}'
                group.mkdir(parents=True,exist_ok=True)
                specs=[];processes=[];handles=[]
                for i in range(concurrency):
                    out=group/str(i);out.mkdir(exist_ok=True)
                    task={'id':f'{mode}_{algorithm}_{concurrency}_{i}','algorithm':algorithm,'backbone':'resnet50_augmix',
                        'env':0,'seed':100+i,'phase':'benchmark','warmup':25,'steps':275,'eval_every':250,'workers':1,
                        'hparams':{'vit':False,'dinov2':False,'resnet18':False,'resnet50_augmix':True},
                        'output':str(out),'barrier':str(group/'GO')}
                    write(out/'task.json',task);specs.append(task)
                    handle=(out/'train.log').open('w');handles.append(handle)
                    processes.append(subprocess.Popen([sys.executable,str(ROOT/'scripts/gpu_benchmark/train_trial.py'),str(out/'task.json')],
                        cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT))
                started=time.time()
                while not all((Path(s['output'])/'READY').exists() for s in specs):
                    if any(p.poll() is not None for p in processes) or time.time()-started>300:break
                    time.sleep(.2)
                started=time.time();(group/'GO').touch()
                telemetry=[]
                while any(p.poll() is None for p in processes):
                    try:
                        raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                        telemetry.append([float(v.strip()) for v in raw.strip().split(',')])
                    except Exception:pass
                    if time.time()-started>900:
                        for p in processes:
                            if p.poll() is None:p.terminate()
                        break
                    time.sleep(1)
                for p in processes:p.wait()
                for h in handles:h.close()
                results=[]
                for task in specs:
                    result=Path(task['output'])/'result.json'
                    if result.exists():results.append(json.loads(result.read_text()))
                ok=len(results)==concurrency and all(p.returncode==0 for p in processes)
                summary={'mode':mode,'algorithm':algorithm,'jobs_per_gpu':concurrency,'success':ok,
                    'results':results,'group_wall_seconds':time.time()-started,
                    'gpu_util_mean':sum(t[0] for t in telemetry)/len(telemetry) if telemetry else None,
                    'peak_gpu_memory_mib':max(t[1] for t in telemetry) if telemetry else None}
                if ok:
                    summary['measured_wall_seconds']=max(r['finished_at'] for r in results)-started
                    summary['aggregate_updates_per_second']=250*concurrency/summary['measured_wall_seconds']
                summaries.append(summary);write(group/'summary.json',summary);write(RUN/f'{mode}_summary.json',summaries)
                print(mode,algorithm,concurrency,'OK' if ok else 'FAILED',summary.get('aggregate_updates_per_second'),flush=True)
    finally:
        if mps:subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=environment,check=False)
    return summaries


if __name__=='__main__':
    RUN.mkdir(exist_ok=True,parents=True)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(visible)==2
    import torch
    props=torch.cuda.get_device_properties(0)
    write(RUN/'hardware.json',{'name':props.name,'sm_count':props.multi_processor_count,
        'total_memory_bytes':props.total_memory,'compute_capability':[props.major,props.minor],
        'host':os.uname().nodename,'job_id':os.environ['SLURM_JOB_ID'],
        'visible_devices':visible})
    # UUIDs avoid MPS ordinal remapping when server exposes only one assigned GPU.
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={line.split(',')[0].strip():line.split(',')[1].strip() for line in ids.splitlines()}
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(run_mode,mode,mapping.get(gpu,gpu)) for mode,gpu in zip(['no_mps','mps'],visible)]
        results=[future.result() for future in futures]
    write(RUN/'all_results.json',[row for group in results for row in group])
    (RUN/'COMPLETE').touch()
