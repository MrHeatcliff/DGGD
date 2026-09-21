"""Memory-aware concurrent workers inside two allocated GPUs, optionally using MPS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/pacs_algorithms'))
from resources import can_admit


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def worker(run,gpu,slot,environment,active,lock,config,peaks):
    owner=f"{os.environ['SLURM_JOB_ID']}_{slot}"
    while not (run/'STOP').exists() and not (run/'COMPLETE').exists():
        claimed=task=None
        with lock:
            for candidate in sorted((run/'queue').glob('*.json')):
                try: proposed=json.loads(candidate.read_text())
                except FileNotFoundError:continue
                if not can_admit(proposed['algorithm'],active[gpu],config,peaks):continue
                destination=run/'running'/(candidate.stem+'__'+owner+'.json')
                try:candidate.rename(destination)
                except FileNotFoundError:continue
                claimed,task=destination,proposed
                active[gpu].append(task['algorithm'])
                break
        if claimed is None:time.sleep(2);continue
        out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
        write(out/'owner.json',{'job_id':os.environ['SLURM_JOB_ID'],'gpu':gpu,'slot':slot,
            'started':time.time(),'attempt':task.get('attempt',1),'mps':config.get('mps',False),
            'jobs_per_gpu_limit':config.get('jobs_per_gpu',1)})
        try:
            with (out/f"train_attempt{task.get('attempt',1)}.log").open('w') as log:
                p=subprocess.Popen([sys.executable,str(ROOT/'scripts/pacs_algorithms/train_trial.py'),str(claimed)],
                    cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    write(out/'heartbeat.json',{'time':time.time(),'job_id':os.environ['SLURM_JOB_ID']})
                    if (run/'STOP').exists():
                        p.terminate()
                        try:p.wait(timeout=30)
                        except subprocess.TimeoutExpired:p.kill();p.wait()
                        return
                    time.sleep(3)
            if p.returncode!=0:write(out/'failure.json',{'returncode':p.returncode,'time':time.time()})
            claimed.unlink(missing_ok=True)
        finally:
            with lock:active[gpu].remove(task['algorithm'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--gpus',type=int,required=True)
    args=parser.parse_args();run=Path(args.run_dir).resolve()
    config=json.loads((run/'config.json').read_text())
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(visible)==args.gpus
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={row.split(',')[0].strip():row.split(',')[1].strip() for row in rows.splitlines()}
    gpus=[mapping.get(g,g) for g in visible]
    summary=json.loads((run/'preflight/summary.json').read_text())
    peaks={name:r['max_gpu_memory_gb'] for name,r in summary.items()}
    environments={};mps_environments=[]
    try:
        for gpu in gpus:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            if config.get('mps',False):
                pipe=Path(tempfile.mkdtemp(prefix=f"pacs-mps-{os.environ['SLURM_JOB_ID']}-"))
                (pipe/'logs').mkdir()
                env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(pipe/'logs'))
                subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
                mps_environments.append(env)
            environments[gpu]=env
        active={gpu:[] for gpu in gpus};lock=threading.Lock();count=config.get('jobs_per_gpu',1)
        print('Allocation',gpus,'max jobs/GPU',count,'MPS',config.get('mps',False),flush=True)
        with ThreadPoolExecutor(args.gpus*count) as pool:
            futures=[pool.submit(worker,run,gpu,i*count+j,environments[gpu],active,lock,config,peaks)
                for i,gpu in enumerate(gpus) for j in range(count)]
            for f in futures:f.result()
    finally:
        for env in mps_environments:
            subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)
