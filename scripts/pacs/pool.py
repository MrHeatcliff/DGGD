"""A Slurm allocation runs one isolated process per allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/pacs'))
from train_trial import atomic_json


def worker(run, gpu, slot):
    owner = f"{os.environ['SLURM_JOB_ID']}_{slot}"
    checked_backbones = set()
    while not (run / 'STOP').exists() and not (run / 'COMPLETE').exists():
        claimed = None
        for candidate in sorted((run / 'queue').glob('*.json')):
            destination = run / 'running' / (candidate.stem + '__' + owner + '.json')
            try:
                candidate.rename(destination)
                claimed = destination
                break
            except FileNotFoundError:
                continue
        if claimed is None:
            time.sleep(5)
            continue
        task = json.loads(claimed.read_text())
        out = Path(task['output'])
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'owner.json', {'job_id':os.environ['SLURM_JOB_ID'],
            'slot':slot, 'gpu':gpu, 'started':time.time(), 'attempt':task.get('attempt',1)})
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
            HF_HUB_OFFLINE='1')
        backbone = task['backbone']
        if backbone.startswith('resnet') and backbone not in checked_backbones:
            check_dir = run / 'gpu_preflight' / owner / backbone
            check_dir.mkdir(parents=True, exist_ok=True)
            with (check_dir / 'preflight.log').open('w') as log:
                check = subprocess.run([sys.executable, str(ROOT/'scripts/pacs/smoke_gpu.py'),
                    '--output', str(check_dir), '--backbone', backbone],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if check.returncode:
                atomic_json(run/'PREFLIGHT_FAILED.json', {'backbone':backbone,
                    'owner':owner, 'log':str(check_dir/'preflight.log')})
                (run/'STOP').touch()
                raise RuntimeError(f'GPU preflight failed: {check_dir}')
            checked_backbones.add(backbone)
        with (out / f"train_attempt{task.get('attempt',1)}.log").open('w') as log:
            process = subprocess.Popen([sys.executable, str(ROOT/'scripts/pacs/train_trial.py'), str(claimed)],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                atomic_json(out / 'heartbeat.json', {'time':time.time(), 'job_id':os.environ['SLURM_JOB_ID']})
                if (run / 'STOP').exists():
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    return
                time.sleep(5)
        if process.returncode != 0:
            atomic_json(out/'failure.json', {'returncode':process.returncode,'job_id':os.environ['SLURM_JOB_ID'], 'time':time.time()})
        claimed.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--gpus',type=int,required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    assert len(visible) == args.gpus and all(visible), (visible,args.gpus)
    print('GPU workers:', visible, flush=True)
    with ThreadPoolExecutor(args.gpus) as pool:
        futures = [pool.submit(worker, run, gpu, slot) for slot,gpu in enumerate(visible)]
        for future in futures: future.result()
