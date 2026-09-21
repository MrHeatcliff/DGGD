"""Native DomainBed algorithm training with source-only checkpoint selection."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/pacs_algorithms"))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc
from compat import algorithm_class


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            prediction = model.predict(x).argmax(1)
            correct += (prediction == y).sum().item()
            total += len(y)
    return correct / total


def run(task):
    out = Path(task['output'])
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'task.json', task)
    seed = task['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    hparams = hparams_registry.default_hparams(task['algorithm'], 'PACS')
    hparams.update(task['hparams'])
    hparams['dinov2_repo'] = str(ROOT / '.cache/dinov2')
    atomic_json(out / 'hparams.json', hparams)
    heldout = task['env']
    data = datasets.PACS(str(ROOT / 'domainbed/data'), [heldout], hparams)
    evaluation_transform = data.datasets[heldout].transform
    train_loaders, validation_loaders = [], []
    workers = task.get('workers', 2)
    for env, source in enumerate(data.datasets):
        if env == heldout:
            continue
        # Fixed source-only split across candidates, confirmation and final seeds.
        validation, train = misc.split_dataset(source, int(0.2 * len(source)), misc.seed_hash(0, env))
        plain = ImageFolder(source.root, transform=evaluation_transform)
        assert plain.samples == source.samples
        validation_plain = Subset(plain, validation.keys[:32] if task['phase']=='smoke' else validation.keys)
        generator = torch.Generator().manual_seed(seed * 10 + env)
        train_loaders.append(DataLoader(train, batch_size=hparams['batch_size'],
            shuffle=True, drop_last=True, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0, generator=generator))
        validation_loaders.append(DataLoader(validation_plain, batch_size=128,
            shuffle=False, num_workers=0, pin_memory=True))
    model = algorithm_class(task['algorithm'])(data.input_shape, data.num_classes, 3, hparams).cuda()
    thresholds=[int(v) for k,v in hparams.items() if k.endswith('anneal_iters') or k in ['iters','eqrm_burnin_iters']]
    if thresholds and hasattr(model,'update_count'):model.update_count.fill_(max(thresholds)+1)
    if task['algorithm']=='ERMPlusPlus':model.global_iter=601
    # Each native algorithm owns its optimizers, objectives and gradient operations.
    iterators = [iter(loader) for loader in train_loaders]
    best_value, best_step = -1.0, 0
    losses = []
    started = time.time()
    for step in range(1, task['steps'] + 1):
        if step == task['warmup'] + 1:
            (out / 'READY').touch()
            while not Path(task['barrier']).exists():time.sleep(0.02)
            torch.cuda.synchronize()
            started=time.perf_counter()
            losses=[]
        minibatches = []
        for i, loader in enumerate(train_loaders):
            try:
                batch = next(iterators[i])
            except StopIteration:
                iterators[i] = iter(loader)
                batch = next(iterators[i])
            minibatches.append(batch)
        model.train()
        if task['phase']=='smoke' and step==4:
            # Exercise post-warmup code paths without running thousands of smoke steps.
            thresholds=[int(v) for k,v in hparams.items() if
                (k.endswith('anneal_iters') or k in ['iters','eqrm_burnin_iters'])]
            if thresholds and hasattr(model,'update_count'):
                model.update_count.fill_(max(thresholds))
            if task['algorithm']=='ERMPlusPlus':
                model.global_iter=601
        minibatches=[(x.cuda(non_blocking=True),y.cuda(non_blocking=True)) for x,y in minibatches]
        metrics=model.update(minibatches,None)
        scalars={k:float(v) for k,v in metrics.items()}
        if not all(math.isfinite(v) for v in scalars.values()):
            raise FloatingPointError(f'Non-finite training metrics: {scalars}')
        losses.append(scalars)
        if step == task['steps']:
            values = [accuracy(model, loader) for loader in validation_loaders]
            value = float(np.mean(values))
            if value > best_value:
                best_value, best_step = value, step
                temporary = out / 'best.pt.tmp'
                torch.save({'model_dict':model.state_dict(), 'hparams':hparams,
                    'step':step, 'source_val':value, 'task':task}, temporary)
                temporary.replace(out / 'best.pt')
            progress = {'step':step, 'source_val':value, 'best_source_val':best_value,
                'best_step':best_step, 'source_val_per_domain':values,
                'train_metrics':{k:float(np.mean([m[k] for m in losses if k in m])) for k in set().union(*losses)}, 'elapsed_seconds':time.perf_counter()-started,
                'max_gpu_memory_gb':torch.cuda.max_memory_allocated()/1024**3}
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(progress) + '\n')
            atomic_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            losses = []
            if task['phase'] == 'search' and (out / 'PRUNE').exists():
                atomic_json(out / 'result.json', dict(progress, state='PRUNED'))
                return
    result = dict(progress, state='COMPLETE', seed=seed, backbone=task['backbone'],
        env=heldout, algorithm=task['algorithm'], phase=task['phase'], slurm_job_id=os.getenv('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(), host=os.uname().nodename)
    if task['phase'] in ['final','smoke']:
        checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_dict'])
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        if task['phase']=='smoke':
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            atomic_json(out / 'result.json', result)
            return
        target = DataLoader(data.datasets[heldout], batch_size=128, shuffle=False,
            num_workers=workers, pin_memory=True)
        result['target_acc'] = accuracy(model, target)
        result['target_samples'] = len(data.datasets[heldout])
    result['measured_steps']=task['steps']-task['warmup']
    result['finished_at']=time.time()
    atomic_json(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    args = parser.parse_args()
    try:
        run(json.loads(Path(args.task).read_text()))
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
