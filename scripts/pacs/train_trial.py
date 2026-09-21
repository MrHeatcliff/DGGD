"""One ERM trial: source-only model selection, target evaluation only for final runs."""
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
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
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
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    hparams = hparams_registry.default_hparams('ERM', 'PACS')
    hparams.update(task['hparams'])
    hparams['dinov2_repo'] = str(ROOT / '.cache/dinov2')
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
        validation_plain = Subset(plain, validation.keys)
        generator = torch.Generator().manual_seed(seed * 10 + env)
        train_loaders.append(DataLoader(train, batch_size=hparams['batch_size'],
            shuffle=True, drop_last=True, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0, generator=generator))
        validation_loaders.append(DataLoader(validation_plain, batch_size=128,
            shuffle=False, num_workers=0, pin_memory=True))
    model = algorithms.ERM(data.input_shape, data.num_classes, 3, hparams).cuda()
    # ERM uses Adam and cross-entropy. BF16 changes precision, not the objective.
    iterators = [iter(loader) for loader in train_loaders]
    best_value, best_step = -1.0, 0
    losses = []
    started = time.time()
    for step in range(1, task['steps'] + 1):
        minibatches = []
        for i, loader in enumerate(train_loaders):
            try:
                batch = next(iterators[i])
            except StopIteration:
                iterators[i] = iter(loader)
                batch = next(iterators[i])
            minibatches.append(batch)
        x = torch.cat([b[0] for b in minibatches]).cuda(non_blocking=True)
        y = torch.cat([b[1] for b in minibatches]).cuda(non_blocking=True)
        model.train()
        model.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model.predict(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        model.optimizer.step()
        losses.append(loss.item())
        if step % task['eval_every'] == 0 or step == task['steps']:
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
                'loss':float(np.mean(losses)), 'elapsed_seconds':time.time()-started,
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
        env=heldout, phase=task['phase'], slurm_job_id=os.getenv('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(), host=os.uname().nodename)
    if task['phase'] == 'final':
        checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_dict'])
        target = DataLoader(data.datasets[heldout], batch_size=128, shuffle=False,
            num_workers=workers, pin_memory=True)
        result['target_acc'] = accuracy(model, target)
        result['target_samples'] = len(data.datasets[heldout])
    atomic_json(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    args = parser.parse_args()
    run(json.loads(Path(args.task).read_text()))
