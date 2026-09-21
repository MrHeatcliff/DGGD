"""Exercise the actual DomainBed train entry point on small real-data subsets."""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys

# Support `python scripts/smoke_domain_gen.py` from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Subset
from domainbed import datasets, algorithms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['PACS','VLCS','TerraIncognita','DomainNet'])
parser.add_argument('--data_dir', default='domainbed/data')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--steps', type=int, default=3)
args = parser.parse_args()
assert torch.cuda.is_available(), 'This smoke test requires CUDA'
torch.set_num_threads(4)
original = getattr(datasets, args.dataset)
expected = {'PACS': (4,7), 'VLCS': (4,5), 'TerraIncognita': (4,10), 'DomainNet': (6,345)}
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)

class SmokeDataset(original):
    N_WORKERS = 0

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams)
        assert (len(self.datasets), self.num_classes) == expected[args.dataset]
        classes = self.datasets[0].class_to_idx
        report = {'dataset':args.dataset, 'cuda':torch.cuda.get_device_name(), 'domains':[]}
        for i, env in enumerate(self.datasets):
            assert env.class_to_idx == classes, 'Class indices differ between domains'
            report['domains'].append({'name': Path(env.root).name, 'images':len(env), 'classes':len(env.classes)})
            # Reproducible sampling throughout the full domain, not just its first class.
            indices = np.random.RandomState(100+i).choice(len(env), min(40,len(env)), replace=False).tolist()
            self.datasets[i] = Subset(env, indices)
        report['full_image_count'] = sum(d['images'] for d in report['domains'])
        (output/'dataset_inventory.json').write_text(json.dumps(report, indent=2))
        print('Full dataset inventory:', json.dumps(report), flush=True)

setattr(datasets, args.dataset, SmokeDataset)
original_update = algorithms.ERM.update
updates = []

def checked_update(self, minibatches, unlabeled=None):
    parameter = next(self.classifier.parameters())
    before = parameter.detach().clone()
    result = original_update(self, minibatches, unlabeled)
    delta = (parameter.detach() - before).abs().max().item()
    assert math.isfinite(result['loss']) and delta > 0
    assert all(torch.isfinite(p.grad).all() for p in self.parameters() if p.grad is not None)
    updates.append({'loss': result['loss'], 'classifier_max_weight_change': delta})
    return result

algorithms.ERM.update = checked_update
sys.argv = ['domainbed.scripts.train', '--data_dir', args.data_dir,
 '--dataset', args.dataset, '--algorithm','ERM','--test_envs','0',
 '--steps',str(args.steps),'--checkpoint_freq',str(args.steps),
 '--hparams','{"batch_size": 2, "resnet18": false}', '--output_dir',str(output)]
runpy.run_module('domainbed.scripts.train',run_name='__main__')
rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
assert rows[-1]['step'] == args.steps-1
assert all(math.isfinite(row['loss']) for row in rows)
assert all(0 <= v <= 1 for row in rows for k,v in row.items() if k.endswith('_acc'))
checkpoint = torch.load(output/'model.pkl', map_location='cpu', weights_only=False)
assert checkpoint['model_num_classes'] == expected[args.dataset][1]
assert all(torch.isfinite(value).all() for value in checkpoint['model_dict'].values())
assert (output/'done').is_file()
(output/'smoke_checks.json').write_text(json.dumps({'passed':True, 'updates':updates,
    'checkpoint_reload':True, 'finite_checkpoint':True, 'held_out_domain':0,
    'samples_per_domain':40}, indent=2))
print('SMOKE PASS:', args.dataset, flush=True)
