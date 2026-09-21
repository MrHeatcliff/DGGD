"""Offline GPU smoke tests using the production PACS ERM training path."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def case(backbone, env, out):
    os.environ['HF_HUB_OFFLINE'] = '1'
    def deny_network(*args, **kwargs):
        raise RuntimeError('Smoke test forbids network access; pre-cache weights first')
    original_connect = socket.socket.connect
    def offline_connect(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        return deny_network()
    socket.socket.connect = offline_connect
    socket.create_connection = deny_network
    from controller import base_hparams
    from train_trial import run
    h = base_hparams(backbone)
    h.update(lr=5e-5, weight_decay=1e-4, batch_size=64,
             resnet_dropout=0.5, freeze_bn=bool(env % 2))
    task = dict(id=f'smoke_{backbone}_e{env}', backbone=backbone, env=env,
                seed=20260920, phase='final', steps=5, eval_every=5,
                workers=2, hparams=h, output=str(out))
    run(task)
    r = json.loads((out / 'result.json').read_text())
    assert r['state'] == 'COMPLETE' and r['step'] == 5
    assert 0 <= r['target_acc'] <= 1 and r['target_samples'] > 0
    assert (out / 'best.pt').is_file()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--backbone', choices=['resnet18', 'resnet50', 'resnet50_augmix'])
    p.add_argument('--env', type=int, choices=range(4))
    a = p.parse_args()
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.env is not None:
        assert a.backbone
        case(a.backbone, a.env, out)
        return
    results = []
    for b in ([a.backbone] if a.backbone else ['resnet18','resnet50','resnet50_augmix']):
        for env in range(4):
            dest = out / f'{b}_e{env}'
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / 'smoke.log').open('w') as log:
                subprocess.run([sys.executable, __file__, '--output', str(dest),
                                '--backbone', b, '--env', str(env)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            results.append(json.loads((dest / 'result.json').read_text()))
            print('PASS', b, env, flush=True)
    (out / 'summary.json').write_text(json.dumps(results, indent=2))
    (out / 'PASSED').write_text('All offline GPU cases passed. Smoke results are excluded from tuning.\n')


if __name__ == '__main__':
    main()
