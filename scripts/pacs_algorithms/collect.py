"""Collect only complete final runs; never publish partial numbers as final results."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def collect(run, backbone=None):
    config=json.loads((run/'config.json').read_text())
    backbones=[backbone] if backbone else config['algorithms']
    raw=[];summaries=[]
    for name in backbones:
        rows=[]
        for task_file in (run/'runs').glob(f'final_{name}_e*_s*/task.json'):
            task=json.loads(task_file.read_text())
            result_file=task_file.parent/'result.json'
            if not result_file.exists():continue
            result=json.loads(result_file.read_text())
            if result['state']!='COMPLETE':continue
            assert result['step']==config['steps']
            rows.append({'algorithm':name,'domain':config['domains'][task['env']],
                'env':task['env'],'seed':task['seed'],'target_acc':result['target_acc'],
                'source_val':result['best_source_val'],'best_step':result['best_step'],
                'target_samples':result['target_samples'],'artifact_dir':str(task_file.parent)})
        expected={(env,seed) for env in range(4) for seed in config['seeds']}
        actual=[(r['env'],r['seed']) for r in rows]
        assert len(actual)==len(expected) and set(actual)==expected, f'{name}: missing or duplicate final runs'
        raw.extend(rows)
        summary={'algorithm':name,'domains':{}}
        for env,domain in enumerate(config['domains']):
            values=[r['target_acc'] for r in rows if r['env']==env]
            summary['domains'][domain]={'mean':statistics.mean(values),'std':statistics.stdev(values),'n':len(values)}
        seed_means=[statistics.mean(r['target_acc'] for r in rows if r['seed']==seed) for seed in config['seeds']]
        summary['overall']={'mean':statistics.mean(seed_means),'std':statistics.stdev(seed_means),'n':len(seed_means)}
        summaries.append(summary)
    prefix=backbone if backbone else 'all_algorithms'
    report=run/'reports';report.mkdir(exist_ok=True)
    (report/f'{prefix}.json').write_text(json.dumps(summaries,indent=2))
    with (report/f'{prefix}_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    lines=['# PACS — 33 algorithms, ResNet50 AugMix', '', 'Target accuracy (%), mean ± sample standard deviation over three independent training seeds.',
        'Overall: equally weighted mean of the four held-out domains for each seed.',
        'Default hyperparameters; checkpoints selected exclusively using source-domain validation.', '',
        '| Algorithm | Art painting | Cartoon | Photo | Sketch | Overall |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for summary in summaries:
        cells=[summary['domains'][d] for d in config['domains']]+[summary['overall']]
        lines.append('| '+summary['algorithm']+' | '+' | '.join(f"{100*c['mean']:.2f} ± {100*c['std']:.2f}" for c in cells)+' |')
    (report/f'{prefix}.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--backbone')
    args=parser.parse_args()
    collect(Path(args.run_dir),args.backbone)
