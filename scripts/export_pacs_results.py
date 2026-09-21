"""Export a lightweight, auditable snapshot without datasets or model weights."""
import csv
import datetime
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/pacs'


def cleaned(value):
    if isinstance(value, dict):
        return {k: cleaned(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cleaned(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '')
    return value


def dump(path, value):
    path.write_text(json.dumps(cleaned(value), indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    seeds = [100, 101, 102]
    for campaign, dest, field in [('pacs_erm_optuna','erm','backbone'), ('pacs_algorithms','algorithms','algorithm')]:
        src = ROOT / 'outputs' / campaign
        dst = OUT / dest
        dst.mkdir(exist_ok=True)
        config = json.loads((src/'config.json').read_text())
        dump(dst/'config.json', config)
        names = config['backbones' if field == 'backbone' else 'algorithms']
        records = []
        for path in sorted((src/'runs').glob('*/result.json')):
            task = json.loads((path.parent/'task.json').read_text())
            result = json.loads(path.read_text())
            if task['phase'] != 'final' or result['state'] != 'COMPLETE': continue
            assert result['step'] == config['steps']
            assert 0 <= result['target_acc'] <= 1
            row = {'task': task, 'result': result}
            hp = path.parent/'hparams.json'
            if hp.exists(): row['effective_hparams'] = json.loads(hp.read_text())
            records.append(row)
        dump(dst/'completed_runs.json', records)
        with (dst/'completed_runs.csv').open('w') as f:
            cols=[field,'domain','seed','target_acc','source_val','best_step','task_id']
            w=csv.DictWriter(f, fieldnames=cols, lineterminator="\n");w.writeheader()
            for r in records:
                t,v=r['task'],r['result']
                w.writerow({field:t[field],'domain':domains[t['env']],'seed':t['seed'],
                    'target_acc':v['target_acc'],'source_val':v['best_source_val'],
                    'best_step':v['best_step'],'task_id':t['id']})
        lines=['# PACS '+dest, '', 'Target accuracy (%), mean ± sample SD over seeds 100, 101, 102. AVG uses equal domain weights per seed. Best checkpoint selected on source validation, not target oracle.', '',
            '| Model | Art Painting | Cartoon | Photo | Sketch | AVG |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        incomplete=[]
        for name in names:
            rr=[r for r in records if r['task'][field]==name]
            actual=[(r['task']['env'],r['task']['seed']) for r in rr]
            assert len(actual)==len(set(actual)), name
            if set(actual)!={(e,s) for e in range(4) for s in seeds}:
                incomplete.append({'name':name,'completed':len(rr),'expected':12});continue
            groups=[[r['result']['target_acc']*100 for r in rr if r['task']['env']==e] for e in range(4)]
            groups.append([statistics.mean(r['result']['target_acc']*100 for r in rr if r['task']['seed']==s) for s in seeds])
            lines.append('| '+name+' | '+' | '.join(f'{statistics.mean(g):.2f} ± {statistics.stdev(g):.2f}' for g in groups)+' |')
        lines += ['',f'Completed final runs: {len(records)}/{len(names)*12}.', '', 'Incomplete algorithms/backbones are not included in the final comparison table:']
        lines += [f"- {i['name']}: {i['completed']}/12" for i in incomplete] or ['- None.']
        (dst/'README.md').write_text('\n'.join(lines)+'\n')
        dump(dst/'completion.json', {'completed_final_runs':len(records),'expected_final_runs':len(names)*12,'incomplete':incomplete})
        if dest=='erm':
            selection=json.loads((src/'reports/selections.json').read_text())
            dump(dst/'selected_hparams.json',selection)
            shutil.copytree(src/'studies',dst/'optuna_trials',dirs_exist_ok=True)
            audit=[]
            for p in sorted((src/'runs').glob('*/result.json')):
                t=json.loads((p.parent/'task.json').read_text())
                if t['phase']!='final': audit.append({'task':t,'result':json.loads(p.read_text())})
            dump(dst/'search_and_confirmation.json',audit)
            hp_lines=['# Selected ERM hyperparameters','', 'One selected configuration per backbone and held-out domain. Selection uses only source validation; see selected_hparams.json for confirmation scores and alternative finalists.', '', '| Backbone | Target | Trial | LR | Weight decay | Batch/source | Dropout | Freeze BN |','| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
            for b in names:
                for e,d in enumerate(domains):
                    v=selection[f'{b}_env{e}'];h=v['hparams']
                    hp_lines.append(f"| {b} | {d} | {v['trial_number']} | {h['lr']:.8g} | {h['weight_decay']:.8g} | {h['batch_size']} | {h.get('vit_dropout',h.get('resnet_dropout'))} | {h['freeze_bn']} |")
            (dst/'hyperparameters.md').write_text('\n'.join(hp_lines)+'\n')
    dump(OUT/'snapshot.json', {'exported_at':datetime.datetime.now().astimezone().isoformat(), 'note':'Snapshot of completed results, not live scheduler status. No checkpoints, data, or secrets included.'})


if __name__ == '__main__':
    main()
