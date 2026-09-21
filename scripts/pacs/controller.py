"""Single-writer Optuna controller; Slurm GPU workers exchange atomic JSON files."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

import optuna
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def base_hparams(backbone):
    return {'vit':backbone=='dinov2', 'dinov2':backbone=='dinov2',
        'vit_attn_tune':False, 'resnet18':backbone=='resnet18',
        'resnet50_augmix':backbone=='resnet50_augmix',
        'data_augmentation':True, 'class_balanced':False, 'nonlinear_classifier':False}


def sample(trial, backbone):
    h = base_hparams(backbone)
    lo, hi = (1e-6, 1e-4) if backbone=='dinov2' else (5e-6, 3e-4)
    h['lr'] = trial.suggest_float('lr',lo,hi,log=True)
    h['weight_decay'] = trial.suggest_float('weight_decay',1e-7,1e-2,log=True)
    h['batch_size'] = trial.suggest_categorical('batch_size',[16,32,64])
    dropout = trial.suggest_categorical('dropout',[0.0,0.1,0.2,0.3,0.5])
    h['vit_dropout' if backbone=='dinov2' else 'resnet_dropout'] = dropout
    h['freeze_bn'] = False if backbone=='dinov2' else trial.suggest_categorical('freeze_bn',[True,False])
    return h


class Controller:
    def __init__(self, run, config):
        self.run, self.config = run, config
        assert sum(p['gpus'] for p in config['pools']) == config['max_gpus'] == 8
        for name in ['queue','running','runs','slurm','studies','reports']:
            (run/name).mkdir(parents=True,exist_ok=True)
        if (run/'state.pkl').exists():
            self.state = pickle.loads((run/'state.pkl').read_bytes())
            assert self.state['config'] == config, 'Cannot silently change a running experiment'
        else:
            self.state = {'config':config,'backbone_index':0,'stage':'search','tasks':{},
                'studies':{},'pools':{},'job_history':[],'selections':{},'created':time.time()}
            self.save()
        atomic_json(run/'config.json',config)

    def save(self):
        tmp = self.run/'state.pkl.tmp'
        tmp.write_bytes(pickle.dumps(self.state))
        tmp.replace(self.run/'state.pkl')

    def get_study(self, backbone, env):
        key = f'{backbone}_env{env}'
        if key not in self.state['studies']:
            study = optuna.create_study(study_name=key,direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.config['sampler_seed']+env,
                    n_startup_trials=10,multivariate=True,constant_liar=True),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10,n_warmup_steps=1500,
                    interval_steps=self.config['eval_every'],n_min_trials=5))
            baseline = {'lr':1e-5 if backbone=='dinov2' else 5e-5,
                'weight_decay':1e-4,'batch_size':32,'dropout':0.0}
            if backbone!='dinov2': baseline['freeze_bn']=True
            study.enqueue_trial(baseline)
            self.state['studies'][key] = study
        return self.state['studies'][key]

    def add_task(self, phase, backbone, env, seed, hparams, trial_number=None):
        suffix = f't{trial_number:03d}' if phase=='search' else f'c{trial_number:03d}_s{seed}'
        task_id = f'{backbone}_e{env}_{phase}_{suffix}'
        if task_id in self.state['tasks']: return
        spec = {'id':task_id,'phase':phase,'backbone':backbone,'env':env,'seed':seed,
            'hparams':hparams,'steps':self.config['steps'],'eval_every':self.config['eval_every'],
            'workers':2,'output':str(self.run/'runs'/task_id),'trial_number':trial_number,'attempt':1}
        self.state['tasks'][task_id] = {'spec':spec,'status':'WAITING','last_report_step':0}
        self.save()

    def slurm(self):
        result = subprocess.run(['squeue','--json','-u',os.environ['USER']],capture_output=True,text=True,check=True)
        jobs = json.loads(result.stdout)['jobs']
        return {str(j['job_id']):j for j in jobs}

    def ensure_pools(self, jobs):
        for index, pool in enumerate(self.config['pools']):
            old = self.state['pools'].get(index)
            if old and old in jobs: continue
            command = ['sbatch','--parsable',f"--partition={pool['partition']}",
                '--job-name=pacs-erm-optuna',f"--gres=gpu:{pool['gpus']}",
                f"--cpus-per-task={8*pool['gpus']}",f"--mem={48*pool['gpus']}G",
                f"--time={pool['time']}",f'--chdir={ROOT}',
                f'--output={self.run}/slurm/pool-%j.log',str(ROOT/'scripts/pacs/pool.sbatch'),
                str(self.run),str(pool['gpus'])]
            reply = subprocess.run(command,capture_output=True,text=True)
            if reply.returncode:
                print('SUBMIT RETRY',reply.stderr,flush=True)
                continue
            job_id = reply.stdout.strip().split(';')[0]
            self.state['pools'][index] = job_id
            self.state['job_history'].append({'job_id':job_id,'pool':index,'submitted':time.time()})
            self.save()
            print('SUBMITTED',job_id,pool,flush=True)

    def retry(self, item, reason):
        spec = item['spec']
        out = Path(spec['output'])
        if spec['attempt'] >= self.config['max_task_attempts']:
            raise RuntimeError(f"Task {spec['id']} failed {spec['attempt']} times: {reason}; inspect {out}")
        archive = out/f"attempt_{spec['attempt']}_incomplete"
        archive.mkdir(parents=True,exist_ok=True)
        for name in ['failure.json','progress.json','metrics.jsonl','PRUNE','owner.json','heartbeat.json','best.pt']:
            p=out/name
            if p.exists(): p.replace(archive/name)
        for p in (self.run/'running').glob(spec['id']+'__*.json'):p.unlink()
        spec['attempt'] += 1
        item['status'],item['last_report_step'] = 'WAITING',0
        self.save()
        print('RETRY',spec['id'],reason,flush=True)

    def process_results(self, jobs):
        for item in list(self.state['tasks'].values()):
            if item['status'] not in ['QUEUED','RUNNING']: continue
            spec=item['spec'];out=Path(spec['output'])
            if (out/'result.json').exists():
                result=json.loads((out/'result.json').read_text())
                if spec['phase']=='search':
                    study=self.get_study(spec['backbone'],spec['env'])
                    if result['state']=='COMPLETE':
                        study.tell(spec['trial_number'],result['best_source_val'])
                    else:
                        study.tell(spec['trial_number'],state=TrialState.PRUNED)
                item['status']=result['state'];item['result']=result
                self.save()
                print('RESULT',spec['id'],result['state'],result['best_source_val'],flush=True)
                continue
            if (out/'failure.json').exists():
                self.retry(item,'worker reported a nonzero exit');continue
            claims=list((self.run/'running').glob(spec['id']+'__*.json'))
            if claims:
                item['status']='RUNNING'
                job_id=claims[0].stem.split('__')[1].split('_')[0]
                if job_id not in jobs:
                    self.retry(item,'allocation ended without a result');continue
            elif not (self.run/'queue'/(spec['id']+'.json')).exists():
                # Recover a controller interruption between state persistence and publication.
                item['status']='WAITING'
            if spec['phase']=='search' and (out/'progress.json').exists():
                p=json.loads((out/'progress.json').read_text())
                if p['step'] > item['last_report_step']:
                    study=self.get_study(spec['backbone'],spec['env'])
                    trial=optuna.trial.Trial(study,study.trials[spec['trial_number']]._trial_id)
                    trial.report(p['best_source_val'],p['step'])
                    item['last_report_step']=p['step']
                    if trial.should_prune() and p['step'] < self.config['steps']:
                        (out/'PRUNE').touch()
                    self.save()

    def fill(self, backbone):
        active=sum(i['status'] in ['QUEUED','RUNNING'] for i in self.state['tasks'].values())
        while active < self.config['max_gpus']:
            waiting=next((i for i in self.state['tasks'].values() if i['status']=='WAITING'),None)
            if waiting is None and self.state['stage']=='search':
                counts=[sum(i['spec']['backbone']==backbone and i['spec']['phase']=='search'
                    and i['spec']['env']==env for i in self.state['tasks'].values()) for env in range(4)]
                env=min(range(4),key=lambda e:counts[e])
                if counts[env]>=self.config['search_trials_per_domain']: break
                study=self.get_study(backbone,env)
                trial=study.ask()
                self.add_task('search',backbone,env,self.config['search_seed'],sample(trial,backbone),trial.number)
                waiting=next(i for i in self.state['tasks'].values() if i['status']=='WAITING')
            if waiting is None:break
            waiting['status']='QUEUED';self.save()
            atomic_json(self.run/'queue'/(waiting['spec']['id']+'.json'),waiting['spec'])
            active+=1

    def transition(self, backbone):
        tasks=[i for i in self.state['tasks'].values() if i['spec']['backbone']==backbone]
        if any(i['status'] in ['WAITING','QUEUED','RUNNING'] for i in tasks):return
        stage=self.state['stage']
        if stage=='search':
            if len(tasks) < 4*self.config['search_trials_per_domain']:return
            # Persist stage before publishing its tasks; reconstruct missing tasks on restart.
            self.state['stage']='confirm';self.save()
            self.plan_confirmation(backbone)
        elif stage=='confirm':
            self.plan_confirmation(backbone)
            if any(i['status']=='WAITING' for i in self.state['tasks'].values()):return
            self.select(backbone)
            self.state['stage']='final';self.save()
            self.plan_final(backbone)
        elif stage=='final':
            self.plan_final(backbone)
            if any(i['status']=='WAITING' for i in self.state['tasks'].values()):return
            self.collect(backbone)
            self.state['backbone_index']+=1
            self.state['stage']='search'
            self.save()

    def top(self, backbone, env):
        study=self.get_study(backbone,env)
        trials=sorted([t for t in study.trials if t.state==TrialState.COMPLETE],key=lambda t:t.value,reverse=True)
        assert len(trials)>=self.config['confirm_top_k']
        return trials[:self.config['confirm_top_k']]

    def search_item(self, backbone, env, number):
        return next(i for i in self.state['tasks'].values() if i['spec']['backbone']==backbone
            and i['spec']['env']==env and i['spec']['phase']=='search' and i['spec']['trial_number']==number)

    def plan_confirmation(self, backbone):
        for env in range(4):
            for trial in self.top(backbone,env):
                hp=self.search_item(backbone,env,trial.number)['spec']['hparams']
                for seed in self.config['confirm_seeds']:
                    self.add_task('confirm',backbone,env,seed,hp,trial.number)

    def select(self, backbone):
        for env in range(4):
            candidates=[]
            for trial in self.top(backbone,env):
                values=[trial.value]+[i['result']['best_source_val'] for i in self.state['tasks'].values()
                    if i['spec']['backbone']==backbone and i['spec']['env']==env
                    and i['spec']['phase']=='confirm' and i['spec']['trial_number']==trial.number]
                assert len(values)==1+len(self.config['confirm_seeds'])
                candidates.append({'trial_number':trial.number,'source_val_seeds':values,'mean':sum(values)/len(values)})
            winner=dict(max(candidates,key=lambda c:c['mean']))
            winner['hparams']=self.search_item(backbone,env,winner['trial_number'])['spec']['hparams']
            winner['candidates']=candidates
            self.state['selections'][f'{backbone}_env{env}']=winner
        self.save()
        atomic_json(self.run/'reports'/'selections.json',self.state['selections'])

    def plan_final(self, backbone):
        for env in range(4):
            selection=self.state['selections'][f'{backbone}_env{env}']
            for seed in self.config['final_seeds']:
                self.add_task('final',backbone,env,seed,selection['hparams'],selection['trial_number'])

    def collect(self, backbone):
        # Collection validates completeness; it never ranks candidates by target accuracy.
        subprocess.run([sys.executable,str(ROOT/'scripts/pacs/collect.py'),str(self.run),
            '--backbone',backbone],check=True)
        for env in range(4):
            self.get_study(backbone,env).trials_dataframe().to_csv(
                self.run/'studies'/f'{backbone}_env{env}.csv',index=False)

    def status(self, jobs):
        index=self.state['backbone_index']
        atomic_json(self.run/'status.json',{'updated':time.time(),
            'backbone':self.config['backbones'][index] if index<len(self.config['backbones']) else None,
            'stage':self.state['stage'],'max_gpus':self.config['max_gpus'],
            'task_counts':dict(Counter(i['status'] for i in self.state['tasks'].values())),
            'pool_jobs':[{**entry,'current_state':jobs.get(entry['job_id'],{}).get('job_state','FINISHED')}
                for entry in self.state['job_history']],
            'progress':[{'id':i['spec']['id'], 'status':i['status'],
                'step':i.get('last_report_step',0)} for i in self.state['tasks'].values()
                if i['status'] in ['QUEUED','RUNNING']]})

    def loop(self):
        while not (self.run/'STOP').exists():
            try: jobs=self.slurm()
            except (subprocess.CalledProcessError,json.JSONDecodeError) as error:
                print('Scheduler query failed; retrying:',error,flush=True);time.sleep(20);continue
            self.process_results(jobs)
            if self.state['backbone_index']>=len(self.config['backbones']):
                subprocess.run([sys.executable,str(ROOT/'scripts/pacs/collect.py'),str(self.run)],check=True)
                (self.run/'COMPLETE').write_text(time.ctime())
                self.status(jobs)
                return
            backbone=self.config['backbones'][self.state['backbone_index']]
            if self.state['stage']=='confirm':self.plan_confirmation(backbone)
            if self.state['stage']=='final':self.plan_final(backbone)
            self.transition(backbone)
            if self.state['backbone_index']>=len(self.config['backbones']):continue
            backbone=self.config['backbones'][self.state['backbone_index']]
            self.fill(backbone)
            self.ensure_pools(jobs)
            self.status(jobs)
            time.sleep(10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/pacs_erm_optuna')
    parser.add_argument('--config',default='experiments/pacs_erm/config.json')
    args=parser.parse_args()
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(Path(args.config).read_text())
    controller=Controller(run,config)
    try:controller.loop()
    except Exception:
        (run/'FAILED.txt').write_text(traceback.format_exc())
        (run/'STOP').touch()
        raise
    finally:
        # Release only this experiment's allocations, never other user jobs.
        ids=list(controller.state['pools'].values())
        if ids:subprocess.run(['scancel',*ids],check=False)


if __name__=='__main__':main()
