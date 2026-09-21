"""Two-GPU queue for 33 native algorithms, with a smoke gate and final collection."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]

def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

class Campaign:
    def __init__(self,run,config):
        self.run,self.config=run,config
        assert config['max_gpus']==2
        for name in ['queue','running','runs','slurm','reports']:(run/name).mkdir(parents=True,exist_ok=True)
        self.state=json.loads((run/'state.json').read_text()) if (run/'state.json').exists() else {'tasks':{},'job_id':None,'job_history':[],'phase':'smoke'}
        write(run/'config.json',config)
        for algorithm in config['algorithms']:self.add(algorithm,0,100,'smoke')
        if self.state['phase']=='final':
            for algorithm in config['algorithms']:
                for env in range(4):
                    for seed in config['seeds']:self.add(algorithm,env,seed,'final')

    def save(self):write(self.run/'state.json',self.state)

    def add(self,algorithm,env,seed,phase):
        identifier=f'{phase}_{algorithm}_e{env}_s{seed}'
        if identifier in self.state['tasks']:return
        spec={'id':identifier,'algorithm':algorithm,'backbone':'resnet50_augmix','env':env,'seed':seed,
            'phase':phase,'steps':5 if phase=='smoke' else self.config['steps'],
            'eval_every':5 if phase=='smoke' else self.config['eval_every'],'workers':2,
            'hparams':{'vit':False,'dinov2':False,'resnet18':False,'resnet50_augmix':True},
            'output':str(self.run/'runs'/identifier),'attempt':1}
        self.state['tasks'][identifier]={'spec':spec,'status':'WAITING'};self.save()

    def reset(self,item):
        s=item['spec'];out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
        archive=out/f"attempt_{s['attempt']}_{int(time.time())}"
        archive.mkdir()
        for name in ['failure.json','progress.json','metrics.jsonl','owner.json','heartbeat.json','best.pt']:
            p=out/name
            if p.exists():p.replace(archive/name)
        for p in (self.run/'running').glob(s['id']+'__*.json'):p.unlink()
        (self.run/'queue'/(s['id']+'.json')).unlink(missing_ok=True)
        s['attempt']+=1;item['status']='WAITING';self.save()

    def failed(self,item,reason):
        print('FAILURE',item['spec']['id'],reason,flush=True)
        if item['spec']['phase']=='final' and item['spec']['attempt']<3:self.reset(item)
        else:item.update(status='FAILED',reason=reason);self.save()

    def results(self,jobs):
        for item in self.state['tasks'].values():
            if item['status'] not in ['QUEUED','RUNNING']:continue
            s=item['spec'];out=Path(s['output'])
            if (out/'result.json').exists():
                item['result']=json.loads((out/'result.json').read_text());item['status']='COMPLETE'
                print('COMPLETE',s['id'],flush=True);self.save();continue
            if (out/'failure.json').exists():self.failed(item,'nonzero process exit');continue
            claims=list((self.run/'running').glob(s['id']+'__*.json'))
            if claims:
                item['status']='RUNNING'
                job=claims[0].stem.split('__')[1].split('_')[0]
                if job not in jobs:self.reset(item)
            elif not (self.run/'queue'/(s['id']+'.json')).exists():item['status']='WAITING'
        self.save()

    def pool(self,jobs):
        if self.state['job_id'] in jobs:return
        command=['sbatch','--parsable',f"--partition={self.config.get('partition','gpu_collaborative')}",'--job-name=pacs-33-algorithms',
            '--gres=gpu:2',f"--cpus-per-task={min(64,max(16,8*self.config.get('jobs_per_gpu',1)))}",'--mem=384G',f"--time={self.config.get('pool_time','2-00:00:00')}",f'--chdir={ROOT}',
            f'--output={self.run}/slurm/pool-%j.log',str(ROOT/'scripts/pacs_algorithms/pool.sbatch'),str(self.run),'2']
        p=subprocess.run(command,text=True,capture_output=True)
        if p.returncode:print('SUBMIT RETRY',p.stderr,flush=True);return
        self.state['job_id']=p.stdout.strip().split(';')[0]
        self.state['job_history'].append(self.state['job_id']);self.save()
        print('SUBMITTED',self.state['job_id'],flush=True)

    def status(self,jobs):
        write(self.run/'status.json',{'updated':time.time(),'phase':self.state['phase'],'max_gpus':2,
            'jobs_per_gpu':self.config.get('jobs_per_gpu',1),'mps':self.config.get('mps',False),
            'job_id':self.state['job_id'],'job_state':jobs.get(self.state['job_id'],{}).get('job_state'),
            'counts':dict(Counter(i['status'] for i in self.state['tasks'].values())),
            'failed':[i['spec']['id'] for i in self.state['tasks'].values() if i['status']=='FAILED'],
            'active':[i['spec']['id'] for i in self.state['tasks'].values() if i['status'] in ['QUEUED','RUNNING']]})

    def loop(self):
        while not (self.run/'STOP').exists():
            try:
                jobs={str(j['job_id']):j for j in json.loads(subprocess.check_output(['squeue','--json','-u',os.environ['USER']],text=True))['jobs']}
            except (subprocess.CalledProcessError,json.JSONDecodeError):time.sleep(15);continue
            if (self.run/'RETRY_FAILED').exists():
                for item in self.state['tasks'].values():
                    if item['status']=='FAILED':self.reset(item)
                (self.run/'RETRY_FAILED').unlink()
            self.results(jobs)
            pending=[i for i in self.state['tasks'].values() if i['status'] in ['WAITING','QUEUED','RUNNING']]
            failed=[i for i in self.state['tasks'].values() if i['status']=='FAILED']
            if not pending:
                if failed:
                    write(self.run/'NEEDS_ATTENTION.json',{'failed':[i['spec']['id'] for i in failed]})
                    self.status(jobs);time.sleep(10);continue
                if self.state['phase']=='smoke':
                    self.state['phase']='final';self.save()
                else:
                    subprocess.run([sys.executable,str(ROOT/'scripts/pacs_algorithms/collect.py'),str(self.run)],check=True)
                    (self.run/'COMPLETE').write_text(time.ctime());self.status(jobs);return
            if self.state['phase']=='final':
                # Idempotent reconstruction after coordinator restart during task creation.
                for algorithm in self.config['algorithms']:
                    for env in range(4):
                        for seed in self.config['seeds']:self.add(algorithm,env,seed,'final')
            active=sum(i['status'] in ['QUEUED','RUNNING'] for i in self.state['tasks'].values())
            for item in self.state['tasks'].values():
                if active>=2*self.config.get('jobs_per_gpu',1):break
                if item['status']!='WAITING':continue
                item['spec']['workers']=self.config.get('workers_per_job',2)
                item['spec']['torch_threads']=self.config.get('torch_threads_per_job',2)
                item['status']='QUEUED';self.save()
                write(self.run/'queue'/(item['spec']['id']+'.json'),item['spec']);active+=1
            if active:self.pool(jobs)
            self.status(jobs);time.sleep(5)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default='outputs/pacs_algorithms');args=parser.parse_args()
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((ROOT/'experiments/pacs_algorithms/config.json').read_text())
    campaign=Campaign(run,config)
    try:campaign.loop()
    except Exception:
        (run/'FAILED.txt').write_text(traceback.format_exc());(run/'STOP').touch();raise
    finally:
        if campaign.state['job_id']:subprocess.run(['scancel',campaign.state['job_id']],check=False)
