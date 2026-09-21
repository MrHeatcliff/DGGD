"""Fast orchestration checks: no GPU jobs and no real training are launched."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.pacs.controller import Controller, atomic_json
from scripts.pacs.collect import collect

ROOT=Path(__file__).resolve().parents[2]

class PipelineTest(unittest.TestCase):
    def test_complete_and_resume(self):
        config=json.loads((ROOT/'experiments/pacs_erm/config.json').read_text())
        config.update(backbones=['dinov2'],search_trials_per_domain=3,confirm_top_k=2,steps=2,eval_every=1)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            controller=Controller(run,config)
            for iteration in range(50):
                if controller.state['backbone_index']==1:break
                if controller.state['stage']=='confirm':controller.plan_confirmation('dinov2')
                if controller.state['stage']=='final':controller.plan_final('dinov2')
                controller.fill('dinov2')
                active=[i for i in controller.state['tasks'].values() if i['status']=='QUEUED']
                self.assertLessEqual(len(active),8)
                for item in active:
                    task=item['spec'];out=Path(task['output']);out.mkdir(parents=True)
                    atomic_json(out/'task.json',task)
                    result={'state':'COMPLETE','best_source_val':.7+task['trial_number']*.01,
                        'step':2,'best_step':2,'target_acc':.1 if task['seed']==100 else .9,'target_samples':100}
                    if task['phase']!='final':result.pop('target_acc')
                    atomic_json(out/'result.json',result)
                    (run/'queue'/(task['id']+'.json')).unlink()
                controller.process_results({})
                # Serialization retains Optuna's sampler and all trial states.
                controller=Controller(run,config)
                controller.transition('dinov2')
            self.assertEqual(controller.state['backbone_index'],1)
            for env in range(4):
                self.assertEqual(controller.state['selections'][f'dinov2_env{env}']['trial_number'],2)
            summary=collect(run)
            self.assertAlmostEqual(summary[0]['overall']['mean'],(0.1+0.9+0.9)/3)
            # Incomplete final runs must fail collection instead of yielding a report.
            path=next((run/'runs').glob('*_final_*/result.json'));path.unlink()
            with self.assertRaises(AssertionError):collect(run)

    def test_retry_does_not_consume_another_optuna_trial(self):
        config=json.loads((ROOT/'experiments/pacs_erm/config.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            c=Controller(Path(temp),config);c.fill('dinov2')
            item=next(iter(c.state['tasks'].values()));task=item['spec']
            out=Path(task['output']);out.mkdir(parents=True)
            atomic_json(out/'failure.json',{'returncode':1})
            (c.run/'queue'/(task['id']+'.json')).unlink()
            c.process_results({})
            self.assertEqual(item['status'],'WAITING')
            self.assertEqual(item['spec']['attempt'],2)
            self.assertEqual(len(c.state['tasks']),8)

if __name__=='__main__':unittest.main()
