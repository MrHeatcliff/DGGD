import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.pacs_algorithms.controller import Campaign,write
from scripts.pacs_algorithms.collect import collect

class CampaignTest(unittest.TestCase):
    def test_resume_reconstructs_final_tasks(self):
        config={'max_gpus':2,'algorithms':['IRM','CORAL'],'seeds':[100,101,102],
                'steps':5,'eval_every':5,'domains':['art','cartoon','photo','sketch']}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);c=Campaign(run,config)
            for item in c.state['tasks'].values():item['status']='COMPLETE'
            c.state['phase']='final';c.save()
            c=Campaign(run,config)
            self.assertEqual(len(c.state['tasks']),2+2*4*3)
            for item in c.state['tasks'].values():
                task=item['spec']
                if task['phase']!='final':continue
                p=Path(task['output']);p.mkdir()
                write(p/'task.json',task)
                write(p/'result.json',{'state':'COMPLETE','step':5,'best_step':5,
                    'best_source_val':.8,'target_acc':.7+task['env']*.01,'target_samples':100})
            reports=collect(run)
            self.assertEqual(len(reports),2)
            self.assertAlmostEqual(reports[0]['overall']['mean'],.715)
            next((run/'runs').glob('final_*/result.json')).unlink()
            with self.assertRaises(AssertionError):collect(run)

if __name__=='__main__':unittest.main()
