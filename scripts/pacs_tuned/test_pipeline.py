import json,tempfile,unittest
from pathlib import Path
from scripts.pacs_tuned.controller import Controller,atomic_json,sample
ROOT=Path(__file__).resolve().parents[2]
class TestCampaign(unittest.TestCase):
 def test_search_spaces(self):
  c=json.loads((ROOT/'experiments/pacs_tuned/config.json').read_text())
  with tempfile.TemporaryDirectory() as tmp:
   x=Controller(Path(tmp),c)
   for pair in c['backbones']:
    h=sample(x.get_study(pair,0).ask(),pair)
    self.assertEqual(h['vit'],pair.endswith('--dinov2'))
    if pair.startswith('DANN--'):self.assertEqual(h['lr_g'],h['lr'])
 def test_resume_claim_and_complete(self):
  c=json.loads((ROOT/'experiments/pacs_tuned/config.json').read_text());pair='CORAL--dinov2'
  c.update(backbones=[pair],search_trials_by_pair={pair:1},confirm_top_k=1,steps=2)
  with tempfile.TemporaryDirectory() as tmp:
   x=Controller(Path(tmp),c);x.fill(pair)
   item=next(iter(x.state['tasks'].values()));spec=item['spec'];src=x.run/'queue'/(spec['id']+'.json');claim=x.run/'running'/(spec['id']+'__123_0.json');src.rename(claim)
   x.process_results({'123':{}});self.assertEqual(item['status'],'RUNNING');self.assertEqual(spec['attempt'],1)
   for _ in range(30):
    if x.state['backbone_index']==1:break
    if x.state['stage']=='confirm':x.plan_confirmation(pair)
    if x.state['stage']=='final':x.plan_final(pair)
    x.fill(pair)
    for i in x.state['tasks'].values():
     if i['status'] not in ['QUEUED','RUNNING']:continue
     t=i['spec'];out=Path(t['output']);out.mkdir(exist_ok=True)
     atomic_json(out/'task.json',t);atomic_json(out/'result.json',{'state':'COMPLETE','step':2,'best_step':2,'best_source_val':.8,'target_acc':.7,'target_samples':100})
     (x.run/'queue'/(t['id']+'.json')).unlink(missing_ok=True)
    x.process_results({'123':{}});x=Controller(Path(tmp),c);x.transition(pair)
   self.assertEqual(x.state['backbone_index'],1)
   self.assertEqual(len([i for i in x.state['tasks'].values() if i['spec']['phase']=='final']),12)
if __name__=='__main__':unittest.main()
