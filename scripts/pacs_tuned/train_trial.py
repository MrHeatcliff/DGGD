"""Native FP32 updates, source-only tuning, no target evaluation during search."""
import json,os,socket,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts/pacs_algorithms'))
from scripts.pacs_algorithms.train_trial import atomic_json,run

def main():
    task=json.loads(Path(sys.argv[1]).read_text())
    algorithm,backbone=task['backbone'].split('--')
    task['algorithm']=algorithm;task['backbone']=backbone
    original=socket.socket.connect
    def offline(s,address):
        if s.family==socket.AF_UNIX:return original(s,address)
        raise RuntimeError('Network access forbidden: pre-cache weights before submitting')
    socket.socket.connect=offline
    try:run(task)
    except FloatingPointError:
        (Path(task['output'])/'NUMERICAL_FAILURE').touch()
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    except Exception:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)

if __name__=='__main__':main()
