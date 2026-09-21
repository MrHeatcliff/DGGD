import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.pacs_algorithms.resources import can_admit,reservation_gb

class ResourceTest(unittest.TestCase):
    def test_memory_and_count_limits(self):
        c={'jobs_per_gpu':8,'gpu_memory_budget_gb':118,'jobs_per_gpu_by_algorithm':{'IGA':2}}
        peaks={'ERMPlusPlus':8.3,'RDM':21.8,'IGA':17.1}
        self.assertTrue(can_admit('ERMPlusPlus',['ERMPlusPlus']*7,c,peaks))
        self.assertFalse(can_admit('ERMPlusPlus',['ERMPlusPlus']*8,c,peaks))
        self.assertTrue(can_admit('RDM',['RDM']*3,c,peaks))
        self.assertFalse(can_admit('RDM',['RDM']*4,c,peaks))
        self.assertFalse(can_admit('ERMPlusPlus',['IGA','ERMPlusPlus'],c,peaks))
        self.assertFalse(can_admit('IGA',['ERMPlusPlus']*2,c,peaks))
        self.assertTrue(can_admit('IGA',['ERMPlusPlus'],c,peaks))

if __name__=='__main__':unittest.main()
