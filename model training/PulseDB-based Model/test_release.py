import csv,json,tempfile,subprocess,sys,unittest
from pathlib import Path
import numpy as np
from inference import Predictor
ROOT=Path(__file__).resolve().parent
EXAMPLE=ROOT.parents[1]/'data/PulseDB/examples/real_20_segments.csv'

class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with EXAMPLE.open() as f:cls.rows=list(csv.DictReader(f))
        cls.x=np.array([[[float(r[f'{c}_{j:04d}']) for j in range(1250)] for c in ['ecg','ppg']] for r in cls.rows],dtype='float32')
        cls.model=Predictor(ROOT/'models')
    def test_reload_and_batch_parity(self):
        a=self.model.predict(self.x)
        b=Predictor(ROOT/'models').predict(self.x)
        c=np.concatenate([self.model.predict(row[None]) for row in self.x])
        np.testing.assert_allclose(a,b,atol=1e-6)
        np.testing.assert_allclose(a,c,atol=1e-4)
        self.assertTrue(np.isfinite(a).all())
    def test_cli_ignores_labels_and_rejects_flat(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'input.csv';dst=Path(td)/'output.csv'
            rows=[dict(self.rows[0]),dict(self.rows[0]),dict(self.rows[0])]
            rows[0]['SBP']='';rows[0]['DBP']=''
            rows[1]['SBP']='99999';rows[1]['DBP']='-99999'
            for j in range(1250):rows[2][f'ecg_{j:04d}']='0'
            with src.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            subprocess.run([sys.executable,str(ROOT/'predict.py'),str(src),'--format','filtered','--output',str(dst)],check=True,capture_output=True)
            with dst.open() as f:result=list(csv.DictReader(f))
            self.assertEqual(result[0]['pred_SBP_mmHg'],result[1]['pred_SBP_mmHg'])
            self.assertEqual(result[0]['pred_DBP_mmHg'],result[1]['pred_DBP_mmHg'])
            self.assertEqual(result[2]['status'],'invalid_input')
    def test_selection(self):
        s=json.loads((ROOT/'models/selection.json').read_text())
        self.assertEqual(s['cnn_name'],'cnn_full')
        self.assertEqual(s['tree_weight'],0)

if __name__=='__main__':unittest.main()
