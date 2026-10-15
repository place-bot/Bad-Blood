"""Software-only synthetic fitting smoke test; never used for performance claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'implementation'))
from personalize import fit,infer,load
from rolling_core import configure


class PortableTest(unittest.TestCase):
    def test_both_input_modes(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(42)
        x=rng.uniform(size=(24,2,1250)).astype('float32')
        y=np.column_stack([110+np.arange(24)*.2,65+np.arange(24)*.1]).astype('float32')
        start=np.arange(24)*20.
        for mode in ['ppg','dual']:
            checkpoint=HERE/'02_personalization/source_weights'/f'{mode}_pair_population_seed42.pt'
            source,saved=load(checkpoint)
            configure(source,'tail')
            self.assertEqual(sum(p.numel() for p in source.parameters() if p.requires_grad),33336)
            with tempfile.TemporaryDirectory(prefix='badblood-portable-test-') as tmp:
                out=Path(tmp)/'fitted'
                fit(checkpoint,dict(x=x,y=y,start=start,end=start+10),out)
                pred=infer(out,x[-2:])
                self.assertEqual(pred.shape,(2,2));self.assertTrue(np.isfinite(pred).all())
                record=json.loads((out/'selection.json').read_text())
                self.assertFalse(record['final_test_used'])
                for d in [0,1]:
                    _,new=load(out/f'tail_endpoint{d}.pt')
                    self.assertEqual(new['endpoint'],d)
                    if new['config'] is not None:self.assertEqual(new['config']['scope'],'tail')
                    for name,weight in saved['state_dict'].items():
                        if not name.startswith(('change.','encoder.body.6.','encoder.gates.6.')) or 'running_' in name or 'num_batches_tracked' in name:
                            self.assertTrue(torch.equal(weight,new['state_dict'][name]),name)
                if mode=='ppg':
                    changed=x[-2:].copy();changed[:,0]=1-changed[:,0]
                    np.testing.assert_array_equal(pred,infer(out,changed))
                    np.testing.assert_array_equal(pred,infer(out,x[-2:,1:,:]))


if __name__=='__main__':unittest.main()
