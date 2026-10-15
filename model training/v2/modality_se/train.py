"""Isolated adapter around the unchanged epoch-resumable population trainer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from attention_cnn import train as original
from modality_se.model import make_model

def configure():
    original.VARIANTS=('se_ppg',)
    original.make_model=make_model
    original.PROTOCOL_COMMIT='modality-se-20261012'
    paths=[ROOT/'attention_cnn/train.py',ROOT/'shared/models.py',ROOT/'shared/data.py',ROOT/'shared/metrics.py']+[Path(__file__).parent/n for n in ('train.py','model.py','PROTOCOL.md')]
    original._source_hashes=lambda:{str(p.relative_to(ROOT)):original.sha256_file(p) for p in paths}

if __name__=='__main__':configure();original.main()
