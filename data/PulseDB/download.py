"""Download the official author-hosted archive (about 17.3 GiB)."""
from pathlib import Path
from urllib.request import urlopen
import shutil

url='https://www.kaggle.com/api/v1/datasets/download/weinanwangrutgers/pulsedb-balanced-training-and-testing'
target=Path(__file__).parent/'pulsedb_official_subsets.zip'
if target.exists():raise SystemExit('Archive already exists; no overwrite.')
temporary=target.with_suffix('.zip.part')
with urlopen(url) as response,temporary.open('wb') as output:
    shutil.copyfileobj(response,output,8*1024*1024)
temporary.replace(target)
print(target)
