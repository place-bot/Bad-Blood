"""Write a checksum manifest for the final, verified release artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verified = json.loads((ROOT / 'release_verification.json').read_text())
    assert verified['status'] == 'passed'
    completed = json.loads((ROOT / 'metrics/complete.json').read_text())
    for key, name in [('summary_sha256', 'summary.json'),
                      ('verification_sha256', 'verification.json')]:
        assert completed[key] == digest(ROOT / 'metrics' / name)
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if (not path.is_file() or path.name == 'manifest.json'
                or '__pycache__' in path.parts or path.suffix == '.pyc'
                or path.name == '.DS_Store'):
            continue
        files[str(path.relative_to(ROOT))] = {
            'bytes': path.stat().st_size, 'sha256': digest(path)}
    assert sum(name.endswith('.pt') for name in files) == 6
    assert sum(name.startswith('02_personalization/personal_weights/')
               and name.endswith('.tar.gz') for name in files) == 6
    assert all(item['bytes'] < 100_000_000 for item in files.values())
    result = {'release': '20261015', 'adopted_method': 'rolling_tail',
              'input_modes': ['ppg', 'dual'], 'files': files}
    (ROOT / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Checksummed {len(files)} files; "
          f"{sum(item['bytes'] for item in files.values()) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
