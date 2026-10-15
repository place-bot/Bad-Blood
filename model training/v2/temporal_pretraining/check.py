"""Pinned development predicate; does not assert independent-test success."""
import json
from pathlib import Path
p=Path(__file__).parent/'results/summary.json'
if not p.exists():
    print('NOT_MET')
else:
    d=json.loads(p.read_text())
    print('PASS' if d['source_selection']['both_validation_endpoints_improved'] and d['development_pass'] else 'NOT_MET')
