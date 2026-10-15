# Rolling, endpoint-specific personal adaptation

Local experiment requested 2026-10-15. See `PROTOCOL.md` for the locked comparison; previous models are retained.

Code implements three within-person temporal validation folds, independent SBP/DBP selection, and matched all-pair head-only versus last-CNN-block/SE adaptation. It uses exactly the previous random 30-person cohort and 80/20 indices, with three pretrained seeds and both input modalities.

Run from the workspace with the existing environment:

```sh
output/bp_model/.venv/bin/python 'output/Bad-Blood/model training/v2/rolling_personal_20261015/test_rolling.py'
output/bp_model/.venv/bin/python 'output/Bad-Blood/model training/v2/rolling_personal_20261015/run.py' prepare
output/bp_model/.venv/bin/python 'output/Bad-Blood/model training/v2/rolling_personal_20261015/run.py' train
output/bp_model/.venv/bin/python 'output/Bad-Blood/model training/v2/rolling_personal_20261015/run.py' evaluate
output/bp_model/.venv/bin/python 'output/Bad-Blood/model training/v2/rolling_personal_20261015/verify.py' --output output/bad_blood_v2_runs/20261015_rolling_personal
```

For two workers, run `train --worker 0 --workers 2 --threads 2` and `train --worker 1 --workers 2 --threads 2`. They have disjoint units; results resume from hashed fold artifacts and completed units. All 180 units must finish before evaluation opens the final test outcomes. Do not edit the frozen training code after `prepare`.

`run_all.py --workers 4` runs one local job through training, evaluation and independent verification automatically, with resumable fold/unit artifacts and separate worker logs. Do not launch it while other workers for this output directory are still running. `job_status.json` records its phase, PID, progress or failure; it does not schedule future runs or send messages.

Output is `output/bad_blood_v2_runs/20261015_rolling_personal/`. `RESULTS.md`, `summary.json`, and `verification.json` will exist only after their respective phases finish. The old post-hoc diagnostic suggested this experiment, so final scores on the same cohort are exploratory, not new independent validation. No GitHub publication is performed by these scripts.
