# PPG-only CNN+SE versus ECG+PPG CNN+SE — 20261012

This is the matched, three-seed modality extension. Existing releases and
models are preserved. Do not infer completion from the existence of this folder:
`results/verification.json` and the runtime `study_complete.json` are written
only after the full inventory and saved predictions have passed verification.

## Completed result

All 816 units are verified. Both input modalities and their three-seed weights
are retained. See [stability review](results/STABILITY.md) for every seed, both
macro and pooled metrics, and participant-bootstrap intervals.

On the 16-person replication cohort, PPG personalization beats personal mean
for both pressures in all three seeds, but loses to PPG bias correction on DBP
in all three seeds. Neither modality establishes a stable joint advantage over
both controls. Lower aggregate Bias alone does not imply more accurate
predictions. Existing published models are unchanged.

## Design

- Same SE-last architecture and 38,296 parameters; seeds42/43/44.
- PPG-only zeroes ECG internally, including the reference encoder. It is trained
  from scratch, not formed by deleting ECG at inference from a dual-trained model.
- Same 1,085 source-training participants /390,600 windows and 192 validation
  participants /69,120 windows. Verified existing dual-channel fits are reused.
- Three new PPG+SE waveform fits, then three PPG reference-conditioned source
  heads. Same PairSE architecture and training procedure as dual-channel heads.
- Same complete early20%-of-time personal support bank, later queries and guard.
- Same four development participants and same nine settings including no fitting:
  head/full × lr0.0003/0.001 ×25/100 steps. Each modality gets a development-only
  selection averaged over three seeds.
- Locked evaluation on six historical and sixteen previously examined target
  people, reported separately. These are NOT new independent test cohorts.
- Controls: personal mean, own-modality bias correction, other-modality bias
  correction, and unadapted population model. Query BP never enters inference.

Report participant-macro and pooled MAE, RMSE, signed Bias, sample error SD,
within5/10/15, tails, all per-participant/seed metrics, seed variability and
paired person-bootstrap intervals. Lower MAE in one pressure alone is not a
joint success. These statistics are not clinical device certification.

See [PROTOCOL.md](PROTOCOL.md) for the predeclared rules. The implementation
reuses frozen modules through isolated adapters rather than changing old source
files or model definitions. Tests verify identical initial weights, dual-path
equivalence, ECG-invariance in both PPG prediction paths, resume equivalence,
test-label isolation and modality-aware saved-checkpoint inference.

## Run

From the V2 root, with the existing Python environment:

```sh
python modality_se/workflow.py \
  --data /path/to/prepared_population \
  --output /path/to/modality_experiment \
  --prior /path/to/completed_dual_attention_runs \
  --personal /path/to/original_personal_mat \
  --fresh /path/to/previous_sixteen_mat
```

The workflow resumes completed waveform runs, trains missing pair heads,
persists personal units, locks selections, evaluates, then invokes `report.py`.
Check progress using `python modality_se/status.py --run /path/to/modality_experiment`.
No automatic Git push, model deployment, or email sending is performed.

After verification, generated outputs are:

- `results/RESULTS.md`: human-readable population and personalization tables.
- `results/comparison.json`: full metrics, seed summaries and paired intervals.
- `results/per_person_seed_metrics.json`: every personal method/person/seed.
- `results/verification.json`: inventory, budget identity and replay checks.
- `results/STABILITY.md` and `results/stability.json`: per-seed stability versus
  each control, with separate macro and pooled interpretation.
- `models/`: compact population/reference models and selected personal weights.
- `models/manifest.json`: checkpoint SHA256 identities.

Use `modality_se.predict.load_predict`, not the old dual-only loader. Compact
population checkpoints accept min-max-normalized N×2×1250 ECG/PPG arrays; the
PPG model ignores the ECG slot. Pair checkpoints additionally require normalized
reference waveforms and their SBP/DBP in mmHg. Output is N×2, SBP/DBP in mmHg.
The encoder and input mask are reconstructed from the checkpoint's `modality`.
Raw waveform data are not added to Git.

Run `python modality_se/stability.py` after the verified comparison to regenerate
the stability review. It does not train, select, or change any model.
