# VitalDB architecture and personalization research

This work extends V1 without modifying its release. The repository base is
`7553248168295bcad593739fdc733dbefff0c3b1`; the V1 network and training code
were introduced at `1f33cdcf090d51133c91dd50af2e575f5eaa0053`.

## Scope

- `attention_cnn/`: controlled attention ablations on the existing ECG/PPG to
  scalar SBP/DBP task, retaining the actual high-resolution V1 backbone.
- `personalized_forecasting/`: personal calibration with early ECG/PPG and BP,
  followed by contemporaneous SBP/DBP estimation from later observed ECG/PPG.
  The original directory name is retained for compatibility. Open-loop future
  waveform prediction is outside the revised assignment.
- `shared/`: reusable implementation after the literature gate is complete.
- `reports/`: combined, result-derived reporting after experiments.

Dataset expansion, CRC training on all of PulseDB, and MIMIC data are outside
this assignment. Existing VitalDB-derived data are the sole experimental source.
The complete updated user prompt is [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md);
the prespecified experiment matrix is [PROTOCOL.md](PROTOCOL.md).

## Ordered gates

1. Audit code, provenance, patients, timestamps, labels and resources.
2. Complete both literature reviews and frozen experimental protocols.
3. Compile and visually inspect both review PDFs.
4. Commit the reviews/protocols before implementing or training.
5. Implement, test, and run the full principal experiment matrix.
6. Generate quantitative reports from saved predictions and run manifests.
7. Push a dedicated branch and open a pull request without merging.

The dated status and experiment inventory distinguish completed work from
pending runs. No unrun results or clinical certification claims are implied.

## Document style

The unmodified official ACL `acl.sty` and `acl_natbib.bst` are vendored in each
document directory for portability. Source:
https://github.com/acl-org/acl-style-files,
commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`.
Reports use the nonanonymous, page-numbered `preprint` option. This is a
technical research report, not a claim of ACL submission or acceptance.

## Artifact storage

Raw recordings, caches, large model weights and detailed predictions stay out
of ordinary Git. Local artifact root: `output/bad_blood_v2_runs/` outside the
repository. Commit only code, configurations, split manifests, small summary
tables, documentation, and rendered research reports. Do not copy credentials
or identifiable ND participant data into this project.
