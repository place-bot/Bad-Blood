# Final verification

Verified on October 9, 2026, America/Chicago. This record describes the completed
fixed study, not a new round of model selection.

## Execution and numeric integrity

- Required population fits: **20/20**. Six dual-channel variants each use seeds
  42, 43 and 44; the two modality controls each use seed 42.
- Personal development: **1,068/1,068** units. Final-target evaluation:
  **684/684** units, six patients and 4,284 common later query windows.
- Strict report generation completed with `generated_from_complete_inventory`.
  Source bytes, frozen manifests, configuration identities, checkpoints and
  predictions were verified before result generation.
- A separate numeric audit recomputed all population metric sets and all
  personal prediction units, 114 personal summaries, paired intervals, and
  all six individual K10 outcomes from saved runtime artifacts.
- Validation and test are explicitly separated. Single-seed modality results
  are distinguished from three-seed architecture results. Test outcomes did
  not alter optimization settings, cohort membership or checkpoint selection.
- Full automated test suite: **98 passed**. The real-artifact personal inference
  replay covers all eight strategies and reproduces 654 saved query predictions
  bitwise on CPU for the first frozen target at K10/seed42. It is a consistency
  check, not independent generalization evidence.

The [inventory](results/inventory.json), [metric overview](results/metric_overview.json),
[attention archive](results/attention.json.gz), and
[personal archive](results/personalization.json.gz) retain complete results and
provenance. Raw recordings, full predictions and weights remain outside Git;
[data access](../DATA_ACCESS.md) distinguishes published summaries from local
artifacts and documents fresh-machine reproduction.

## PDF checks

The official ACL literature reviews were compiled and visually checked before
implementation at the [review gate](../REVIEW_GATE.md). The final report was
generated only after all required experiments completed, compiled with
`pdflatex`, `bibtex`, and two further `pdflatex` passes, then rendered at 110 dpi.

Every final-report page was inspected: pages 1-14 by the main reviewer and
15-25 by a second reviewer, with the appendix opening also checked by the main
reviewer. After the final edits, unchanged pages 1-4 and 10-14 were verified
byte-identical to their previously inspected PNGs; changed pages were reinspected.
Tables and figure labels are legible, with no clipping or overlapping content.
Compilation reports no undefined references or overfull boxes. Normal whitespace
remains around some floats; no content is lost.

| Deliverable | Pages | SHA-256 |
|---|---:|---|
| Attention literature review | 7 | `c1e5c14167b5d0a4a24f677aa30dfb1b63b1161a53a9ec35400ba3f0c962c4dd` |
| Personal-calibration literature review | 6 | `3fb93240583f07e419174e691bacc52218c64d60b04fad960b4283626bf908e5` |
| Combined actual-results report | 25 | `88555a2caf8234c27d2e565a6811594ad4db7cc87672075b603ef4f5f206226a` |

## Scope and delivery checks

All changes against research base `7553248168295bcad593739fdc733dbefff0c3b1`
are confined to `model training/v2/`. V1 and unrelated files are unchanged.
Repository inspection found no raw waveform arrays, checkpoints or obvious
credential strings in the added files or decompressed result archives.
Both compressed metric archives decode successfully; all six documented CLI
entry points provide working help output. Reproduction commands are in the
[README](../README.md).

No required experiment remains unfinished. ECA is supplementary and was not
run. The study does not establish real cuff-calibration accuracy, cross-visit
durability, clinical certification or Raspberry Pi latency. It uses the
user-confirmed same-window BP estimation task, not future-waveform forecasting.
