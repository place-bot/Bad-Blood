# Personal calibration runtime integrity check

On 2026-10-09, eight completed development units for `vitaldb:p005548`, case `c4678`, seed 42, K=10 passed read-only artifact, source-alignment, parameter-update, and prediction-replay checks. The sample was fixed by patient, seed, budget, and settings before inspecting scores. This is a runtime-integrity check, not a comparison of prediction accuracy or a claim that the experiment inventory is complete.

The artifact root was `/Users/jguo9/Desktop/26fall/LLM/output/bad_blood_v2_runs/20261009`. All paths below are relative to that root. Unit records are under `personalization/development/vitaldb:p005548/seed42/K10/`.

| Method | Setting directory | Trainable scalar parameters | Changed parameter tensors |
|---|---|---:|---:|
| Head | `head/lr0.0001_steps20` | 1,634 | 4 |
| Last block and head | `last_block/lr0.0001_steps20` | 25,730 | 13 |
| Full model | `full/lr0.0001_steps20` | 37,666 | 34 |
| Population | `population/fixed` | 0 | 0 |
| Additive bias | `bias/fixed` | 0 | 0 |
| Last reference | `last_reference/fixed` | 0 | 0 |
| Calibration mean | `calibration_mean/fixed` | 0 | 0 |
| Affine | `affine/lambda1` | 0 | 0 |

The last five methods change no neural parameters; non-population scalar controls fit their specified coefficients from support references.

## Verified checks

1. The source hashes in `personalization/run_manifest.json` matched the frozen code/protocol. The authoritative personal loader revalidated complete raw-file identities, the data manifest, and waveform/reference/timestamp/window-index hashes against that run's provenance.
2. `attention/baseline_seed42/best.pt` matched both the personal run's recorded baseline hash and its population completion marker. Population `config.json`, `manifest.json`, and `completion.json` also matched the byte hashes recorded by the personal run.
3. Every sampled `unit.json` matched its configuration hash and the hashes of its saved prediction and adaptation artifacts. The unit verifier checked patient, case, seed, budget, setting, provenance, and source-row alignment.
4. All ten support rows and all 481 query rows matched the raw-source references and timestamp intervals exactly. Support source indices were `[0,5,20,29,38,53,63,72,86,97]`; query source indices were 103 through 583. The final support endpoint was 2640.008 s, the calibration boundary 2686.008 s, and the first query start 2700.008 s. Thus this sample respects the required boundary-plus-ten-second query exclusion.
5. Direct differences between each saved adapted state and the original population state matched the recorded changed-parameter names and L2 changes. Head-only updates were exactly `head.1.{weight,bias}` and `head.3.{weight,bias}`. Last-block updates added the nine parameter tensors in `body.6`; full adaptation changed all 34 parameter tensors. No parameter outside the selected trainable mask changed.
6. All 30 saved buffers, including BatchNorm running statistics and batch counters, were bitwise identical to the population baseline for each neural unit. Recorded dropout state was inactive, BatchNorm running statistics were frozen, and query use for adaptation was false.
7. Predictions and references were finite. CPU inference from each saved neural checkpoint reproduced every saved query prediction exactly using the original inputs, target normalization, evaluation mode, four CPU threads, and the existing prediction batch size of 256. Recomputed population support predictions also matched the saved support predictions exactly.
8. Independently refitting each scalar calibration from those support predictions and original support labels reproduced its saved calibration coefficients exactly. Applying it to the population query predictions reproduced every saved query prediction exactly.

No neural adaptation or training was rerun, no hyperparameter was changed, and no scores were used to recommend settings. Full-inventory verification remains required after development and final-target execution finish.

## Artifact identity receipt

Baseline checkpoint SHA-256:

```text
972125f69da1b01d827e2d8162228311987bb781da5640b72309d2b36cf15f35
```

Personal data manifest hash:

```text
687a571b42e6431ce5f33da35882d1f863c38930b16154b3fe1a1f1c70009f73
```

`personalization/run_manifest.json` file SHA-256:

```text
435f2f1c7d80ab0fb2ccf6b209dfb4abce8edc73b94637f79511da963aca55d2
```

The following SHA-256 values identify the exact `unit.json` records inspected. Paths are relative to the K10 unit directory above.

```text
3d04c5579609aa1bf0cff31674ed787ca501c2209449b41235e720292c727092  head/lr0.0001_steps20/unit.json
c0b53eb4b53ee7803f7f5ed14fb39ce50690234cf5b0fbd5d94de873e64515c9  last_block/lr0.0001_steps20/unit.json
59048cafcb42229f2414f5166b1da9a04ca5744c29af23a91c6527829e326328  full/lr0.0001_steps20/unit.json
4912c6d661e5db43ca025696b1b450817f2220b506efd04adf1e3ee3deca643a  population/fixed/unit.json
3e645333851848f402161170fa7936269b6b071ea1e2863390e8949cc23f5c3e  bias/fixed/unit.json
d66e1a0527415037ef28146027e505a1efdda07a13e4820d3ecbee5b0e29ada2  last_reference/fixed/unit.json
2edcf1720e89344193f9144e3e3b47fb5ada2a381db02248e0aeb098bc699ccb  calibration_mean/fixed/unit.json
0993bfe4c2cb98680a22175f357c9446c19ac699617a0e7d66f02e3694043b0e  affine/lambda1/unit.json
```

## Host continuity snapshot

At approximately 16:23 America/Chicago on 2026-10-09, both suite processes and their active child jobs were running. Population training used MPS while personal development used CPU on the same workstation. The host was on AC power with the battery charging at 70%, had 612 GiB disk space available, and reported 43% system-wide memory free through `memory_pressure -Q`. Swap usage was 2708.69 MiB; this snapshot alone does not establish active swapping or a memory-pressure failure. No thermal or performance warning was recorded by `pmset`.

Existing `caffeinate -i` assertions targeted both suite PIDs, 8321 and 12337, preventing idle system sleep while those processes remain alive. They do not guarantee continuity through manual sleep, lid closure, shutdown, or process interruption. Shared-host activity also limits interpreting timings as isolated benchmarks. The inspection changed no power settings or running jobs.
