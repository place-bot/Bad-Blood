# V2 repository and existing-data audit

Audit date: 2026-10-09. This audit used read-only code, checkpoint, metadata, and numeric validity inspection. No model implementation, fitting, or training was performed. Counts and per-record facts are preserved in [audit_snapshot.json](audit_snapshot.json).

## Repository and baseline identity

The working repository is `/Users/jguo9/Desktop/26fall/LLM/output/Bad-Blood`. The research base is upstream commit `7553248168295bcad593739fdc733dbefff0c3b1`; work is on `research/v2-vitaldb-attention-personalization`. The initially inspected checkout was clean at `d6b73cc419b6b6dad2b66d70dcc9558041f35753`; updating to the research base did not change the V1 network.

The baseline is the actual `BPNet(pool=1)` in [the V1 network](../PulseDB-based%20Model-V1/network.py). Its last source-changing commit is `1f33cdcf090d51133c91dd50af2e575f5eaa0053`; Git blob `e400ce5994205dc1406752b767da9315105e313f`. The source SHA-256 is `650e9d7b43cc8599a35c7075ea12347efb6f33c31046b0843bdbca0d47bf2d5d`. The selected checkpoint is `models/cnn_full.pt`, SHA-256 `bb6742489f64128e714b365ebb2dd659b82ed2a8b8476a61416f9dff8af42ef1`.

“Higher resolution” refers to removal of the initial fivefold average pooling used by the alternate small CNN. It is not a different published network. Both networks have 37,666 trainable parameters; the full-resolution variant retains longer feature sequences.

| Stage | Exact operation | Output for input B × 2 × 1250 |
|---|---|---|
| Input | ECG and PPG, 125 Hz, 10 seconds | B × 2 × 1250 |
| Initial pooling | AvgPool1d(1) | B × 2 × 1250 |
| Stem | Conv1d(2,8,k=7,s=2,p=3,no bias), BatchNorm, SiLU | B × 8 × 625 |
| Residual block 1 | 8→16; convolutions k=7,s=2 then k=5,s=1; BN; projected skip | B × 16 × 313 |
| Residual block 2 | 16→32; same block layout | B × 32 × 157 |
| Residual block 3 | 32→48; same block layout | B × 48 × 79 |
| Pool | Adaptive average pool to length 1, flatten | B × 48 |
| Head | Dropout(0.15), Linear(48,32), SiLU, Linear(32,2) | B × 2 |

Each block applies SiLU after its first convolution/BN and after adding the second convolution/BN to the projected skip. The skip uses a stride-two 1×1 convolution and BN. The model predicts contemporaneous segment SBP and DBP, in that order. It has no waveform decoder and performs no future forecasting.

The checked checkpoint records pool=1 and epoch=3. Target center is [115.3868637084961, 62.89986801147461] mmHg and target scale is [18.96441078186035, 12.095508575439453] mmHg. Historical training used normalized-target MSE, AdamW (learning rate 0.001, weight decay 0.001), batch 256, Gaussian waveform noise SD 0.003, gradient clipping at norm 5, seed 20260923, ReduceLROnPlateau(factor=0.5, patience=3), and stopping after eight non-improving epochs. The validation criterion was the arithmetic mean of patient-macro SBP and DBP MAE. The selected run stopped after epoch 11; epoch 3 was best. Its recorded total epoch time was 661.32 seconds; these are historical measurements, not a new hardware benchmark.

## Actual local data and provenance

There are two existing representations, outside ordinary Git history:

| Representation | Actual location | What it supports |
|---|---|---|
| Official PulseDB VitalDB subsets and processed arrays | `/Users/jguo9/Desktop/26fall/LLM/output/bp_model_v2/data/` | Population contemporaneous SBP/DBP estimation |
| PulseDB v2 individual subject files | `/Users/jguo9/Desktop/26fall/LLM/output/bp_model/data/vital/` | Chronological within-record studies and personal adaptation |

The directory named `bp_model_v2` predates this research assignment and holds the data behind the published repository V1. Its name does not indicate that the newly requested V2 experiments already exist.

Provenance is not inferred solely from filenames. Three independent checks support the VitalDB origin:

1. Every decoded official-subset Subject value has suffix `_1`. The [official generator](https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Generate_Subsets.m) explicitly uses source 1 for VitalDB and source 0 for MIMIC.
2. Individual files contain original SubjectID and CaseID fields. All 31 observed case-to-subject pairs in the 30 complete files match the public [VitalDB clinical metadata](https://api.vitaldb.net/cases), inspected on the audit date. For example, local subject p006081/case c2280 matches caseid 2280/subjectid 6081.
3. The [official field dictionary](https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Supplementary%20Materials.pdf) defines the subject IDs as original dataset identifiers, while CaseID distinguishes multiple recordings of one subject.

Canonical identity should be source-qualified, for example `vitaldb:p006081`. The suffix `_1` is a source identifier, **not a visit or session identifier**. Strip it only after verifying the source. Use `(source, SubjectID, CaseID)` for temporal construction and SubjectID for patient-disjoint splitting.

## Official subsets and V1 preprocessing

All three official files contain the same fields: Age, BMI, DBP, Gender, Height, SBP, Signals, Subject, and Weight. They do not retain T, CaseID, SegmentID, cuff readings, or event times.

| Official file | Participants | Segments | File bytes |
|---|---:|---:|---:|
| VitalDB_Train_Subset.mat | 1,293 | 465,480 | 13,797,698,523 |
| VitalDB_CalFree_Test_Subset.mat | 144 | 57,600 | 1,706,947,648 |
| VitalDB_AAMI_Test_Subset.mat | 116 | 666 | 19,806,481 |

The local source archive is 18,592,635,040 bytes. Signals have HDF5 axes 1250 × 3 × N. The generator identifies the channels as ECG_F, PPG_F, and ABP_Raw. V1 uses only the first two channels, converts them to float32 N × 2 × 1250, and applies per-window, per-channel min-max normalization. Its validity checks are finite inputs, nonzero channel span, finite labels, and SBP > DBP. All existing processed rows pass those checks. SBP and DBP are derived from invasive ABP; they are not cuff measurements.

The original deterministic split (seed 20260923) has 1,099 training patients/395,640 segments and 194 validation patients/69,840 segments. Decoded train, calibration-free test, and AAMI patient sets are pairwise disjoint, and internal training/validation are disjoint. The official subset row order cannot recover chronological adjacency. Even adjacent rows for one patient can represent distant selected segments.

V1's optional raw-input adapter filters each already-observed 10-second input independently using zero-phase Butterworth ECG (0.5–40 Hz) and Chebyshev-II PPG (0.5–8 Hz), then normalizes it. The prepared population inputs instead use upstream filtered PulseDB fields. These are related but not automatically identical preprocessing pathways.

Historical validation MAE is 12.3842/8.3347 mmHg for SBP/DBP. The official calibration-free historical test MAE is 12.4593/8.4566 and RMSE 16.0097/10.7860 mmHg. These scores describe the old split and old weights only. They must not be used as the baseline score against newly retrained models on a corrected population split.

## Individual records, completeness, and temporal support

The original download manifest lists 32 files. There are 31 present files, of which 30 match expected byte counts and can be read completely. Those 30 files contain 30 patients, 31 records, and 15,486 retained 10-second windows: 154,860 seconds, or 43.0167 hours, of observed data. Gaps mean this is not one continuous 43-hour time series.

| Exclusion | Evidence |
|---|---|
| p005628.mat | Manifest entry exists; local file is absent |
| p005648.mat | Present size 17,825,792 bytes versus expected 117,080,432 bytes; exclude as truncated |
| p005918.mat | Two isolated windows, no adjacent pair; inadequate adaptation/durability support |
| p006048.mat | Two isolated windows, no adjacent pair; inadequate adaptation/durability support |

Complete-file inspection checked every retained window for length 1250, finite monotonically regular timestamps with interval 0.008 seconds, and finite non-flat ECG_Record, PPG_Record, and ABP_Raw. There were zero failures of these numeric checks. This is not a physiological quality guarantee: PulseDB already performs signal and reference-dependent quality selection.

The individual v2 files provide T, SubjectID, CaseID, SegmentID, WinID, WinSeqID, ECG_Record, PPG_Record, normalized ECG_Raw/PPG_Raw, filtered variants, ABP_Raw, scalar SegSBP/SegDBP, and reference-derived landmarks and QC values. T is relative to the retrieved clip, not a recoverable wall-clock timestamp. The [upstream v2 documentation](https://raw.githubusercontent.com/pulselabteam/PulseDB/main/README.md) identifies ECG_Record/PPG_Record as non-normalized raw-amplitude fields. Their availability allows input-only normalization.

Within each case, all inspected starts are ordered and unique. A critical exception to treating a *file* as one case is p005878:

| Case | Windows | Age | First/last retained sample, clip seconds |
|---|---:|---:|---|
| c1574 | 391 | 56 | 0.008 / 6840.000 |
| c4742 | 713 | 57 | 1590.008 / 16020.008 |

Both records are independently linked to VitalDB subject 5878 by the clinical API. Their timestamps reset and overlap. Concatenating this file by T would mix sessions. The two cases provide evidence of separate recordings, but absolute dates and the inter-session interval are unavailable. Random CaseID order is not chronology; the age difference is insufficient to quantify elapsed duration. A cross-case sensitivity analysis may be possible, but a quantified cross-visit persistence claim is unsupported by the present metadata.

## Cuff references and scope resolution

The user confirmed that Track B should calibrate with early personal ECG/PPG and scalar BP references, then estimate contemporaneous SBP/DBP from later observed ECG/PPG. Genuine future-waveform prediction is outside the confirmed experiment scope. The directory name `personalized_forecasting` is retained for continuity with the original request; reports and result tables must describe the actual contemporaneous estimation task. The following forecasting-specific audit findings are retained as provenance limitations and do not redefine the confirmed task.

No NIBP/cuff fields or independent cuff events exist in either local representation. VitalDB's [native parameter list](https://vitaldb.net/dataset/) includes NIBP tracks, but those tracks are not present in the working PulseDB files. A budget of one, three, or ten existing segment labels counts invasive-ABP-derived reference windows, not cuff events. Any sparse reference calibration using them must be explicitly called an ABP-reference simulation. It cannot validate actual cuff calibration.

Chronological waveform targets can be constructed only from the individual files, grouped by patient and record and indexed by T. Raw ABP retains pressure units and has not been amplitude normalized. ABP_Lag is stored cross-correlation/QC metadata; it must not be used to shift a target into agreement with a forecast. ECG_RPeaks, ABP peaks/turns, and PPG_ABP_Corr also must not become unavailable inference inputs.

The local data are already downsampled to 125 Hz. The published source describes downsampling from native VitalDB recordings but does not provide the exact upstream resampling kernel/support in its distributed code inventory. Consequently, absence of *new* future-dependent operations can be tested, while a strict native-device-time causality guarantee remains unverified. Full-record filtered fields must not be assumed safe for forecasting. A future-waveform protocol should use unfiltered fields and define observed-window processing, explicitly report the upstream limitation, and resolve whether the needed guarantee requires the original same-case recordings. This is a data-provenance issue, not a reason to silently substitute scalar estimation for waveform forecasting.

The [PulseDB paper](https://doi.org/10.3389/fdgth.2022.1090854) describes reference-dependent quality selection. It also defines segment BP as the average of beat-level invasive pressures. Those selection effects remain in all retained local windows; performance is conditional on this curated sample.

## Corrected comparison and frozen personal cohort

The historical exploratory subject list contains 31 IDs and includes the truncated subject. Exclude all 31 from *all newly trained population models and population preprocessing-statistic fitting*, including attention variants and their full-resolution CNN reference. This removes 14 subjects from original training and two from validation:

| Corrected population partition | Patients | Segments |
|---|---:|---:|
| Training | 1,085 | 390,600 |
| Validation | 192 | 69,120 |

The historical V1 checkpoint has already seen several candidate personal patients. It is therefore an architectural reference and historical artifact, not a leakage-free initialization for these patients. New comparable population weights must be trained on the corrected split. Three exploratory IDs also appear in the official calibration-free test (p005897, p005968, p005977); report the 141-patient/56,400-segment exposure-excluded test separately if that test is retained.

The initial availability criteria were: complete file, one case, at least 400 observed windows, at least 300 adjacent 10-second pairs, at least 100 windows beginning one hour after the first retained sample, and all numeric checks passing. Seventeen patients qualified. Their sorted IDs were shuffled with NumPy default_rng(20261009); the first six were proposed as targets and the next four as adaptation development. The unchanged original proposal is preserved in the snapshot's `candidate_personalization_cohort` field.

An additional pre-experiment eligibility check required ten calibration windows, each starting at least 60 seconds after the previous selected window, within the first 20% of elapsed recording span. Original target p005983 had only six such windows and was excluded from the final target cohort. It was replaced by p005977, the next unused subject in the original seeded permutation. No model performance was inspected or used. The four development assignments remain unchanged.

- Final target patients: p005879, p005885, p005977, p005984, p006009, p006052.
- Development patients: p005548, p005647, p005968, p006090.
- Unallocated patients meeting the final availability criteria: p005884, p005897, p005903, p005906, p006006, p006056.

The six final targets have 5,103 retained windows/850.5 observed minutes across six records; the four development patients have 2,409 windows/401.5 minutes across four records. Final queries contain 4,284 target windows and 1,855 development windows. “Observed duration” sums retained windows and does not imply continuous elapsed recording.

The frozen per-patient manifests are in `final_personalization_protocol.patients` in the snapshot. Each identifies the canonical patient, original subject and case, source file, zero-based source-row indices, and exact sample intervals for every selected support and query window. Their construction is entirely metadata-based:

1. Let the last sample's exclusive endpoint be last_sample + 0.008 seconds. Set boundary b = first_sample + 0.20 × (last_endpoint − first_sample). This is 20% of elapsed span, not 20% of retained rows.
2. Retain calibration candidates whose entire 10-second interval ends at or before b. Traverse them chronologically and greedily select a pool with start-to-start spacing of at least 60 seconds.
3. Select ten positions from that pool using `np.rint(np.linspace(0, len(pool)-1, 10)).astype(int)`, with NumPy's ties-to-even rounding. Budgets 1, 2, 3, 5, and 10 use nested chronological prefixes of those ten windows. Their observed waveform durations are 10, 20, 30, 50, and 100 seconds, separately from their elapsed spans.
4. Apply each adaptation at the common boundary b. Include only query windows starting at or after b + 10 seconds. All methods and budgets use the same query rows for a patient. Support and query intervals are disjoint.

The sparse references are segment-averaged invasive ABP labels. A 60-second gap limits immediate redundancy but does not turn these into actual cuff events or prove statistical independence.

| Role | Subject | Boundary b, seconds | Spaced support pool | Selected support | Query windows |
|---|---|---:|---:|---:|---:|
| Target | p005879 | 6170.008 | 28 | 10 | 654 |
| Target | p005885 | 3836.008 | 49 | 10 | 1220 |
| Target | p005977 | 3960.008 | 28 | 10 | 569 |
| Target | p005984 | 2338.008 | 30 | 10 | 666 |
| Target | p006009 | 4506.008 | 13 | 10 | 761 |
| Target | p006052 | 1934.008 | 18 | 10 | 414 |
| Development | p005548 | 2686.008 | 22 | 10 | 481 |
| Development | p005647 | 2884.008 | 45 | 10 | 486 |
| Development | p005968 | 1314.008 | 18 | 10 | 401 |
| Development | p006090 | 1932.008 | 28 | 10 | 487 |

Timing bins [0,15), [15,30), [30,60), [60,90), [90,120), and [120,infinity) minutes are recorded relative to adaptation time b. Reports must call this elapsed time since adaptation. Actual time since the final observed reference is a different quantity and varies with budget; every budget's final-reference time and its age at adaptation are also preserved. For p006009, even the tenth selected reference ends at 2090.008 seconds, 2416 seconds before adaptation. Such missing-data gaps and empty timing bins must remain visible in patient-level interpretation.

## Environment and next decisions

The existing Python environment is `/Users/jguo9/Desktop/26fall/LLM/output/bp_model/.venv/bin/python`, Python 3.12.13 with PyTorch 2.14.0, NumPy, SciPy, h5py, scikit-learn and existing project dependencies. The machine is macOS 26.6.2 ARM64. An in-sandbox probe reports 10 logical CPUs with MPS inaccessible; the main task separately verified MPS availability outside the sandbox on Apple M5 with 24 GiB memory. Device availability therefore depends on execution permissions. Record the actual execution device and software versions for every new run. Historical epoch timings cannot substitute for measured new-run costs.

The primary-task decision and metadata-based personal cohort are now frozen. The user's confirmed scope supersedes the original future-waveform requirement. This audit does not implement or train models. Both literature-review PDFs must be complete and visually inspected before implementation or training begins.
