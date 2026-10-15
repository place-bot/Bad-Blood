# Data access and fresh-machine reproduction

This guide supplements [README.md](README.md). It distinguishes acquiring the existing public source data, retraining the frozen experiments, and verifying an existing runtime-artifact bundle. It does not authorize adding new patients, changing the frozen cohort, or substituting a newly collected dataset.

As of **2026-10-09 22:49 America/Chicago**, all **20 required population fits, 1,068 personal development units, and 684 final-target units are complete and verified**. The repository includes the generated [combined report PDF](reports/final_report.pdf), [verified inventory](reports/results/inventory.json), [metric overview](reports/results/metric_overview.json), and detailed [attention](reports/results/attention.json.gz) / [personal](reports/results/personalization.json.gz) metric archives. Training and target evaluation are finished, all 25 final-report pages have been visually inspected, and 98 automated tests pass. See [final verification](reports/FINAL_VERIFICATION.md) and the [README findings](README.md#main-findings-from-the-completed-runs) for the three-seed SE/CNN comparison, the single-seed modality result, and the personal K=10 reference-mean comparison.

No external V2 checkpoint/prediction/runtime-artifact bundle has been published. The runtime paths in the README are local paths, not download links, and the report's metric archives do not contain raw waveforms, full prediction archives, or model weights. A fresh user can reproduce the workflow by acquiring the sources below and retraining the full inventory. The code, PDFs, and result summaries alone do not supply the complete inputs and runtime artifacts needed to rerun verification or inference against the original fitted models.

## 1. Environment and population arrays

First clone the research branch and complete the README's environment setup, including `BB_REPO`, `BB_PY`, and a runtime root outside the repository. Use a new checkout for source preparation; if prepared arrays already exist, reuse them rather than running preparation over an active study directory.

The preserved V1 preparation pathway downloads the author-hosted [PulseDB balanced training/testing archive](https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing), extracts only the three VitalDB-derived population subsets, and builds the required arrays. The official archive is approximately 17.3 GiB; allow roughly 50 GB free for the archive, extracted MATLAB sources, and prepared arrays. The source attribution and license are in [data/PulseDB](../../data/PulseDB/README.md).

On a fresh checkout, run:

```sh
cd "$BB_REPO"
"$BB_PY" data/PulseDB/download.py
cd "$BB_REPO/model training/PulseDB-based Model-V1"
"$BB_PY" prepare.py
export BB_POP="$BB_REPO/data/PulseDB"
cd "$BB_REPO/model training/v2"
```

The destination is fixed by the preserved `paths.py`: **`$BB_REPO/data/PulseDB`**, not the local author's `bp_model_v2/data` path. Do not append `--features`: V2 CNNs do not require the optional handcrafted-feature arrays. These commands prepare inputs only; they do not run V1 training or overwrite V1 model weights. The downloader refuses to overwrite an existing archive. Existing completed array directories are reused, but `prepare.py` rewrites its deterministic `split.json`; do not run it against data being used by an active experiment.

The resulting layout is:

```text
split.json
train/{waves.npy,y.npy,subjects.npy,valid.npy,complete.json}
test/{waves.npy,y.npy,subjects.npy,valid.npy,complete.json}
aami/{waves.npy,y.npy,subjects.npy,valid.npy,complete.json}
```

The bulky archive, extracted sources, and array subdirectories are ignored by Git in this preparation location. For physically external data storage, copy the three completed subdirectories and `split.json` unchanged to a new external directory and set `BB_POP` to that directory. Do not edit `paths.py`, rewrite identity metadata, or commit raw files. The V2 loader checks axes, values, disjoint patient partitions, and the corrected counts, then hashes every source file and split/identity record. Its additional 31-patient exclusion is applied when loading; do not pre-delete rows from the prepared arrays.

## 2. Ten individual personal-calibration files

Choose an external directory and point `BB_PERSONAL` to it:

```sh
export BB_PERSONAL="/path/to/external/pulsedb-individual"
mkdir -p "$BB_PERSONAL"
```

Download the ten original MATLAB files below from the public source links and save each with the exact filename shown, directly inside `BB_PERSONAL`. These are the links and byte counts recorded in the frozen [audit_snapshot.json](audit_snapshot.json), not a newly sampled cohort. Link availability and provider download limits can change; this documentation update did not bulk-download or revalidate the remote links. If a file is inaccessible or has a different identity, stop and resolve source access rather than substituting another participant.

| Filename / source link | Role | Case | Exact bytes |
|---|---|---|---:|
| [p005548.mat](https://drive.google.com/file/d/1hK-qzscgsRih1tf38XIkkxmBGC6NYAqg/view?usp=drivesdk) | Development | c4678 | 71,157,032 |
| [p005647.mat](https://drive.google.com/file/d/1ibJ2JAx0O0nIiAwgOf4Z_mk4gOK163g1/view?usp=drivesdk) | Development | c4746 | 85,547,336 |
| [p005879.mat](https://drive.google.com/file/d/1l-V5-eHk-NJGY3UmEdsH616IfrtIjzXV/view?usp=drivesdk) | Target | c5552 | 98,042,392 |
| [p005885.mat](https://drive.google.com/file/d/1l2_N4vh6yv0mDKGN8zvtqCVOF2XFf6Y3/view?usp=drivesdk) | Target | c0637 | 176,333,813 |
| [p005968.mat](https://drive.google.com/file/d/1lapSEgzfJQXiYOHdTFa5jI_BWGxQ25-y/view?usp=drivesdk) | Development | c3618 | 60,498,632 |
| [p005977.mat](https://drive.google.com/file/d/1leBEMFiNm_PDQemyY5_pedAjshzwz3R9/view?usp=drivesdk) | Target | c3725 | 87,713,552 |
| [p005984.mat](https://drive.google.com/file/d/1lgzQXIE1iEZ4SUQOGYGYy8mX67OZtBxE/view?usp=drivesdk) | Target | c4777 | 96,951,896 |
| [p006009.mat](https://drive.google.com/file/d/1lvdWYm5NF0b8nc5OCV5NVQMdjeSrNIus/view?usp=drivesdk) | Target | c2935 | 99,656,016 |
| [p006052.mat](https://drive.google.com/file/d/1mQeD3voBSsmfR0DtL045d2WTCcVvuFIy/view?usp=drivesdk) | Target | c4462 | 62,001,496 |
| [p006090.mat](https://drive.google.com/file/d/1mkuqH0XP84h7V3G5Qsz9TubfnGVFWvVn/view?usp=drivesdk) | Development | c2134 | 76,019,064 |

The ten files total 913,921,229 bytes. The other historical candidate files are not required to execute the frozen personal cohort. Their audit history remains in the repository; it is not permission to expand the cohort.

### Raw-file SHA-256 identity snapshot

The frozen audit contains URLs and byte counts but did **not** contain full raw-file hashes. The following hashes were computed read-only on 2026-10-09 from the exact local files identified by that audit. They are an additional byte-identity documentation snapshot, not a retroactive change to the frozen protocol or audit. The loader independently hashes these complete files and records those identities in each personal run manifest.

On macOS or another system with `shasum`, verify all ten files with:

```sh
cd "$BB_PERSONAL"
shasum -a 256 -c <<'SHA256'
37a15a40bd19efb5caa039d50c9750650e42df81f81085d01ed0240bbf25c9d9  p005548.mat
57d1df91faf35f7272269fa92ad7a28a09974e26c0eef5a571f27f74d54567c7  p005647.mat
8ee95c65b29d60edada190a0b4bacd14d4b310b600670df104db3ade34fe830e  p005879.mat
37cf310edcf7f427cadd71220fc1dbd4c44bcc93eba756ff4f7a045213f5f356  p005885.mat
693f0af0c4377114ca8c6e1c05ab1c736736829b5a466ce682f911f45edf1861  p005968.mat
b1d396edc4d519471f4e723736db2489455fa8c36404537d48077a840390120f  p005977.mat
2dd25d9c65d7248e35856ec29de363a058dbe57405a9e51f67743d8292be749a  p005984.mat
b86a432018998885a31c18ad2ec073381ab7153e116689522eb15759c786b1a9  p006009.mat
0e69b8a94e6e0bd66da39710c0e11f8cd1f9cfe7d00f959843a8388cbb3a3c8c  p006052.mat
f2801ea051629eda42aea01a84b9fc3aaf1b89a41ff35c2078ef81f86629a88e  p006090.mat
SHA256
cd "$BB_REPO/model training/v2"
```

On Linux with GNU coreutils, replace `shasum -a 256 -c` with `sha256sum -c`. All ten lines must report `OK`; a matching filename or byte count alone is insufficient.

PulseDB's existing filtering and fields must be preserved. Do not convert or resave the MAT files before verification. These are VitalDB-derived PulseDB recordings with arterial-pressure-derived scalar references, **not actual cuff/NIBP calibration events**. Follow the source's CC BY-NC-SA 4.0 terms and attribution when using or redistributing data.

## 3. Retraining and portable result verification

With `BB_POP`, `BB_PERSONAL`, `BB_RUN_ROOT`, and the environment set, return to the README's full-suite command. It trains 20 required population fits, performs 1,068 development units, freezes the development-selected settings, and evaluates 684 final-target units. Different supported devices can produce different floating-point trajectories; preserve the resulting source/data/configuration/environment identities rather than editing them to resemble another run.

If an authorized runtime bundle becomes available later, preserve its complete `attention/` and `personalization/` sibling layout, including manifests, completion/index files, selected checkpoints, and full predictions. Keep it outside Git. With byte-identical data located anywhere on the new machine, the report command is:

```sh
cd "$BB_REPO/model training/v2"
"$BB_PY" reports/build_report.py \
  --artifact-root "$BB_RUN_ROOT" \
  --data-root "$BB_POP" \
  --individual-root "$BB_PERSONAL"
```

The location overrides affect file discovery only. Original config/provenance hashes, source hashes, complete raw bytes, patient/window identities, and expected inventories are still checked. Never rewrite a recorded historical absolute path or hash inside a hash-bound artifact to make it load. A relocated data directory supports report verification, but **resuming an old training job at changed paths is not the same operation**: trainers deliberately reject changed configurations. Reproduce from scratch in a new runtime root when the original resume contract cannot be retained.

Only deserialize checkpoints from a trusted source; the training checkpoint format contains Python objects. The standalone prediction CLI additionally requires explicit `--trust-checkpoint`, and can use a trusted copied population `best.pt` with the matching source checkout. No published checkpoint bundle should be implied until an actual externally accessible location and its integrity metadata have been provided.
