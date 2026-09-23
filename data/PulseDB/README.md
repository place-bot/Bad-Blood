# PulseDB source and processing

Source: Wang, Mohseni, Kilgore and Najafizadeh, *PulseDB*, Frontiers in
Digital Health (2023), DOI [10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854).
Official [repository](https://github.com/pulselabteam/PulseDB) and
[author-hosted subsets](https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing).

## Download and prepare

```
python data/PulseDB/download.py
cd "model training/PulseDB-based Model"
python prepare.py --features
```

Allow roughly 50 GB of free space for the archive, extracted source and
processed arrays. The preparation step extracts three files only:

| Official file | Participants | Segments |
|---|---:|---:|
| VitalDB_Train_Subset.mat | 1,293 | 465,480 |
| VitalDB_CalFree_Test_Subset.mat | 144 | 57,600 |
| VitalDB_AAMI_Test_Subset.mat | 116 | 666 |

`processing_summary.json` records actual validity counts. The training subset
is split into 1,099 training and 194 validation participants (seed 20260923).
All segments of a participant remain together. The separate calibration-based
subset in the archive is not used. AAMI/test results are separate evaluations,
not pooled as an additional independent cohort.

## Data representation

The HDF5-backed MATLAB structure is `Subset`. In h5py, `Signals` has axes
`1250 × 3 × N`: ECG_F, PPG_F, ABP_Raw. Only the first two channels enter the
model. `SBP` and `DBP` are scalar arterial-pressure-derived labels in mmHg.
`Subject` holds MATLAB string references, decoded into participant IDs.

Preparation converts the input to float32 `N × 2 × 1250`, checks finite,
non-flat channels and finite labels with SBP > DBP, then applies channel-wise
min-max normalization. Source channels are already filtered. No ABP samples,
IDs or labels are used as input features. Handcrafted features describe
waveform shape, spectra, intervals and beat templates; the CNN uses waveforms
directly. The 122 feature names are in `feature_names.json`.

## Real examples

- `examples/real_20_segments.csv`: 20 genuine 10-second records, all 2,506 columns.
- `examples/complete_header.csv`: the exact complete header, without data rows.
- `examples/column_dictionary.csv`: every column's index, name, unit and meaning.
- `examples/real_20_segments_metadata.csv`: readable metadata/label view.
- `examples/real_20_samples.csv`: the first 20 sampling points from the first segment.
- `examples/provenance.json`: source file, official subset row numbers and transforms.

Examples are selected from development training, not chosen for low prediction
error. They come from one person and are intended to explain the format, not
to characterize the dataset. Values are real, not synthetic. CSV is our
interchange format; the source is MATLAB, and bulk training uses NumPy arrays.

The public samples are adapted from PulseDB under CC BY-NC-SA 4.0. The
transformation is filtering already supplied by the authors plus per-channel
normalization, float32 conversion and CSV formatting. The original license is
in `LICENSE.txt`; retain this attribution and license when redistributing them.
