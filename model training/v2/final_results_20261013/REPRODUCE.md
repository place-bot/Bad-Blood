# Reproduction and provenance

## Predict or adapt without rerunning population training

Use the commands in the two model directories and the portable implementation.
Only NumPy and PyTorch are needed for inference. Checkpoints use Python pickle;
load only these trusted, checksum-verified files. All predictions are scalar
SBP/DBP in mmHg. Input normalization is required; the loader does not invent a
filter or resampling policy for arbitrary new sensors.

## Reproduce the completed study

The sibling experiment folders remain the authoritative complete workflow.
This package also freezes their Python implementation under `reproduction_source`.
That snapshot is for inspection/provenance; use the complete repository workflow
for audit manifests, cached prior fits and raw-data preparation.

From `model training/v2`:

```sh
python -m pip install -r requirements.txt
python modality_se/workflow.py \
  --data /path/to/prepared_population \
  --output /path/to/modality_experiment \
  --prior /path/to/completed_dual_attention_runs \
  --personal /path/to/original_personal_mat \
  --fresh /path/to/previous_sixteen_mat
python -m unittest discover -s tests
```

The recorded workflow uses the local MPS training device. Adapting its training
backend to CUDA/CRC is engineering work, not a bit-identical existing run.
Portable inference/adaptation supports CPU and a requested PyTorch device.
Do not supply participant-overlapping population files or alter frozen splits.
Official public data download/preparation and source hashes are documented in
the V2 root, `shared/data.py`, and the audit manifests.

## Rebuild this report

From this folder, `python build_package.py` regenerates numeric tables and copies
verified existing artifacts; it does not train or tune. Then:

```sh
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The distributed PDF is named `Bad_Blood_V2_Final_Report_20261013.pdf`.
`manifest.json` hashes release artifacts, and `verification.json` records
checkpoint-loading, inference equivalence and selected personal-fit checks.
Dates distinguish this consolidated release from the 20261012 experiment date.

## Evidence boundaries

- Population development and test participants are separated; personal targets
  are excluded from the population fits.
- Four distinct development participants select personal hyperparameters.
- Both personal target cohorts and the population test cohort were evaluated
  earlier in the project. Report them as previously examined cohorts.
- The actual known reference is invasive-ABP-derived BP, not a measured ND cuff
  value. ND transfer and hardware behavior require their own data collection.
- The new residual follow-up was verified but not promoted: improved SBP did
  not compensate for worsened DBP under the joint endpoint criterion.
