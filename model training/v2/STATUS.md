# Workflow status

## 2026-10-09

- Dedicated branch: `research/v2-vitaldb-attention-personalization`.
- V1 release retained. Latest team changes fetched; no unrelated files edited.
- Audit and both literature reviews are complete. The main agent inspected
  all seven attention-review pages and all six personalization-review pages
  after compilation; both have resolved citations and readable layouts.
  The protocol is frozen before implementation. No new model-training
  experiment has started at this gate.
- The project owner confirmed the revised task: early ECG/PPG and BP personal
  calibration followed by same-window BP estimation from later observed
  ECG/PPG. RESEARCH_PROMPT.md supersedes the original waveform-forecasting
  scope; PROTOCOL.md specifies the experiments.
- Local MPS access works outside the restricted process sandbox: Apple M5,
  24 GiB RAM, 10 logical CPUs, PyTorch 2.14.0. No CRC allocation is used here.
- Temporal analysis must use individual-record files with timestamps and
  CaseID. Balanced subset row order cannot establish chronology.
