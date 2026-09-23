# ND collection interface

This public directory contains the data contract and blank templates only.
Store actual participant waveforms, cuff records and photographs in the
approved restricted ND storage, not this public repository.

Each accepted 10-second segment has two synchronized channels at 125 Hz.
The acquisition software exports one raw segment CSV with these exact columns:

```
time_s,ecg_raw,ppg_raw
```

There are 1,250 rows, at 0.000, 0.008, ..., 9.992 seconds. Amplitudes are
device-native values; keep their units and acquisition settings in a session
configuration file. Preserve the continuous source recording as well.

The paper form records the participant token, visit, cycle, segment ID and
cuff readings. After transcription and review, its machine-readable log is:

```
participant_id,visit,cycle,segment_id,waveform_file,SBP,DBP,status
```

`waveform_file` is relative to the log. `status=accepted` selects a completed
pair for conversion. Retain unsuccessful attempts with their reason in the
collection log; do not replace a failed measurement with an invented value.

Run `convert_nd.py` from the model directory to filter and normalize these
raw segments, producing the same 2,506-column schema as the PulseDB examples.
Run prediction with `--format filtered` on that converted file, avoiding a
second filtering pass. Raw data at other sampling rates require synchronized,
anti-aliased resampling before this converter; it rejects inconsistent timing.

The converter enforces file/timing/value checks, not clinical signal quality.
The hardware team must verify waveform polarity, synchronization, placement
and acquisition settings against the approved collection procedure.
