# Bad Blood V1 — Raspberry Pi implementation pack

This folder is the handoff for the work requested by
[`../PROMPT_ZH.md`](../PROMPT_ZH.md). Clone the **whole repository**: the
deployable code, released model weights and public example input are linked
below. This folder records how to reproduce and assess the implementation;
there is only one copy of the runnable source under `deployment/raspberry-pi/`.

| Read | Purpose |
| --- | --- |
| [STATUS.md](STATUS.md) | What was actually verified on the Pi, numeric results, and open work |
| [SOURCE_MAP.md](SOURCE_MAP.md) | Every relevant source, model, data, test and configuration file |
| [REPRODUCE.md](REPRODUCE.md) | Commands and checks for a fresh Raspberry Pi 4B deployment |
| [EVIDENCE_SUMMARY.json](EVIDENCE_SUMMARY.json) | Sanitized, machine-readable Pi measurement summary |
| [CODEX_HANDOFF.md](CODEX_HANDOFF.md) | Copyable task for another Codex on a different computer/Pi |

The implemented service is at [`../../raspberry-pi/`](../../raspberry-pi/).
Its own [README](../../raspberry-pi/README.md) explains the input contract and
runtime. The current source is a public PulseDB recording, **not** a connected
medical sensor. A future device adapter must emit the same validated ECG/PPG
window before the existing V1 predictor can use it.

For another Codex agent: read this folder, the linked source and the original
prompt; inspect the destination Pi and network rather than assuming the
original computer's IP, username or package cache. Re-run the Pi-only checks.
Do not report OMEN measurements as Pi measurements, upload the ignored raw
result files, or claim device/clinical validation.
