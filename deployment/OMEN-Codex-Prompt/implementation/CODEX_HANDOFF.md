# Copyable handoff for another Codex

I want to reproduce the Bad Blood V1 Raspberry Pi deployment from this full
repository. Start by reading `deployment/OMEN-Codex-Prompt/PROMPT_ZH.md` and
everything in `deployment/OMEN-Codex-Prompt/implementation/`, especially
`REPRODUCE.md` and `SOURCE_MAP.md`. Use the single runnable source under
`deployment/raspberry-pi/`; do not make a duplicate implementation.

Inspect **this** Raspberry Pi's model, OS, architecture, Python version,
network and free space before installing anything. Keep its existing microSD
contents unless I separately authorize a destructive change. Install the
pinned Pi environment, run the deployment tests and `run_pi_checks.py` on the
Pi, and report new Pi evidence rather than reusing the original machine's
numbers. Start the service bound to Pi loopback, connect my computer through
an SSH tunnel, and verify advancing predictions, ECG/PPG input waveforms,
recorded-data labeling and stale/disconnected clearing. The computer is only
the display; the released V1 predictor must run on the Pi.

The present source is recorded PulseDB data. A future device must supply the
same validated `SignalWindow`; no real-device or clinical validation exists.
Tell me clearly what succeeded and what remains. Do not post credentials,
SSH keys, raw local evidence or private data to GitHub. Do not enable boot
startup or reboot the Pi until I agree, as the original prompt requires.
