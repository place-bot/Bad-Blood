# Population models retained unchanged

Use the [CNN+SE architecture, six population checkpoints and training results](../../final_results_20261013/01_pretrained/README.md).
Source training: 1,085 VitalDB-derived PulseDB participants / 390,600 ten-second
windows; validation: 192 / 69,120. Main test: 141 / 56,400.
The 20261015 change is to **personal adaptation**, not new population training.
Both PPG-only and ECG+PPG remain available. The paired source initializers for
personalization are also copied into this release's `02_personalization/source_weights/`.
