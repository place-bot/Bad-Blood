# Implementation-time source diagnostics

October 9, 2026, before examining new model test results.

The real-data loader checks all frozen personal timestamps against their
source MATLAB columns. Two later-query pairs overlap by one 125Hz sample
(8ms) in upstream timestamps: p005885 rows1328/1329 and p006009 rows739/740
(zero-based). Both pairs are wholly in the query period, not across a
calibration/evaluation boundary. They remain in the frozen query sets.

The loader records their exact times in
`manifest.source_time_diagnostics`; it does not shift or silently delete the
windows. This minor within-query overlap further supports patient-level,
rather than independent-window, uncertainty calculations. Support/query gaps
and reference spacing still pass their explicit checks.

Real population identity hash:
`88984e9b5a412cc20c3f9ee48badc9c82502cf982fedb7ae9e43315f314a3fca`.
Real personal identity hash:
`687a571b42e6431ce5f33da35882d1f863c38930b16154b3fe1a1f1c70009f73`.
These include full input-file content hashes and frozen selection identity.
