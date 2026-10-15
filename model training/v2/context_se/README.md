# Alternative personalization architectures — development follow-up, 20261012

**Status: bounded development screen complete; no replacement model established.**
This experiment follows the fixed-method new-participant result in
[V2 20261012.1](../releases/20261012/). The released CNN+SE and paired model
are preserved. These experimental candidates are not added to that frozen release.

The prior paired model did not consistently beat bias correction on 16 new
people. We therefore tested three different ways of using the full personal
reference set, rather than continuing to average pairwise MLP estimates.

## Architectures tested

1. **Learned-metric residual attention:** project SE features, learn query-to-
   reference similarity, and estimate a waveform-dependent personal correction
   from reference residuals. The output gain is trainable and unconstrained.
2. **Context-conditioned residual network:** encode the whole reference set;
   condition a nonlinear waveform correction on that context and center the
   correction on the observed references. This uses personal reference values
   without requiring the individual encoder to relearn waveform features.
3. **Learned-feature kernel residual regression:** learn feature projection,
   kernel length scale and noise from source-person episodes, then solve each
   person's residual regression from their own reference set.

These are engineering adaptations inspired by [Attentive Neural Processes](https://arxiv.org/abs/1901.05761)
and [Deep Kernel Learning](https://proceedings.mlr.press/v51/wilson16.html), not
reproductions of those papers. The fixed CNN+SE representation and calibration
budget remain the same. All three start at simple bias correction, then learn
departures. No query BP enters an inference context.

## Development results

Four original development people, SE seed42 only, all eligible early-20%-of-time
references. MAE is participant-macro SBP / DBP in mmHg. These numbers must not
be compared as if they were scores on the six- or sixteen-person target cohorts.

| Method | SBP MAE | DBP MAE |
|---|---:|---:|
| Personal mean | 12.53 | **5.26** |
| Simple SE bias correction | **6.84** | 6.18 |
| Residual attention | 7.07 | 6.98 |
| Context-conditioned network | 6.97 | 5.57 |
| Learned-feature kernel | 7.09 | 6.55 |
| Best tested context personal-gradient adaptation | 6.95 | 5.66 |

None lowered both SBP and DBP MAE below both controls. The context network was
closest; a separate prospectively specified follow-up tested nine personal
adaptation settings (learning rates 0.0001/0.0003/0.001 and 10/25/100 updates).
It also did not pass. The best adaptation by worst-control ratio was lr=0.0001,
10 updates. Longer adaptation was often worse, so increased fitting alone is
not evidence of better personalization.

Because no candidate passed development, seeds43/44 and new final testing were
not triggered. No candidate was run on the historical six or the newly examined
sixteen. There is no independent-test success claim for these alternatives.

## Files and reproduction

- [Initial frozen protocol](PROTOCOL.md), [personal-fitting follow-up](ADAPTATION_PROTOCOL.md).
- [Architecture](model.py), [source/development runner](run.py), [personal adaptation](adapt.py).
- [Full results](results/), [trained experimental checkpoints](models/), [checkpoint hashes](models/manifest.json).
- [Verification](results/verification.json): independent recalculation of all 48 person-level prediction files.

Population episodes sample 32 references and 16 disjoint queries per person
from the verified cached features; random source episodes have no temporal
claim because the population arrays have no timestamps. Personal fitting uses
chronological reference masks and all early observations; source and later-query
labels cannot leak into their own reference contexts. See the protocols for
all fixed optimizer, epoch and selection details.

```sh
python context_se/run.py --cache /path/to/paired_se/population/seed42 \
  --individual-root /path/to/vital_mat --output /path/to/context_screen
python context_se/adapt.py --source-run /path/to/context_screen \
  --individual-root /path/to/vital_mat --output /path/to/context_adaptation
```

The personal-gradient checkpoints contain the context-network parameters, not
the waveform encoder or reference bank; they are research artifacts tied to the
named development people, not deployment recommendations.

A next experiment should address the difference between random same-person
source episodes and chronological early-to-late calibration, ideally using
timestamped source-person records and a larger development cohort. That is a
testable direction, not an established explanation of the failed comparisons.
