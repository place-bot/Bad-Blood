# Context-network personal fitting follow-up — 20261012

The initial three-architecture seed42 development screen is complete and none
passed the joint criterion. Preserve those results. This separate bounded
follow-up tests personal gradient adaptation of the closest architecture,
`context`, instead of merely conditioning on a reference set at inference.

Start from the already selected population-context checkpoint (source-validation
epoch 7). For each of the four original development people, use all eligible
early-20%-of-time observations. Each early query sees only earlier references
whose windows end at least 10 seconds before that query begins. All support rows
must participate as a reference or a training query. Query labels cannot enter
their own reference context, including through context normalization.

Fine-tune the entire context network, not the original SE encoder. Adam,
no weight decay/L2-SP, gradient clip 5. Candidate learning rates
{0.0001, 0.0003, 0.001}, update counts {10, 25, 100}: nine bounded settings.
Each optimizer step covers every eligible early query with all its allowed
earlier references. Standardized-target MSE. At evaluation, use all early
references, unchanged later windows and the matched seed42 SE baseline.

Select only using the four development people. Primary criterion remains
simultaneous lower SBP/DBP MAE than both personal mean and simple bias correction;
rank by worst ratio, then mean MAE and fixed candidate order. No historical six
or fresh-follow-up sixteen are loaded or scored. Stop this follow-up if none
passes; successful development requires seed replication and another unused
cohort before claiming superiority. Do not relabel a development result as test.
