# CNN plus SE personalization with complete early records

Protocol fixed on October 12, 2026 before running this extension. Earlier sparse-reference results remain unchanged.

## Pretrained model

Use `se_last` (38,296 parameters), seeds 42, 43, 44, loading the completed
validation-selected checkpoint within each SE run. CNN+SE has the lowest observed
primary-test MAE among the six required three-seed V2 architecture comparisons
(12.386 / 8.002 mmHg SBP/DBP). This is the user's chosen initializer for this
extension, not a retroactive replacement of the original validation winner.
PPG-only had lower single-seed errors and is not part of that three-seed ranking.

## Data and controls

Reuse the frozen four development and six target participants, source hashes,
per-case elapsed-time boundary, and exact later query indices. All valid windows
whose exclusive end is within the first 20% of elapsed recording time enter
personal training, not just the ten sparse references. Retain the existing
ten-second boundary guard. All methods use identical support and query rows.
Whole-record normalization is prohibited: existing per-window ECG/PPG min-max
normalization and population-only target mean/SD are retained.

Controls are unadapted SE, support-reference mean, and support-mean error
correction added to SE predictions. All support references are ABP-derived labels,
not actual cuff measurements. Dense early supervision cannot be claimed feasible
with only a few ND cuff readings. The six targets have already been examined in
the prior study: this follow-up is exploratory, not fresh external validation.

## Bounded development search

Run 24 preset configurations: three trainable scopes (head plus SE gate; final
residual block plus head and SE gate; all parameters), two learning rates
(0.0001, 0.001), two full-batch update counts (20, 100), and two objectives.
The plain objective is standardized-target MSE. The centered objective subtracts
the current support prediction mean and adds the support reference mean, learning
within-person variation; the final support-derived offset is saved for inference.
Every update uses ALL support windows. AdamW weight decay 0.001, gradient clipping
5, BatchNorm running statistics frozen, dropout disabled. Every fit starts afresh
from the corresponding SE checkpoint; seeds are separate fits, not an ensemble.

Each configuration is evaluated on all four development people and all three
seeds. For each pressure, calculate patient-and-seed-macro MAE relative to each
of the two personal controls. Select the smallest worst ratio across these four
comparisons, then the lower average MAE, then earlier fixed candidate order.
A worst ratio <1 means beating both controls on both pressures in development.
Save the full grid and a hashed selection before any new target predictions.
Evaluate only that selected configuration on the six target participants. Do not
change settings based on target performance. Failure to beat a control is a result,
not authorization to suppress it or to redefine the test set.

## Outputs and acceptance

Keep per-person weights, offsets, support/query indices, source and checkpoint
hashes, support losses, prediction arrays, and complete metrics locally. Publish
reproduction code and aggregate/per-person metrics, not new raw participant data.
Report MAE, RMSE, signed bias, sample error SD, fractions within 5/10/15 mmHg,
per-participant MAE, and descriptive patient-bootstrap differences against both
controls after averaging seeds. Report both pooled and patient-macro summaries.
The requested performance objective is target patient-macro MAE below both
controls for BOTH SBP and DBP, without claiming clinical certification or universal
per-person improvement. Do not claim success before the measurements establish it.

Verify frozen buffers, inclusion of the SE gate, exact support coverage, fixed
query indices, no query-label access in adaptation, fresh initialization, resumable
artifact checksums, and saved-model prediction replay. Preserve V1 and prior V2.
