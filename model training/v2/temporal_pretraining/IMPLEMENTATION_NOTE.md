# Matched sampling correction

The initial execution used the same RNG seed but a single RNG stream for person
choice and window sampling. Random and chronological window sampling consume
different numbers of draws, so subsequent person choices were not exactly
matched. The initial run is retained locally under `20261012_temporal/run`;
it is not the authoritative comparison.

Before accepting any result, participant choice was separated into a fixed
epoch-specific RNG schedule shared by both modes (`episode_people`). The
corrected execution uses `20261012_temporal/matched_run`. No cohort, model,
optimizer, calibration budget, stopping rule or selection threshold changed.
Both variants have identical participant exposure for each epoch they run;
their validation-based stopping epochs may differ. The correction is covered
by a deterministic-exposure test. No target participants were evaluated.
