"""Pure, outcome-derived interpretation of a complete verified Track A inventory.

No files are read or written, no models are fitted, and no final-test outcome
selects an architecture. The caller must verify raw prediction pairing first.
The validation rankings distinguish the prospective four-model attention set
from the larger required comparison including capacity and placement controls.
"""
from __future__ import annotations

import numpy as np

from shared.metrics import paired_patient_change

SEEDS = (42, 43, 44)
ATTENTION_SET = ("baseline", "se_last", "temporal_last", "cbam_last")
REQUIRED_THREE_SEED = (*ATTENTION_SET, "additive_last", "se_all")
MODALITIES = ("baseline_ecg", "baseline_ppg")
PARAMETERS = {"baseline": 37666, "se_last": 38296, "temporal_last": 37680,
              "cbam_last": 38310, "additive_last": 38296, "se_all": 38670,
              "baseline_ecg": 37666, "baseline_ppg": 37666, "eca_last": 37669}
DISPLAY = {"baseline": "CNN", "se_last": "SE-last", "temporal_last": "Temporal",
           "cbam_last": "CBAM1D", "additive_last": "Additive", "se_all": "SE-all",
           "baseline_ecg": "ECG only", "baseline_ppg": "PPG only", "eca_last": "ECA"}
TARGETS = ("SBP", "DBP")
TOLERANCE = 1e-9
# These are descriptive contrasts of existing fixed fits, not new experiments.
SECONDARY_CONTRASTS = (
    ("se_last", "additive_last", "capacity-matched operator"),
    ("se_all", "se_last", "insertion depth and capacity"),
    ("cbam_last", "se_last", "combined versus channel gate"),
    ("cbam_last", "temporal_last", "combined versus temporal gate"),
    ("baseline_ecg", "baseline", "remove PPG, one seed"),
    ("baseline_ppg", "baseline", "remove ECG, one seed"),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seeds(variant):
    return (42,) if variant in MODALITIES else SEEDS


def _validate(aruns):
    """Check complete summary/config identities after upstream byte verification."""
    variants = (*REQUIRED_THREE_SEED, *MODALITIES)
    if any(key.startswith("eca_last_seed") for key in aruns):
        variants += ("eca_last",)
    expected = {f"{variant}_seed{seed}" for variant in variants for seed in _seeds(variant)}
    _require(set(aruns) == expected, "Attention interpretation requires the complete fixed run inventory")
    reference_members = {}
    for key, run in aruns.items():
        config, completion = run["config"], run["completion"]
        variant, seed = config["variant"], config["seed"]
        _require(key == f"{variant}_seed{seed}" and variant in variants and seed in _seeds(variant),
                 "Run key and configured variant/seed disagree")
        _require(completion["status"] == "complete" and completion["toy_test_only"] is False,
                 "Interpretation cannot consume incomplete or toy-test results")
        _require(completion["variant"] == variant and completion["seed"] == seed,
                 "Completion identity differs from configured run")
        _require(completion["parameters"] == PARAMETERS[variant], "Unexpected architecture parameter count")
        for split, n_patients, n_windows in (("validation", 192, 69120), ("test", 141, 56400)):
            summary = run["metrics"][split]
            people = summary["per_patient"]
            _require(summary["target_order"] == list(TARGETS), "Pressure-target order is not SBP/DBP")
            _require(summary["n_patients"] == n_patients and len(people) == n_patients
                     and summary["n_windows"] == n_windows,
                     f"Wrong corrected population cohort size: {split}")
            counts = {patient: row["n_windows"] for patient, row in people.items()}
            _require(all(isinstance(value, int) and value > 0 for value in counts.values())
                     and sum(counts.values()) == n_windows, "Invalid per-patient window counts")
            reference_members.setdefault(split, counts)
            _require(counts == reference_members[split], "Patient membership or coverage differs across runs")
            for metric in ("mae", "rmse"):
                values = np.asarray([people[patient][metric] for patient in sorted(people)], float)
                macro = np.asarray(summary["macro"][metric], float)
                _require(values.shape == (n_patients, 2) and np.isfinite(values).all()
                         and np.all(values >= 0) and macro.shape == (2,)
                         and np.allclose(macro, values.mean(0), atol=TOLERANCE, rtol=0),
                         f"Invalid equal-patient {metric} summary: {key}/{split}")
    _require(not (set(reference_members["validation"]) & set(reference_members["test"])),
             "Validation and primary-test patients overlap")
    return variants


def interval_relation(lower, upper):
    """Directional descriptive evidence, never a significance or equivalence test."""
    _require(np.isfinite(lower) and np.isfinite(upper) and lower <= upper, "Invalid interval")
    if upper < -TOLERANCE:
        return "wholly below zero"
    if lower > TOLERANCE:
        return "wholly above zero"
    return "includes or touches zero"


def _comparison(aruns, left, right):
    seeds = _seeds(left)
    _require(set(seeds).issubset(_seeds(right)), "Contrast has unmatched seeds")
    result = {"left": left, "right": right, "n_seeds": len(seeds), "seeds": list(seeds)}
    for metric in ("mae", "rmse"):
        pair = paired_patient_change(
            [aruns[f"{left}_seed{seed}"]["metrics"]["test"] for seed in seeds],
            [aruns[f"{right}_seed{seed}"]["metrics"]["test"] for seed in seeds], metric)
        patient_changes = np.asarray(list(pair["patient_changes"].values()), float)
        seed_changes = np.asarray(pair["seed_macro_changes"], float)
        pair["target_details"] = {}
        for q, target in enumerate(TARGETS):
            pair["target_details"][target] = {
                "interval_relation": interval_relation(pair["ci_lower"][q], pair["ci_upper"][q]),
                "patients_lower_error": int(np.sum(patient_changes[:, q] < -TOLERANCE)),
                "patients_higher_error": int(np.sum(patient_changes[:, q] > TOLERANCE)),
                "patients_unchanged": int(np.sum(np.abs(patient_changes[:, q]) <= TOLERANCE)),
                "seeds_lower_macro_error": int(np.sum(seed_changes[:, q] < -TOLERANCE)),
                "seeds_higher_macro_error": int(np.sum(seed_changes[:, q] > TOLERANCE)),
                "seeds_unchanged": int(np.sum(np.abs(seed_changes[:, q]) <= TOLERANCE)),
            }
        result[metric] = pair
    return result


def _delta_text(pair):
    return "; ".join(f"{target} {pair['estimate'][q]:+.2f} "
                     f"[{pair['ci_lower'][q]:+.2f}, {pair['ci_upper'][q]:+.2f}]"
                     for q, target in enumerate(TARGETS))


def _direction_sentence(pair, left_name, right_name):
    relations = [pair["target_details"][target]["interval_relation"] for target in TARGETS]
    if relations == ["wholly below zero"] * 2:
        return f"Both descriptive MAE intervals favor {left_name} over {right_name} under the fixed fitting settings."
    if relations == ["wholly above zero"] * 2:
        return f"Both descriptive MAE intervals favor {right_name} over {left_name} under the fixed fitting settings."
    return ("The two targets do not show a uniform directional advantage with both intervals on one side of zero; "
            "this is not evidence of equivalence.")


def _secondary_tables(comparisons):
    tables = []
    for metric in ("mae", "rmse"):
        rows = []
        for left, right, role in SECONDARY_CONTRASTS:
            contrast = comparisons[f"{left}-minus-{right}"]
            pair = contrast[metric]
            cells = [r"\shortstack{" + f"{DISPLAY[left]} minus {DISPLAY[right]}" + r"\\{" + role + "}}",
                     str(contrast["n_seeds"])]
            for q in range(2):
                # Braces prevent '[' from being parsed as an optional row skip.
                cells.append(r"\shortstack{" + f"{pair['estimate'][q]:+.2f}" + r"\\{" +
                             f"[{pair['ci_lower'][q]:+.2f}, {pair['ci_upper'][q]:+.2f}]" + "}}")
            rows.append(" & ".join(cells) + r"\\")
        tables.append("\n".join([
            r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{lrrr}\toprule",
            r"Contrast (left minus right) & Seeds & SBP change [95\% CI] & DBP change [95\% CI]\\\midrule",
            *rows, r"\bottomrule\end{tabular}",
            f"\\caption{{Direct architectural {metric.upper()} contrasts on the primary test, in mmHg. "
            "Each estimate averages matched-seed differences within each of the 141 patients before a 2,000-resample patient bootstrap. "
            "Negative changes favor the left model. All intervals are descriptive, not multiplicity-adjusted or seed-population confidence intervals; "
            "single-seed modality contrasts condition on seed 42. These contrasts do not alter model selection.}\\end{table*}",
        ]))
    return "\n\n".join(tables)


def build_attention_interpretation(aruns, validation_winner=None):
    """Return (LaTeX, JSON-ready evidence) from complete verified real summaries.

    ``validation_winner``, if supplied, is the caller's all-six-required choice.
    It is checked against validation only. The main four-model attention choice
    is reported separately; optional ECA never enters either primary ranking.
    """
    variants = _validate(aruns)
    scores = {variant: float(np.mean([aruns[f"{variant}_seed{seed}"]["metrics"]["validation"]["macro"]["mae"]
                                     for seed in SEEDS])) for variant in REQUIRED_THREE_SEED}
    main_winner = min(ATTENTION_SET, key=scores.get)
    required_winner = min(REQUIRED_THREE_SEED, key=scores.get)
    _require(validation_winner is None or validation_winner == required_winner,
             "Supplied winner is not the fixed all-six-required validation choice")
    comparisons = {}
    for variant in (*ATTENTION_SET[1:], *(('eca_last',) if 'eca_last' in variants else ())):
        comparisons[f"{variant}-minus-baseline"] = _comparison(aruns, variant, "baseline")
    for left, right, _ in SECONDARY_CONTRASTS:
        comparisons[f"{left}-minus-{right}"] = _comparison(aruns, left, right)
    rankings = {"attention_candidates": list(ATTENTION_SET), "attention_winner": main_winner,
                "required_three_seed_candidates": list(REQUIRED_THREE_SEED), "required_winner": required_winner,
                "same_winner": main_winner == required_winner, "validation_mean_macro_mae": scores,
                "tie_rule": "first model in the stated fixed candidate order; exact score ties only",
                "selection_uses": "validation patient-macro MAE averaged over SBP/DBP and three seeds only",
                "supplementary_eca_excluded": True}
    details = {"selection": rankings, "comparisons": comparisons, "target_order": list(TARGETS),
               "primary_test_patients": 141, "primary_test_windows": 56400,
               "uncertainty": "2,000 patient resamples after within-patient matched-seed averaging; descriptive, not multiplicity-adjusted",
               "conditioning": "fixed fitted checkpoints and available offline-filtered cohort; does not sample a population of training seeds",
               "parameters": {variant: PARAMETERS[variant] for variant in variants}}
    paragraphs = [r"\paragraph{Validation choices.}",
        f"Within the four-model attention comparison (CNN, SE-last, Temporal and CBAM1D), validation selects "
        f"{DISPLAY[main_winner]} (mean SBP/DBP patient-macro MAE {scores[main_winner]:.2f} mmHg). "
        f"The separate required three-seed comparison, which also includes Additive and SE-all, selects "
        f"{DISPLAY[required_winner]} ({scores[required_winner]:.2f} mmHg). "
        + ("The choices coincide. " if main_winner == required_winner else "The choices differ; the larger set is not relabeled as the primary attention family. ")
        + "Both choices use validation only; capacity, placement and modality contrasts retain their distinct roles, and optional ECA is supplementary.",
        r"\paragraph{Attention effects and heterogeneity.}",
        "The following are primary-test MAE changes with descriptive paired 95\\% patient intervals, in mmHg, relative to the same-seed CNN. "
        "Negative values reduce error. SBP and DBP remain separate rather than being replaced by the selection scalar."]
    for variant in ATTENTION_SET[1:]:
        pair = comparisons[f"{variant}-minus-baseline"]["mae"]
        counts = [pair["target_details"][target] for target in TARGETS]
        paragraphs.append(f"{DISPLAY[variant]}: {_delta_text(pair)}. "
            f"SBP/DBP macro MAE is lower in {counts[0]['seeds_lower_macro_error']}/3 and {counts[1]['seeds_lower_macro_error']}/3 matched seeds; "
            f"seed-averaged patient MAE is lower for {counts[0]['patients_lower_error']}/141 and {counts[1]['patients_lower_error']}/141 patients. "
            + _direction_sentence(pair, DISPLAY[variant], "CNN"))
    paragraphs.extend([
        "Intervals spanning zero do not establish equivalence, while intervals excluding zero remain unadjusted for the multiple architecture/endpoint contrasts. "
        "These intervals condition on the fitted seeds: the separate seed counts and SDs describe training variability. "
        "An average gain is not a guarantee for every patient or pressure range, and no minimum clinically useful effect was prespecified.",
        r"\paragraph{Architectural controls.}",
    ])
    capacity = comparisons["se_last-minus-additive_last"]["mae"]
    placement = comparisons["se_all-minus-se_last"]["mae"]
    paragraphs.append(f"SE-last minus Additive changes MAE by {_delta_text(capacity)}. "
        + _direction_sentence(capacity, "SE-last", "Additive")
        + " Both contain 38,296 parameters, so parameter count alone cannot distinguish these specified operators. "
        "This is not proof that all attention mechanisms or all optimizers would behave similarly.")
    paragraphs.append(f"SE-all minus SE-last gives {_delta_text(placement)}. "
        + _direction_sentence(placement, "SE-all", "SE-last")
        + " SE-all inserts gates at three residual outputs and has 374 more parameters than SE-last; the contrast combines insertion depth with capacity, not a pure relocation. "
        "The direct CBAM1D-versus-SE and CBAM1D-versus-Temporal MAE/RMSE contrasts are tabulated below. "
        "CBAM1D also changes channel pooling from SE's average-only descriptor to shared average/max descriptors, so it is not a factorial test isolating the added temporal gate.")
    paragraphs.extend([r"\paragraph{Observed-sensor ablations.}",
        "The zero-channel models are retrained baselines, not missing-sensor tests of the trained two-sensor model. "
        + "; ".join(f"{DISPLAY[variant]} minus CNN MAE: {_delta_text(comparisons[f'{variant}-minus-baseline']['mae'])}"
                    for variant in MODALITIES) + ". "
        "ECG only removes PPG, whereas PPG only removes ECG. Their patient intervals condition on seed 42 and cannot establish seed-robust sensor importance or an attention-by-modality interaction. "
        "Learned feature-channel gates do not themselves identify physiological sensor importance.",
        "Accuracy gains must be weighed against the separately measured parameters and CPU/MPS latency; parameter overhead alone is not measured speed or evidence of native-device readiness.",
    ])
    if "eca_last" in variants:
        paragraphs.append(r"\paragraph{Supplementary ECA.} "
            + f"ECA minus CNN MAE: {_delta_text(comparisons['eca_last-minus-baseline']['mae'])}. "
            "It remains outside both primary validation rankings and does not replace a required comparison.")
    return "\n\n".join(paragraphs) + "\n\n" + _secondary_tables(comparisons), details
