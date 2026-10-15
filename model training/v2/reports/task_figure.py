"""Vector task schematic for the revised, contemporaneous Track B protocol.

The diagram describes information flow only: it contains no performance values,
synthetic waveforms or claim that later signals are forecast. Source: PROTOCOL.md.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build_task_figure(out):
    """Write figures/personal_task.pdf under ``out`` and return its path."""
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    destination = figures / "personal_task.pdf"
    colors = {"ink": "#26323D", "line": "#566775", "input": "#EAF0F6",
              "model": "#E6F1EE", "reference": "#FBF0DE", "reference_line": "#987033",
              "neutral": "#F3F4F5", "border": "#A2ADB5"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
        ax.set_xlim(0, 10.25)
        ax.set_ylim(0, 6.15)
        ax.axis("off")

        def node(x, y, width, height, text, fill="neutral", fontsize=9):
            patch = FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.045,rounding_size=0.08",
                                  linewidth=.8, edgecolor=colors["border"], facecolor=colors[fill])
            ax.add_patch(patch)
            ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                    fontsize=fontsize, linespacing=1.3, color=colors["ink"])

        def arrow(start, end, *, dashed=False, color="line", connectionstyle="arc3"):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                         linewidth=1.1, color=colors[color],
                                         linestyle=(0, (3, 2)) if dashed else "solid",
                                         connectionstyle=connectionstyle, shrinkA=1, shrinkB=1))

        ax.text(.1, 5.99, "Separate population training", ha="left", va="center",
                fontsize=10, weight="bold", color=colors["ink"])
        node(.1, 5.03, 2.9, .68, "ECG/PPG + scalar BP\nDisjoint population patients", "input")
        node(3.73, 5.03, 2.55, .68, "Population CNN\nUnchanged baseline", "model")
        ax.text(8.41, 5.37, "b: end of the first 20%\nof elapsed case span", ha="center", va="center",
                fontsize=8.5, linespacing=1.3, color=colors["line"])
        arrow((3.06, 5.37), (3.65, 5.37))
        arrow((5.005, 4.96), (5.005, 4.59))
        ax.text(5.18, 4.79, "initialize", fontsize=8, va="center", color=colors["line"])

        ax.text(.1, 4.82, "Early support: first 20% elapsed", ha="left", va="center",
                fontsize=8.7, weight="bold", color=colors["ink"])
        node(.1, 3.52, 2.9, 1.0,
             "Observed ECG/PPG\n+ ABP-derived SBP/DBP\nK = 1, 2, 3, 5, 10\nWhole windows end at/before b",
             "input", fontsize=8.3)
        node(3.73, 3.52, 2.55, 1.0, "Personal calibration\nSupport references only\nDevelopment-fixed settings", "model", fontsize=8.5)
        arrow((3.06, 4.02), (3.65, 4.02))
        node(7.06, 3.52, 2.7, 1.0, "Later reference SBP/DBP\nABP-derived labels\nEvaluator only", "reference", fontsize=8.6)

        ax.text(.1, 2.99, "Later queries: start at/after b + 10 s", ha="left", va="center",
                fontsize=8.7, weight="bold", color=colors["ink"])
        ax.text(.1, 2.76, "10-s guard; same queries for every K", ha="left", va="center",
                fontsize=8.1, color=colors["line"])
        arrow((5.005, 3.45), (5.005, 2.61))
        ax.text(5.18, 3.01, "calibrated\nparameters", fontsize=8, va="center", color=colors["line"])
        node(.1, 1.71, 2.9, .8, "Later observed ECG/PPG\nCurrent 10-second window", "input")
        node(3.73, 1.71, 2.55, .8, "Personalized estimator\nNo query-time fitting", "model")
        node(7.06, 1.71, 2.7, .8, "Predicted SBP / DBP\nSame observed window", "model")
        arrow((3.06, 2.11), (3.65, 2.11))
        arrow((6.35, 2.11), (6.98, 2.11))

        node(7.06, .75, 2.7, .53, "Compare with\nheld-out reference", "neutral", fontsize=8.2)
        arrow((8.41, 1.64), (8.41, 1.35))
        # Evaluation labels travel outside the inference path. There is no
        # reference-label arrow to adaptation or to the personalized estimator.
        ax.plot([9.83, 10.03, 10.03], [4.02, 4.02, 1.015], color=colors["reference_line"],
                linewidth=1.1, linestyle=(0, (3, 2)))
        arrow((10.03, 1.015), (9.83, 1.015), dashed=True, color="reference_line")
        ax.text(.1, 1.10, "Later BP never enters calibration or inference.", ha="left", va="center",
                fontsize=8.4, color=colors["ink"])
        ax.text(.1, .43, "ABP-derived sparse-reference simulation; no actual cuff measurements in these local files.",
                ha="left", va="center", fontsize=8.1, color=colors["ink"])
        ax.text(.1, .18, "Elapsed-time split retains gaps. Inputs are the provided offline-filtered ECG/PPG fields.",
                ha="left", va="center", fontsize=8.1, color=colors["line"])
        fig.savefig(destination, format="pdf", metadata={"Title": "Personal calibration and later contemporaneous BP estimation",
                                                        "Subject": "Frozen Bad-Blood v2 Track B information flow; source PROTOCOL.md",
                                                        "Creator": "Bad-Blood v2 reports/task_figure.py"})
        plt.close(fig)
    return destination


if __name__ == "__main__":
    print(build_task_figure(Path(__file__).resolve().parent))
