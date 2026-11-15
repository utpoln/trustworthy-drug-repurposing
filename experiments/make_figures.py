"""Paper figures (PDF, vector) from results/tables/results.json.

Run make_results first. Writes paper/figures/fig_{accuracy,calibration,faithfulness,poisoning}.pdf

Palette: validated categorical slots (blue, orange, aqua; dataviz reference palette),
diverging blue / gray / red for supported / partial / unsupported.

Usage: python -m experiments.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.make_results import MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
RES = json.loads((ROOT / "results" / "tables" / "results.json").read_text())

BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"
SERIES5 = [BLUE, ORANGE, AQUA, "#eda100", MAGENTA]  # fixed slot order 1-5
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SUP, PART, UNS = "#2a78d6", "#c3c2b7", "#e34948"
NAMES = dict(MODELS)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ accuracy
def fig_accuracy():
    acc = RES["accuracy"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    y = np.arange(len(MODELS))[::-1]
    for (slug, name), yi in zip(MODELS, y):
        for mode, col, off, lab in (("no_paths", ORANGE, 0.14, "LLM, no paths"),
                                    ("grounded", BLUE, -0.14, "LLM + KG paths")):
            d = acc[slug][mode]["dKGE"]
            ax.errorbar(d["mean"], yi + off, xerr=[[d["mean"] - d["ci"][0]], [d["ci"][1] - d["mean"]]],
                        fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=1.4, capsize=0,
                        label=lab if yi == y[0] else None, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [n for _, n in MODELS])
    ax.set_xlabel(r"$\Delta$MRR vs. RotatE (95% CI)")
    grid(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7, handletextpad=0.3)
    fig.savefig(OUT / "fig_accuracy.pdf"); plt.close(fig)


# ------------------------------------------------------------------ calibration
def fig_calibration():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for (slug, name), col in zip(MODELS, SERIES5):
        rel = RES["accuracy"][slug]["calibration"]["grounded"]["reliability"]
        pts = [(r["mean_conf"], r["frac_gold"]) for r in rel if r["n"] >= 30]
        x, yv = zip(*pts)
        a.plot(x, yv, "-o", color=col, lw=2, ms=3.5, label=name, zorder=3)
    a.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8)
    a.text(0.62, 0.68, "perfect calibration", rotation=37, color=INK2, fontsize=6.5)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel("Mean stated confidence (bin)"); a.set_ylabel("Fraction gold indications")
    a.set_title("(a) Reliability, grounded LLMs (raw)", loc="left")
    grid(a, "both"); a.legend(fontsize=6.5, loc="upper left")

    rc = RES["recalibration"]
    maps = [("raw", "none", INK2), ("temperature", "temperature", ORANGE), ("platt", "Platt", BLUE)]
    w = 0.26; x = np.arange(len(MODELS))
    for i, (key, lab, col) in enumerate(maps):
        v = [rc[s][key]["ECE"] for s, _ in MODELS]
        b.bar(x + (i - 1) * (w + 0.02), v, w, color=col, label=lab, zorder=3)
    b.set_xticks(x, [n.replace("-", "\n", 1) for _, n in MODELS], fontsize=7)
    b.set_ylabel("Test ECE (10 bins)")
    b.set_title("(b) ECE after maps fitted on validation", loc="left")
    b.set_ylim(0, 0.36)
    grid(b, "y"); b.legend(fontsize=7, ncol=3, loc="upper left", title="recalibration map", title_fontsize=7,
                           alignment="left")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig_calibration.pdf"); plt.close(fig)


# ------------------------------------------------------------------ faithfulness
def fig_faithfulness():
    f, v2 = RES["faithfulness"]["models"], RES["prompt_v2"]
    rows = []
    for slug, name in MODELS:
        rows.append((f"{name}", "Claims", f[slug]["share"]["claim"]))
    for slug, name in MODELS:
        rows.append((f"{name}", "Mechanism, prompt v1", f[slug]["share"]["mechanism"]))
    for slug in v2:
        rows.append((NAMES[slug], "Mechanism, prompt v2", v2[slug]["faith_v2"]["share"]["mechanism"]))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ypos, labels, y, prev = [], [], 0, None
    for name, grp, sh in rows:
        if grp != prev and prev is not None:
            y += 0.7
        prev = grp
        ypos.append(y); labels.append(name); y += 1
    ypos = -np.array(ypos)
    for yi, (name, grp, sh) in zip(ypos, rows):
        left = 0
        for k, col in (("SUPPORTED", SUP), ("PARTIAL", PART), ("UNSUPPORTED", UNS)):
            val = sh.get(k, 0)
            ax.barh(yi, val, 0.78, left=left, color=col, edgecolor="white", lw=1, zorder=3)
            if val >= 0.06:
                ax.text(left + val / 2, yi, f"{100 * val:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if col != PART else INK)
            left += val
    ax.set_yticks(ypos, labels, fontsize=7)
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    # group labels on the right
    grp_names = []
    for g in dict.fromkeys(r[1] for r in rows):
        idx = [i for i, r in enumerate(rows) if r[1] == g]
        grp_names.append((g, ypos[idx].mean()))
    for g, yy in grp_names:
        ax.text(1.01, yy, g, va="center", ha="left", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SUP, label="supported"), Patch(color=PART, label="partial"),
                       Patch(color=UNS, label="unsupported")], ncol=3, fontsize=7,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_xlabel(f"Share of statements (LLM judge: {RES['faithfulness']['judge']})")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "fig_faithfulness.pdf"); plt.close(fig)


# ------------------------------------------------------------------ poisoning
def fig_poisoning():
    rob = RES["robustness"]
    rows = [("RotatE (retrained)", rob["kge"]["targeted (n=3)"]["decoy"]["top1_clean"],
             rob["kge"]["targeted (n=3)"]["decoy"]["top1_poisoned"])]
    rows += [(f"{n} + paths", *rob["llm_targeted"][s]["decoy_top1"]) for s, n in MODELS]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y = np.arange(len(rows))[::-1]
    for (name, c, p), yi in zip(rows, y):
        ax.plot([c, p], [yi, yi], color=GRID, lw=2.2, zorder=2, solid_capstyle="round")
        ax.plot(c, yi, "o", color=INK2, ms=5, zorder=3, label="clean graph" if yi == y[0] else None)
        ax.plot(p, yi, "o", color=UNS, ms=5, zorder=3, label="3 fake chains / decoy" if yi == y[0] else None)
        ax.text(p + 0.006, yi, f"{100 * p:.0f}%", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 0.24); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Decoy drug ranked #1 (share of test diseases)")
    grid(ax); ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(OUT / "fig_poisoning.pdf"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_accuracy, fig_calibration, fig_faithfulness, fig_poisoning):
        f(); print("wrote", f.__name__)


if __name__ == "__main__":
    main()
