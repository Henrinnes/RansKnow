"""
Generates the paper's Phase 2/3/4 figures directly from
outputs/phase{2,3,4}_*.json -- no numbers are hand-copied into plotting
code, so a re-run after any pipeline change regenerates figures that
stay consistent with the tables in main.tex.

Usage:
    python3 Scripts/make_paper_figures.py
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150,
})

TASK_ORDER = ["relevance", "dominant_tactic", "platform", "tool", "family"]
TASK_LABEL = {"relevance": "Relevance", "dominant_tactic": "Dom. tactic",
              "platform": "Platform", "tool": "Tool", "family": "Family"}


def fig_reliability():
    d = json.loads((ROOT / "outputs" / "phase2_calibration.json").read_text())
    tasks = [t for t in TASK_ORDER if not d[t]["calibration"].get("skipped")]

    fig, axes = plt.subplots(1, len(tasks), figsize=(2.3 * len(tasks), 2.5), sharey=True)
    for ax, t in zip(axes, tasks):
        bins = d[t]["calibration"]["reliability_bins"]
        centers, acc, conf = [], [], []
        for b in bins:
            if b["n"] == 0:
                continue
            centers.append((b["lo"] + b["hi"]) / 2)
            acc.append(b["accuracy"])
            conf.append(b["avg_confidence"])
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
        ax.plot(conf, acc, "o-", color="#c0392b", markersize=3, label="observed")
        ax.set_title(f"{TASK_LABEL[t]}\nECE={d[t]['calibration']['ece']:.3f}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("agreement with\nweak label")
    axes[-1].legend(loc="upper left", fontsize=6, frameon=False)
    fig.suptitle("Reliability diagrams against weak labels (Section 5)", y=1.05, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig_uncertainty_reliability.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def fig_conformal_coverage():
    d = json.loads((ROOT / "outputs" / "phase3_conformal.json").read_text())
    tasks = ["dominant_tactic", "platform", "family", "tool"]
    labels = [TASK_LABEL[t] for t in tasks]

    marginal, post2025, conference = [], [], []
    for t in tasks:
        v = d[t]
        if "marginal_coverage_pooled" in v:
            marginal.append(v["marginal_coverage_pooled"])
            post2025.append(v["slice_post2025"]["coverage"])
            conference.append(v["slice_conference"]["coverage"])
        else:
            marginal.append(v["marginal_all_labels_jointly_covered_pooled"])
            post2025.append(v["slice_post2025_joint"]["coverage"])
            conference.append(v["slice_conference_joint"]["coverage"])

    x = np.arange(len(tasks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.2, 3))
    ax.bar(x - w, marginal, w, label="marginal (pooled test)", color="#2c3e50")
    ax.bar(x, post2025, w, label="slice: Year$\\geq$2026", color="#2980b9")
    ax.bar(x + w, conference, w, label="slice: Conference channels", color="#e67e22")
    ax.axhline(0.90, color="k", linestyle="--", linewidth=0.8, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("Split-conformal coverage: marginal vs. conditional slices (Section 6)")
    ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    fig.tight_layout()
    out = FIGDIR / "fig_conformal_coverage.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def fig_ood_radar():
    d = json.loads((ROOT / "outputs" / "phase4_ood.json").read_text())
    setups = ["temporal_emergence", "channel_type_shift", "leave_rare_family_out", "external_corpus"]
    setup_label = {"temporal_emergence": "Temporal\nemergence", "channel_type_shift": "Channel-type\nshift",
                   "leave_rare_family_out": "Leave-rare-\nfamily-out", "external_corpus": "External\ncorpus"}
    methods = ["max_softmax_probability", "energy", "mahalanobis", "conformal_nonconformity"]
    method_label = {"max_softmax_probability": "MSP", "energy": "Energy",
                     "mahalanobis": "Mahalanobis", "conformal_nonconformity": "Conformal"}
    colors = {"max_softmax_probability": "#2980b9", "energy": "#27ae60",
              "mahalanobis": "#c0392b", "conformal_nonconformity": "#8e44ad"}

    n = len(setups)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    for m in methods:
        vals = []
        for s in setups:
            v = d[s]["auroc"].get(m)
            vals.append(v if v is not None else 0.0)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.4, markersize=3, label=method_label[m], color=colors[m])
        ax.fill(angles, vals, alpha=0.06, color=colors[m])

    chance = [0.5] * (n + 1)
    ax.plot(angles, chance, "k--", linewidth=0.8, label="chance (0.5)")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([setup_label[s] for s in setups])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("OOD detection AUROC by method and setup (Section 7)", y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7, frameon=False)
    fig.tight_layout()
    out = FIGDIR / "fig_ood_radar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def fig_baselines_full_sweep():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "outputs" / "experiments" / "phase1_*.csv")))]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["model"] != "rule_based_ka"]  # circular baseline, excluded from every reported comparison

    splits = ["stratified_random", "channel_grouped", "temporal"]
    split_label = {"stratified_random": "Stratified random", "channel_grouped": "Channel-grouped",
                   "temporal": "Temporal (train pre-2024)"}
    models = ["logreg", "svm", "random_forest", "gbm"]
    model_label = {"logreg": "LogReg", "svm": "SVM", "random_forest": "RF", "gbm": "GBM"}

    # pcolormesh + a dedicated colorbar axis (width_ratios), not a shared
    # multi-axes fig.colorbar(ax=axes) call -- the latter produced a
    # reproducible diagonal rendering artifact across the middle panel
    # (present in the saved PDF itself, not just a viewer/DPI quirk;
    # confirmed by re-rendering at 300 DPI) that this avoids entirely.
    fig, axes = plt.subplots(1, 4, figsize=(11.3, 3.4),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    cax = axes[3]
    im = None
    for ax, split in zip(axes[:3], splits):
        sub = df[df["split"] == split]
        best = sub.groupby(["task", "model"])["macro_f1"].max().unstack()
        best = best.reindex(index=TASK_ORDER, columns=models)
        n_rows, n_cols = best.shape
        im = ax.pcolormesh(best.to_numpy(), vmin=0, vmax=1, cmap="YlOrRd",
                            edgecolors="white", linewidth=1.5)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels([model_label[m] for m in models])
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels([TASK_LABEL[t] for t in TASK_ORDER] if split == splits[0] else [])
        ax.set_title(split_label[split])
        for i in range(n_rows):
            for j in range(n_cols):
                v = best.iloc[i, j]
                if pd.notna(v):
                    ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 0.6 else "white")
    fig.colorbar(im, cax=cax, label="macro-F1 (best representation)")
    fig.suptitle("Full baseline sweep: best-representation macro-F1 by task, model, and CV split "
                  "(weak-label agreement, Section 4.3)", y=1.03, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig_baselines_full_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def fig_baselines_3d():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    dfs = [pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "outputs" / "experiments" / "phase1_*.csv")))]
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["model"] != "rule_based_ka") & (df["split"] == "stratified_random")]
    models = ["logreg", "svm", "random_forest", "gbm"]
    model_label = {"logreg": "LogReg", "svm": "SVM", "random_forest": "RF", "gbm": "GBM"}
    best = df.groupby(["task", "model"])["macro_f1"].max().unstack().reindex(index=TASK_ORDER, columns=models)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    n_tasks, n_models = best.shape
    xpos, ypos = np.meshgrid(np.arange(n_models), np.arange(n_tasks))
    xpos, ypos = xpos.ravel(), ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    dz = best.to_numpy().ravel()
    dx = dy = 0.7

    cmap = plt.get_cmap("YlOrRd")
    colors = cmap(dz / dz.max())

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(n_models) + dx / 2)
    ax.set_xticklabels([model_label[m] for m in models])
    ax.set_yticks(np.arange(n_tasks) + dy / 2)
    ax.set_yticklabels([TASK_LABEL[t] for t in TASK_ORDER])
    ax.set_zlabel("macro-F1")
    ax.set_zlim(0, 1)
    ax.view_init(elev=25, azim=-50)
    ax.set_title("Baseline sweep in 3D: best-representation macro-F1 by task and model\n"
                  "(stratified-random split, weak-label agreement, Section 4.3)", fontsize=9, y=1.0)
    fig.tight_layout()
    out = FIGDIR / "fig_baselines_3d.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    fig_baselines_full_sweep()
    fig_baselines_3d()
    fig_reliability()
    fig_conformal_coverage()
    fig_ood_radar()
