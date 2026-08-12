"""
3D ridgeline (distribution) figure for Phase 4's OOD detection results --
plots the actual ID-test vs. OOD max-softmax-probability score
*distributions* per setup, not just the summary AUROC number the radar
chart (fig_ood_radar.pdf) already shows. Reuses phase4_ood_detection.py's
setup logic directly rather than re-deriving it, so the underlying
train/ID/OOD splits are identical to what produced Section 7's numbers.

Usage:
    python3 Scripts/make_ood_distribution_figure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "paper" / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase4_ood_detection as p4  # noqa: E402
from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402


def _msp(train_df, id_test_df, ood_df):
    est, builder, _ = p4._fit_classifier(train_df)
    id_msp, _, _ = p4._msp_and_energy(est, builder, id_test_df)
    ood_msp, _, _ = p4._msp_and_energy(est, builder, ood_df)
    return id_msp, ood_msp


def _external_corpus_msp(df):
    task = TASKS.get(p4.TASK_NAME)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, random_state=p4.SEED)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    id_test_df = df.iloc[test_idx].reset_index(drop=True)

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    train_texts = [load_transcript(p) for p in train_df["Transcript_Path"]]
    X_train = vec.fit_transform(train_texts)
    y_train = task.build_y(train_df)
    est = MODELS.get("gbm").build(task.label_type, task)
    est.fit(X_train, y_train)

    id_texts = [load_transcript(p) for p in id_test_df["Transcript_Path"]]
    X_id = vec.transform(id_texts)
    news = fetch_20newsgroups(subset="train", categories=["rec.sport.baseball", "sci.space"],
                               remove=("headers", "footers", "quotes"))
    X_ood = vec.transform(news.data)

    id_msp = 1 - est.predict_proba(X_id).max(axis=1)
    ood_msp = 1 - est.predict_proba(X_ood).max(axis=1)
    return id_msp, ood_msp


def main():
    df = load_features()
    register_tasks(df)

    print("Recomputing raw MSP score distributions per OOD setup...")

    pre2024 = df[df["Year"] < 2024].reset_index(drop=True)
    post2024 = df[df["Year"] >= 2024].reset_index(drop=True)
    ood_mask = np.zeros(len(post2024), dtype=bool)
    for f in p4.POST2023_EMERGING_FAMILIES:
        ood_mask |= p4._has_family(post2024, f).to_numpy()
    ood_df = post2024[ood_mask].reset_index(drop=True)
    train_idx, test_idx = train_test_split(np.arange(len(pre2024)), test_size=0.25, random_state=p4.SEED)
    id_msp_1, ood_msp_1 = _msp(pre2024.iloc[train_idx].reset_index(drop=True),
                                pre2024.iloc[test_idx].reset_index(drop=True), ood_df)

    id_pool = df[df["Channel_Type"].isin(["Vendor", "DFIR"])].reset_index(drop=True)
    ood_pool = df[df["Channel_Type"].isin(["Conference", "Independent"])].reset_index(drop=True)
    train_idx, test_idx = train_test_split(np.arange(len(id_pool)), test_size=0.25, random_state=p4.SEED)
    id_msp_2, ood_msp_2 = _msp(id_pool.iloc[train_idx].reset_index(drop=True),
                                id_pool.iloc[test_idx].reset_index(drop=True), ood_pool)

    is_rare = p4._has_family(df, p4.RARE_FAMILY)
    id_pool3 = df[~is_rare].reset_index(drop=True)
    ood_pool3 = df[is_rare].reset_index(drop=True)
    train_idx, test_idx = train_test_split(np.arange(len(id_pool3)), test_size=0.25, random_state=p4.SEED)
    id_msp_3, ood_msp_3 = _msp(id_pool3.iloc[train_idx].reset_index(drop=True),
                                id_pool3.iloc[test_idx].reset_index(drop=True), ood_pool3)

    id_msp_4, ood_msp_4 = _external_corpus_msp(df)

    setups = [
        ("Temporal\nemergence", id_msp_1, ood_msp_1),
        ("Channel-type\nshift", id_msp_2, ood_msp_2),
        ("Leave-rare-\nfamily-out", id_msp_3, ood_msp_3),
        ("External\ncorpus", id_msp_4, ood_msp_4),
    ]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xs = np.linspace(0, 1, 200)
    for i, (label, id_scores, ood_scores) in enumerate(setups):
        for scores, color, offset in [(id_scores, "#2980b9", -0.18), (ood_scores, "#c0392b", 0.18)]:
            scores = np.asarray(scores)
            scores = scores[np.isfinite(scores)]
            if len(scores) < 2 or np.std(scores) < 1e-6:
                continue
            kde = gaussian_kde(scores, bw_method=0.25)
            density = kde(xs)
            density = density / density.max() * 0.8  # normalize per-curve height for comparability
            y = np.full_like(xs, i + offset)

            ax.plot(xs, y, density, color=color, linewidth=1.3)
            # Ribbon fill down to the baseline, as a single 3D polygon
            verts = [list(zip(xs, y, density)) + [(xs[-1], y[-1], 0), (xs[0], y[0], 0)]]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=color, alpha=0.35, edgecolor="none"))

    ax.set_xlabel("max-softmax-probability score (1 - top-class confidence)")
    ax.set_yticks(range(len(setups)))
    ax.set_yticklabels([s[0] for s in setups])
    ax.set_zlabel("density (normalized)")
    ax.set_zlim(0, 1)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("OOD score distributions by setup: in-distribution (blue) vs.\n"
                  "out-of-distribution (red) -- Section 7", fontsize=9)

    # Legend proxy
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#2980b9", lw=2, label="ID-test"),
               Line2D([0], [0], color="#c0392b", lw=2, label="OOD")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    out = FIGDIR / "fig_ood_distributions_3d.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
