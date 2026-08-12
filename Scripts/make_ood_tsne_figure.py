"""
3D t-SNE projection of the 384-dim sentence embeddings
(outputs/embedding_cache_1034.npz) that Phase 4's Mahalanobis-distance
method (Section 7, Eq. maha) is computed on directly, colored by
ID/OOD membership per setup. Answers visually what the Mahalanobis
AUROC numbers only summarize: do the ID and OOD populations actually
separate in this embedding space, or not.

external_corpus is excluded -- 20 Newsgroups has no entry in the
RansKnow-only embedding cache (same reason Table/Section 7 reports
Mahalanobis as unavailable for that setup).

Usage:
    python3 Scripts/make_ood_tsne_figure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "paper" / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase4_ood_detection as p4  # noqa: E402
from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.tasks import register_tasks  # noqa: E402


def main():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    df = load_features()
    register_tasks(df)
    embeddings = p4._load_embeddings()

    pre2024 = df[df["Year"] < 2024].reset_index(drop=True)
    post2024 = df[df["Year"] >= 2024].reset_index(drop=True)
    ood_mask = np.zeros(len(post2024), dtype=bool)
    for f in p4.POST2023_EMERGING_FAMILIES:
        ood_mask |= p4._has_family(post2024, f).to_numpy()
    ood_1 = post2024[ood_mask].reset_index(drop=True)
    id_1 = pre2024.reset_index(drop=True)

    id_2 = df[df["Channel_Type"].isin(["Vendor", "DFIR"])].reset_index(drop=True)
    ood_2 = df[df["Channel_Type"].isin(["Conference", "Independent"])].reset_index(drop=True)

    is_rare = p4._has_family(df, p4.RARE_FAMILY)
    id_3 = df[~is_rare].reset_index(drop=True)
    ood_3 = df[is_rare].reset_index(drop=True)

    setups = [
        ("Temporal emergence", id_1, ood_1),
        ("Channel-type shift", id_2, ood_2),
        ("Leave-rare-family-out", id_3, ood_3),
    ]

    fig = plt.figure(figsize=(14, 5))
    for i, (title, id_df, ood_df) in enumerate(setups):
        id_emb = np.stack([embeddings[vid] for vid in id_df["Video_ID"]])
        ood_emb = np.stack([embeddings[vid] for vid in ood_df["Video_ID"]])
        all_emb = np.vstack([id_emb, ood_emb])
        labels = np.array([0] * len(id_emb) + [1] * len(ood_emb))

        perplexity = min(30, max(5, (len(all_emb) - 1) // 3))
        proj = TSNE(n_components=3, perplexity=perplexity, random_state=0,
                    init="pca", learning_rate="auto").fit_transform(all_emb)

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(*proj[labels == 0].T, c="#2980b9", s=14, alpha=0.6, label="ID", depthshade=True)
        ax.scatter(*proj[labels == 1].T, c="#c0392b", s=22, alpha=0.85, label="OOD",
                   edgecolor="black", linewidth=0.3, depthshade=True)
        ax.set_title(f"{title}\n(n_ID={len(id_emb)}, n_OOD={len(ood_emb)})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=18, azim=-60)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle("3D t-SNE of the 384-dim sentence embeddings, colored by ID/OOD membership "
                  "(Section 7's Mahalanobis-distance input space)", fontsize=10, y=1.03)
    fig.tight_layout()
    out = FIGDIR / "fig_ood_tsne_3d.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
