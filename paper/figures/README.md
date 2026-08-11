# Figures

Empty for now — the paper has no real results to plot yet (see the
`[TODO]` markers in `main.tex` from Section 4 onward). Expected first
occupants, in the order they'll actually become available:

- **Section 4 (Task Definitions & Baselines)** — a grouped bar chart of
  macro-F1 across the 5 tasks x 5 models x 3 CV protocols, once the full
  smoke test (structured + TF-IDF + embedding) is gold-set-verified, not
  just weak-label. Probably also a confusion matrix for `dominant_tactic`
  and `platform` (the two multiclass tasks).
- **Section 5 (Uncertainty)** — reliability diagrams (predicted confidence
  vs. empirical accuracy) per task/model, scored against the gold set.
- **Section 6 (Conformal Prediction)** — coverage-vs-set-size plots, and
  the conditional-coverage-by-slice bar chart (gold subset / post-2025 /
  conference-channel).
- **Section 7 (OOD Detection)** — ROC curves for the 4 OOD setups x 4
  methods (max-softmax, Mahalanobis, energy, conformal-nonconformity).

Naming convention once files land: `fig_<section>_<short-description>.pdf`
(vector PDF preferred over PNG/JPG for anything that's a plot, not a
screenshot).
