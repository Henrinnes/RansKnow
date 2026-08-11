"""
RansKnow experiment CLI -- Phase 1 baseline ladder runner.

Usage:
    python3 Scripts/run_experiment.py --tasks family dominant_tactic \
        --features structured tfidf --models logreg random_forest gbm rule_based_ka \
        --splits stratified_random channel_grouped temporal

    python3 Scripts/run_experiment.py --list      # show registered tasks/features/models/splits
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.registry import FEATURES, MODELS, SPLITS, TASKS  # noqa: E402
from rk_pipeline.runner import run  # noqa: E402
from rk_pipeline.tasks import register_tasks  # noqa: E402


def list_registered():
    df = load_features()
    register_tasks(df)
    for reg_name, reg in [("tasks", TASKS), ("features", FEATURES), ("models", MODELS), ("splits", SPLITS)]:
        print(f"\n{reg_name}:")
        for name in reg.names():
            spec = reg.get(name)
            print(f"  {name:20} {getattr(spec, 'describe', '')}")


def main():
    parser = argparse.ArgumentParser(description="RansKnow Phase 1 experiment runner")
    parser.add_argument("--tasks", nargs="+",
                         default=["family", "dominant_tactic", "platform", "tool", "relevance"])
    parser.add_argument("--features", nargs="+", default=["structured", "tfidf"])
    parser.add_argument("--models", nargs="+", default=["logreg", "random_forest", "gbm"])
    parser.add_argument("--splits", nargs="+",
                         default=["stratified_random", "channel_grouped", "temporal"])
    parser.add_argument("--out-name", default="results")
    parser.add_argument("--list", action="store_true", help="List registered components and exit")
    args = parser.parse_args()

    if args.list:
        list_registered()
        return

    run(args.tasks, args.features, args.models, args.splits, out_name=args.out_name)


if __name__ == "__main__":
    main()
