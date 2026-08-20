#!/usr/bin/env python3
"""Run the v6.0-exp6.8.5 experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_8_5 import run_exp6_8_5


def main():
    parser = argparse.ArgumentParser(description="Run exp6.8.5")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-cal", type=int, default=80)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000, 2000])
    parser.add_argument("--features", nargs="+", default=["F1_current", "F4_full"])
    parser.add_argument("--mechanisms", nargs="+", default=["connectivity_threshold", "redundancy_threshold"])
    args = parser.parse_args()

    print("=" * 70)
    print("v6.0-exp6.8.5: Full Structural Advantage Features")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Do F4 (full structural) features break the F1 ceiling?")
    print("  Does held-out Spearman improve as N increases under F4?")
    print()
    print("Hard stop condition:")
    print("  If F4 materially improves: integrate and freeze planner")
    print("  If F4 plateaus: freeze anyway, move to exp7-real-ai-topology")
    print()

    result = run_exp6_8_5(
        n_train_per_mechanism=args.n_train,
        n_calibration=args.n_cal,
        n_test=args.n_test,
        data_sizes=args.sizes,
        feature_levels=args.features,
        mechanisms=args.mechanisms,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_8_5")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nDecision: {result.decision}")
    print(f"All gates passed: {result.all_gates_passed}")

    return 0  # always return 0 — the decision is the output, not pass/fail


if __name__ == "__main__":
    sys.exit(main())
