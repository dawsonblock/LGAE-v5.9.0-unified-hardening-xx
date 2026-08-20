#!/usr/bin/env python3
"""Run the v6.0-exp6.8.4 experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_8_4 import run_exp6_8_4


def main():
    parser = argparse.ArgumentParser(description="Run exp6.8.4")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-cal", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--models", nargs="+", default=["M1_ridge", "M2_gbt", "M3_mlp", "M4_pairwise"])
    parser.add_argument("--targets", nargs="+", default=["T1_raw", "T2_normalized", "T3_sign", "T4_ordinal", "T5_downside"])
    parser.add_argument("--features", nargs="+", default=["F1_current", "F4_full"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000])
    parser.add_argument("--mechanisms", nargs="+", default=["connectivity_threshold", "redundancy_threshold"])
    args = parser.parse_args()

    print("=" * 70)
    print("v6.0-exp6.8.4: Advantage Model Identification")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Is structural advantage A* = Q_L - Q_B actually learnable")
    print("  from information available at decision time?")
    print()
    print("Four-axis sweep:")
    print("  Target:   raw, normalized, sign, ordinal, downside-adjusted")
    print("  Features: current, full (action effects + topology + global)")
    print("  Model:    ridge, GBT, MLP, pairwise")
    print("  Data:     250, 500, 1k examples/mechanism")
    print()
    print("Success: one combination shows a clear learning curve")
    print("         AND gives positive mean advantage with tail-risk control")
    print()

    result = run_exp6_8_4(
        n_train_per_mechanism=args.n_train,
        n_calibration=args.n_cal,
        n_test=args.n_test,
        data_sizes=args.sizes,
        models=args.models,
        targets=args.targets,
        feature_levels=args.features,
        mechanisms=args.mechanisms,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_8_4")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
