#!/usr/bin/env python3
"""Run the v6.0-exp6.8.3 experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_8_3 import run_exp6_8_3


def main():
    parser = argparse.ArgumentParser(description="Run exp6.8.3")
    parser.add_argument("--model", default="A4_bootstrap_mlp",
                        help="Advantage model name (A0_zero, A1_linear, A2_ridge, "
                             "A3_mlp, A4_bootstrap_mlp, A5_quantile_mlp)")
    parser.add_argument("--n-train", type=int, default=200,
                        help="Training tasks per mechanism")
    parser.add_argument("--n-cal", type=int, default=50,
                        help="Calibration tasks per mechanism")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Test tasks per mechanism")
    args = parser.parse_args()

    print("=" * 70)
    print("v6.0-exp6.8.3: Conformal Structural Advantage")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can conformal calibration of an advantage predictor allow")
    print("  the learned planner to override the deterministic baseline")
    print("  only when there is calibrated evidence that it is better?")
    print()
    print("Architecture:")
    print("  A* = Q_H(S, a_learned) - Q_H(S, a_baseline)  [exact advantage]")
    print("  A_hat = f(state, actions, objective)           [advantage model]")
    print("  LCB_A = A_hat - q_{1-alpha}                    [conformal LCB]")
    print("  override only if LCB_A > 0                     [arbitration]")
    print("  alpha chosen on calibration, evaluated on test [no leak]")
    print()
    print("Risk metrics:")
    print("  - OverridePrecision = P(A* > 0 | override)")
    print("  - FalseOverrideRate = P(A* <= 0 | override)")
    print("  - OverrideCoverage = #overrides / #decisions")
    print("  - MeanOverrideAdvantage = E[A* | override]")
    print("  - CVaR95, P95, P99 regret")
    print("  - Confidence decile monotonicity")
    print("  - OOD coverage analysis")
    print()

    result = run_exp6_8_3(
        n_train_per_mechanism=args.n_train,
        n_calibration=args.n_cal,
        n_test=args.n_test,
        gamma=0.9,
        horizon=2,
        beam_width=3,
        model_name=args.model,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_8_3")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
