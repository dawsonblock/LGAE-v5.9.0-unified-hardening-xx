#!/usr/bin/env python3
"""Run the v6.0-exp6.8.2 experiment."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_8_2 import run_exp6_8_2


def main():
    print("=" * 70)
    print("v6.0-exp6.8.2: Calibrated Selective Planning")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can ensemble-based LCB-margin arbitration with calibrated")
    print("  kappa preserve non-greedy gains while eliminating tail risk?")
    print()
    print("Architecture:")
    print("  Ensemble of M=5 LearnedStateModel instances")
    print("  LCB(margin) = mu_margin - kappa * sigma_margin")
    print("  Use learned only if LCB(margin) > 0")
    print("  kappa chosen on calibration split, evaluated on locked test")
    print()
    print("Risk metrics:")
    print("  - CVaR95 (Conditional Value at Risk)")
    print("  - Median, P95, P99 regret")
    print("  - Uncertainty-error correlation")
    print("  - Monotonic risk-by-uncertainty deciles")
    print()

    result = run_exp6_8_2(
        n_train_per_mechanism=200,
        n_calibration=50,
        n_test=50,
        gamma=0.9,
        horizon=2,
        beam_width=3,
        n_ensemble=5,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_8_2")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
