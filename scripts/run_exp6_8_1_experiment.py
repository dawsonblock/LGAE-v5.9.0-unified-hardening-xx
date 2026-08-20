#!/usr/bin/env python3
"""Run the v6.0-exp6.8.1 experiment."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_8_1 import run_exp6_8_1


def main():
    print("=" * 70)
    print("v6.0-exp6.8.1: Selective Hybrid Structural Planning")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can selective arbitration preserve non-greedy gains while")
    print("  preventing the learned planner from overriding strong")
    print("  deterministic decisions when its prediction is unreliable?")
    print()
    print("Architecture:")
    print("  ExactState: components, degrees, topology (exact)")
    print("  CertifiedApproxState: spectral gap, resistance, curvature (deterministic)")
    print("  LearnedState: path length, efficiency, future opportunity (learned)")
    print()
    print("Arbitration:")
    print("  use learned only if sigma < tau_sigma AND margin > tau_margin")
    print("  otherwise: fall back to greedy")
    print()
    print("Risk-aware metrics:")
    print("  - NormalizedPlanningRegret (primary)")
    print("  - MedianRegret, P95Regret, P99Regret")
    print("  - P(Regret > tau)")
    print("  - Coverage-vs-risk curve")
    print()

    result = run_exp6_8_1(
        n_train_per_mechanism=200,
        n_target_suboptimal=100,
        gamma=0.9,
        horizon=2,
        beam_width=3,
        tau_sigma=2.0,
        tau_margin=0.5,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_8_1")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
