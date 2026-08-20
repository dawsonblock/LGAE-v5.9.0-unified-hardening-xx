#!/usr/bin/env python3
"""Run the v6.0-exp6.8 experiment."""
import sys
import os
import json

# Add src to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_8 import run_exp6_8


def main():
    print("=" * 70)
    print("v6.0-exp6.8: Exact-Transition Model-Based Structural Planning")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Does recursively rolling the causal model with exact graph")
    print("  transitions recover non-greedy actions?")
    print()
    print("Architecture:")
    print("  G_{t+1} = T_exact(G_t, a_t)     [exact graph transition]")
    print("  z_{t+1} = F(G_t, z_t, a_t)      [learned consequential state]")
    print()
    print("Four systems compared:")
    print("  1. Greedy: exact, no foresight")
    print("  2. Exact MPC: exact, exact foresight")
    print("  3. One-step causal: exact, one-step learned")
    print("  4. Recursive causal MPC: exact, multi-step learned")
    print()
    print("Metrics:")
    print("  - NonGreedyRecoveryRate (with ActionIdentity)")
    print("  - NormalizedRegret = |Q* - Q_model| / (|Q*| + eps)")
    print("  - Search savings")
    print("  - Error by horizon: E_1, E_2, E_3")
    print("  - Teacher-forced vs free rollout")
    print()

    result = run_exp6_8(
        n_train_per_mechanism=200,
        n_target_suboptimal=100,
        gamma=0.9,
        horizon=2,
        beam_width=3,
    )

    # Save report.
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_8")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
