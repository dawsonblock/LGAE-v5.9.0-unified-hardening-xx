#!/usr/bin/env python3
"""Run the v7.0-exp4-learned-routing-policy experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp7_4 import run_exp7_4
from lgae_v3.experimental.exp7_2 import ObjectiveWeights


def main():
    parser = argparse.ArgumentParser(description="Run exp7.4")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--cal-interval", type=int, default=20)
    parser.add_argument("--shadow-batch", type=int, default=5)
    args = parser.parse_args()

    print("=" * 95)
    print("v7.0-exp4: Learned Routing Policy")
    print("=" * 95)
    print()
    print("Objective: Match or approach the dynamic router's token efficiency")
    print("without giving LGAE explicit task-class labels.")
    print()
    print("Target: Quality ≥ Dynamic-ε, Tokens ≤ 1.5× Dynamic")
    print()
    print("Four conditions:")
    print("  A. Fixed topology")
    print("  B. Rule-based dynamic router")
    print("  C. LGAE task-conditioned (exp7.3)")
    print("  D. LGAE learned node-necessity router (exp7.4)")
    print()
    print("Key mechanism: Per-node marginal value estimation")
    print("  ΔJ_n = J(with node n) - J(without node n)")
    print("  Learned via k-NN regression on task embeddings")
    print()

    weights = ObjectiveWeights(
        w_quality=1.0,
        lambda_tokens=0.3,
        lambda_latency=0.2,
        lambda_calls=0.2,
        lambda_failures=0.5,
        token_budget=2000,
        latency_budget_ms=5000.0,
        call_budget=6,
    )

    result = run_exp7_4(
        n_tasks_per_class=args.n_tasks,
        backend_type=args.backend,
        calibration_interval=args.cal_interval,
        shadow_batch_size=args.shadow_batch,
        weights=weights,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v7_exp4")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
