#!/usr/bin/env python3
"""Run the v7.0-exp3-task-conditioned-topology-learning experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp7_3 import run_exp7_3
from lgae_v3.experimental.exp7_2 import ObjectiveWeights


def main():
    parser = argparse.ArgumentParser(description="Run exp7.3")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--adapt-interval", type=int, default=20)
    parser.add_argument("--shadow-batch", type=int, default=20)
    parser.add_argument("--no-shadow-sweep", action="store_true", help="skip shadow batch sweep")
    args = parser.parse_args()

    print("=" * 90)
    print("v7.0-exp3: Task-Conditioned Topology Learning")
    print("=" * 90)
    print()
    print("Central hypothesis:")
    print("  LGAE with text-derived task features (NO labels) can close")
    print("  the gap to the rule-based dynamic router.")
    print()
    print("Four conditions:")
    print("  A. Fixed topology (no adaptation)")
    print("  B. Rule-based dynamic router (human rules, task-aware)")
    print("  C. LGAE telemetry-only (no task features)")
    print("  D. LGAE task-conditioned (text-derived features, NO labels)")
    print()
    print("State: S = (G, x_task, telemetry, budget, history)")
    print("Policy: π_topology(G, x_task, telemetry)")
    print()
    print("Shadow batch sweep: 5, 10, 20, 50 — measure ShadowTransferCorrelation")
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

    result = run_exp7_3(
        n_tasks_per_class=args.n_tasks,
        backend_type=args.backend,
        adaptation_interval=args.adapt_interval,
        shadow_batch_size=args.shadow_batch,
        run_shadow_sweep=not args.no_shadow_sweep,
        weights=weights,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v7_exp3")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
