#!/usr/bin/env python3
"""Run the v7.0-exp2-live-model-topology-benchmark experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp7_2 import run_exp7_2, ObjectiveWeights


def main():
    parser = argparse.ArgumentParser(description="Run exp7.2")
    parser.add_argument("--n-tasks", type=int, default=50,
                        help="Tasks per class (50-100 recommended)")
    parser.add_argument("--backend", type=str, default="mock",
                        choices=["mock", "openai"],
                        help="Model backend type")
    parser.add_argument("--adapt-interval", type=int, default=20,
                        help="LGAE adaptation interval")
    parser.add_argument("--shadow-batch", type=int, default=5,
                        help="Shadow evaluation batch size")
    args = parser.parse_args()

    print("=" * 80)
    print("v7.0-exp2: Live Model Topology Benchmark")
    print("=" * 80)
    print()
    print("Central hypothesis:")
    print("  Changing AI execution topology changes quality/cost")
    print("  enough for LGAE to learn useful routing interventions.")
    print()
    print("Three conditions:")
    print("  A. Fixed topology (no adaptation)")
    print("  B. Hand-written dynamic router (human rules, task-aware)")
    print("  C. LGAE adaptive topology (learned, NO task labels)")
    print()
    print("Held constant: model backend, prompts, tasks, token limits, hardware")
    print("LGAE controls only: routing topology (edges)")
    print("LGAE sees only: telemetry (NOT task metadata)")
    print()
    print("Objective: J = w_Q*Q - λ_T*(T/T_budget) - λ_L*(L/L_budget) - λ_C*(C/C_budget) - λ_F*F")
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

    result = run_exp7_2(
        n_tasks_per_class=args.n_tasks,
        backend_type=args.backend,
        adaptation_interval=args.adapt_interval,
        shadow_batch_size=args.shadow_batch,
        weights=weights,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v7_exp2")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
