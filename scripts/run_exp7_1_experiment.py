#!/usr/bin/env python3
"""Run the v7.0-exp1-real-ai-topology experiment."""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp7_1 import run_exp7_1, ObjectiveWeights


def main():
    parser = argparse.ArgumentParser(description="Run exp7.1")
    parser.add_argument("--n-tasks", type=int, default=10,
                        help="Tasks per class")
    parser.add_argument("--adapt-interval", type=int, default=10,
                        help="LGAE adaptation interval")
    args = parser.parse_args()

    print("=" * 70)
    print("v7.0-exp1: Real AI Topology")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can structural adaptation make an AI system perform")
    print("  better per unit compute?")
    print()
    print("Three conditions:")
    print("  A. Fixed topology (no adaptation)")
    print("  B. Hand-written dynamic router (rule-based)")
    print("  C. LGAE adaptive topology (structural planning)")
    print()
    print("Held constant: models, prompts, tasks, tokens, tools, hardware")
    print("LGAE controls only: routing topology (edges)")
    print()
    print("Objective: J = w_Q*Q - λ_T*Tokens - λ_L*Latency - λ_F*Failures - λ_C*Calls")
    print()

    weights = ObjectiveWeights(
        w_quality=1.0,
        lambda_tokens=0.001,
        lambda_latency=0.01,
        lambda_failures=0.5,
        lambda_calls=0.05,
    )

    result = run_exp7_1(
        n_tasks_per_class=args.n_tasks,
        weights=weights,
        adaptation_interval=args.adapt_interval,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v7_exp1")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
