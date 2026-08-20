#!/usr/bin/env python
"""Run the v6.0-exp6.7 experiment."""
import sys
import os
import json
import time

# Ensure src is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp6_7 import run_exp6_7


def main() -> int:
    print("=" * 70)
    print("v6.0-exp6.7: Multi-Operator Causal Structural Model")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can the causal structural effect model generalize across")
    print("  heterogeneous mutations AND reward formulations?")
    print()
    print("Mutation types: ADD_EDGE, REMOVE_EDGE, REWEIGHT_EDGE, EDGE_SWAP")
    print("Effect heads: 7 (components, redundancy, hub, spectral, path, efficiency, curvature)")
    print("Generalization axes:")
    print("  1. Leave-one-mechanism-out (LOMO)")
    print("  2. Reward-formulation hold-out (threshold -> linear/composite)")
    print("Statistical rigor: paired bootstrap CIs for Recovery_C - Recovery_A")
    print()

    t0 = time.time()
    result = run_exp6_7()
    elapsed = time.time() - t0

    # Save report.
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v6_exp6_7")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "EXPERIMENT_RESULT.json")
    with open(report_path, "w") as f:
        json.dump(result.to_log(), f, indent=2)

    print(f"\nExperiment completed in {elapsed:.1f}s")
    print(f"Report saved to {os.path.abspath(report_path)}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
