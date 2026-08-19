#!/usr/bin/env python3
"""Run v6.0-exp6.4: Learned non-additive value."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgae_v3.experimental.exp6_4.experiment_runner import run_exp6_4


def main() -> int:
    print("=" * 70)
    print("v6.0-exp6.4: Learned Non-Additive Value")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can learned bonus prediction recover non-greedy first actions")
    print("  without access to exact future utility?")
    print()
    print("Key metric: NonGreedyRecoveryRate")
    print("  = P(a_model == a_exact_mpc | a_greedy != a_exact_mpc)")
    print()
    print("Architecture (NO information leakage):")
    print("  Q_hat = delta_U_additive + gamma * V_bonus_hat(S')")
    print("  - delta_U_additive: exact analytical O(1)")
    print("  - V_bonus_hat: learned from structural features (no utility_fn)")
    print()

    start_t = time.time()
    result = run_exp6_4()
    elapsed = time.time() - start_t

    print(f"\nExperiment completed in {elapsed:.1f}s")

    report_dir = ROOT / "reports" / "v6_exp6_4"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = result.to_log()
    report["elapsed_seconds"] = round(elapsed, 2)

    with open(report_dir / "EXPERIMENT_RESULT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Report saved to {report_dir / 'EXPERIMENT_RESULT.json'}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
