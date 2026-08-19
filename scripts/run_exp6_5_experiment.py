#!/usr/bin/env python3
"""Run v6.0-exp6.5: Cross-mechanism foresight generalization."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgae_v3.experimental.exp6_5.experiment_runner import run_exp6_5


def main() -> int:
    print("=" * 70)
    print("v6.0-exp6.5: Cross-Mechanism Foresight Generalization")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can the learned future-residual model generalize across")
    print("  different forms of non-additive structural value?")
    print()
    print("Method: Leave-one-mechanism-out (LOMO) evaluation")
    print("  For each mechanism M_i:")
    print("    Train on {M_j : j != i}")
    print("    Test on M_i")
    print()
    print("Mechanisms:")
    print("  - connectivity_threshold")
    print("  - redundancy_threshold")
    print("  - hub_load_threshold")
    print("  - spectral_gap_threshold")
    print()
    print("Architecture (NO information leakage):")
    print("  Q_hat = delta_U_additive + gamma * V_residual_hat(S')")
    print("  - No mechanism label in features")
    print("  - No utility_fn access during search")
    print("  - Adaptive beam width based on ensemble uncertainty")
    print()

    start_t = time.time()
    result = run_exp6_5()
    elapsed = time.time() - start_t

    print(f"\nExperiment completed in {elapsed:.1f}s")

    report_dir = ROOT / "reports" / "v6_exp6_5"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = result.to_log()
    report["elapsed_seconds"] = round(elapsed, 2)

    with open(report_dir / "EXPERIMENT_RESULT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Report saved to {report_dir / 'EXPERIMENT_RESULT.json'}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
