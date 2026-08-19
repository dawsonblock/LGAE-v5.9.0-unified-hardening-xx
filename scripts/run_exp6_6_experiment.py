#!/usr/bin/env python3
"""Run v6.0-exp6.6: Objective-conditioned causal foresight."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgae_v3.experimental.exp6_6.experiment_runner import run_exp6_6


def main() -> int:
    print("=" * 70)
    print("v6.0-exp6.6: Objective-Conditioned Causal Foresight")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can LGAE separate the physics of structural change from")
    print("  the objective being optimized well enough to reuse its")
    print("  foresight across new goals?")
    print()
    print("Three architectures compared:")
    print("  A. Scalar residual:       F(S,a) -> R")
    print("  B. Objective-conditioned: F(S,a,O) -> R")
    print("  C. Causal effect:         F(S,a) -> effects, O(effects) -> R")
    print()
    print("Method: Leave-one-mechanism-out (LOMO) with 100+ suboptimal cases")
    print()
    print("Key hypothesis: Architecture C generalizes because it separates")
    print("  structural physics from objective evaluation.")
    print()

    start_t = time.time()
    result = run_exp6_6()
    elapsed = time.time() - start_t

    print(f"\nExperiment completed in {elapsed:.1f}s")

    report_dir = ROOT / "reports" / "v6_exp6_6"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = result.to_log()
    report["elapsed_seconds"] = round(elapsed, 2)

    with open(report_dir / "EXPERIMENT_RESULT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Report saved to {report_dir / 'EXPERIMENT_RESULT.json'}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
