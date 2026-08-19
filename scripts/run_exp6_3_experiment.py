#!/usr/bin/env python3
"""Run the v6.0-exp6.3 long-horizon structural value experiment."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgae_v3.experimental.exp6_3 import run_exp6_3


def main() -> int:
    print("=" * 70)
    print("v6.0-exp6.3: Long-Horizon Structural Value")
    print("=" * 70)
    print()
    print("Research question:")
    print("  Can learned structural dynamics reduce long-horizon search cost")
    print("  while preserving or improving exact MPC decisions?")
    print()

    start_t = time.time()
    result = run_exp6_3()
    elapsed = time.time() - start_t

    print(f"\nExperiment completed in {elapsed:.1f}s")

    # Save report.
    report_dir = ROOT / "reports" / "v6_exp6_3"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = result.to_log()
    report["elapsed_seconds"] = round(elapsed, 2)

    with open(report_dir / "EXPERIMENT_RESULT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Report saved to {report_dir / 'EXPERIMENT_RESULT.json'}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
