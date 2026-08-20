#!/usr/bin/env python3
"""Run the v7.0-exp5-real-llm-routing-validation experiment.

Usage:
  # Mock backend (for testing infrastructure)
  python scripts/run_exp7_5_experiment.py --backend mock

  # Real OpenAI backend
  OPENAI_API_KEY=sk-... python scripts/run_exp7_5_experiment.py \
      --backend openai \
      --model gpt-4o-mini \
      --input-price 0.15 \
      --output-price 0.60

  # Fine-tuned model
  OPENAI_API_KEY=sk-... LGAE_MODEL_ID=ft:gpt-4o-mini:org:run:abc123 \
      python scripts/run_exp7_5_experiment.py --backend openai --model "$LGAE_MODEL_ID"

Security:
  - API key is read from OPENAI_API_KEY environment variable
  - Never passed on command line or committed to Git
  - Only OPENAI_API_KEY_PRESENT=true is recorded in artifacts
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp7_5 import (
    run_exp7_5, BackendConfig, MOCK_CONFIG, make_openai_config, BudgetGuard,
)
from lgae_v3.experimental.exp7_2 import ObjectiveWeights


def main():
    parser = argparse.ArgumentParser(description="Run exp7.5")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model ID (or set LGAE_MODEL_ID env var)")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--input-price", type=float, default=0.15,
                        help="Input token price per 1M tokens")
    parser.add_argument("--output-price", type=float, default=0.60,
                        help="Output token price per 1M tokens")
    parser.add_argument("--cached-price", type=float, default=0.075,
                        help="Cached input token price per 1M tokens")
    parser.add_argument("--max-cost", type=float, default=50.0,
                        help="Maximum dollar cost budget")
    parser.add_argument("--max-calls", type=int, default=10000,
                        help="Maximum API calls")
    parser.add_argument("--no-smoke", action="store_true", help="Skip smoke test")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity check")
    parser.add_argument("--no-ablation", action="store_true", help="Skip node ablation")
    parser.add_argument("--no-main", action="store_true", help="Skip main experiment")
    args = parser.parse_args()

    print("=" * 95)
    print("v7.0-exp5: Real LLM Routing Validation")
    print("=" * 95)
    print()
    print("Scientific question:")
    print("  Does LGAE's learned sparse routing policy still beat fixed and")
    print("  hand-designed routing when every cognitive node is backed by a real LLM?")
    print()
    print("Three conditions:")
    print("  A. Fixed topology")
    print("  B. Human dynamic router")
    print("  C. LGAE node-necessity router")
    print()
    print("15 predeclared gates (A-O)")
    print()

    # Check for API key if using OpenAI.
    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("Set it with: export OPENAI_API_KEY=sk-...")
            return 1
        # Use LGAE_MODEL_ID if set, otherwise --model.
        model_id = os.environ.get("LGAE_MODEL_ID", args.model)
        print(f"  Model ID: {model_id}")
        print(f"  API key: present (not shown)")

        config = make_openai_config(
            model_id=model_id,
            input_price=args.input_price,
            output_price=args.output_price,
            cached_price=args.cached_price,
            temperature=0.0,
        )
    else:
        print("  Backend: mock (for infrastructure testing)")
        config = MOCK_CONFIG

    budget = BudgetGuard(
        max_api_calls=args.max_calls,
        max_dollar_cost=args.max_cost,
    )
    print(f"  Budget: max ${args.max_cost}, max {args.max_calls} calls")

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

    result = run_exp7_5(
        backend_config=config,
        n_tasks_per_class=args.n_tasks,
        run_smoke=not args.no_smoke,
        run_sensitivity=not args.no_sensitivity,
        run_ablation=not args.no_ablation,
        run_main_experiment=not args.no_main,
        budget=budget,
        weights=weights,
    )

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v7_exp5")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "EXPERIMENT_RESULT.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"\nReport saved to {os.path.join(report_dir, 'EXPERIMENT_RESULT.json')}")
    print(f"\nAll gates passed: {result.all_gates_passed}")
    print(f"Budget: {result.budget_summary}")

    return 0 if result.all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
