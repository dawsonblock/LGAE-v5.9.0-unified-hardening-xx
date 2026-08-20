#!/usr/bin/env python3
"""Run the v7.0-exp5-real-llm-routing-validation experiment.

Usage:
  # Mock backend (for testing infrastructure)
  python scripts/run_exp7_5_experiment.py --backend mock

  # DeepSeek backend
  DEEPSEEK_API_KEY=sk-... python scripts/run_exp7_5_experiment.py \
      --backend deepseek --model deepseek-v4-flash

  # OpenAI backend
  OPENAI_API_KEY=sk-... python scripts/run_exp7_5_experiment.py \
      --backend openai --model gpt-4o-mini

Security:
  - API key is read from environment variable (DEEPSEEK_API_KEY or OPENAI_API_KEY)
  - Never passed on command line or committed to Git
  - Only *_PRESENT=true is recorded in artifacts
"""
import sys
import os
import json
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.experimental.exp7_5 import (
    run_exp7_5, BackendConfig, MOCK_CONFIG, make_openai_config, BudgetGuard,
    create_snapshot, GATE_DEFINITIONS, get_prompt_hashes, make_split,
)
from lgae_v3.experimental.exp7_2 import ObjectiveWeights


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip()[:7]
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Run exp7.5")
    parser.add_argument("--backend", type=str, default="mock",
                        choices=["mock", "openai", "deepseek"])
    parser.add_argument("--model", type=str, default="deepseek-v4-flash",
                        help="Model ID")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--input-price", type=float, default=0.27,
                        help="Input token price per 1M tokens")
    parser.add_argument("--output-price", type=float, default=1.10,
                        help="Output token price per 1M tokens")
    parser.add_argument("--cached-price", type=float, default=0.07,
                        help="Cached input token price per 1M tokens")
    parser.add_argument("--max-cost", type=float, default=50.0,
                        help="Maximum dollar cost budget")
    parser.add_argument("--max-calls", type=int, default=10000,
                        help="Maximum API calls")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-main", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip snapshot creation")
    args = parser.parse_args()

    print("=" * 95)
    print("v7.0-exp5: Real LLM Routing Validation")
    print("=" * 95)
    print()
    print("Scientific question:")
    print("  Does LGAE's learned sparse routing policy still beat fixed and")
    print("  hand-designed routing when every cognitive node is backed by a real LLM?")
    print()

    # Determine config based on backend.
    if args.backend == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
            return 1
        model_id = args.model
        print(f"  Provider: DeepSeek")
        print(f"  Model: {model_id}")
        print(f"  API key: present (not shown)")

        import openai
        sdk_ver = openai.__version__
        config = BackendConfig(
            provider="deepseek",
            model_id=model_id,
            temperature=0.0,
            max_output_tokens=1024,
            timeout_seconds=60.0,
            max_retries=3,
            input_price_per_mtok=args.input_price,
            output_price_per_mtok=args.output_price,
            cached_input_price_per_mtok=args.cached_price,
            sdk_version=sdk_ver,
            backend_version="exp7.5-deepseek-v1",
        )
    elif args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            return 1
        model_id = os.environ.get("LGAE_MODEL_ID", args.model)
        print(f"  Provider: OpenAI")
        print(f"  Model: {model_id}")
        config = make_openai_config(
            model_id=model_id,
            input_price=args.input_price,
            output_price=args.output_price,
            cached_price=args.cached_price,
        )
    else:
        print("  Backend: mock (for infrastructure testing)")
        config = MOCK_CONFIG

    budget = BudgetGuard(
        max_api_calls=args.max_calls,
        max_dollar_cost=args.max_cost,
    )
    print(f"  Budget: max ${args.max_cost}, max {args.max_calls} calls")

    # Create experiment snapshot.
    if not args.no_snapshot:
        print("\n=== Creating experiment snapshot ===")
        split = make_split(n_per_class=args.n_tasks, seed=42)
        commit = get_git_commit()
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
        snap = create_snapshot(
            source_commit=commit,
            provider=config.provider,
            model_id=config.model_id,
            backend_config_hash=config.config_hash,
            prompt_hashes=get_prompt_hashes(),
            train_ids=[t.task_id for t in split.train],
            calibration_ids=[t.task_id for t in split.calibration],
            test_ids=[t.task_id for t in split.test],
            objective_weights=weights.to_dict(),
            routing_config={
                "calibration_interval": 20,
                "shadow_batch_size": 5,
                "k_neighbors": 5,
                "min_samples": 3,
            },
            budget_ceilings={
                "max_api_calls": args.max_calls,
                "max_dollar_cost": args.max_cost,
            },
            gate_definitions=GATE_DEFINITIONS,
        )
        report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "v7_exp5")
        os.makedirs(report_dir, exist_ok=True)
        snapshot_path = os.path.join(report_dir, "EXPERIMENT_SNAPSHOT.json")
        snap_hash = snap.save(snapshot_path)
        print(f"  Snapshot hash: {snap_hash}")
        print(f"  Saved to: {snapshot_path}")
        print(f"  Train: {len(split.train)}, Cal: {len(split.calibration)}, Test: {len(split.test)}")

    # Run experiment.
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
