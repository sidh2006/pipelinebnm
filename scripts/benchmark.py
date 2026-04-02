"""
scripts/benchmark.py

Run inference.py against all three tasks 10 times each with seeds 0-9.
Record mean ± std score per task.

Usage:
    python -m scripts.benchmark

Note: Requires a running server and valid API key.
      Set API_BASE_URL, OPENAI_API_KEY or HF_TOKEN in .env
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Run benchmark and print results table."""
    print("=" * 70)
    print("  DataPipelineEnv — Benchmark Suite")
    print("  Running 10 episodes per task with seeds 0-9")
    print("=" * 70)

    try:
        import requests
        import numpy as np
    except ImportError:
        print("ERROR: Install requests and numpy to run benchmarks.")
        sys.exit(1)

    api_base = "http://localhost:7860"

    # Check server is running
    try:
        resp = requests.get(f"{api_base}/ping", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Server not available at {api_base}: {e}")
        print("Start the server first: uvicorn env.server:app --port 7860")
        sys.exit(1)

    tasks = [1, 2, 3]
    results: dict[int, list[float]] = {t: [] for t in tasks}

    for task_id in tasks:
        print(f"\n--- Task {task_id} ---")
        for seed in range(10):
            try:
                # Reset with seed
                resp = requests.post(
                    f"{api_base}/reset",
                    params={"task_id": task_id, "seed": seed},
                    timeout=10,
                )
                resp.raise_for_status()
                obs = resp.json()
                max_steps = obs.get("max_steps", 20)

                # Run NOOP baseline (placeholder — real benchmark needs LLM)
                for _ in range(max_steps):
                    step_resp = requests.post(
                        f"{api_base}/step",
                        json={
                            "action_type": "NOOP",
                            "justification": f"benchmark seed={seed}",
                        },
                        params={"task_id": task_id},
                        timeout=10,
                    )
                    step_resp.raise_for_status()
                    result = step_resp.json()
                    if result.get("done"):
                        break

                # Get grader score
                grade_resp = requests.get(
                    f"{api_base}/grader",
                    params={"task_id": task_id},
                    timeout=10,
                )
                grade_resp.raise_for_status()
                score = grade_resp.json().get("score", 0.0)
                results[task_id].append(score)
                print(f"  Seed {seed}: score={score:.4f}")

            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
                results[task_id].append(0.0)

    # Print results table
    print(f"\n{'=' * 70}")
    print("  BENCHMARK RESULTS (NOOP Baseline)")
    print(f"{'=' * 70}")
    print(f"{'Task':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)

    for task_id in tasks:
        scores = results[task_id]
        if scores:
            arr = np.array(scores)
            print(
                f"  Task {task_id:<4} "
                f"{arr.mean():>10.4f} "
                f"{arr.std():>10.4f} "
                f"{arr.min():>10.4f} "
                f"{arr.max():>10.4f}"
            )
        else:
            print(f"  Task {task_id:<4}  No data")

    print(f"\nNote: These are NOOP baseline scores.")
    print(f"For smart agent scores, run inference.py with an LLM API key.")


if __name__ == "__main__":
    main()
