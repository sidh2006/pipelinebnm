"""
scripts/validate_diversity.py

Generates 100 scenarios per task using seeds 0-99 and validates
that no two scenarios share the same (bug_type, target_column, row_index)
triple. Prints a diversity report confirming generalization coverage.

This script doubles as documentation for judges.

Usage:
    python -m scripts.validate_diversity
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.data.bug_injector import generate_scenario


def main() -> None:
    print("=" * 70)
    print("  DataPipelineEnv — Scenario Diversity Validation")
    print("=" * 70)

    tasks = [
        ("task1", "easy"),
        ("task2", "medium"),
        ("task3", "hard"),
    ]

    all_passed = True

    for task_id, difficulty in tasks:
        print(f"\n--- Task: {task_id} (difficulty={difficulty}) ---")

        triples: list[tuple[str, str | None, int | None]] = []
        bug_type_counts: Counter[str] = Counter()
        scenarios_generated = 0

        for seed in range(100):
            scenario = generate_scenario(seed=seed, task_id=task_id, difficulty=difficulty)
            scenarios_generated += 1

            for bug in scenario:
                bug_type = bug["type"]
                col = bug.get("column") or bug.get("new_col")
                row = bug.get("row")
                triples.append((bug_type, col, row))
                bug_type_counts[bug_type] += 1

        # Check uniqueness of triples
        triple_counter = Counter(triples)
        duplicates = {k: v for k, v in triple_counter.items() if v > 1}

        unique_triples = len(set(triples))
        total_triples = len(triples)
        duplicate_count = len(duplicates)

        print(f"  Scenarios generated: {scenarios_generated}")
        print(f"  Total bug triples:   {total_triples}")
        print(f"  Unique triples:      {unique_triples}")
        print(f"  Duplicate triples:   {duplicate_count}")
        print(f"  Bug type distribution:")
        for bt, count in sorted(bug_type_counts.items()):
            print(f"    {bt:25s} = {count}")

        # For task1, verify at least 3 different bug types per scenario
        if task_id == "task1":
            low_diversity = 0
            for seed in range(100):
                scenario = generate_scenario(seed=seed, task_id=task_id, difficulty=difficulty)
                types_in_scenario = set(b["type"] for b in scenario)
                if len(types_in_scenario) < 3:
                    low_diversity += 1
            print(f"  Scenarios with <3 bug types: {low_diversity}/100")
            if low_diversity > 0:
                print("  ⚠ WARNING: Some scenarios have low bug type diversity")
                all_passed = False

        if duplicate_count == 0:
            print("  ✓ PASS: No repeated (bug_type, column, row) triples")
        else:
            print(f"  ✗ FAIL: {duplicate_count} duplicate triples found")
            for triple, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
                print(f"    {triple} appeared {count} times")
            all_passed = False

    print(f"\n{'=' * 70}")
    if all_passed:
        print("  ALL CHECKS PASSED — Scenario diversity confirmed ✓")
    else:
        print("  SOME CHECKS FAILED — Review output above")
    print(f"{'=' * 70}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
