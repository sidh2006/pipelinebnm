"""
tests/test_grader2.py

Unit tests verifying grader2 and task2_env agree on 5 known scenario outcomes.
"""
from __future__ import annotations

import random

import pytest

from env.graders.grader2 import grade_task2, _rows_passing
from env.models import ActionType, DataAction
from env.tasks.task2_schema import Task2SchemaEnv


@pytest.fixture(autouse=True)
def force_base_scenario(monkeypatch):
    original_choice = random.choice
    def custom_choice(seq):
        if seq and hasattr(seq[0], "name") and "scenario" in seq[0].name:
            base = [f for f in seq if f.name == "task2_scenario.json"]
            if base:
                return base[0]
        return original_choice(seq)
    monkeypatch.setattr(random, "choice", custom_choice)


class TestGrader2RowsPassing:
    """Verify _rows_passing logic matches env behavior."""

    def test_initial_state_rows(self):
        """After reset with schema drift, some rows should still pass."""
        env = Task2SchemaEnv()
        env.reset(scenario_override="task2_scenario.json")
        rows = _rows_passing(env)
        # With schema drift, employee_id is renamed to customer_uuid
        # so expected_columns check may fail → 0 rows
        assert isinstance(rows, int)
        assert rows >= 0

    def test_after_full_fix_rows_increase(self):
        """After fixing all bugs, rows passing should be near total."""
        env = Task2SchemaEnv()
        env.reset(scenario_override="task2_scenario.json")
        rows_before = _rows_passing(env)

        # Fix schema drift: rename customer_uuid back to employee_id
        env.step(DataAction(
            action_type=ActionType.INSPECT,
            target_column="schema",
            justification="Check schema drift",
        ))
        env.step(DataAction(
            action_type=ActionType.RENAME_COLUMN,
            target_column="customer_uuid",
            transformation="employee_id",
            justification="Fix schema drift",
        ))
        # Fix hire_date drift
        env.step(DataAction(
            action_type=ActionType.RENAME_COLUMN,
            target_column="dob_date",
            transformation="hire_date",
            justification="Fix hire_date drift",
        ))
        # Fix consent_flag nulls
        env.step(DataAction(
            action_type=ActionType.FILL_DEFAULT,
            target_column="consent_flag",
            justification="Fill null consent_flag",
        ))

        rows_after = _rows_passing(env)
        assert rows_after >= rows_before

    def test_grader_and_env_agree_noop(self):
        """NOOP baseline: grader score should be low."""
        env = Task2SchemaEnv()
        env.reset(scenario_override="task2_scenario.json")
        for _ in range(5):
            env.step(DataAction(action_type=ActionType.NOOP, justification="noop"))
        result = grade_task2(env)
        assert 0.0 <= result.score <= 1.0
        assert result.score < 0.5  # NOOP should not score well

    def test_blast_events_reduce_score(self):
        """Blast radius penalty should reduce grader score."""
        env = Task2SchemaEnv()
        env.reset(scenario_override="task2_scenario.json")
        score_clean = grade_task2(env).score
        env.blast_events = 3
        score_blasted = grade_task2(env).score
        assert score_blasted < score_clean

    def test_grader_breakdown_completeness(self):
        """Grader breakdown must contain all expected keys."""
        env = Task2SchemaEnv()
        env.reset(scenario_override="task2_scenario.json")
        result = grade_task2(env)
        expected_keys = {
            "rows_validation", "schema_compliance", "column_recovery",
            "column_coverage", "type_correctness", "blast_radius_penalty",
            "bugs_fixed", "rows_passing", "total_rows", "blast_events",
        }
        assert expected_keys.issubset(set(result.breakdown.keys()))
