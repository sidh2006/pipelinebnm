"""
tests/test_inference.py

Tests for Member 2: graders and inference loop.
Runs against the live task classes (no mocking of env logic).
inference.py LLM calls are mocked to avoid API dependency in CI.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from env.graders.grader1 import grade_task1
from env.graders.grader2 import grade_task2
from env.graders.grader3 import grade_task3
from env.models import ActionType, AERRecord, DataAction
from env.tasks.task1_audit import Task1AuditEnv
from env.tasks.task2_schema import Task2SchemaEnv
from env.tasks.task3_incident import Task3IncidentEnv


# -- Grader 1 tests --------------------------------------------------------
class TestGrader1:
    def test_zero_score_at_reset(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        result = grade_task1(env)
        assert result.score == 0.0
        assert 0.0 <= result.score <= 1.0

    def test_score_increases_after_fix(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        
        env.step(
            DataAction(
                action_type=ActionType.INSPECT,
                target_column="salary",
                justification="Checking salary first",
            )
        )
        before = grade_task1(env).score
        
        env.step(
            DataAction(
                action_type=ActionType.FILL_DEFAULT,
                target_column="salary",
                transformation="fill_median",
                justification="Fixing null salary.",
            )
        )
        after = grade_task1(env).score
        assert after > before

    def test_perfect_score_formula(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        for bug in env.ground_truth:
            env.identified_bug_ids.add(bug["bug_id"])
            env.fixed_bug_ids.add(bug["bug_id"])
        result = grade_task1(env)
        assert result.score == 1.0

    def test_identification_only_score(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        for bug in env.ground_truth:
            env.identified_bug_ids.add(bug["bug_id"])
        result = grade_task1(env)
        assert result.score == pytest.approx(0.4, abs=0.001)

    def test_breakdown_keys_present(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        result = grade_task1(env)
        assert "identification" in result.breakdown
        assert "remediation" in result.breakdown
        assert "total_bugs" in result.breakdown

    def test_deterministic(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        s1 = grade_task1(env).score
        s2 = grade_task1(env).score
        assert s1 == s2

    def test_noop_baseline_below_threshold(self):
        env = Task1AuditEnv()
        env.reset(scenario_override="task1_scenario.json")
        for _ in range(10):  # MAX_STEPS = 10
            env.step(DataAction(action_type=ActionType.NOOP, justification="noop"))
        assert grade_task1(env).score < 0.1


# -- Grader 2 tests --------------------------------------------------------
class TestGrader2:
    def test_returns_valid_range(self):
        env = Task2SchemaEnv()
        env.reset()
        result = grade_task2(env)
        assert 0.0 <= result.score <= 1.0

    def test_blast_penalty_reduces_score(self):
        env = Task2SchemaEnv()
        env.reset()
        score_clean = grade_task2(env).score
        env.blast_events = 2
        score_after = grade_task2(env).score
        assert score_after <= score_clean

    def test_breakdown_keys_present(self):
        env = Task2SchemaEnv()
        env.reset()
        result = grade_task2(env)
        assert "schema_compliance" in result.breakdown
        assert "blast_radius_penalty" in result.breakdown
        assert "bugs_fixed" in result.breakdown

    def test_score_rounded_4_places(self):
        env = Task2SchemaEnv()
        env.reset()
        score = grade_task2(env).score
        assert score == round(score, 4)


# -- Grader 3 tests --------------------------------------------------------
class TestGrader3:
    def test_pii_penalty_when_unmasked(self):
        env = Task3IncidentEnv()
        env.reset()
        result = grade_task3(env)
        assert result.breakdown["pii_compliance_penalty"] == pytest.approx(-0.20)

    def test_no_pii_penalty_when_masked(self):
        env = Task3IncidentEnv()
        env.reset()
        env.pii_masked = True
        result = grade_task3(env)
        assert result.breakdown["pii_compliance_penalty"] == 0.0

    def test_perfect_score_all_true(self):
        env = Task3IncidentEnv()
        env.reset()
        env.diagnosis_correct = True
        env.fix_applied = True
        env.pii_masked = True
        env.validation_passed = True
        result = grade_task3(env)
        assert result.score == pytest.approx(1.0, abs=0.001)

    def test_reasoning_bonus_from_aer(self):
        env = Task3IncidentEnv()
        env.reset()
        env.aer_history = [
            AERRecord(
                step_id=1,
                action_type="INSPECT",
                target="logs",
                justification="I see schema drift at stage 3 join. SSN data leaked. Revenue type mismatch.",
                reward_earned=0.25,
                issues_identified=[],
                issues_fixed=[],
            )
        ]
        result = grade_task3(env)
        assert result.breakdown["reasoning_bonus"] > 0.0

    def test_score_clamped_to_range(self):
        env = Task3IncidentEnv()
        env.reset()
        result = grade_task3(env)
        assert 0.0 <= result.score <= 1.0

    def test_breakdown_all_keys_present(self):
        env = Task3IncidentEnv()
        env.reset()
        result = grade_task3(env)
        for key in [
            "diagnosis",
            "fix",
            "pii_sweep",
            "validation",
            "pii_compliance_penalty",
            "reasoning_bonus",
            "signals_unlocked",
            "downstream_health",
        ]:
            assert key in result.breakdown, f"Missing key: {key}"


# -- Inference tests (mocked LLM) -----------------------------------------
class TestInference:
    def _make_mock_client(self, action_dict: dict):
        """Create a mock OpenAI client that returns a fixed action."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(action_dict)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_parse_clean_json(self):
        from inference import _parse_json_from_text, _validate_action

        raw = json.dumps(
            {
                "action_type": "NOOP",
                "target_column": None,
                "transformation": None,
                "justification": "test",
                "identified_issues": None,
            }
        )
        parsed = _parse_json_from_text(raw)
        assert parsed is not None
        assert _validate_action(parsed)

    def test_parse_fenced_json(self):
        from inference import _parse_json_from_text

        raw = '```json\n{"action_type":"NOOP","justification":"test"}\n```'
        parsed = _parse_json_from_text(raw)
        assert parsed is not None
        assert parsed["action_type"] == "NOOP"

    def test_invalid_action_returns_none(self):
        from inference import _parse_json_from_text, _validate_action

        raw = '{"action_type": "INVALID_TYPE", "justification": "test"}'
        parsed = _parse_json_from_text(raw)
        assert not _validate_action(parsed)

    def test_parse_garbage_returns_none(self):
        from inference import _parse_json_from_text

        result = _parse_json_from_text("this is not json at all %%%")
        assert result is None

    def test_belief_state_update(self):
        from inference import _update_belief_state

        belief = {"candidates": [], "confirmed": [], "fixes_done": []}
        action = {
            "action_type": "CAST_TYPE",
            "target_column": "age",
            "justification": "schema drift at stage 3 caused type mismatch",
        }
        result = {"reward": 0.20, "info": {"fixed": ["B002"], "identified": ["B001", "B002"]}}
        updated = _update_belief_state(belief, action, result)
        assert "stage 3" in updated["candidates"] or "schema drift" in updated["candidates"]
        assert "B002" in updated["fixes_done"]

    def test_pii_sanitizer(self):
        from inference import _sanitize_pii

        text = "The SSN is 123-45-6789 and email is test@corp.com"
        sanitized = _sanitize_pii(text)
        assert "123-45-6789" not in sanitized
        assert "test@corp.com" not in sanitized
        assert "[SSN-REDACTED]" in sanitized
        assert "[EMAIL-REDACTED]" in sanitized

    def test_compaction_summary_structure(self):
        from inference import _compaction_summary

        belief = {
            "candidates": ["schema drift", "ssn"],
            "confirmed": ["CAST_TYPE:rev_amt"],
            "fixes_done": ["B001"],
        }
        summary = _compaction_summary(belief, ["TypeError at stage 3"])
        assert "schema drift" in summary
        assert "CAST_TYPE:rev_amt" in summary
        assert "B001" in summary

    def test_rolling_window_truncation(self):
        from inference import ROLLING_WINDOW, _truncate_messages

        system = {"role": "system", "content": "sys"}
        msgs = [system]
        for i in range(20):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        truncated = _truncate_messages(msgs, system)
        non_sys = [m for m in truncated if m["role"] != "system"]
        assert len(non_sys) <= ROLLING_WINDOW * 2
        assert truncated[0]["role"] == "system"

    @patch.dict(
        "os.environ",
        {
            "API_BASE_URL": "http://localhost:8000",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        },
    )
    def test_grader_scores_are_floats(self):
        """Verify all graders return floats in [0,1]."""
        for EnvClass, grade_fn in [
            (Task1AuditEnv, grade_task1),
            (Task2SchemaEnv, grade_task2),
            (Task3IncidentEnv, grade_task3),
        ]:
            env = EnvClass()
            env.reset()
            result = grade_fn(env)
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0

    def test_episode_runtime_guard(self):
        """Verify runtime guard exits before 20 minutes."""
        import inference as inf

        original = inf._EPISODE_START
        inf._EPISODE_START = time.time() - (19 * 60 + 5)
        with pytest.raises(SystemExit):
            inf._check_runtime()
        inf._EPISODE_START = original


class TestBeliefState:
    def test_confidence_increases_on_positive_reward(self):
        from inference import BeliefState

        belief = BeliefState()
        belief.update_confidence(0.20)
        assert belief.confidence > 0.0

    def test_confidence_decreases_on_negative_reward(self):
        from inference import BeliefState

        belief = BeliefState(confidence=0.5)
        belief.update_confidence(-0.10)
        assert belief.confidence < 0.5

    def test_confidence_clamped_at_1(self):
        from inference import BeliefState

        belief = BeliefState(confidence=0.99)
        belief.update_confidence(0.20)
        assert belief.confidence <= 1.0

    def test_confidence_clamped_at_0(self):
        from inference import BeliefState

        belief = BeliefState(confidence=0.01)
        belief.update_confidence(-0.10)
        assert belief.confidence >= 0.0

    def test_to_prompt_str_empty(self):
        from inference import BeliefState

        belief = BeliefState()
        assert belief.to_prompt_str() == "No hypothesis yet."

    def test_to_prompt_str_with_data(self):
        from inference import BeliefState

        belief = BeliefState(
            candidate_causes=["schema drift"],
            confirmed_fixes=["B001"],
            confidence=0.75,
        )
        prompt_str = belief.to_prompt_str()
        assert "schema drift" in prompt_str
        assert "B001" in prompt_str

    def test_belief_update_adds_candidates(self):
        from inference import BeliefState, _update_belief

        belief = BeliefState()
        action = {
            "action_type": "INSPECT",
            "target_column": "logs",
            "justification": "I suspect schema drift at stage 3 join",
        }
        result = {
            "reward": 0.15,
            "info": {
                "fixed": [],
                "identified": [],
                "signals_unlocked": ["logs"],
            },
        }
        updated = _update_belief(belief, action, result)
        assert "schema drift" in updated.candidate_causes
        assert "logs" in updated.signals_unlocked

    def test_belief_update_records_eliminated(self):
        from inference import BeliefState, _update_belief

        belief = BeliefState()
        action = {
            "action_type": "DROP_COLUMN",
            "target_column": "name",
            "justification": "dropping name column",
        }
        result = {
            "reward": -0.10,
            "info": {
                "fixed": [],
                "identified": [],
                "signals_unlocked": [],
            },
        }
        updated = _update_belief(belief, action, result)
        assert "DROP_COLUMN:name" in updated.eliminated_causes


class TestEscalation:
    def test_escalation_summary_contains_key_fields(self):
        from inference import BeliefState, _build_escalation_summary

        belief = BeliefState(
            candidate_causes=["pii leak"],
            confirmed_fixes=["B001"],
            confidence=0.6,
        )
        summary = _build_escalation_summary(belief, step_num=5)
        assert "pii leak" in summary
        assert "B001" in summary
        assert "0.60" in summary
        assert "2 steps left" in summary or "steps left" in summary


class TestGrader1Efficiency:
    def test_efficiency_bonus_when_all_fixed_early(self):
        env = Task1AuditEnv()
        env.reset()

        for bug in env.ground_truth:
            env.identified_bug_ids.add(bug["bug_id"])
            env.fixed_bug_ids.add(bug["bug_id"])

        env.step_count = 3
        result = grade_task1(env)
        assert result.breakdown["efficiency_bonus"] > 0.0
        assert result.score > 0.999

    def test_no_efficiency_bonus_when_bugs_remain(self):
        env = Task1AuditEnv()
        env.reset()
        result = grade_task1(env)
        assert result.breakdown["efficiency_bonus"] == 0.0


class TestGrader2TypeCorrectness:
    def test_type_correctness_in_breakdown(self):
        env = Task2SchemaEnv()
        env.reset()
        result = grade_task2(env)
        assert "type_correctness" in result.breakdown
        assert "column_recovery" in result.breakdown
        assert 0.0 <= result.score <= 1.0


class TestGrader3Bonuses:
    def test_all_bonuses_in_breakdown(self):
        env = Task3IncidentEnv()
        env.reset()
        result = grade_task3(env)
        for key in [
            "reasoning_bonus",
            "root_cause_attribution",
            "signals_investigation",
            "efficiency_bonus",
            "total_bonus",
        ]:
            assert key in result.breakdown, f"Missing: {key}"

    def test_root_cause_attribution_exact_keyword(self):
        from env.graders.grader3 import _root_cause_attribution

        env = Task3IncidentEnv()
        env.reset()
        env.aer_history = [
            AERRecord(
                step_id=1,
                action_type="INSPECT",
                target="logs",
                justification="Corruption at stage_3_join identified clearly.",
                reward_earned=0.25,
                issues_identified=[],
                issues_fixed=[],
            )
        ]
        bonus = _root_cause_attribution(env)
        assert bonus == 0.05

    def test_signals_bonus_scales_with_unlocked(self):
        from env.graders.grader3 import _signals_investigation_bonus

        env = Task3IncidentEnv()
        env.reset()
        env.signals_unlocked = {"logs", "metrics", "compliance", "dag"}
        bonus = _signals_investigation_bonus(env)
        assert bonus == pytest.approx(0.08)

    def test_pii_penalty_is_not_100(self):
        """CRITICAL: verify PII penalty never returns -100 (DQ violation)."""
        env = Task3IncidentEnv()
        env.reset()
        result = grade_task3(env)
        assert result.score >= 0.0, "Score must never go below 0.0"
        assert result.breakdown["pii_compliance_penalty"] == pytest.approx(-0.20)
        assert result.breakdown["pii_compliance_penalty"] != -100.0

    def test_perfect_agent_scores_above_09(self):
        env = Task3IncidentEnv()
        env.reset()
        env.diagnosis_correct = True
        env.fix_applied = True
        env.pii_masked = True
        env.validation_passed = True
        env.step_count = 3
        env.signals_unlocked = {"logs", "metrics", "compliance", "dag"}
        env.aer_history = [
            AERRecord(
                step_id=1,
                action_type="INSPECT",
                target="logs",
                justification=(
                    "I see schema drift at stage_3_join. "
                    "SSN data leaked. Revenue type mismatch. "
                    "Aggregation inflated."
                ),
                reward_earned=0.25,
                issues_identified=[],
                issues_fixed=[],
            )
        ]
        result = grade_task3(env)
        assert result.score >= 0.90, f"Perfect agent scored {result.score}"
