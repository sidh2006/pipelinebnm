import pytest
import random

from env.data.generator import generate_employee_dataset
from env.graders import grade_task1, grade_task3
from env.models import ActionType, DataAction, DataObservation
from env.tasks import Task1AuditEnv, Task2SchemaEnv, Task3IncidentEnv


@pytest.fixture(autouse=True)
def force_base_scenarios(monkeypatch):
	original_choice = random.choice
	def custom_choice(seq):
		if seq and hasattr(seq[0], "name") and "scenario" in seq[0].name:
			base = [f for f in seq if f.name in ["task1_scenario.json", "task2_scenario.json", "task3_scenario.json"]]
			if base:
				return base[0]
		return original_choice(seq)
	monkeypatch.setattr(random, "choice", custom_choice)


def test_reset_returns_valid_observation():
	env = Task1AuditEnv()
	obs = env.reset(scenario_override="task1_scenario.json")
	assert isinstance(obs, DataObservation)
	assert len(obs.dataset_preview) > 0
	assert 0.0 <= obs.downstream_health <= 1.0
	assert obs.time_remaining == 10  # MAX_STEPS = 10
	assert obs.task_id == 1


def test_step_loop_terminates_at_max_steps():
	env = Task1AuditEnv()
	env.reset(scenario_override="task1_scenario.json")
	done = False
	for _ in range(10):  # MAX_STEPS = 10
		result = env.step(DataAction(action_type=ActionType.NOOP, justification="test"))
		if result.done:
			done = True
			break
	assert done


def test_grader_is_deterministic():
	env = Task1AuditEnv()
	env.reset(scenario_override="task1_scenario.json")
	s1 = grade_task1(env).score
	s2 = grade_task1(env).score
	assert s1 == s2


def test_noop_baseline_scores_below_threshold():
	env = Task1AuditEnv()
	env.reset(scenario_override="task1_scenario.json")
	for _ in range(10):  # MAX_STEPS = 10
		env.step(DataAction(action_type=ActionType.NOOP, justification="noop"))
	assert grade_task1(env).score < 0.1


def test_correct_fix_increases_score():
	env = Task1AuditEnv()
	env.reset(scenario_override="task1_scenario.json")
	score_before = grade_task1(env).score
	env.step(
		DataAction(
			action_type=ActionType.INSPECT,
			target_column="salary",
			justification="Checking salary",
		)
	)
	env.step(
		DataAction(
			action_type=ActionType.FILL_DEFAULT,
			target_column="salary",
			transformation="fill_median",
			justification="Filling null salary values.",
		)
	)
	score_after = grade_task1(env).score
	assert score_after > score_before


def test_generator_is_deterministic():
	df1 = generate_employee_dataset(seed=42)
	df2 = generate_employee_dataset(seed=42)
	assert df1.equals(df2)


def test_server_ping():
	from fastapi.testclient import TestClient

	from env.server import app

	with TestClient(app) as client:
		response = client.get("/ping")
		assert response.status_code == 200
		assert response.json() == {"status": "ok"}


def test_server_reset_and_step():
	from fastapi.testclient import TestClient

	from env.server import app

	with TestClient(app) as client:
		obs = client.post("/reset", params={"task_id": 1}).json()
		assert "dataset_preview" in obs
		assert "schema" in obs
		result = client.post(
			"/step",
			json={"action_type": "NOOP", "justification": "test"},
			params={"task_id": 1},
		).json()
		assert "reward" in result
		assert "done" in result


def test_task3_pii_penalty():
	env = Task3IncidentEnv()
	env.reset(scenario_override="task3_scenario.json")
	for _ in range(8):
		env.step(DataAction(action_type=ActionType.NOOP, justification="noop"))
	result = grade_task3(env)
	assert result.breakdown["pii_compliance_penalty"] == -0.2


def test_blast_radius_penalizes_score():
	env = Task2SchemaEnv()
	env.reset(scenario_override="task2_scenario.json")
	health_before = env.downstream_health
	env.step(
		DataAction(
			action_type=ActionType.DROP_COLUMN,
			target_column="salary",
			justification="Dropping salary column.",
		)
	)
	assert env.downstream_health < health_before or env.blast_events > 0
