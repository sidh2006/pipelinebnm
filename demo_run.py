import requests


API_BASE_URL = "http://localhost:8000"
MAX_STEPS = 8


def run_noop_demo(task_id: int) -> float:
    requests.post(f"{API_BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
    for _ in range(MAX_STEPS):
        payload = {
            "action_type": "NOOP",
            "target_column": None,
            "transformation": None,
            "justification": "demo noop",
            "identified_issues": None,
        }
        result = requests.post(
            f"{API_BASE_URL}/step",
            params={"task_id": task_id},
            json=payload,
            timeout=30,
        ).json()
        if result.get("done"):
            break

    grade = requests.get(f"{API_BASE_URL}/grader", params={"task_id": task_id}, timeout=30).json()
    return float(grade.get("score", 0.0))


if __name__ == "__main__":
    scores = {task_id: run_noop_demo(task_id) for task_id in (1, 2, 3)}
    print(scores)
