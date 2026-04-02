# DataPipelineEnv — Architecture

## System Overview

```mermaid
graph TD
    A["inference.py / External Agent"] -->|POST /reset| B["server.py FastAPI"]
    A -->|POST /step| B
    B --> C["tasks/task1_audit.py"]
    B --> D["tasks/task2_schema.py"]
    B --> E["tasks/task3_incident.py"]
    C --> F["data/generator.py"]
    D --> F
    E --> F
    C --> G["data/bug_injector.py"]
    D --> G
    E --> G
    B -->|GET /grader| H["graders/grader1-3.py"]
    B --> I["leaderboard.json"]
```

## Request Flow

```mermaid
sequenceDiagram
    participant Agent as inference.py
    participant Server as FastAPI server
    participant Env as TaskEnv
    participant Gen as generator.py
    participant Inj as bug_injector.py
    participant Grader as grader.py

    Agent->>Server: POST /reset (task_id, seed)
    Server->>Env: reset(seed)
    Env->>Gen: generate_employee_dataset(seed)
    Gen-->>Env: clean DataFrame
    Env->>Inj: inject_bugs(df, scenario)
    Inj-->>Env: corrupted df + ground_truth
    Env-->>Server: DataObservation
    Server-->>Agent: DataObservation (with max_steps)

    loop For each step
        Agent->>Server: POST /step (action, task_id)
        Server->>Env: step(action)
        Env-->>Server: StepResult
        Server-->>Agent: StepResult (observation, reward, done, info)
    end

    Agent->>Server: GET /grader?task_id=X
    Server->>Grader: grade_taskX(env)
    Grader-->>Server: GraderResult
    Server-->>Agent: GraderResult (score, breakdown, explanation)
```

## File Structure

```
├── env/
│   ├── __init__.py
│   ├── models.py           # Pydantic models: DataAction, DataObservation, StepResult, etc.
│   ├── server.py           # FastAPI endpoints: /reset, /step, /grader, /demo, etc.
│   ├── data/
│   │   ├── generator.py    # Seed-parameterized employee dataset generation
│   │   ├── bug_injector.py # Bug injection + procedural scenario generation
│   │   └── scenarios/      # Static JSON scenarios (used by /demo only)
│   ├── graders/
│   │   ├── grader1.py      # Task 1 scorer: identification + remediation
│   │   ├── grader2.py      # Task 2 scorer: rows_passing + column_recovery + type_correctness
│   │   └── grader3.py      # Task 3 scorer: diagnosis + fix + pii + validation + bonuses
│   └── tasks/
│       ├── task1_audit.py   # Data Quality Audit (easy, 10 steps)
│       ├── task2_schema.py  # Schema Drift Remediation (medium, 15 steps)
│       └── task3_incident.py# Full Data Incident Response (hard, 20 steps)
├── inference.py             # LLM agent loop with belief state tracking
├── scripts/
│   ├── validate_diversity.py# Scenario diversity validation
│   └── benchmark.py        # Automated benchmarking
├── tests/
│   ├── test_env.py          # Environment unit tests
│   ├── test_inference.py    # Inference + grader tests
│   └── test_grader2.py      # Grader 2 specific tests
├── docs/
│   ├── architecture.md      # This file
│   └── reward_design.md     # Complete reward table
├── openenv.yaml             # OpenEnv specification
├── Dockerfile               # Container deployment
└── README.md                # Project documentation
```

## Deployment Model

- **Workers**: 1 (required — state is in-process)
- **State backend**: In-process dict (`_envs`)
- **Leaderboard backend**: File-backed (`leaderboard.json`)
- **Port**: 7860 (default for HF Spaces)
