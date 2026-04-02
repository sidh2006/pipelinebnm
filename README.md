---
title: Broken Pipeline Env
emoji: 🔧
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# DataPipelineEnv

Every company's revenue dashboard has been wrong at 2 AM.
A data engineer gets the call. We built the environment to
benchmark whether an AI agent can do their job.

## The Problem

Bad data costs companies $12.9M/year on average (IBM, 2022).
Every data team faces broken pipelines. Until now, there was
no standardized benchmark to evaluate whether an AI agent
can diagnose and fix them. DataPipelineEnv fills that gap.

## What Makes This Different

**Progressive Discovery**: The agent cannot see bugs until it inspects.
Unlike static environments, DataPipelineEnv requires the agent to
investigate before it can act — just like a real engineer would.

**Blast Radius**: Wrong actions cascade. Drop the wrong column
and downstream tables break too. The agent must understand
data dependencies, not just column names.

**Curriculum Design**: Three tasks form a strict skill progression.
Skills learned in Task 1 transfer directly to Tasks 2 and 3.
This enables meaningful RL training signal across difficulty levels.

**Reproducible Grading**: Bugs are injected from fixed scenario files.
The grader compares agent output against the exact ground truth
we planted. Scores are 100% deterministic.

## The 3 Tasks

| Task | Difficulty | What the Agent Does | Baseline (NOOP) | Empirical Agent Score |
|------|-----------|---------------------|-----------------|-----------------------|
| 1 — Data Quality Audit | Easy | Find and fix nulls, type errors, duplicates | < 0.10 | 0.85 – 0.95 |
| 2 — Schema Drift | Medium | Fix renamed columns, type changes, missing fields | < 0.80 | 0.70 – 0.85 |
| 3 — Incident Response | Hard | Trace 5-stage pipeline, fix, PII sweep, validate | < 0.10 | 0.40 – 0.60 |

## Action Space

| Action | Description | When to Use |
|--------|-------------|-------------|
| `INSPECT` | Reveals bugs in a column or facet | Always first |
| `FILL_DEFAULT` | Fill NULLs with median/zero | After inspecting null column |
| `CAST_TYPE` | Fix column data type | After finding type corruption |
| `RENAME_COLUMN` | Fix schema drift | After schema_diff reveals rename |
| `MASK_PII` | Redact sensitive data | Immediately on SSN detection |
| `VALIDATE` | Confirm all fixes and close episode | After all fixes applied |
| `DROP_COLUMN` | Remove column (triggers blast radius) | Avoid unless necessary |
| `NOOP` | No operation | Fallback only |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `dataset_preview` | list[dict] | First 10 rows |
| `schema` | dict | Column types and nullable flags |
| `pipeline_stage` | str | Current ETL stage |
| `validation_report` | list | Bugs discovered so far (empty at reset) |
| `time_remaining` | int | Steps left |
| `downstream_health` | float | 0.0=broken, 1.0=fixed |
| `agent_context` | dict | Investigation state, recommendations |
| `pipeline_stage_health` | dict | Per-stage health (Task 3 only) |

## Reward Function

| Event | Reward |
|-------|--------|
| Broad scan (metrics/logs/pii/schema) | +0.05 |
| Discover real bug via INSPECT | +0.15 |
| Correct fix applied | +0.20 |
| VALIDATE after all fixes | +0.25 |
| All bugs fixed (completion) | +0.30 |
| Re-inspect same target | -0.05 |
| Fix before discovering | -0.10 |
| PII not masked | -0.20 |

## Quickstart
```bash
git clone https://github.com/Nithesh1109/broken-pipeline-env
cd broken-pipeline-env
pip install -r requirements.txt
uvicorn env.server:app --port 7860
```

## Docker
```bash
docker build -t pipeline-env .
docker run -p 7860:7860 pipeline-env
curl http://localhost:7860/ping
```

## Run the Agent
```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_token

# Run all 3 tasks sequentially
python inference.py

# Run a specific task with a procedural seed
python inference.py --task 2 --seed 42
```

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |
| `/tasks` | GET | Task list with curriculum info |
| `/reset` | POST | Start episode (accepts optional `seed` and `scenario_override` for procedural generation/static JSONs) |
| `/step` | POST | Submit action `action_type`, `target_column`, `transformation`, `justification` |
| `/grader` | GET | Generate detailed grading report and score 0.0–1.0 |
| `/tools` | GET | Available investigation tools |
| `/demo` | POST | Automatically evaluate NOOP or baseline agent solve |
| `/replay` | GET | Replay any episode step by step |
| `/leaderboard` | GET | Score history across runs |
| `/baseline` | GET | NOOP agent scores |
| `/mcp` | POST | MCP tool discovery (JSON-RPC 2.0) |

## Live Demo

Space: https://nidhishmg10-broken-pipeline-env.hf.space
Docs: https://nidhishmg10-broken-pipeline-env.hf.space/docs