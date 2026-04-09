"""
inference.py - DataPipelineEnv Peak Agent Loop

Architecture: Observe -> Hypothesize -> Tool Call -> Update Belief -> Fix -> Verify

Features:
- Rolling window memory (last 6 pairs)
- BeliefState tracking (candidates, eliminated, confidence)
- Self-correction retry (2 retries before NOOP)
- Escalation summary at step 6
- Context compaction at step 6
- PII sanitizer on reasoning traces
- Runtime guard (19 min hard limit)
- Strict env var validation on startup
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import requests as http
from dotenv import load_dotenv
from openai import OpenAI

from env.models import ActionType

load_dotenv()


def get_runtime_config() -> dict[str, str]:
    """Load and validate runtime configuration lazily (import-safe)."""
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:7860")

    # Spec requires OPENAI_API_KEY — also support HF_TOKEN as fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if openai_key:
        token = openai_key
        llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    elif hf_token:
        token = hf_token
        llm_base_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    else:
        raise EnvironmentError(
            "Set OPENAI_API_KEY (required by OpenEnv spec) or HF_TOKEN as fallback."
        )

    return {
        "api_base_url": api_base_url,
        "model_name": model_name,
        "token": token,
        "llm_base_url": llm_base_url,
    }


# -- Constants -------------------------------------------------------------
# MAX_STEPS is read dynamically from /reset response (see run_episode).
# These are defaults used only if the server does not provide max_steps.
DEFAULT_MAX_STEPS = 20
MAX_PARSE_RETRIES = 2
ROLLING_WINDOW = 6
MAX_RUNTIME_SECS = 19 * 60
HTTP_TIMEOUT = 30

VALID_ACTION_TYPES = {
    "INSPECT",
    "RENAME_COLUMN",
    "CAST_TYPE",
    "FILL_DEFAULT",
    "DROP_COLUMN",
    "VALIDATE",
    "MASK_PII",
    "NOOP",
}

FALLBACK_ACTION = {
    "action_type": ActionType.NOOP.value,
    "target_column": None,
    "transformation": None,
    "justification": "Fallback NOOP - could not parse valid action.",
    "identified_issues": None,
}

_EPISODE_START: float = 0.0

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")


# -- Structured output for validator --------------------------------------
def print_start(task_name: str):
    print(f"[START] task={task_name}", flush=True)


def print_step(step_num: int, reward: float, action_type: str):
    print(f"[STEP] step={step_num} reward={round(reward, 4)} action={action_type}", flush=True)


def print_end(task_name: str, score: float, steps: int):
    print(f"[END] task={task_name} score={round(score, 4)} steps={steps}", flush=True)


# -- Belief state ----------------------------------------------------------
@dataclass
class BeliefState:
    candidate_causes: list[str] = field(default_factory=list)
    eliminated_causes: list[str] = field(default_factory=list)
    confirmed_fixes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    signals_unlocked: list[str] = field(default_factory=list)
    step_errors: list[str] = field(default_factory=list)

    def to_prompt_str(self) -> str:
        lines = []
        if self.candidate_causes:
            lines.append(f"Candidates: {self.candidate_causes}")
        if self.eliminated_causes:
            lines.append(f"Eliminated: {self.eliminated_causes}")
        if self.confirmed_fixes:
            lines.append(f"Fixed so far: {self.confirmed_fixes}")
        if self.confidence > 0:
            lines.append(f"Confidence: {self.confidence:.2f}")
        return " | ".join(lines) if lines else "No hypothesis yet."

    def update_confidence(self, reward: float):
        """Bayesian-lite confidence update from step reward."""
        delta = 0.15 if reward > 0.1 else (-0.10 if reward < 0 else 0.0)
        self.confidence = round(max(0.0, min(1.0, self.confidence + delta)), 3)


# -- System prompt ---------------------------------------------------------
SYSTEM_PROMPT = """You are a senior data engineer investigating a production incident.
The pipeline is broken. Revenue numbers are wrong. CEO is asking questions.

INVESTIGATION PROTOCOL - MANDATORY:
Step 1-2: ALWAYS start with broad tool scans:
  INSPECT target="metrics"    → reveals null counts
  INSPECT target="schema"     → reveals renamed columns  
  INSPECT target="pii"        → reveals SSN leaks
  INSPECT target="logs"       → reveals pipeline errors

Step 3+: Inspect specific columns based on what tools revealed.
  validation_report starts EMPTY — you cannot see bugs until you inspect.
  Do NOT attempt fixes before inspecting. You will get zero reward.

For Task 3 specifically:
  INSPECT target="stage_5" → see output anomaly
  INSPECT target="stage_4" → trace aggregation
  INSPECT target="stage_3" → find corruption entry point
  Only then apply fixes at stage_3.

Check agent_context.recommended_next in each observation.
It tells you exactly what to do next.

PHASE 2 — FIX (steps 4-6):
  Only fix what you have actually found through inspection.
  Fix highest severity bugs first (critical > high > medium > low).

  Available fixes:
  - FILL_DEFAULT target="column" transformation="fill_median" → fix nulls
  - CAST_TYPE target="column" transformation="cast_to_int/float" → fix types
  - RENAME_COLUMN target="old_name" transformation="correct_name" → fix schema drift
  - MASK_PII target="ssn" → fix PII leak (do this immediately if found)
  - DROP_COLUMN → DANGEROUS, triggers blast radius penalty

PHASE 3 — VALIDATE (steps 7-8):
  - VALIDATE target="pipeline" → confirms all fixes applied correctly
  - MASK_PII if not done yet → PII penalty is -0.20, never skip this

CRITICAL RULES:
  - If validation_report is empty, you have NOT inspected yet. INSPECT FIRST.
  - If you see SSN data anywhere, MASK_PII immediately (next step).
  - Never DROP_COLUMN without checking schema dependencies first.
  - Your justification field is graded — explain your reasoning clearly.
  - Mention specific evidence: "salary column shows 3 NULL values at rows 23,47,89"

OUTPUT: Reply ONLY with valid JSON, zero markdown, zero explanation:
{
  "action_type": "INSPECT"|"FILL_DEFAULT"|"CAST_TYPE"|"RENAME_COLUMN"|"VALIDATE"|"MASK_PII"|"DROP_COLUMN"|"NOOP",
  "target_column": "column_name or facet_name or null",
  "transformation": "cast_to_int"|"cast_to_float"|"fill_median"|"fill_zero"|"drop_duplicates" or null,
  "justification": "What evidence you saw and why you chose this action.",
  "identified_issues": [
    {
      "issue_type": "null_injection"|"type_corruption"|"out_of_range"|"format_inconsistency"|"schema_drift"|"pii_leak"|"duplicate_rows",
      "column": "column_name" or null,
      "description": "specific description of what you observed",
      "severity": "low"|"medium"|"high"|"critical"
    }
  ] or null
}"""


# -- Utility functions -----------------------------------------------------
def _check_runtime():
    if time.time() - _EPISODE_START > MAX_RUNTIME_SECS:
        print(f"\n[TIMEOUT] Exceeded {MAX_RUNTIME_SECS // 60}min. Stopping.")
        sys.exit(1)


def _sanitize_pii(text: str) -> str:
    text = _SSN_RE.sub("[SSN-REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL-REDACTED]", text)
    return text


def _parse_json_from_text(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_action(action: Optional[dict]) -> bool:
    if not isinstance(action, dict):
        return False
    if action.get("action_type") not in VALID_ACTION_TYPES:
        return False
    if not action.get("justification"):
        return False
    return True


def _truncate_messages(messages: list[dict], system_msg: dict) -> list[dict]:
    non_sys = [m for m in messages if m["role"] != "system"]
    if len(non_sys) > ROLLING_WINDOW * 2:
        non_sys = non_sys[-(ROLLING_WINDOW * 2) :]
    return [system_msg] + non_sys


def _build_escalation_summary(belief: BeliefState, step_num: int, max_steps: int = 20) -> str:
    """
    Inject compressed incident summary at compaction step.
    Replaces verbose history. Focuses agent on resolution.
    Saves token cost for remaining steps.
    """
    return (
        f"[ESCALATION SUMMARY - Step {step_num + 1}]\n"
        f"Root cause hypothesis: {belief.candidate_causes or ['undetermined']}\n"
        f"Eliminated: {belief.eliminated_causes or ['none']}\n"
        f"Fixes confirmed: {belief.confirmed_fixes or ['none']}\n"
        f"Confidence: {belief.confidence:.2f}\n"
        f"Signals unlocked: {belief.signals_unlocked or ['none']}\n"
        f"Recent errors: {belief.step_errors[-3:] or ['none']}\n"
        f"--- You have {max_steps - step_num - 1} steps left. "
        f"If all fixes applied, use VALIDATE. "
        f"If PII exposed, use MASK_PII on 'ssn' immediately. ---"
    )


def _observation_to_prompt(obs: dict, belief: BeliefState, step_num: int, max_steps: int = 20) -> str:
    lines = [
        f"=== STEP {step_num + 1}/{max_steps} ===",
        f"Stage: {obs.get('pipeline_stage', '?')} | "
        f"Remaining: {obs.get('time_remaining', 0)} | "
        f"Health: {obs.get('downstream_health', 0):.2f}",
    ]

    vis = obs.get("visible_signals") or {}
    alert = vis.get("alert")
    if alert:
        lines.append(
            f"\n[ALERT] {alert.get('severity', '?').upper()} "
            f"risk={alert.get('risk_score', 0):.2f}: {alert.get('message', '')}"
        )

    logs = vis.get("logs")
    if logs:
        lines.append(f"\n[LOGS] status={logs.get('last_run_status', '?')}")
        for err in (logs.get("recent_errors") or [])[:3]:
            lines.append(f"  x {err}")

    metrics = vis.get("metrics")
    if metrics:
        lines.append(
            f"\n[METRICS] rows={metrics.get('row_count')} "
            f"avg={metrics.get('historical_avg')} "
            f"null_ratio={metrics.get('null_ratio', 0):.3f}"
        )

    compliance = vis.get("compliance")
    if compliance:
        lines.append(
            f"\n[COMPLIANCE] pii_detected={compliance.get('pii_detected')} "
            f"risky={compliance.get('risky_columns')}"
        )

    lines.append(f"\nSchema:\n{json.dumps(obs.get('schema', {}), indent=2)}")
    preview = obs.get("dataset_preview", [])[:5]
    lines.append(f"\nDataset preview (5 rows):\n{json.dumps(preview, indent=2)}")

    if obs.get("validation_report"):
        lines.append(f"\nOpen issues:\n{json.dumps(obs['validation_report'], indent=2)}")

    belief_str = belief.to_prompt_str()
    if belief_str != "No hypothesis yet.":
        lines.append(f"\n[BELIEF STATE] {belief_str}")

    # Extract and prominently display agent_context
    agent_ctx = obs.get("agent_context", {})
    if agent_ctx:
        lines.append("\n[YOUR INVESTIGATION STATE]")
        
        bugs_found = agent_ctx.get("bugs_found", [])
        if bugs_found:
            lines.append(f"Bugs discovered so far: {bugs_found}")
        else:
            lines.append("Bugs discovered so far: NONE — you must INSPECT first")
        
        bugs_fixed = agent_ctx.get("bugs_fixed", [])
        if bugs_fixed:
            lines.append(f"Bugs fixed so far: {bugs_fixed}")
        
        tools = agent_ctx.get("tools_available", [])
        if tools:
            lines.append(f"Available tools: {tools}")
        
        stages = agent_ctx.get("stages_inspected", [])
        if stages:
            lines.append(f"Pipeline stages inspected: {stages}")
        
        recommended = agent_ctx.get("recommended_next", "")
        if recommended:
            lines.append(f"\n*** RECOMMENDED NEXT ACTION: {recommended} ***")
            lines.append("Following this recommendation will maximize your score.")

    lines.append("\nWhat is your next action?")
    return "\n".join(lines)


def _update_belief(belief: BeliefState, action: dict, result: dict) -> BeliefState:
    action_type = action.get("action_type", ActionType.NOOP.value)
    target = action.get("target_column", "") or ""
    justif = _sanitize_pii(action.get("justification", "")).lower()
    reward = float(result.get("reward", 0.0))
    info = result.get("info", {})

    keywords = [
        "stage 3",
        "schema drift",
        "type mismatch",
        "ssn",
        "pii",
        "revenue",
        "aggregation",
        "null",
        "duplicate",
        "join",
    ]

    for kw in keywords:
        if kw in justif and kw not in belief.candidate_causes:
            belief.candidate_causes.append(kw)

    if reward < -0.05 and target:
        attempt = f"{action_type}:{target}"
        if attempt not in belief.eliminated_causes:
            belief.eliminated_causes.append(attempt)

    for bug_id in info.get("fixed", []):
        if bug_id not in belief.confirmed_fixes:
            belief.confirmed_fixes.append(bug_id)

    for sig in info.get("signals_unlocked", []):
        if sig not in belief.signals_unlocked:
            belief.signals_unlocked.append(sig)

    belief.update_confidence(reward)
    return belief


# Compatibility shims for existing tests and imports

def _update_belief_state(belief: dict, action: dict, result: dict) -> dict:
    state = BeliefState(
        candidate_causes=list(belief.get("candidates", [])),
        eliminated_causes=list(belief.get("eliminated", [])),
        confirmed_fixes=list(belief.get("fixes_done", [])),
        confidence=float(belief.get("confidence", 0.0)),
        signals_unlocked=list(belief.get("signals_unlocked", [])),
    )
    updated = _update_belief(state, action, result)
    belief["candidates"] = list(updated.candidate_causes)
    belief["eliminated"] = list(updated.eliminated_causes)
    belief["fixes_done"] = list(updated.confirmed_fixes)
    belief["signals_unlocked"] = list(updated.signals_unlocked)
    belief["confidence"] = float(updated.confidence)
    return belief


def _compaction_summary(belief_state: dict, step_errors: list[str], max_steps: int = 20) -> str:
    compaction_step = max(5, max_steps // 3)
    state = BeliefState(
        candidate_causes=list(belief_state.get("candidates", [])),
        eliminated_causes=list(belief_state.get("eliminated", [])),
        confirmed_fixes=list(belief_state.get("confirmed", []))
        + list(belief_state.get("fixes_done", [])),
        confidence=float(belief_state.get("confidence", 0.0)),
        signals_unlocked=list(belief_state.get("signals_unlocked", [])),
        step_errors=list(step_errors),
    )
    return _build_escalation_summary(state, compaction_step, max_steps=max_steps)


# -- Episode loop ----------------------------------------------------------
def run_episode(task_id: int, config: dict[str, str], client: OpenAI, seed: int | None = None) -> float:
    _check_runtime()

    TASK_NAMES = {
        1: "DataQualityAudit",
        2: "SchemaDriftRemediation",
        3: "FullIncidentResponse",
    }
    task_name = TASK_NAMES.get(task_id, f"Task{task_id}")

    print_start(task_name)

    params = {"task_id": task_id}
    if seed is not None:
        params["seed"] = seed

    resp = http.post(
        f"{config['api_base_url']}/reset",
        params=params,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    obs = resp.json()

    # Read max_steps from reset response (Fix 1.2 / 6.1)
    max_steps = int(obs.get("max_steps", DEFAULT_MAX_STEPS))
    compaction_step = max(5, max_steps // 3)

    # Fix 2.3: visible_signals is empty after reset (no step has occurred)
    # Only populate from StepResult.info starting at step 1
    obs["visible_signals"] = {}

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Task {task_id}: max_steps={max_steps}, initial_signals=[]")

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system_msg]
    belief = BeliefState()

    for step_num in range(max_steps):
        _check_runtime()

        if step_num == compaction_step:
            summary = _build_escalation_summary(belief, step_num, max_steps=max_steps)
            non_sys = [m for m in messages if m["role"] != "system"]
            last_two = non_sys[-2:] if len(non_sys) >= 2 else non_sys
            messages = [system_msg, {"role": "user", "content": summary}] + last_two
            print(
                f"  [ESCALATION] Injected summary - "
                f"confidence={belief.confidence:.2f} "
                f"candidates={belief.candidate_causes}"
            )

        user_msg = _observation_to_prompt(obs, belief, step_num, max_steps=max_steps)
        messages.append({"role": "user", "content": user_msg})
        messages = _truncate_messages(messages, system_msg)

        action = None
        last_error = ""

        for attempt in range(MAX_PARSE_RETRIES + 1):
            if attempt > 0:
                correction = (
                    f"Your previous output was invalid: {last_error}. "
                    f"Reply ONLY with valid JSON. No markdown. No text."
                )
                messages.append({"role": "user", "content": correction})

            try:
                response = client.chat.completions.create(
                    model=config["model_name"],
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
            except Exception as exc:
                last_error = str(exc)
                belief.step_errors.append(f"LLM error: {exc}")
                print(f"  [LLM ERROR] {exc}")
                raw = ""

            parsed = _parse_json_from_text(raw) if raw else None
            if parsed and _validate_action(parsed):
                action = parsed
                messages.append({"role": "assistant", "content": raw})
                break

            last_error = (
                f"Invalid action_type "
                f"'{parsed.get('action_type') if parsed else 'none'}' "
                f"or missing justification"
            )

        if action is None:
            action = FALLBACK_ACTION
            messages.append({"role": "assistant", "content": json.dumps(FALLBACK_ACTION)})
            print(f"  [FALLBACK] NOOP after {MAX_PARSE_RETRIES} retries")

        try:
            step_resp = http.post(
                f"{config['api_base_url']}/step",
                json=action,
                params={"task_id": task_id},
                timeout=HTTP_TIMEOUT,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as exc:
            print(f"  [STEP ERROR] {exc}")
            belief.step_errors.append(str(exc))
            break

        obs = result.get("observation", obs)
        done = result.get("done", False)
        reward = float(result.get("reward", 0.0))
        print_step(step_num + 1, reward, action.get("action_type", "NOOP"))
        info = result.get("info", {})
        obs["visible_signals"] = info.get("visible_signals", {})

        belief = _update_belief(belief, action, result)

        justif_short = _sanitize_pii(action.get("justification", ""))[:55]
        print(
            f"  Step {step_num + 1:02d} | "
            f"{action['action_type']:15s} | "
            f"target={str(action.get('target_column', ''))[:10]:10s} | "
            f"reward={reward:+.3f} | "
            f"health={obs.get('downstream_health', 0):.2f} | "
            f"conf={belief.confidence:.2f} | "
            f"done={done}"
        )
        print(f"          '{justif_short}...'")

        if done:
            print(f"  [DONE] Completed at step {step_num + 1}")
            break

    try:
        grade_resp = http.get(
            f"{config['api_base_url']}/grader",
            params={"task_id": task_id},
            timeout=HTTP_TIMEOUT,
        )
        grade_resp.raise_for_status()
        grade = grade_resp.json()
    except Exception as exc:
        print(f"  [GRADER ERROR] {exc}")
        return 0.0

    score = float(grade.get("score", 0.0))
    breakdown = grade.get("breakdown", {})
    explanation = grade.get("explanation", "")

    print("\n  GRADER:")
    for key, value in breakdown.items():
        print(f"    {key:30s} = {value}")
    print(f"  {explanation}")
    print("\n  BeliefState final:")
    print(f"    candidates  = {belief.candidate_causes}")
    print(f"    confirmed   = {belief.confirmed_fixes}")
    print(f"    confidence  = {belief.confidence:.2f}")
    print(f"    signals     = {belief.signals_unlocked}")

    print_end(task_name, score, step_num + 1)

    return score


# -- Entry point -----------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="DataPipelineEnv Inference Loop")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="Specific task ID to run (1, 2, or 3)")
    parser.add_argument("--seed", type=int, help="Seed for procedural generation")
    args = parser.parse_args()

    global _EPISODE_START
    config = get_runtime_config()
    client = OpenAI(
        api_key=config["token"],
        base_url=config["llm_base_url"],
    )

    _EPISODE_START = time.time()
    scores: dict[int, float] = {}

    tasks_to_run = [args.task] if args.task else [1, 2, 3]

    for task_id in tasks_to_run:
        print(f"\n{'=' * 65}")
        print(f"  TASK {task_id}")
        print(f"{'=' * 65}")
        try:
            scores[task_id] = run_episode(task_id, config, client, seed=args.seed)
        except Exception as exc:
            print(f"  [TASK {task_id} FAILED] {exc}")
            scores[task_id] = 0.0

        elapsed = time.time() - _EPISODE_START
        print(
            f"\n  -> Task {task_id} Score: {scores[task_id]:.4f} "
            f"(elapsed: {elapsed:.0f}s/{MAX_RUNTIME_SECS}s)"
        )

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n{'=' * 65}")
    print("  FINAL RESULTS")
    print(f"  Task 1: {scores.get(1, 0):.4f}")
    print(f"  Task 2: {scores.get(2, 0):.4f}")
    print(f"  Task 3: {scores.get(3, 0):.4f}")
    print(f"  Average: {avg:.4f}")
    print(f"{'=' * 65}")

    try:
        http.post(
            f"{config['api_base_url']}/record_score",
            json={
                "task_1": scores.get(1, 0.0),
                "task_2": scores.get(2, 0.0),
                "task_3": scores.get(3, 0.0),
                "average": avg,
                "model": config["model_name"],
            },
            timeout=10,
        )
    except Exception:
        pass  # never crash on leaderboard update


if __name__ == "__main__":
    main()
