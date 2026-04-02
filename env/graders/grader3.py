from __future__ import annotations

from env.models import GraderResult
from env.tasks.task3_incident import Task3IncidentEnv

WEIGHTS = {
    "diagnosis": 0.25,
    "fix": 0.35,
    "pii_sweep": 0.20,
    "validation": 0.20,
}

DIAGNOSIS_KEYWORDS = [
    "stage 3",
    "join stage",
    "schema drift",
    "ssn",
    "pii",
    "type mismatch",
    "revenue",
    "aggregation",
    "rev_amt",
    "corruption",
    "join failure",
    "type error",
]

EXACT_STAGE_KEYWORDS = [
    "stage_3_join",
    "stage 3 join",
    "join stage corruption",
    "corruption at stage 3",
    "stage3",
]


def _action_is_substantive(action_type: str) -> bool:
    """Return True if the action_type represents a real investigation or fix step."""
    return action_type.upper() not in {"NOOP", ""}


def _contextual_reasoning_bonus(env: Task3IncidentEnv) -> float:
    """
    Award up to +0.05 for correct diagnostic language in justifications,
    but ONLY when the agent has actually performed substantive actions
    AND the keywords relate to signals the agent has already unlocked.

    Scoring rules:
      - NOOP actions are excluded entirely (prevents keyword stuffing)
      - Keywords only count if the agent has unlocked the corresponding signal:
        * "ssn"/"pii" keywords require "compliance" signal unlocked
        * "schema drift"/"rev_amt"/"type mismatch" require "schema" or "logs" unlocked
        * "stage 3"/"join"/"aggregation"/"revenue" require any stage inspected
      - Max bonus: 0.05 (1 qualifying keyword × 0.05, hard cap)
    """
    if not getattr(env, "aer_history", None):
        return 0.0

    signals = getattr(env, "signals_unlocked", set())
    stages = getattr(env, "stages_inspected", set())

    # Only consider justifications from substantive (non-NOOP) actions
    substantive_justifications = " ".join(
        r.justification.lower()
        for r in env.aer_history
        if _action_is_substantive(r.action_type)
    )
    if not substantive_justifications:
        return 0.0

    # Context-aware keyword groups: each requires a specific signal/stage
    CONTEXTUAL_KEYWORDS: list[tuple[list[str], set[str], set[str]]] = [
        # (keywords, required_signals, required_stages)
        (["ssn", "pii"], {"compliance"}, set()),
        (["schema drift", "rev_amt", "type mismatch", "type error"], {"schema", "logs"}, set()),
        (["stage 3", "join stage", "join", "corruption"], set(), {"stage_3"}),
        (["revenue", "aggregation"], set(), {"stage_4", "stage_5"}),
    ]

    hits = 0
    for keywords, req_signals, req_stages in CONTEXTUAL_KEYWORDS:
        # Check if the agent has unlocked the required context
        has_signal_context = not req_signals or bool(req_signals & signals)
        has_stage_context = not req_stages or bool(req_stages & stages)
        if not (has_signal_context or has_stage_context):
            continue
        for kw in keywords:
            if kw in substantive_justifications:
                hits += 1
                break  # One hit per group is enough

    return round(min(0.05 * hits, 0.05), 4)


def _root_cause_attribution(env: Task3IncidentEnv) -> float:
    """
    Bonus for agents that identify the exact corruption entry point.
    Requires precise stage identification, not only broad keywords.
    """
    if not getattr(env, "aer_history", None):
        return 0.0
    combined = " ".join(r.justification.lower() for r in env.aer_history)
    for kw in EXACT_STAGE_KEYWORDS:
        if kw in combined:
            return 0.05
    return 0.0


def _signals_investigation_bonus(env: Task3IncidentEnv) -> float:
    """
    Reward systematic investigation across facets.
    +0.02 per unlocked facet, max +0.08.
    """
    unlocked = len(getattr(env, "signals_unlocked", set()))
    return round(min(0.02 * unlocked, 0.08), 4)


def _efficiency_bonus(env: Task3IncidentEnv) -> float:
    """Small bonus for completing all sub-tasks efficiently.
    
    Uses env.MAX_STEPS (default 20) instead of hardcoded value.
    Result is clamped to >= 0.0 to prevent negative bonus.
    """
    all_done = (
        env.diagnosis_correct
        and env.fix_applied
        and env.pii_masked
        and env.validation_passed
    )
    if not all_done:
        return 0.0
    max_steps = max(1, int(getattr(env, "MAX_STEPS", 20)))
    steps_used = int(getattr(env, "step_count", max_steps))
    steps_used = max(0, steps_used)
    return round(max(0.0, 0.03 * (1.0 - (steps_used / max_steps))), 4)


def grade_task3(env: Task3IncidentEnv) -> GraderResult:
    """
    Task 3: Full Incident Response scorer.

    Weighted sub-scores:
      diagnosis  × 0.25
      fix        × 0.35
      pii_sweep  × 0.20
      validation × 0.20

        Bonuses (additive, capped by final clamp):
            contextual_reasoning_bonus  up to +0.05 (anti-gaming: requires substantive actions + signal context)
            root_cause_attribution      up to +0.05
            signals_investigation       up to +0.08
            efficiency_bonus            up to +0.03

        Penalties:
            pii_compliance_penalty = -0.20 if pii_masked is False
            (never -100)

        Final: clamp(weighted + penalties + bonuses, 0.0, 1.0)
    """
    sub = {
        "diagnosis": 1.0 if env.diagnosis_correct else 0.0,
        "fix": 1.0 if env.fix_applied else 0.0,
        "pii_sweep": 1.0 if env.pii_masked else 0.0,
        "validation": 1.0 if env.validation_passed else 0.0,
    }
    weighted = sum(WEIGHTS[k] * v for k, v in sub.items())
    pii_penalty = -0.20 if not env.pii_masked else 0.0
    reasoning_bon = _contextual_reasoning_bonus(env)
    root_cause_bon = _root_cause_attribution(env)
    signals_bon = _signals_investigation_bonus(env)
    efficiency_bon = _efficiency_bonus(env)

    total_bonus = reasoning_bon + root_cause_bon + signals_bon + efficiency_bon
    score = round(max(0.0, min(1.0, weighted + pii_penalty + total_bonus)), 4)

    return GraderResult(
        score=score,
        breakdown={
            **{k: round(v, 4) for k, v in sub.items()},
            "pii_compliance_penalty": round(pii_penalty, 4),
            "reasoning_bonus": reasoning_bon,
            "root_cause_attribution": root_cause_bon,
            "signals_investigation": signals_bon,
            "efficiency_bonus": efficiency_bon,
            "total_bonus": round(total_bonus, 4),
            "signals_unlocked": float(len(getattr(env, "signals_unlocked", set()))),
            "downstream_health": round(env.downstream_health, 4),
        },
        explanation=(
            f"D:{sub['diagnosis']} F:{sub['fix']} "
            f"P:{sub['pii_sweep']} V:{sub['validation']} | "
            f"weighted={round(weighted, 3)} "
            f"pii_pen={pii_penalty} "
            f"bonuses={round(total_bonus, 3)} "
            f"-> {score}"
        ),
    )