from __future__ import annotations

from env.models import GraderResult
from env.tasks.task1_audit import Task1AuditEnv


def grade_task1(env: Task1AuditEnv) -> GraderResult:
    """
    Task 1: Data Quality Audit scorer.

    Formula:
      id_score         = len(identified_bug_ids) / TOTAL_BUGS
      fix_score        = len(fixed_bug_ids)      / TOTAL_BUGS
      base             = 0.4 * id_score + 0.6 * fix_score
      efficiency_bonus = 0.05 * (1 - steps_used/max_steps)
                         only when all bugs fixed
      score            = clamp(base + efficiency_bonus, 0.0, 1.0)
    """
    total = env.TOTAL_BUGS
    if total == 0:
        return GraderResult(
            score=0.0,
            breakdown={},
            explanation="No bugs defined in scenario.",
        )

    num_id = len(env.identified_bug_ids)
    num_fix = len(env.fixed_bug_ids)

    id_score = num_id / total
    fix_score = num_fix / total
    raw = 0.4 * id_score + 0.6 * fix_score

    max_steps = max(1, int(getattr(env, "MAX_STEPS", 8)))
    steps_used = int(getattr(env, "step_count", max_steps))
    steps_used = max(0, steps_used)
    if num_fix == total:
        efficiency_bonus = round(0.05 * (1.0 - (steps_used / max_steps)), 4)
    else:
        efficiency_bonus = 0.0

    score = round(max(0.0, min(1.0, raw + efficiency_bonus)), 4)

    return GraderResult(
        score=score,
        breakdown={
            "identification": round(id_score, 4),
            "remediation": round(fix_score, 4),
            "efficiency_bonus": efficiency_bonus,
            "bugs_identified": float(num_id),
            "bugs_fixed": float(num_fix),
            "total_bugs": float(total),
            "steps_used": float(steps_used),
        },
        explanation=(
            f"Identified {num_id}/{total} (weight 0.4), "
            f"fixed {num_fix}/{total} (weight 0.6), "
            f"efficiency_bonus={efficiency_bonus}. "
            f"Score={score}"
        ),
    )