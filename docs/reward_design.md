# DataPipelineEnv — Reward Design

This document describes the complete reward function used across all three tasks.

## Core Principles

1. **Inspect-before-fix**: Attempting a fix before discovering the bug results in a penalty (-0.10)
2. **Progressive discovery**: Bugs are only visible in `validation_report` after inspection
3. **Shaped completion**: The +0.30 completion bonus is distributed as progress signals throughout the episode, not as a single terminal spike
4. **Re-inspection penalty**: Re-inspecting the same target costs -0.10 (discourages random exploration)

## Reward Table

| Action | Condition | Reward | Rationale |
|--------|-----------|--------|-----------|
| `INSPECT` (broad scan) | First scan of metrics/logs/pii/schema/dag | +0.05 | Encourages systematic investigation |
| `INSPECT` (column) | Discovers a real bug | +0.15 | Rewards targeted inspection |
| `INSPECT` (column) | No bug found | -0.05 | Discourages random column sampling |
| `INSPECT` (any) | Re-inspecting same target | -0.10 | Prevents exploration loops |
| `FILL_DEFAULT` | Correct fix on discovered null bug | +0.20 | Rewards correct remediation |
| `CAST_TYPE` | Correct fix on discovered type bug | +0.20 | Rewards correct remediation |
| `RENAME_COLUMN` | Correct schema drift fix | +0.20 | Rewards correct remediation |
| `MASK_PII` | Correct PII masking | +0.20 | Rewards compliance action |
| `VALIDATE` | All bugs fixed — completion | +0.25 | Immediate validation bonus |
| `VALIDATE` | All bugs fixed — terminal | +residual of 0.30 | Shaped completion residual (see below) |
| `VALIDATE` | Not all bugs fixed | -0.05 | Premature validation penalty |
| `DROP_COLUMN` | Any column | -0.10 | Blast radius risk |
| Fix before discovery | Bug not yet inspected | -0.10 | Enforces inspect-before-fix |
| `NOOP` | Always | 0.0 | Neutral fallback |
| PII not masked (grader) | End of episode | -0.20 | Compliance penalty |

## Shaped Completion Bonus

The +0.30 completion bonus is **not** awarded as a single spike at VALIDATE.
Instead, it is distributed using potential-based shaping:

```
When bugs_fixed increases (and not all bugs are fixed yet):
    progress_reward = 0.30 * (delta_fixed / total_bugs)

When VALIDATE fires (all bugs fixed):
    residual = 0.30 - (0.30 * prev_fixed / total_bugs)
```

This means:
- Each fix contributes a proportional share of the 0.30 bonus
- By the time VALIDATE fires, most of the 0.30 has already been distributed
- The terminal spike is at most +0.25 + small residual (instead of +0.55)
- Total reward across the episode is unchanged

## Diminishing Discovery Bonus

- First INSPECT of a column that discovers a bug: +0.15
- Second time the same target is inspected: -0.10 (penalty, not bonus)
- This removes the "random inspection" strategy

## Reward Range

- Step reward: [-0.5, 1.0] (clamped)
- Grader output: [0.0, 1.0]
