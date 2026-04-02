from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from env.data.bug_injector import (
    build_logs_facet,
    build_metrics_facet,
    generate_scenario,
    get_failure_signature,
    inject_bugs,
    load_scenario,
    matches_ground_truth,
)
from env.data.generator import generate_employee_dataset
from env.models import (
    AERRecord,
    ActionType,
    AlertSignal,
    ComplianceFacet,
    DagOverview,
    DataAction,
    DataObservation,
    DetectedIssue,
    StepResult,
    VisibleSignals,
)


class Task1AuditEnv:
    """Task 1 environment for data quality audit and direct remediation."""

    MAX_STEPS = 10
    TOTAL_BUGS = 5
    SCENARIO_DIR = Path(__file__).parent.parent / "data" / "scenarios"

    def __init__(self) -> None:
        """Initialize Task1 mutable state containers."""
        self.df: pd.DataFrame = pd.DataFrame()
        self.ground_truth: list[dict] = []
        self.step_count: int = 0
        self.identified_bug_ids: set[str] = set()
        self.fixed_bug_ids: set[str] = set()
        self.discovered_bugs: set[str] = set()          # NEW: progressive discovery
        self.downstream_health: float = 1.0
        self.visible_signals: VisibleSignals | None = None
        self.signals_unlocked: set[str] = set()
        self.aer_history: list[AERRecord] = []
        self.step_errors: list[str] = []
        self.inspected_targets: set[str] = set()

    def reset(self, seed: int = 42, scenario_override: str | None = None) -> DataObservation:
        """Reset state, generate deterministic data, inject bugs, and return observation."""
        if scenario_override:
            chosen = self.SCENARIO_DIR / scenario_override
            scenario = load_scenario(str(chosen))
        else:
            # Procedural generation for live episodes
            scenario = generate_scenario(seed=seed, task_id="task1", difficulty="easy")
        clean_df = generate_employee_dataset(seed=seed)
        self.df, self.ground_truth = inject_bugs(clean_df, scenario)
        self.step_count = 0
        self.identified_bug_ids = set()
        self.fixed_bug_ids = set()
        self.discovered_bugs = set()                     # NEW: starts empty
        self.downstream_health = 1.0

        failure_sig = get_failure_signature(self.ground_truth)
        initial_alert = AlertSignal(
            severity="high",
            message=f"Pipeline anomaly: {failure_sig.detection_hint}",
            risk_score=0.65,
        )
        self.visible_signals = VisibleSignals(alert=initial_alert)
        self.signals_unlocked = set()
        self.aer_history = []
        self.step_errors = []
        self.inspected_targets = set()

        return self._build_observation()

    def step(self, action: DataAction) -> StepResult:
        """Apply an agent action and return the resulting transition tuple."""
        reward = 0.0
        done = False
        prev_fixed_count = len(self.fixed_bug_ids)

        if action.action_type == ActionType.INSPECT:
            target = (action.target_column or "").lower()

            _TOOL_ALIASES = {
                "run_null_check": "metrics",
                "run_type_check": "schema",
                "run_duplicate_check": "metrics",
                "run_pii_scan": "pii",
                "run_schema_diff": "schema",
                "trace_pipeline_stage": "dag",
            }
            target = _TOOL_ALIASES.get(target, target)

            reinspecting = target in self.inspected_targets
            if reinspecting:
                reward -= 0.10  # Raised re-inspect penalty
            else:
                self.inspected_targets.add(target)

            if target == "metrics" and "metrics" not in self.signals_unlocked:
                self.visible_signals.metrics = build_metrics_facet(self.df)
                self.signals_unlocked.add("metrics")
                if not reinspecting: reward += 0.05
            elif target == "logs" and "logs" not in self.signals_unlocked:
                self.visible_signals.logs = build_logs_facet(self.step_errors or ["No errors logged"])
                self.signals_unlocked.add("logs")
                if not reinspecting: reward += 0.05
            elif target == "dag" and "dag" not in self.signals_unlocked:
                self.visible_signals.dag = DagOverview(current_node="stage_1_audit", upstream_nodes=["ingestion"], downstream_nodes=["reporting"])
                self.signals_unlocked.add("dag")
                if not reinspecting: reward += 0.05
            elif target in ["pii", "ssn", "compliance"] and "compliance" not in self.signals_unlocked:
                pii_cols = [col for col in self.df.columns if "ssn" in col.lower()]
                self.visible_signals.compliance = ComplianceFacet(pii_detected=len(pii_cols) > 0, risky_columns=pii_cols)
                self.signals_unlocked.add("compliance")
                if not reinspecting: reward += 0.05
            elif target == "schema" and "schema" not in self.signals_unlocked:
                for bug in self.ground_truth:
                    if bug["type"] == "schema_drift" and bug["bug_id"] not in self.discovered_bugs:
                        self.discovered_bugs.add(bug["bug_id"])
                        if bug["bug_id"] not in self.identified_bug_ids:
                            self.identified_bug_ids.add(bug["bug_id"])
                        if not reinspecting: reward += 0.15
                self.signals_unlocked.add("schema")
                if not reinspecting: reward += 0.05
            else:
                found_any = False
                for bug in self.ground_truth:
                    bug_col = (bug.get("column") or "").lower()
                    if bug_col and bug_col == target and bug["bug_id"] not in self.discovered_bugs:
                        self.discovered_bugs.add(bug["bug_id"])
                        self.identified_bug_ids.add(bug["bug_id"])
                        found_any = True
                if not reinspecting:
                    if found_any:
                        reward += 0.15
                    else:
                        reward -= 0.05

        elif action.action_type == ActionType.FILL_DEFAULT:
            col = action.target_column
            matching_bug = next((b for b in self.ground_truth if b["type"] == "null_injection" and b.get("column") == col and b["bug_id"] not in self.fixed_bug_ids), None)
            if matching_bug:
                if matching_bug["bug_id"] not in self.discovered_bugs:
                    reward -= 0.10
                elif action.transformation == "fill_median":
                    median_val = pd.to_numeric(self.df[col], errors="coerce").median()
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(median_val)
                    reward += 0.20
                    self.fixed_bug_ids.add(matching_bug["bug_id"])
                elif action.transformation == "fill_zero":
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
                    reward += 0.20
                    self.fixed_bug_ids.add(matching_bug["bug_id"])
                else:
                    reward -= 0.05
            else:
                reward -= 0.05

        elif action.action_type == ActionType.CAST_TYPE:
            col = action.target_column
            matching_bugs = [b for b in self.ground_truth if b["type"] in ["type_corruption", "out_of_range"] and b.get("column") == col and b["bug_id"] not in self.fixed_bug_ids]
            if matching_bugs:
                if not all(b["bug_id"] in self.discovered_bugs for b in matching_bugs):
                    reward -= 0.10
                elif action.transformation in ["cast_to_int", "cast_to_float"]:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                    if action.transformation == "cast_to_int":
                        median_val = int(self.df[col].median())
                        self.df[col] = self.df[col].fillna(median_val).astype(object)
                    for b in matching_bugs:
                        self.fixed_bug_ids.add(b["bug_id"])
                        reward += 0.20
                else:
                    reward -= 0.05
            else:
                reward -= 0.05

        elif action.action_type == ActionType.VALIDATE:
            fixed_this_step = False
            for bug in self.ground_truth:
                if bug["bug_id"] in self.fixed_bug_ids or bug["bug_id"] not in self.discovered_bugs: continue
                if bug["type"] == "format_inconsistency":
                    col = bug.get("column", "phone")
                    if col in self.df.columns:
                        self.df[col] = self.df[col].astype(str).str.replace(r"[^\d]", "", regex=True).str[-10:]
                    self.fixed_bug_ids.add(bug["bug_id"])
                    fixed_this_step = True
                elif bug["type"] == "duplicate_rows":
                    self.df = self.df.drop_duplicates().reset_index(drop=True)
                    self.fixed_bug_ids.add(bug["bug_id"])
                    fixed_this_step = True
            if fixed_this_step:
                reward += 0.20
            if len(self.fixed_bug_ids) == self.TOTAL_BUGS:
                reward += 0.25
                # Shaped completion: residual bonus (most already distributed via progress)
                already_shaped = 0.30 * (prev_fixed_count / self.TOTAL_BUGS)
                reward += round(0.30 - already_shaped, 4)
                done = True
            elif not fixed_this_step:
                reward -= 0.05

        elif action.action_type == ActionType.DROP_COLUMN:
            reward -= 0.10

        elif action.action_type == ActionType.NOOP:
            reward = 0.0

        else:
            reward -= 0.10

        reward = max(-0.5, min(1.0, reward))

        # Shaped completion bonus: per-step progress signal when bugs_fixed increases
        new_fixed_count = len(self.fixed_bug_ids)
        if new_fixed_count > prev_fixed_count and new_fixed_count < self.TOTAL_BUGS:
            progress_reward = 0.30 * ((new_fixed_count - prev_fixed_count) / self.TOTAL_BUGS)
            reward += round(progress_reward, 4)
            reward = max(-0.5, min(1.0, reward))

        self.step_count += 1
        self.downstream_health = len(self.fixed_bug_ids) / self.TOTAL_BUGS
        done = done or (self.step_count >= self.MAX_STEPS)

        aer = AERRecord(
            step_id=self.step_count,
            action_type=action.action_type.value,
            target=action.target_column,
            justification=action.justification,
            reward_earned=round(reward, 4),
            issues_identified=list(self.identified_bug_ids),
            issues_fixed=list(self.fixed_bug_ids),
        )
        self.aer_history.append(aer)

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "fixed": list(self.fixed_bug_ids),
                "identified": list(self.identified_bug_ids),
                "signals_unlocked": list(self.signals_unlocked),
                "visible_signals": self.visible_signals.model_dump() if self.visible_signals else {},
                "aer_last": aer.model_dump(),
                "step": self.step_count,
                "action_space_hints": {
                    "accepted_casts": ["cast_to_int", "cast_to_float"],
                    "accepted_fills": ["fill_median", "fill_zero"],
                },
            },
        )

    def _build_observation(self) -> DataObservation:
        """Construct a DataObservation from current in-memory state.

        Progressive discovery: only bugs in `discovered_bugs` AND not yet
        fixed are shown in validation_report.  At reset this is empty.
        """
        # Only show bugs the agent has actually discovered via INSPECT
        visible_bugs = [
            t for t in self.ground_truth
            if t["bug_id"] in self.discovered_bugs and t["bug_id"] not in self.fixed_bug_ids
        ]
        validation_report = [
            DetectedIssue(
                issue_type=b["type"],
                column=b.get("column"),
                description=b["description"],
                severity=b["severity"],
            )
            for b in visible_bugs
        ]

        schema_dict = {
            col: {
                "type": str(dtype),
                "nullable": bool(self.df[col].isna().any()),
            }
            for col, dtype in self.df.dtypes.items()
        }

        # Build agent_context for belief tracking
        agent_context = {
            "inspected_columns": sorted(
                {(b.get("column") or "").lower() for b in self.ground_truth if b["bug_id"] in self.discovered_bugs}
                | self.signals_unlocked
            ),
            "bugs_found": [
                f"{b['type']}:{b.get('column', 'N/A')}"
                for b in self.ground_truth
                if b["bug_id"] in self.discovered_bugs
            ],
            "bugs_fixed": [
                f"{b['type']}:{b.get('column', 'N/A')}"
                for b in self.ground_truth
                if b["bug_id"] in self.fixed_bug_ids
            ],
            "tools_available": ["metrics", "logs", "dag", "pii", "schema"],
        }

        return DataObservation(
            dataset_preview=self.df.head(10).to_dict(orient="records"),
            column_schema=schema_dict,
            pipeline_stage="AUDIT",
            validation_report=validation_report,
            time_remaining=self.MAX_STEPS - self.step_count,
            downstream_health=self.downstream_health,
            step_count=self.step_count,
            task_id=1,
            max_steps=self.MAX_STEPS,
            pipeline_stage_health=None,
            agent_context=agent_context,
        )

    def state(self) -> DataObservation:
        """Return current observation snapshot without side effects."""
        return self._build_observation()
