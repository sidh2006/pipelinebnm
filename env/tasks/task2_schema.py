from __future__ import annotations

import json
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


class Task2SchemaEnv:
    """Task 2 environment for schema drift diagnosis and remediation."""

    MAX_STEPS = 15
    TOTAL_BUGS = 3
    SCENARIO_DIR = Path(__file__).parent.parent / "data" / "scenarios"

    def __init__(self) -> None:
        """Initialize dependency graph and task state containers."""
        default_scenario = self.SCENARIO_DIR / "task2_scenario.json"
        with default_scenario.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.COLUMN_DEPENDENCIES: dict[str, list[str]] = payload.get("column_dependencies", {})

        self.df: pd.DataFrame = pd.DataFrame()
        self.ground_truth: list[dict] = []
        self.step_count: int = 0
        self.identified_bug_ids: set[str] = set()
        self.fixed_bug_ids: set[str] = set()
        self.discovered_bugs: set[str] = set()          # NEW: progressive discovery
        self.downstream_health: float = 1.0
        self.blast_events: int = 0
        self.inspected_targets = set()
        self.visible_signals: VisibleSignals | None = None
        self.signals_unlocked: set[str] = set()
        self.aer_history: list[AERRecord] = []
        self.current_scenario_path: Path | None = None
        self.inspected_targets: set[str] = set()

    def reset(self, seed: int = 42, scenario_override: str | None = None) -> DataObservation:
        """Reset state and initialize a fresh corrupted Task2 dataframe."""
        if scenario_override:
            chosen = self.SCENARIO_DIR / scenario_override
            self.current_scenario_path = chosen
            # Reload dependencies from chosen scenario
            with chosen.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.COLUMN_DEPENDENCIES = payload.get("column_dependencies", self.COLUMN_DEPENDENCIES)
            scenario_bugs = load_scenario(str(chosen))
        else:
            # Procedural generation for live episodes
            scenario_bugs = generate_scenario(seed=seed, task_id="task2", difficulty="medium")
            self.current_scenario_path = None
        # Cache scenario data for step() — NO file I/O in step path
        self._scenario_bugs = scenario_bugs
        self._expected_renames = {b["new_col"]: b["old_col"] for b in scenario_bugs if b.get("type") == "schema_drift"}
        clean_df = generate_employee_dataset(seed=seed)
        self.df, self.ground_truth = inject_bugs(clean_df, scenario_bugs)
        self.step_count = 0
        self.identified_bug_ids = set()
        self.fixed_bug_ids = set()
        self.discovered_bugs = set()                     # NEW: starts empty
        self.downstream_health = 1.0
        self.blast_events: int = 0
        self.inspected_targets = set()

        failure_sig = get_failure_signature(self.ground_truth)
        initial_alert = AlertSignal(
            severity="high",
            message=f"Schema drift detected: {failure_sig.detection_hint}",
            risk_score=0.78,
        )
        self.visible_signals = VisibleSignals(
            alert=initial_alert,
            dag=DagOverview(
                current_node="stage_2_schema_validation",
                upstream_nodes=["stage_1_ingest"],
                downstream_nodes=["stage_3_join", "stage_4_aggregate"],
            ),
        )
        self.signals_unlocked = {"dag"}
        self.aer_history = []

        return self._build_observation()

    def _rows_passing(self) -> int:
        """Count rows passing essential schema conditions in current dataframe."""
        if self.df.empty:
            return 0

        expected_columns = {
            "employee_id",
            "name",
            "age",
            "salary",
            "department",
            "phone",
            "ssn",
            "hire_date",
            "consent_flag",
        }
        if not expected_columns.issubset(set(self.df.columns)):
            return 0

        salary_ok = pd.to_numeric(self.df["salary"], errors="coerce").notna()
        age_ok = pd.to_numeric(self.df["age"], errors="coerce").between(0, 120, inclusive="both")
        hire_ok = pd.to_datetime(self.df["hire_date"], errors="coerce").notna()
        consent_ok = self.df["consent_flag"].notna()
        return int((salary_ok & age_ok & hire_ok & consent_ok).sum())

    def step(self, action: DataAction) -> StepResult:
        """Apply one schema remediation step and return transition output."""
        reward = 0.0
        done = False
        prev_fixed_count = len(self.fixed_bug_ids)

        # Read from cached scenario data (no file I/O in step path)
        expected_renames = self._expected_renames

        if action.action_type == ActionType.INSPECT:
            target = (action.target_column or "").lower()
            _TOOL_ALIASES = {
                "run_null_check": "metrics", "run_type_check": "schema",
                "run_duplicate_check": "metrics", "run_pii_scan": "pii",
                "run_schema_diff": "schema", "trace_pipeline_stage": "dag",
            }
            target = _TOOL_ALIASES.get(target, target)

            reinspecting = target in self.inspected_targets
            if reinspecting: reward -= 0.10  # Raised re-inspect penalty
            else: self.inspected_targets.add(target)

            if target == "metrics" and "metrics" not in self.signals_unlocked:
                self.visible_signals.metrics = build_metrics_facet(self.df)
                self.signals_unlocked.add("metrics")
                if not reinspecting: reward += 0.05
            elif target == "logs" and "logs" not in self.signals_unlocked:
                self.visible_signals.logs = build_logs_facet(["SchemaError: column employee_id not found", "Warning: consent_flag all NULL"])
                self.signals_unlocked.add("logs")
                if not reinspecting: reward += 0.05
            elif target in ["pii", "ssn", "compliance"] and "compliance" not in self.signals_unlocked:
                pii_cols = [col for col in self.df.columns if "ssn" in col.lower()]
                self.visible_signals.compliance = ComplianceFacet(pii_detected=len(pii_cols) > 0, risky_columns=pii_cols)
                self.signals_unlocked.add("compliance")
                if not reinspecting: reward += 0.05
            elif target == "schema" and "schema" not in self.signals_unlocked:
                new_discoveries = 0
                for bug in self.ground_truth:
                    if bug["type"] == "schema_drift" and bug["bug_id"] not in self.discovered_bugs:
                        self.discovered_bugs.add(bug["bug_id"])
                        self.identified_bug_ids.add(bug["bug_id"])
                        new_discoveries += 1
                self.signals_unlocked.add("schema")
                if not reinspecting:
                    reward += 0.05
                    if new_discoveries > 0: reward += 0.15
            else:
                found_any = False
                for bug in self.ground_truth:
                    bug_col = (bug.get("column") or "").lower()
                    drift_cols = [(bug.get("new_col") or "").lower(), (bug.get("old_col") or "").lower()]
                    match_cols = ([bug_col] if bug_col else []) + drift_cols
                    if target in match_cols and bug["bug_id"] not in self.discovered_bugs:
                        self.discovered_bugs.add(bug["bug_id"])
                        self.identified_bug_ids.add(bug["bug_id"])
                        found_any = True
                if not reinspecting:
                    if found_any: reward += 0.15
                    else: reward -= 0.05

        elif action.action_type == ActionType.DROP_COLUMN:
            dependents = self.COLUMN_DEPENDENCIES.get(action.target_column, [])
            if dependents:
                penalty = -0.10 * len(dependents)
                self.downstream_health = max(0.0, self.downstream_health - 0.15 * len(dependents))
                self.blast_events += 1
                reward += penalty
            else:
                reward -= 0.10

        elif action.action_type == ActionType.RENAME_COLUMN:
            if action.target_column in expected_renames and action.transformation == expected_renames[action.target_column]:
                src = action.target_column
                dst = action.transformation
                matching_bug = next((b for b in self.ground_truth if b["type"] == "schema_drift" and b.get("old_col") == dst and b.get("new_col") == src), None)
                if matching_bug:
                    if matching_bug["bug_id"] not in self.discovered_bugs:
                        reward -= 0.10
                    else:
                        if src in self.df.columns and dst not in self.df.columns:
                            self.df.rename(columns={src: dst}, inplace=True)
                        self.fixed_bug_ids.add(matching_bug["bug_id"])
                        reward += 0.20
                else:
                    reward -= 0.05
            else:
                reward -= 0.05

        elif action.action_type == ActionType.CAST_TYPE:
            if action.target_column in {"hire_date", "dob_date"} and action.transformation == "cast_to_date":
                matching_bug = next((b for b in self.ground_truth if b["type"] == "type_corruption" and b.get("column") == "hire_date"), None)
                if matching_bug:
                    if matching_bug["bug_id"] not in self.discovered_bugs:
                        reward -= 0.10
                    else:
                        col = "hire_date" if "hire_date" in self.df.columns else "dob_date"
                        if col in self.df.columns:
                            self.df[col] = pd.to_datetime(self.df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                            self.fixed_bug_ids.add(matching_bug["bug_id"])
                            reward += 0.20
                else: reward -= 0.05
            else: reward -= 0.05

        elif action.action_type == ActionType.FILL_DEFAULT:
            if action.target_column == "consent_flag" and "consent_flag" in self.df.columns:
                matching_bug = next((b for b in self.ground_truth if b["type"] == "null_injection" and b.get("column") == "consent_flag"), None)
                if matching_bug:
                    if matching_bug["bug_id"] not in self.discovered_bugs:
                        reward -= 0.10
                    else:
                        self.df["consent_flag"] = self.df["consent_flag"].fillna(False)
                        self.fixed_bug_ids.add(matching_bug["bug_id"])
                        reward += 0.20
                else: reward -= 0.05
            else: reward -= 0.05

        elif action.action_type == ActionType.VALIDATE:
            rows_passing = self._rows_passing()
            if len(self.fixed_bug_ids) == self.TOTAL_BUGS:
                reward += 0.25
                # Shaped completion: residual bonus
                already_shaped = 0.30 * (prev_fixed_count / self.TOTAL_BUGS)
                reward += round(0.30 - already_shaped, 4)
                done = True
            else:
                reward -= 0.05

        elif action.action_type == ActionType.NOOP:
            reward = 0.0
        else:
            reward -= 0.10

        reward = max(-0.5, min(1.0, reward))

        # Shaped completion bonus: per-step progress signal
        new_fixed_count = len(self.fixed_bug_ids)
        if new_fixed_count > prev_fixed_count and new_fixed_count < self.TOTAL_BUGS:
            progress_reward = 0.30 * ((new_fixed_count - prev_fixed_count) / self.TOTAL_BUGS)
            reward += round(progress_reward, 4)
            reward = max(-0.5, min(1.0, reward))

        self.step_count += 1
        self.downstream_health = len(self.fixed_bug_ids) / self.TOTAL_BUGS
        done = done or (self.step_count >= self.MAX_STEPS)

        aer = AERRecord(
            step_id=self.step_count, action_type=action.action_type.value, target=action.target_column,
            justification=action.justification, reward_earned=round(reward, 4),
            issues_identified=list(self.identified_bug_ids), issues_fixed=list(self.fixed_bug_ids),
        )
        self.aer_history.append(aer)

        return StepResult(
            observation=self._build_observation(), reward=round(reward, 4), done=done,
            info={
                "blast_events": self.blast_events, "fixed": list(self.fixed_bug_ids),
                "identified": list(self.identified_bug_ids), "signals_unlocked": list(self.signals_unlocked),
                "visible_signals": self.visible_signals.model_dump() if self.visible_signals else {},
                "aer_last": aer.model_dump(),
                "action_space_hints": {
                    "accepted_casts": ["cast_to_date", "cast_to_int", "cast_to_float"],
                    "accepted_fills": ["fill_median", "fill_zero"],
                    "cast_target_columns": {
                        "hire_date": "cast_to_date",
                        "dob_date": "cast_to_date",
                        "salary": "cast_to_int",
                        "age": "cast_to_int",
                    },
                },
            },
        )

    def _build_observation(self) -> DataObservation:
        """Construct DataObservation from current dataframe.

        Progressive discovery: only bugs in `discovered_bugs` AND not yet
        fixed are shown in validation_report.  At reset this is empty.
        """
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
            col: {"type": str(dtype), "nullable": bool(self.df[col].isna().any())}
            for col, dtype in self.df.dtypes.items()
        }

        agent_context = {
            "inspected_columns": sorted(
                {(b.get("column") or b.get("new_col") or "").lower()
                 for b in self.ground_truth if b["bug_id"] in self.discovered_bugs}
                | self.signals_unlocked
            ),
            "bugs_found": [
                f"{b['type']}:{b.get('column') or b.get('new_col', 'N/A')}"
                for b in self.ground_truth
                if b["bug_id"] in self.discovered_bugs
            ],
            "bugs_fixed": [
                f"{b['type']}:{b.get('column') or b.get('new_col', 'N/A')}"
                for b in self.ground_truth
                if b["bug_id"] in self.fixed_bug_ids
            ],
            "tools_available": ["metrics", "logs", "dag", "pii", "schema"],
        }

        return DataObservation(
            dataset_preview=self.df.head(10).to_dict(orient="records"),
            column_schema=schema_dict,
            pipeline_stage="SCHEMA_REMEDIATION",
            validation_report=validation_report,
            time_remaining=self.MAX_STEPS - self.step_count,
            downstream_health=self.downstream_health,
            step_count=self.step_count,
            task_id=2,
            max_steps=self.MAX_STEPS,
            pipeline_stage_health=None,
            agent_context=agent_context,
        )

    def state(self) -> DataObservation:
        """Return current state without changing environment variables."""
        return self._build_observation()
