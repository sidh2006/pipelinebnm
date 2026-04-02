from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    INSPECT = "INSPECT"
    RENAME_COLUMN = "RENAME_COLUMN"
    CAST_TYPE = "CAST_TYPE"
    FILL_DEFAULT = "FILL_DEFAULT"
    DROP_COLUMN = "DROP_COLUMN"
    VALIDATE = "VALIDATE"
    MASK_PII = "MASK_PII"
    NOOP = "NOOP"


class ScenarioBug(BaseModel):
    """A single bug specification for procedural generation."""
    bug_id: str
    type: str
    column: Optional[str] = None
    old_col: Optional[str] = None
    new_col: Optional[str] = None
    row: Optional[int] = None
    rows: Optional[list[int] | str] = None  # list of ints or "ALL"
    value: Optional[str | int | float] = None
    indices: Optional[list[int]] = None
    severity: str = "medium"
    description: str = ""
    stage: Optional[str] = None


class Scenario(BaseModel):
    """A complete scenario specification for a task."""
    bugs: list[ScenarioBug]
    task_id: str = "task1"
    seed: int = 42
    difficulty: str = "easy"


class DetectedIssue(BaseModel):
    issue_type: str
    column: Optional[str] = None
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class DataAction(BaseModel):
    action_type: ActionType
    target_column: Optional[str] = None
    transformation: Optional[str] = None
    justification: str
    identified_issues: Optional[List[DetectedIssue]] = None


class DataObservation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # ONE config block only — never duplicate this

    dataset_preview: List[dict]
    column_schema: dict = Field(alias="schema")
    # alias="schema" preserves wire format without shadowing BaseModel.schema
    pipeline_stage: str
    validation_report: List[DetectedIssue]
    time_remaining: int
    downstream_health: float
    step_count: int
    task_id: int
    max_steps: int = 20  # Default to largest task budget
    pipeline_stage_health: Optional[dict[str, float]] = None
    agent_context: Optional[dict] = None


class StepResult(BaseModel):
    observation: DataObservation
    reward: float
    done: bool
    info: dict


class GraderResult(BaseModel):
    score: float
    breakdown: dict[str, float]
    explanation: str


# -- Observability facets -- unlocked progressively via INSPECT actions --

class AlertSignal(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    risk_score: float = Field(ge=0.0, le=1.0)


class DagOverview(BaseModel):
    current_node: str
    upstream_nodes: List[str]
    downstream_nodes: List[str]


class MetricsFacet(BaseModel):
    row_count: int
    historical_avg: int
    null_ratio: float = Field(ge=0.0, le=1.0)
    storage_bytes: Optional[int] = None


class LogsFacet(BaseModel):
    recent_errors: List[str]
    last_run_status: Literal["success", "failed", "warning", "running"]


class ComplianceFacet(BaseModel):
    pii_detected: bool
    risky_columns: List[str]


class VisibleSignals(BaseModel):
    """
    Returned inside StepResult.info as progressively unlocked facets.
    At reset: only alert is populated; other facets are unlocked by actions.
    """

    alert: AlertSignal
    dag: Optional[DagOverview] = None
    metrics: Optional[MetricsFacet] = None
    logs: Optional[LogsFacet] = None
    compliance: Optional[ComplianceFacet] = None


# -- Agent execution record -- reasoning trace --

class AERRecord(BaseModel):
    """
    Tracks agent reasoning per step and can be used for grader partial credit.
    """

    step_id: int
    action_type: str
    target: Optional[str]
    justification: str
    reward_earned: float
    issues_identified: List[str]
    issues_fixed: List[str]


# -- Failure mode signatures -- used by bug injector and tasks --

class FailureSignature(BaseModel):
    """Describes the type of industry failure being simulated."""

    failure_type: Literal[
        "zombie_partition",
        "silent_drop",
        "schema_drift",
        "pii_leak",
        "swallowed_exception",
        "duplicate_aggregation",
        "type_corruption",
    ]
    affected_stage: str
    blast_radius: Literal["low", "medium", "high", "critical"]
    detection_hint: str