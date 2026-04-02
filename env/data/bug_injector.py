import json
from pathlib import Path

import pandas as pd

from env.models import DetectedIssue


# Canonical matching function — used by ALL graders
def matches_ground_truth(detected: DetectedIssue, truth: dict) -> bool:
    """
    Returns True if detected issue matches a ground truth entry.
    Compares issue_type AND column only. Severity is irrelevant.
    """
    return detected.issue_type == truth["type"] and detected.column == truth.get("column")


def load_scenario(scenario_path: str) -> list[dict]:
    """Load bug spec from JSON scenario file."""
    path = Path(scenario_path)
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Scenario file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in scenario file {path}: {exc}") from exc
    # Support both list format (task1) and dict format (task2/task3)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "bugs" in data:
        return data["bugs"]
    raise ValueError(f"Unrecognized scenario format in {path}")


def inject_bugs(df: pd.DataFrame, bug_spec: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject bugs into df in strict order. Returns (corrupted_df, ground_truth_list).
    Injection order is MANDATORY:
      1. null_injection
      2. type_corruption
      3. out_of_range
      4. format_inconsistency
      5. schema_drift
      6. pii_leak
      7. duplicate_rows  ← ALWAYS LAST
    """
    INJECTION_ORDER = [
        "null_injection",
        "type_corruption",
        "out_of_range",
        "format_inconsistency",
        "schema_drift",
        "pii_leak",
        "duplicate_rows",
    ]

    corrupted = df.copy().astype(object)
    ground_truth: list[dict] = []

    for bug_type in INJECTION_ORDER:
        bugs_of_type = [b for b in bug_spec if b["type"] == bug_type]
        for bug in bugs_of_type:
            affected_rows: list[int] = []

            if bug_type == "null_injection":
                col = bug["column"]
                rows = bug.get("rows", [])
                if rows == "ALL":
                    corrupted[col] = None
                    affected_rows = list(range(len(corrupted)))
                else:
                    corrupted.loc[rows, col] = None
                    affected_rows = rows

            elif bug_type == "type_corruption":
                col = bug["column"]
                row = bug["row"]
                if col not in corrupted.columns and col == "rev_amt" and "revenue_amount" in corrupted.columns:
                    col = "revenue_amount"
                # CRITICAL: cast to object BEFORE assignment to suppress FutureWarning
                corrupted[col] = corrupted[col].astype(object)
                corrupted.loc[row, col] = bug["value"]
                affected_rows = [row]

            elif bug_type == "out_of_range":
                col = bug["column"]
                row = bug["row"]
                corrupted.loc[row, col] = bug["value"]
                affected_rows = [row]

            elif bug_type == "format_inconsistency":
                col = bug.get("column", "phone")
                row = bug["row"]
                existing = str(corrupted.loc[row, col])
                # valid: "98XXXXXXXX" → corrupted: "+91-98-XXXXXXXX"
                if existing.startswith("98") and len(existing) == 10:
                    corrupted.loc[row, col] = f"+91-{existing[:2]}-{existing[2:]}"
                else:
                    corrupted.loc[row, col] = f"+91-{existing}"
                affected_rows = [row]

            elif bug_type == "schema_drift":
                old_col = bug["old_col"]
                new_col = bug["new_col"]
                if old_col in corrupted.columns:
                    corrupted.rename(columns={old_col: new_col}, inplace=True)
                affected_rows = []

            elif bug_type == "pii_leak":
                # Dynamically inject an employee_ssn column with plausible fake values
                # This makes PII detection a real discovery task, not just a column lookup
                import numpy as np
                _pii_rng = np.random.default_rng(hash(bug.get("bug_id", "B003")) % (2**31))
                ssn_values = [
                    f"{_pii_rng.integers(100,999)}-{_pii_rng.integers(10,99)}-{_pii_rng.integers(1000,9999)}"
                    for _ in range(len(corrupted))
                ]
                pii_col = bug.get("column", "employee_ssn")
                if pii_col not in corrupted.columns:
                    corrupted[pii_col] = ssn_values
                affected_rows = list(range(len(corrupted)))

            elif bug_type == "duplicate_rows":
                # ALWAYS LAST — pd.concat, never df.append (deprecated)
                indices = bug.get("indices", [])
                if indices:
                    extra = corrupted.iloc[indices].copy()
                    corrupted = pd.concat([corrupted, extra], ignore_index=True)
                affected_rows = indices

            ground_truth.append(
                {
                    "bug_id": bug["bug_id"],
                    "type": bug_type,
                    "column": bug.get("column"),
                    "description": bug.get("description", ""),
                    "severity": bug.get("severity", "medium"),
                    "affected_rows": affected_rows,
                }
            )

    return corrupted, ground_truth


def generate_scenario(seed: int, task_id: str, difficulty: str = "easy") -> list[dict]:
    """
    Procedurally generate a bug scenario based on seed, task_id, and difficulty.
    
    Uses the seed to select bug types from weighted distribution, randomly
    selects target columns and row indices. Returns a list of bug dicts
    compatible with inject_bugs().
    
    Static JSON scenarios are preserved as fallback for /demo only.
    All live /reset calls should use this function.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    
    # Column pools per task
    NUMERIC_COLS = ["salary", "age"]
    STRING_COLS = ["phone", "name", "department"]
    DATE_COLS = ["hire_date"]
    ALL_COLS = NUMERIC_COLS + STRING_COLS + DATE_COLS
    
    # Bug type weights by difficulty
    BUG_POOLS = {
        "task1": {
            "types": ["null_injection", "type_corruption", "out_of_range", "format_inconsistency", "duplicate_rows"],
            "weights": [0.25, 0.25, 0.2, 0.15, 0.15],
            "count": 5,
        },
        "task2": {
            "types": ["schema_drift", "schema_drift", "type_corruption", "null_injection"],
            "weights": [0.35, 0.25, 0.2, 0.2],
            "count": 3,
        },
        "task3": {
            "types": ["schema_drift", "type_corruption", "pii_leak", "duplicate_rows"],
            "weights": [0.25, 0.30, 0.25, 0.20],
            "count": 4,
        },
    }
    
    pool = BUG_POOLS.get(task_id, BUG_POOLS["task1"])
    bug_count = pool["count"]
    
    # Select bug types ensuring diversity (at least 3 different types for task1)
    selected_types: list[str] = []
    available_types = list(set(pool["types"]))
    
    if task_id == "task1":
        # Enforce at least 3 different types
        min_diverse = min(3, len(available_types))
        diverse_picks = rng.choice(available_types, size=min_diverse, replace=False).tolist()
        selected_types.extend(diverse_picks)
        remaining = bug_count - len(selected_types)
        if remaining > 0:
            extra = rng.choice(pool["types"], size=remaining, p=pool["weights"] if len(pool["weights"]) == len(pool["types"]) else None).tolist()
            selected_types.extend(extra)
    else:
        # For task2/task3, use predefined structure
        selected_types = pool["types"][:bug_count]
    
    # Severity mapping by difficulty
    severity_pool = {
        "easy": ["medium", "high"],
        "medium": ["high", "critical"],
        "hard": ["critical", "critical", "high"],
    }
    severities = severity_pool.get(difficulty, ["medium", "high"])
    
    # Rename column mappings for schema_drift
    RENAME_MAPPINGS = [
        ("employee_id", "customer_uuid"),
        ("hire_date", "dob_date"),
        ("salary", "rev_amt"),
        ("department", "dept_code"),
        ("name", "full_name"),
    ]
    
    bugs: list[dict] = []
    used_rename_indices: set[int] = set()
    used_rows: set[int] = set()
    
    for i, bug_type in enumerate(selected_types[:bug_count]):
        bug_id = f"B{i+1:03d}"
        severity = severities[rng.integers(0, len(severities))]
        
        if bug_type == "null_injection":
            col = rng.choice(NUMERIC_COLS)
            rows = sorted(rng.choice(range(5, 195), size=rng.integers(2, 5), replace=False).tolist())
            bugs.append({
                "bug_id": bug_id, "type": "null_injection", "column": col,
                "rows": rows, "severity": severity,
                "description": f"NULL values injected in {col} column at rows {rows}",
            })
            
        elif bug_type == "type_corruption":
            col = rng.choice(NUMERIC_COLS)
            row = int(rng.integers(1, 190))
            while row in used_rows:
                row = int(rng.integers(1, 190))
            used_rows.add(row)
            values = ["twenty-three", "N/A", "unknown", "1234.56"]
            bugs.append({
                "bug_id": bug_id, "type": "type_corruption", "column": col,
                "row": row, "value": rng.choice(values),
                "severity": severity,
                "description": f"{col} stored as string at row {row}",
            })
            
        elif bug_type == "out_of_range":
            row = int(rng.integers(1, 190))
            while row in used_rows:
                row = int(rng.integers(1, 190))
            used_rows.add(row)
            bugs.append({
                "bug_id": bug_id, "type": "out_of_range", "column": "age",
                "row": row, "value": int(rng.choice([999, -1, 500, 255])),
                "severity": severity,
                "description": f"Age value out of valid range at row {row}",
            })
            
        elif bug_type == "format_inconsistency":
            row = int(rng.integers(1, 190))
            while row in used_rows:
                row = int(rng.integers(1, 190))
            used_rows.add(row)
            bugs.append({
                "bug_id": bug_id, "type": "format_inconsistency", "column": "phone",
                "row": row, "severity": "low",
                "description": f"Phone reformatted at row {row}",
            })
            
        elif bug_type == "schema_drift":
            available = [j for j in range(len(RENAME_MAPPINGS)) if j not in used_rename_indices]
            if not available:
                available = list(range(len(RENAME_MAPPINGS)))
            idx = rng.choice(available)
            used_rename_indices.add(idx)
            old_col, new_col = RENAME_MAPPINGS[idx]
            bugs.append({
                "bug_id": bug_id, "type": "schema_drift",
                "old_col": old_col, "new_col": new_col,
                "severity": severity,
                "description": f"{old_col} renamed to {new_col} upstream",
            })
            
        elif bug_type == "pii_leak":
            bugs.append({
                "bug_id": bug_id, "type": "pii_leak", "column": "ssn",
                "severity": "critical",
                "description": "SSN column propagated into analytics output",
            })
            
        elif bug_type == "duplicate_rows":
            indices = sorted(rng.choice(range(10, 190), size=rng.integers(2, 4), replace=False).tolist())
            bugs.append({
                "bug_id": bug_id, "type": "duplicate_rows", "column": None,
                "indices": indices, "severity": severity,
                "description": f"Rows {indices} duplicated",
            })
    
    return bugs


def get_failure_signature(bug_spec: list[dict]):
    """
    Derives a FailureSignature from the bug spec.
    Used by task classes to build the initial AlertSignal.
    Maps bug types to industry failure mode names.
    """
    from env.models import FailureSignature

    type_map = {
        "null_injection": "silent_drop",
        "type_corruption": "type_corruption",
        "out_of_range": "type_corruption",
        "format_inconsistency": "schema_drift",
        "schema_drift": "schema_drift",
        "pii_leak": "pii_leak",
        "duplicate_rows": "duplicate_aggregation",
    }

    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    sorted_bugs = sorted(
        bug_spec,
        key=lambda b: severity_order.get(b.get("severity", "low"), 0),
        reverse=True,
    )
    top_bug = sorted_bugs[0] if sorted_bugs else {}
    failure_type = type_map.get(top_bug.get("type", "schema_drift"), "schema_drift")

    critical_count = sum(1 for b in bug_spec if b.get("severity") == "critical")
    if critical_count >= 3:
        blast = "critical"
    elif critical_count == 2:
        blast = "high"
    elif critical_count == 1:
        blast = "medium"
    else:
        blast = "low"

    hint_map = {
        "silent_drop": "Row count anomaly detected. Check null ratio.",
        "type_corruption": "Type validation failed. Check column dtypes.",
        "schema_drift": "Schema mismatch vs upstream contract.",
        "pii_leak": "Sensitive data detected in analytics output.",
        "duplicate_aggregation": "Aggregation inflated. Check for duplicate rows.",
        "swallowed_exception": "Job succeeded but output invalid. Check logs.",
        "zombie_partition": "Partition exists but storage is 0 bytes.",
    }

    return FailureSignature(
        failure_type=failure_type,
        affected_stage=top_bug.get("stage", "stage_3_join"),
        blast_radius=blast,
        detection_hint=hint_map.get(failure_type, "Unknown failure pattern."),
    )


def build_metrics_facet(df, historical_avg: int = 200) -> "MetricsFacet":
    """Build MetricsFacet from current DataFrame state."""
    from env.models import MetricsFacet

    null_ratio = float(df.isnull().sum().sum() / max(df.size, 1))
    return MetricsFacet(
        row_count=len(df),
        historical_avg=historical_avg,
        null_ratio=round(null_ratio, 4),
        storage_bytes=len(df) * 64 if len(df) > 0 else 0,
    )


def build_logs_facet(error_list: list[str], status: str = "failed") -> "LogsFacet":
    """Build LogsFacet from accumulated error messages."""
    from env.models import LogsFacet

    return LogsFacet(
        recent_errors=error_list[-5:],
        last_run_status=status,
    )


if __name__ == "__main__":
    from env.data.generator import generate_employee_dataset

    scenarios = [
        "env/data/scenarios/task1_scenario.json",
        "env/data/scenarios/task2_scenario.json",
        "env/data/scenarios/task3_scenario.json",
    ]
    df_clean = generate_employee_dataset(seed=42)
    for path in scenarios:
        spec = load_scenario(path)
        corrupted, gt = inject_bugs(df_clean.copy(), spec)
        print(f"{path}: {len(gt)} bugs, corrupted shape={corrupted.shape}")