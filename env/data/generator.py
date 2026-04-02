import numpy as np
import pandas as pd


CLEAN_SCHEMA = {
    "employee_id": "object",
    "name": "object",
    "age": "int64",
    "salary": "int64",
    "department": "object",
    "phone": "object",
    "ssn": "object",
    "hire_date": "object",
    "consent_flag": "bool",
}


def generate_employee_dataset(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Produces a deterministic employee DataFrame.
    Same seed always returns identical output. Uses numpy default_rng exclusively.
    """
    rng = np.random.default_rng(seed)
    departments = ["Engineering", "Sales", "HR", "Finance"]
    df = pd.DataFrame(
        {
            "employee_id": [f"EMP{i:04d}" for i in range(n_rows)],
            "name": [f"Employee_{i}" for i in range(n_rows)],
            "age": rng.integers(22, 66, n_rows).tolist(),
            "salary": rng.integers(30000, 150001, n_rows).tolist(),
            "department": rng.choice(departments, n_rows).tolist(),
            "phone": [f"98{rng.integers(10000000, 99999999)}" for _ in range(n_rows)],
            "ssn": [f"XXX-XX-{rng.integers(1000, 9999)}" for _ in range(n_rows)],
            "hire_date": pd.date_range("2015-01-01", periods=n_rows, freq="7D")
            .strftime("%Y-%m-%d")
            .tolist(),
            "consent_flag": rng.choice([True, False], n_rows).tolist(),
        }
    )
    return df.astype(object)


if __name__ == "__main__":
    df = generate_employee_dataset()
    print(df.shape)
    print(df.dtypes)
    print(df.head(3))