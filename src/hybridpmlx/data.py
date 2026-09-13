from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    frame: pd.DataFrame
    development_ids: np.ndarray
    test_ids: np.ndarray


def load_and_split(cfg: dict) -> PreparedData:
    spec, split = cfg["data"], cfg["split"]
    path = Path(spec["file"])
    if not path.is_absolute():
        path = Path(cfg["_config_dir"]) / path
    df = pd.read_csv(path)
    required = [spec["id_column"], spec["idv_column"], spec["dv_column"], *spec["feature_columns"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Missing values were found in required columns")

    id_col = spec["id_column"]
    # Baseline features must be constant within an individual.
    nunique = df.groupby(id_col)[spec["feature_columns"]].nunique(dropna=False)
    if (nunique > 1).any().any():
        bad = nunique.columns[(nunique > 1).any()].tolist()
        raise ValueError(f"Features must be constant within ID; varying columns: {bad}")

    id_rows = df.drop_duplicates(id_col)
    ids = id_rows[id_col].to_numpy()
    if split["method"] == "predefined":
        column = split["split_column"]
        if column not in id_rows:
            raise ValueError(f"Missing predefined split column: {column}")
        dev = id_rows.loc[id_rows[column] == split.get("development_label", "development"), id_col].to_numpy()
        test = id_rows.loc[id_rows[column] == split.get("test_label", "test"), id_col].to_numpy()
    else:
        stratify = None
        strat_col = split.get("stratify_by")
        if strat_col:
            stratify = id_rows[strat_col].to_numpy()
        dev, test = train_test_split(ids, test_size=float(split["test_ratio"]), random_state=int(split["seed"]), stratify=stratify)
    if set(dev) & set(test):
        raise RuntimeError("Development/test ID overlap detected")
    return PreparedData(df, np.asarray(dev), np.asarray(test))


def fit_id_scaler(df: pd.DataFrame, ids, id_col: str, features: list[str]) -> StandardScaler:
    unique = df[df[id_col].isin(ids)].drop_duplicates(id_col)
    return StandardScaler().fit(unique[features])

