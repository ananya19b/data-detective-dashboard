from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Investigation:
    profile: dict[str, Any]
    ranked_rows: pd.DataFrame
    numeric_summary: pd.DataFrame
    rare_categories: pd.DataFrame
    column_profile: pd.DataFrame
    correlations: pd.DataFrame


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def profile_data(df: pd.DataFrame) -> dict[str, Any]:
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(int(df.size), 1)
    completeness = 100 * (1 - missing_cells / total_cells)
    uniqueness_penalty = 100 * int(df.duplicated().sum()) / max(len(df), 1)
    health_score = max(0.0, completeness - uniqueness_penalty)
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": missing_cells,
        "completeness_pct": round(completeness, 1),
        "data_health_score": round(health_score, 1),
        "column_types": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_column": {
            c: int(v) for c, v in df.isna().sum().items() if int(v) > 0
        },
    }


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        s = df[col]
        record = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "unique": int(s.nunique(dropna=True)),
            "min": None,
            "max": None,
            "mean": None,
            "std_dev": None,
        }
        if pd.api.types.is_numeric_dtype(s):
            clean = s.replace([np.inf, -np.inf], np.nan)
            record.update(
                {
                    "min": _json_safe(clean.min()),
                    "max": _json_safe(clean.max()),
                    "mean": round(float(clean.mean()), 3) if clean.notna().any() else None,
                    "std_dev": round(float(clean.std()), 3) if clean.notna().sum() > 1 else None,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def strongest_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["column_1", "column_2", "correlation", "strength"])
    corr = numeric.corr()
    records = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                records.append(
                    {
                        "column_1": left,
                        "column_2": right,
                        "correlation": round(float(value), 3),
                        "strength": round(abs(float(value)), 3),
                    }
                )
    return pd.DataFrame(records).sort_values("strength", ascending=False).reset_index(drop=True)


def investigate(
    df: pd.DataFrame,
    iqr_multiplier: float = 2.5,
) -> Investigation:
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    working = df.copy().reset_index(drop=True)
    working.insert(0, "_row_id", np.arange(1, len(working) + 1))
    numeric_cols = [
        c for c in working.select_dtypes(include=np.number).columns if c != "_row_id"
    ]
    reasons: list[list[str]] = [[] for _ in range(len(working))]
    for idx in np.flatnonzero(working.drop(columns="_row_id").isna().any(axis=1).to_numpy()):
        reasons[idx].append("one or more missing values")
    numeric_records: list[dict[str, Any]] = []

    for col in numeric_cols:
        series = working[col].replace([np.inf, -np.inf], np.nan)
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        numeric_records.append(
            {
                "column": col,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(mask.sum()),
            }
        )
        for idx in np.flatnonzero(mask.fillna(False).to_numpy()):
            reasons[idx].append(f"{col}={_json_safe(series.iloc[idx])} outside IQR bounds [{lower:.2f}, {upper:.2f}]")

    categorical_cols = [
        c
        for c in working.select_dtypes(include=["object", "category", "bool"]).columns
        if working[c].nunique(dropna=True) <= 50
    ]
    rare_records: list[dict[str, Any]] = []
    rare_cutoff = max(2, int(np.ceil(len(working) * 0.01)))
    for col in categorical_cols:
        counts = working[col].value_counts(dropna=True)
        rare_values = counts[(counts <= rare_cutoff) & (counts.index.astype(str) != "")]
        for value, count in rare_values.items():
            rare_records.append({"column": col, "value": value, "count": int(count)})
            mask = working[col].eq(value)
            for idx in np.flatnonzero(mask.fillna(False).to_numpy()):
                reasons[idx].append(f"rare {col}={value!r} appears {int(count)} time(s)")

    duplicate_mask = working.drop(columns="_row_id").duplicated(keep=False)
    for idx in np.flatnonzero(duplicate_mask.to_numpy()):
        reasons[idx].append("exact duplicate record")

    working["anomaly_score"] = [len(r) for r in reasons]
    working["is_flagged"] = [bool(r) for r in reasons]
    working["reasons"] = ["; ".join(r) if r else "" for r in reasons]
    ranked = working.sort_values(["is_flagged", "anomaly_score"], ascending=[False, False])

    return Investigation(
        profile=profile_data(df),
        ranked_rows=ranked,
        numeric_summary=pd.DataFrame(numeric_records),
        rare_categories=pd.DataFrame(rare_records),
        column_profile=build_column_profile(df),
        correlations=strongest_correlations(df),
    )


def evidence_package(result: Investigation, limit: int = 10) -> dict[str, Any]:
    flagged = result.ranked_rows[result.ranked_rows["is_flagged"]].head(limit)
    keep = ["_row_id", "anomaly_score", "reasons"]
    return {
        "profile": result.profile,
        "numeric_checks": result.numeric_summary.to_dict(orient="records"),
        "rare_categories": result.rare_categories.to_dict(orient="records"),
        "column_profile": result.column_profile.to_dict(orient="records"),
        "strongest_correlations": result.correlations.head(8).to_dict(orient="records"),
        "top_flagged_rows": [
            {k: _json_safe(v) for k, v in row.items()}
            for row in flagged[keep].to_dict(orient="records")
        ],
    }
