"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def _parquet_ready(table: pd.DataFrame) -> pd.DataFrame:
    """Parquet exige nombres de columna de texto (los pivot_table traen enteros)."""
    out = table.copy()
    out.columns = [str(c) for c in out.columns]
    return out


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Paso 28: cuatro tablas descriptivas por grupos."""
    def rate(col: str) -> pd.DataFrame:
        return df.groupby(col)["Survived"].agg(["mean", "sum", "count"]).reset_index()

    return {
        "age_by_survival": _parquet_ready(df.groupby("Survived")["Age"].describe().reset_index()),
        "survival_by_sex": _parquet_ready(rate("Sex")),
        "survival_by_pclass": _parquet_ready(rate("Pclass")),
        "survival_by_alone": _parquet_ready(rate("Alone")),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Paso 29: prueba de independencia chi-cuadrado entre Sex y Survived."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, dof, _ = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: tasa de supervivencia por clase (filas) y sexo (columnas)."""
    t = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    return _parquet_ready(t.reset_index())


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (1): tasa y número de pasajeros por puerto."""
    t = df.groupby("Embarked")["Survived"].agg(["mean", "count"]).reset_index()
    return _parquet_ready(t)


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (2): tasa por puerto estratificada por clase."""
    t = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    return _parquet_ready(t.reset_index())


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (1): cuartil de tarifa calculado dentro de cada clase (columna 'FareQ')."""
    out = df.copy()
    out["FareQ"] = out.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return out


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (2): tasa de supervivencia por cuartil de tarifa y clase."""
    d = add_fare_quartile(df)
    t = d.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean")
    return _parquet_ready(t.reset_index())


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: tasa y número de pasajeros por tamaño de familia."""
    t = df.groupby("Familia")["Survived"].agg(["mean", "count"]).reset_index()
    return _parquet_ready(t)


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Paso 38 (1): True donde el valor queda fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38 (2-3): cuántos atípicos hay en Fare, Age y Familia y cómo sobrevivieron."""
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    filas = []
    for nombre, col in [("Fare", "Fare"), ("Age", age_col), ("Familia", "Familia")]:
        mask = outlier_mask_iqr(df[col])
        filas.append({
            "variable": nombre,
            "n": int(df[col].notna().sum()),
            "n_outliers": int(mask.sum()),
            "pct_outliers": float(mask.sum() / df[col].notna().sum() * 100),
            "survival_outliers": float(df.loc[mask, "Survived"].mean()),
            "survival_rest": float(df.loc[~mask & df[col].notna(), "Survived"].mean()),
        })
    return pd.DataFrame(filas)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet(Path(in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
