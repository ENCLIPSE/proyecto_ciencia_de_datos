"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]

SILVER_NOT_NULL = [
    "Survived", "Pclass", "Age", "Fare", "Embarked", "Person", "Alone", "TCabin",
    "TTitulo", "NSex", "NEmbarked", "NCabin", "NTitulo",
]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de entrada: columnas esperadas y PassengerId sin duplicados."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en Bronze: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Hay PassengerId duplicados en Bronze")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Survived, Pclass, SibSp, Parch -> int; Age y Fare -> float."""
    out = df.copy()
    for col in ["Survived", "Pclass", "SibSp", "Parch"]:
        out[col] = out[col].astype(int)
    for col in ["Age", "Fare"]:
        out[col] = out[col].astype(float)
    return out


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): NaN < 16 es False, así que se usa Sex.
    Esto subestima a los niños sin edad registrada (ver Reto 8.1).
    """
    out = df.copy()
    out["Person"] = np.where(out["Age"] < 16, "child", out["Sex"])
    return out


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    out = df.copy()
    out["TCabin"] = out["Cabin"].str[0].fillna("X")
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    return out


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    out = df.copy()
    out["Familia"] = out["SibSp"] + out["Parch"]
    out["Alone"] = np.where(out["Familia"] > 0, "with Family", "Alone")
    return out


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    out = df.copy()
    out["Survivor"] = out["Survived"].map({0: "No", 1: "Yes"})
    return out


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TTitulo': el título que va entre la coma y el punto del nombre."""
    out = df.copy()
    out["TTitulo"] = out["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return out


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los títulos raros: usa TITLE_MAP y deja el resto como 'Otro'."""
    out = df.copy()
    out["TTitulo"] = out["TTitulo"].replace(TITLE_MAP)
    out["TTitulo"] = out["TTitulo"].where(out["TTitulo"].isin(MAIN_TITLES), "Otro")
    return out


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los faltantes de 'Embarked' con la moda."""
    out = df.copy()
    out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode()[0])
    return out


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Se usa para EXPLORAR; el modelo
    (Parte 11) vuelve a imputar solo con datos de entrenamiento.
    """
    out = df.copy()
    out["AgeOriginal"] = out["Age"]
    out["AgeImputada"] = out["Age"].isna()
    std_before = out["Age"].std()
    promedios = out.groupby("Sex")["Age"].mean()
    out["Age"] = out["Age"].fillna(out["Sex"].map(promedios))
    log.info("Desviación estándar de Age: antes %.2f, después %.2f", std_before, out["Age"].std())
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las versiones numéricas NSex, NEmbarked, NCabin y NTitulo."""
    out = df.copy()
    out["NSex"] = out["Sex"].map({"female": 0, "male": 1})
    out["NEmbarked"] = out["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    deck_map = {letra: i for i, letra in enumerate(DECKS)}
    out["NCabin"] = out["TCabin"].map(deck_map)
    out["NTitulo"] = out["TTitulo"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3})
    return out


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    return df.drop(columns=["Name", "Sex", "Embarked", "Cabin"])


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de salida: las columnas clave existen y no tienen nulos."""
    missing = [c for c in SILVER_NOT_NULL if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en Silver: {missing}")
    nulls = df[SILVER_NOT_NULL].isnull().sum()
    bad = nulls[nulls > 0]
    if len(bad) > 0:
        raise ValueError(f"Silver tiene nulos en: {bad.to_dict()}")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
