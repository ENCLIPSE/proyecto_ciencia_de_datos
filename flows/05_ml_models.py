"""Modelado: Silver -> modelo entrenado y guardado."""
from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import MODELS, SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]            # features "sospechosas": se evalúan en el Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# --- Decisiones congeladas (resultado del cuaderno 04; justificadas en reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"                 # nombre en get_candidates()
FINAL_PARAMS = {                        # ejemplo de la guía: reemplázalo con tus mejores del Paso 57
    "model__learning_rate": 0.05,
    "model__max_depth": 3,
    "model__max_iter": 200,
    "model__l2_regularization": 1.0,
}
USE_CABIN_FEATURES = False              # decisión del Paso 61
THRESHOLD = 0.5                         # decisión del Paso 53


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver. Usa la edad ORIGINAL (con NaN): el modelo imputa por su cuenta."""
    df = pd.read_parquet(Path(path) if path else SILVER / SILVER_NAME)
    X = df[CANDIDATE_COLUMNS].copy()
    X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None,
          random_state: int | None = None):
    """Devuelve X_train, X_test, y_train, y_test, estratificado por y."""
    return train_test_split(
        X, y,
        test_size=settings.test_size if test_size is None else test_size,
        random_state=settings.random_state if random_state is None else random_state,
        stratify=y,
    )


class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Rellena 'Age' con el promedio de su grupo, aprendido SOLO en fit()."""

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        self.group_means_ = X.groupby(cols)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        fill = X[cols].merge(means, on=cols, how="left")["_fill"].to_numpy(dtype=float)
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X


def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """GroupAgeImputer -> ColumnTransformer -> modelo, todo en un Pipeline."""
    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])
    prep = ColumnTransformer(
        [("num", num, list(num_features)), ("cat", cat, list(cat_features))],
        sparse_threshold=0,
    )
    return Pipeline([
        ("age_imputer", GroupAgeImputer(group_cols=("Sex", "Pclass"))),
        ("prep", prep),
        ("model", model),
    ])


def get_candidates(random_state: int = 42) -> dict:
    """Diccionario {nombre: estimador sin entrenar}."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


def cross_validate_candidates(X, y, num_features=None, cat_features=None,
                              random_state=None, cv_folds=None) -> pd.DataFrame:
    """Devuelve un DataFrame con media y desviación de cada métrica por modelo."""
    random_state = settings.random_state if random_state is None else random_state
    cv_folds = settings.cv_folds if cv_folds is None else cv_folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    filas = []
    for nombre, modelo in get_candidates(random_state).items():
        pipe = build_pipeline(modelo, num_features, cat_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # el baseline no predice positivos: precisión indefinida
            res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
        fila = {"model": nombre}
        for metrica in SCORING:
            fila[f"{metrica}_mean"] = float(np.mean(res[f"test_{metrica}"]))
            fila[f"{metrica}_std"] = float(np.std(res[f"test_{metrica}"]))
        fila[f"train_{PRIMARY_METRIC}_mean"] = float(np.mean(res[f"train_{PRIMARY_METRIC}"]))
        fila["gap_train_val"] = fila[f"train_{PRIMARY_METRIC}_mean"] - fila[f"{PRIMARY_METRIC}_mean"]
        log.info("CV %s: %s = %.3f ± %.3f", nombre, PRIMARY_METRIC,
                 fila[f"{PRIMARY_METRIC}_mean"], fila[f"{PRIMARY_METRIC}_std"])
        filas.append(fila)
    return (pd.DataFrame(filas)
            .sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)
            .reset_index(drop=True))


def get_param_grids() -> dict[str, dict]:
    """Rejillas por modelo, con los nombres que espera el Pipeline (prefijo 'model__')."""
    return {
        "logreg": {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]},
        "forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "hgb": {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__max_iter": [100, 200],
            "model__l2_regularization": [0.0, 1.0],
        },
    }


def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Devuelve exactitud, F1, ROC-AUC y el tamaño de la prueba (dict serializable a JSON)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": len(y_test),
    }


def bootstrap_ci(y_true, proba, metric=roc_auc_score, n_boot: int = 1000,
                 random_state: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap (remuestreo con reemplazo de los pares)."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_true)
    p = np.asarray(proba)
    puntajes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        puntajes.append(metric(y[idx], p[idx]))
    bajo, alto = np.percentile(puntajes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(bajo), float(alto)


def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Construye el pipeline con las decisiones congeladas y lo entrena con TODO (X, y)."""
    num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    modelo = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    pipe = build_pipeline(modelo, num, cat).set_params(**FINAL_PARAMS)
    return pipe.fit(X, y)


def save_model(pipeline: Pipeline, metadata: dict, name: str = MODEL_NAME,
               models_dir: Path = MODELS) -> Path:
    """Guarda el pipeline (.joblib) y un archivo de metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "model_sha256": sha256_of(model_path),
    }
    (models_dir / f"{name}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", model_path)
    return model_path


def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Carga el pipeline y sus metadatos; avisa si la versión de scikit-learn cambió."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Modelo entrenado con scikit-learn %s; versión actual %s",
                    meta.get("sklearn_version"), sklearn.__version__)
    return pipeline, meta


def predict(pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Devuelve un DataFrame con 'proba_survive' y 'pred' (0/1) para cada fila de X."""
    proba = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {"proba_survive": proba, "pred": (proba >= threshold).astype(int)},
        index=X.index,
    )


def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Silver -> evaluación honesta -> modelo final entrenado y guardado."""
    silver_path = Path(in_dir) / SILVER_NAME if in_dir else SILVER / SILVER_NAME
    X, y = load_dataset(silver_path)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # Evaluación honesta: modelo entrenado solo con X_tr, probado una vez en X_te
    pipe_eval = train_final(X_tr, y_tr)
    metrics = evaluate_on_test(pipe_eval, X_te, y_te, THRESHOLD)
    proba_te = pipe_eval.predict_proba(X_te)[:, 1]
    ci_bajo, ci_alto = bootstrap_ci(y_te, proba_te, random_state=settings.random_state)
    metrics["roc_auc_ci95"] = [ci_bajo, ci_alto]
    log.info("Prueba: exactitud=%.3f f1=%.3f roc_auc=%.3f (IC95 %.3f-%.3f), n=%d",
             metrics["accuracy"], metrics["f1"], metrics["roc_auc"], ci_bajo, ci_alto,
             metrics["n_test"])

    # Modelo definitivo: con TODOS los datos etiquetados
    pipeline = train_final(X, y)
    num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    metadata = {
        "estimator": FINAL_ESTIMATOR,
        "params": FINAL_PARAMS,
        "features": {"numeric": num, "categorical": cat},
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": settings.random_state,
        "n_train": len(X),
        "data_sha256": sha256_of(silver_path),
        "metrics": metrics,
    }
    return save_model(pipeline, metadata, models_dir=models_dir)
