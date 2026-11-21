"""Точка входа песочницы оценки. Код движка, но исполняется ВНУТРИ контейнера:
здесь загружается предиктор (pickle = исполнение кода) и код фичей агента.

Три режима, каждый в своём контейнере:
  inspect  пишет /out/info.json: что на самом деле лежит в предикторе и
           умолчания AutoGluon для сверки. Код агента в контейнер не монтируется,
           поэтому features.py не может подделать сводку.
  predict  пишет /out/proba.json: вероятности положительного класса для строк
           отложенной выборки, и /out/features.parquet: признаки после
           features.py (для гейта утечки). Здесь исполняется features.py агента.
  lineage  пишет /out/lineage.json: от каких колонок train зависит каждый признак
           после features.py. Колонки (и таргет) по очереди перемешиваются,
           каждый вызов features.py в отдельном fork, чтобы состояние модуля
           не переходило между вызовами. Движок сверяет это с планом фичей.
Меток отложенной выборки в контейнерах нет, метрики считает движок снаружи.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def default_hyperparameters(cfg: dict) -> dict:
    """Умолчания AutoGluon этой версии для семейств из конфига (до загрузки pickle)."""
    from autogluon.tabular.models import CatBoostModel, LGBModel, XGBoostModel
    MODEL_CLASSES = {"GBM": LGBModel, "CAT": CatBoostModel, "XGB": XGBoostModel}
    out = {}
    for family in cfg["automl"]["hyperparameters"]:
        if family not in MODEL_CLASSES:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            m = MODEL_CLASSES[family](problem_type=cfg["task"]["type"],
                                      eval_metric=cfg["metric"]["eval_metric"],
                                      path=tmp + "/", name="ref")
            m.initialize(num_classes=2 if cfg["task"]["type"] == "binary" else None)
            out[family] = dict(m.params)
    return out


def inspect(cfg: dict) -> None:
    from autogluon.tabular import TabularPredictor
    defaults = default_hyperparameters(cfg)
    predictor = TabularPredictor.load("/predictor", require_py_version_match=False)
    info = predictor.info()
    summary = {
        "label": predictor.label,
        "problem_type": predictor.problem_type,
        "eval_metric": predictor.eval_metric.name,
        "features": predictor.features(),
        "original_features": list(predictor.original_features),
        "num_bag_folds": info["num_bag_folds"],
        "max_core_stack_level": info["max_core_stack_level"],
        "time_limit": info["time_limit"],
        "models": {name: {"model_type": m["model_type"], "hyperparameters": m["hyperparameters"],
                          **({"bagged_info": {"child_model_type": m["bagged_info"].get("child_model_type"),
                                              "child_hyperparameters": m["bagged_info"].get("child_hyperparameters"),
                                              "num_child_models": m["bagged_info"].get("num_child_models")}}
                             if m.get("bagged_info") else {})}
                   for name, m in info["model_info"].items()},
        "default_hyperparameters": defaults,
    }
    Path("/out/info.json").write_text(json.dumps(summary, default=str), encoding="utf-8")


def predict(cfg: dict) -> None:
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load("/predictor", require_py_version_match=False)
    sys.path.insert(0, "/work")
    from train.features import build_features

    df = build_features(pd.read_parquet("/data/holdout_features.parquet"), cfg)
    proba = predictor.predict_proba(df)[cfg["data"]["positive_class"]]
    Path("/out/proba.json").write_text(json.dumps([float(x) for x in proba]), encoding="utf-8")
    df.reset_index(drop=True).to_parquet("/out/features.parquet", index=False)


def _same(a: pd.Series, b: pd.Series) -> bool:
    return len(a) == len(b) and a.reset_index(drop=True).equals(b.reset_index(drop=True))


def lineage(cfg: dict) -> None:
    sys.path.insert(0, "/work")
    from train.features import build_features

    df = pd.read_parquet("/data/train.parquet")
    label = cfg["data"]["label"]

    def call(frame: pd.DataFrame, k: int) -> pd.DataFrame | None:
        path = f"/tmp/lineage_{k}.pkl"
        pid = os.fork()
        if pid == 0:  # ребёнок: свежая копия состояния после импорта
            try:
                build_features(frame.copy(), cfg).to_pickle(path)
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        return pd.read_pickle(path) if status == 0 and os.path.exists(path) else None

    base = call(df, 0)
    if base is None:
        raise SystemExit("features.py упал на train")
    again = call(df, 1)
    outputs = [c for c in base.columns if c != label]
    nondet = [c for c in outputs if again is None or c not in again or not _same(base[c], again[c])]
    deps: dict[str, list[str]] = {c: [] for c in outputs}
    failed = []
    for i, col in enumerate(df.columns):
        shuffled = df.copy()
        shuffled[col] = df[col].sample(frac=1.0, random_state=1000 + i).to_numpy()
        if _same(shuffled[col], df[col]):
            continue  # константа: перемешивание ничего не меняет
        out = call(shuffled, i + 2)
        if out is None:
            failed.append(col)
            continue
        for f in outputs:
            if f not in out or not _same(base[f], out[f]):
                deps[f].append(col)
    Path("/out/lineage.json").write_text(json.dumps(
        {"outputs": outputs, "deps": deps, "nondeterministic": nondet, "failed_on": failed}), encoding="utf-8")


if __name__ == "__main__":
    config = yaml.safe_load(Path("/engine/config.yaml").read_text(encoding="utf-8"))
    {"inspect": inspect, "predict": predict, "lineage": lineage}[sys.argv[1]](config)
