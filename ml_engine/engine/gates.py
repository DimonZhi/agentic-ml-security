"""Гейты валидации (spec, раздел 7). Доверенное ядро, работает на хосте.

Каждый гейт возвращает {"gate", "passed", "detail"}. Вердикт движка
«пройдено» только если прошли все гейты.
"""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_ROWS_FOR_CATEGORY = 10


def gate(name: str, passed: bool, detail: str) -> dict:
    return {"gate": name, "passed": bool(passed), "detail": detail}


def _auc(x: pd.Series, y: pd.Series) -> float | None:
    if y.nunique() < 2 or x.nunique() < 2:
        return None
    x = x.astype(float)
    if not np.isfinite(x).all():
        return None
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc)


def single_feature_auc(x: pd.Series, y: pd.Series) -> float | None:
    """Насколько один признак сам по себе предсказывает таргет: max(AUC, 1-AUC).
    Категории кодируются долей таргета внутри категории (на тех же данных,
    поэтому редкие категории сливаются, чтобы не завышать оценку).
    Пропуск тоже сигнал: колонка, заполненная только у одного класса (например,
    причина ухода), угадывает таргет одним фактом заполненности."""
    x, y = x.reset_index(drop=True), y.reset_index(drop=True)
    scores = []
    if not pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
        s = x.astype(str).where(x.notna(), "__missing__")
        counts = s.value_counts()
        s = s.where(s.map(counts) >= MIN_ROWS_FOR_CATEGORY, "__rare__")
        scores.append(_auc(s.map(y.groupby(s).mean()), y))
    else:
        mask = x.notna()
        scores.append(_auc(x[mask], y[mask]))
        if (~mask).any():
            scores.append(_auc((~mask).astype(int), y))
    scores = [v for v in scores if v is not None]
    return max(scores) if scores else None


def leakage(df: pd.DataFrame, y: pd.Series, limit: float, where: str) -> dict:
    scores = {c: single_feature_auc(df[c], y) for c in df.columns}
    scores = {c: round(s, 4) for c, s in scores.items() if s is not None}
    suspicious = {c: s for c, s in scores.items() if s > limit}
    top = dict(sorted(scores.items(), key=lambda kv: -kv[1])[:3])
    if suspicious:
        return gate(f"leakage_{where}", False,
                    f"признаки сами по себе почти угадывают таргет (AUC > {limit}): {suspicious}")
    return gate(f"leakage_{where}", True, f"максимум AUC одного признака {top}")


def quality(metric: float, cfg_gates: dict, baseline: float | None) -> list[dict]:
    out = []
    floor = cfg_gates.get("min_holdout_roc_auc")
    if floor is not None:
        out.append(gate("quality_floor", metric >= floor, f"ROC-AUC {metric:.3f}, порог {floor}"))
    if cfg_gates.get("baseline_run_id"):
        need = cfg_gates.get("min_gain_over_baseline", 0.0)
        if baseline is None:
            out.append(gate("baseline", False, "у бейзлайна нет оценки движка на отложенной выборке"))
        else:
            out.append(gate("baseline", metric - baseline >= need,
                            f"ROC-AUC {metric:.3f}, бейзлайн {baseline:.3f}, нужен прирост {need}"))
    return out


def stability(val_metric: float | None, holdout_metric: float, limit: float | None) -> dict:
    if limit is None:
        return gate("stability", True, "порог не задан")
    if val_metric is None or not math.isfinite(val_metric):
        return gate("stability", False, "нет метрики на val")
    gap = abs(val_metric - holdout_metric)
    return gate("stability", gap <= limit,
                f"val {val_metric:.3f} против отложенной {holdout_metric:.3f}, разрыв {gap:.3f}, порог {limit}")
