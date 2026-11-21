"""Политика платформы (policy.yaml) и проверка предложений агента по ней.

Политику пишет владелец платформы. Агент предлагает задачу и конфиг, человек
утверждает, но ни то, ни другое не может выйти за эти границы.
"""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from engine import gates

POLICY = Path("policy.yaml")


@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load(POLICY.read_text(encoding="utf-8"))


def protected_columns(columns: list[str]) -> list[str]:
    pats = [re.compile(p) for p in load()["protected_column_patterns"]]
    return [c for c in columns if any(p.search(c.lower()) for p in pats)]


def leaky_columns(df: pd.DataFrame, label: str) -> dict[str, float]:
    """Колонки, которые сами по себе почти угадывают таргет."""
    limit = load()["max_single_feature_auc"]
    y = df[label]
    out = {}
    for c in df.columns:
        if c == label:
            continue
        auc = gates.single_feature_auc(df[c], y)
        if auc is not None and auc > limit:
            out[c] = round(auc, 3)
    return out


def id_like_columns(df: pd.DataFrame) -> list[str]:
    """Текстовые колонки, где почти все значения уникальны: идентификаторы клиентов,
    договоров и т. п. Модель на них запоминает строки, а не учится."""
    n = len(df)
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and n and df[c].nunique() / n > 0.9]


def check_automl(ag: dict) -> list[str]:
    pol = load()["automl"]
    p = []
    fams = ag.get("hyperparameters")
    if not isinstance(fams, dict) or not fams or not set(fams) <= set(pol["allowed_families"]):
        p.append(f"automl.hyperparameters: только семейства {pol['allowed_families']}")
    elif not all(isinstance(v, dict) for v in fams.values()):
        p.append("гиперпараметры семейства задаются словарём")
    tl = ag.get("time_limit_sec")
    if not isinstance(tl, int) or not 10 <= tl <= pol["max_time_limit_sec"]:
        p.append(f"automl.time_limit_sec: целое от 10 до {pol['max_time_limit_sec']}")
    if ag.get("num_bag_folds") not in pol["num_bag_folds"]:
        p.append(f"automl.num_bag_folds: одно из {pol['num_bag_folds']}")
    st = ag.get("num_stack_levels")
    if not isinstance(st, int) or not 0 <= st <= pol["max_num_stack_levels"]:
        p.append(f"automl.num_stack_levels: от 0 до {pol['max_num_stack_levels']}")
    elif st > 0 and not ag.get("num_bag_folds"):
        p.append("стекинг в AutoGluon требует бэггинга: num_bag_folds > 0")
    if ag.get("presets") not in pol["allowed_presets"]:
        p.append(f"automl.presets: одно из {pol['allowed_presets']}")
    if not isinstance(ag.get("seed"), int) or ag.get("seed") < 0:
        p.append("automl.seed: неотрицательное целое")
    if ag.get("engine") != "autogluon":
        p.append("automl.engine: autogluon")
    return p


def check_gates(g) -> list[str]:
    bounds = load()["gates"]
    if not isinstance(g, dict):
        return ["нужен раздел gates с порогами (границы в policy.yaml)"]
    p = []
    for k, b in bounds.items():
        v = g.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not b["min"] <= v <= b["max"]:
            p.append(f"gates.{k}: число от {b['min']} до {b['max']}")
    if "baseline_run_id" in g and g["baseline_run_id"] is not None and not isinstance(g["baseline_run_id"], str):
        p.append("gates.baseline_run_id: строка или null")
    extra = set(g) - set(bounds) - {"baseline_run_id"}
    if extra:
        p.append(f"gates: неизвестные пороги {sorted(extra)}")
    return p
