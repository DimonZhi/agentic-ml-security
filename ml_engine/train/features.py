"""Фича инжиниринг. Зона DS-агента: этот файл агент меняет свободно в рамках
плана фичей из конфига. Одна и та же функция применяется к train и к
отложенной выборке, поэтому она не должна смотреть на таргет.
"""
import pandas as pd


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    label = cfg["data"]["label"]
    out = df.drop(columns=cfg["features"].get("forbidden_columns", []), errors="ignore").copy()

    out["credit_per_month"] = out["credit_amount"] / out["duration"].clip(lower=1)
    out["installment_burden"] = out["installment_commitment"] * out["credit_per_month"]
    out["age_bucket"] = pd.cut(
        out["age"], bins=[0, 25, 35, 50, 120], labels=["<25", "25-35", "35-50", "50+"]
    ).astype(str)

    if label in out.columns:  # таргет переносим в конец без изменений
        out[label] = out.pop(label)
    return out
