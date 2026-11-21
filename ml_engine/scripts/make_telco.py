"""Готовит датасет IBM Telco Customer Churn (xlsx) для ML Engine.

Пишет:
  data/telco_train.parquet                    train для DS-агента (80%)
  holdout/telco-churn-001/holdout.parquet     отложенная выборка, только для движка (20%)
  holdout/telco-churn-001/holdout.sha256
и вписывает sha256 train в tasks/telco_churn.yaml.

Что делает с данными:
  - имена колонок в snake_case (план фичей требует латиницу, цифры и _);
  - таргет churn = Churn Value (1 = ушёл);
  - Churn Label (копия таргета) удаляется; Churn Score, Churn Reason, CLTV
    остаются в данных, но запрещены в постановке: это проверка, что движок их не пустит;
  - константы (Count, Country, State) и дубль координат (Lat Long) удаляются;
  - Total Charges в число (у новых клиентов пустая строка -> NaN).

Запуск: python scripts/make_telco.py --xlsx data/Telco_customer_churn.xlsx
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TASK_ID = "telco-churn-001"
TASK_FILE = Path("tasks/telco_churn.yaml")
TRAIN = Path("data/telco_train.parquet")
HOLDOUT_DIR = Path("holdout") / TASK_ID


def snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = pd.read_excel(args.xlsx)
    df.columns = [snake(c) for c in df.columns]
    df = df.drop(columns=["count", "country", "state", "lat_long", "churn_label"])
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["churn"] = df.pop("churn_value").astype(int)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype(str).where(df[c].notna(), None)

    train, holdout = train_test_split(df, test_size=args.holdout_fraction,
                                      stratify=df["churn"], random_state=args.seed)
    TRAIN.parent.mkdir(exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    ho = HOLDOUT_DIR / "holdout.parquet"
    train.reset_index(drop=True).to_parquet(TRAIN, index=False)
    holdout.reset_index(drop=True).to_parquet(ho, index=False)
    (HOLDOUT_DIR / "holdout.sha256").write_text(sha256(ho) + "\n", encoding="utf-8")

    train_sha = sha256(TRAIN)
    text = TASK_FILE.read_text(encoding="utf-8")
    TASK_FILE.write_text(re.sub(r"train_sha256: \S+", f"train_sha256: {train_sha}", text, count=1), encoding="utf-8")
    print(f"train   {len(train):5d} строк, churn {train['churn'].mean():.3f}  sha256={train_sha}")
    print(f"holdout {len(holdout):5d} строк  sha256={sha256(ho)}")
    print(f"sha256 train вписан в {TASK_FILE}")


if __name__ == "__main__":
    main()
