"""Обучение: parquet -> фичи -> AutoGluon. Зона DS-агента.

Запускается движком внутри песочницы обучения (без сети, файловая система
только на чтение): python -m engine.train_run --config-sha <sha256>

Скрипт получает утверждённый конфиг и train только на чтение, делит train на
train/val, обучает TabularPredictor и кладёт результат в --out:
  predictor/         папка предиктора
  val_metrics.json   метрики на val (заявлены кодом агента, движок им не верит)
  leaderboard.csv
В MLflow скрипт не пишет и отложенную выборку не видит.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from train.features import build_features

# AutoGluon по умолчанию ставит seed=0 в каждой модели; имя параметра зависит от семейства
SEED_PARAM = {"GBM": "seed", "CAT": "random_seed", "XGB": "seed"}


def with_seeds(hyperparameters: dict, seed: int) -> dict:
    return {fam: {SEED_PARAM[fam]: seed, **(hp or {})} for fam, hp in hyperparameters.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    label, ag = cfg["data"]["label"], cfg["automl"]
    out = Path(args.out)

    df = build_features(pd.read_parquet(args.data), cfg)
    train_df, val_df = train_test_split(
        df, test_size=cfg["split"]["val_fraction"], stratify=df[label], random_state=cfg["split"]["seed"]
    )

    bagging = ag.get("num_bag_folds", 0) > 0
    predictor = TabularPredictor(
        label=label,
        problem_type=cfg["task"]["type"],
        eval_metric=cfg["metric"]["eval_metric"],
        path=str(out / "predictor"),
        verbosity=1,
    ).fit(
        train_data=train_df,
        # с бэггингом AutoGluon валидируется на out-of-fold, val остаётся честной проверкой
        tuning_data=None if bagging else val_df,
        presets=ag["presets"],
        hyperparameters=with_seeds(ag["hyperparameters"], ag["seed"]),
        num_bag_folds=ag.get("num_bag_folds", 0),
        num_stack_levels=ag.get("num_stack_levels", 0),
        dynamic_stacking=False,   # DyStack делает лишние обучения и съедает лимит времени
        time_limit=ag["time_limit_sec"],
    )

    val_scores = {k: float(v) for k, v in predictor.evaluate(val_df, silent=True).items()}
    (out / "val_metrics.json").write_text(json.dumps(val_scores, indent=2))
    predictor.leaderboard(val_df, silent=True).to_csv(out / "leaderboard.csv", index=False)
    print(json.dumps({"n_train": len(train_df), "n_val": len(val_df), "val": val_scores}))


if __name__ == "__main__":
    main()
