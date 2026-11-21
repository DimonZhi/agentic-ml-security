"""Запуск обучения движком. Доверенное ядро.

1. берёт утверждённый конфиг по sha256, сверяет sha256 train;
2. готовит копию train без запрещённых колонок и снимок кода агента (train/);
3. запускает train/train.py в песочнице обучения;
4. сам пишет run в MLflow: теги ставит движок, а не код агента.

Запуск: python -m engine.train_run --config-sha <sha256 из engine.approve>
Релизный прогон делает engine.release, он вызывает train() со снимком кода.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from engine import feature_plan, sandbox
from engine.config_store import load_approved

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
CODE_DIR = Path("train")


def dir_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def train(config_sha: str, mode: str = "explore", code_dir: Path = CODE_DIR,
          extra_tags: dict | None = None) -> str:
    """Обучает в песочнице и возвращает run_id."""
    cfg, approval = load_approved(config_sha)
    ag = cfg["automl"]
    if ag["time_limit_sec"] > sandbox.MAX_TIME_LIMIT_SEC:
        raise SystemExit(f"time_limit_sec больше потолка движка {sandbox.MAX_TIME_LIMIT_SEC}")

    if not feature_plan.input_columns(cfg):
        raise SystemExit("в утверждённом конфиге нет плана фичей (features.plan)")
    train_path = Path(cfg["data"]["train"])
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    if train_sha != cfg["data"]["train_sha256"]:
        raise SystemExit(f"train не совпадает с утверждённым: {train_sha}")

    image = sandbox.image_id()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # песочница получает только колонки из плана фичей и таргет
        data = tmp / "data" / "train.parquet"
        data.parent.mkdir()
        pd.read_parquet(train_path)[feature_plan.input_columns(cfg) + [cfg["data"]["label"]]] \
            .to_parquet(data, index=False)

        code = tmp / "code" / "train"
        shutil.copytree(code_dir, code, ignore=shutil.ignore_patterns("__pycache__"))
        config = tmp / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        out = tmp / "out"

        res = sandbox.run(
            ["python", "-m", "train.train", "--config", "/engine/config.yaml",
             "--data", "/data/train.parquet", "--out", "/out"],
            ro={code: "/work/train", config: "/engine/config.yaml", data.parent: "/data"},
            out_dir=out,
            timeout_sec=ag["time_limit_sec"] + sandbox.OVERHEAD_SEC,
        )

        with mlflow.start_run() as run:
            mlflow.set_tags({
                "mode": mode,
                "task_id": cfg["task"]["id"],
                "config_sha256": config_sha,
                "approved_by": approval["approved_by"],
                "train_sha256": train_sha,
                "code_sha256": dir_sha256(code),
                "image_id": image,
                "tags_written_by": "engine",
                **(extra_tags or {}),
            })
            mlflow.log_params({
                "label": cfg["data"]["label"],
                "eval_metric": cfg["metric"]["eval_metric"],
                "presets": ag["presets"],
                "model_families": ",".join(ag["hyperparameters"]),
                "time_limit_sec": ag["time_limit_sec"],
                "seed": ag["seed"],
            })
            mlflow.log_artifact(str(config), "config")
            mlflow.log_artifacts(str(code), "code/train")
            mlflow.log_text(res.log, "sandbox/train_log.txt")  # недоверенный текст из песочницы

            ok = res.returncode == 0 and (out / "predictor").is_dir()
            status = "ok" if ok else ("timeout" if res.timed_out else "failed")
            mlflow.set_tag("train_status", status)
            if not ok:
                raise SystemExit(f"Обучение в песочнице: {status} (run {run.info.run_id})\n{res.log[-2000:]}")

            # метрики на val заявлены кодом агента: берём только конечные числа
            try:
                val = json.loads((out / "val_metrics.json").read_text(encoding="utf-8"))
                val = {f"val_{k}": float(v) for k, v in val.items()
                       if isinstance(v, (int, float)) and math.isfinite(v)}
            except (OSError, ValueError, AttributeError):
                val = {}
            mlflow.log_metrics(val)
            mlflow.set_tag("val_metrics_source", "agent_code")
            if (out / "leaderboard.csv").is_file():
                mlflow.log_artifact(str(out / "leaderboard.csv"))
            mlflow.log_artifacts(str(out / "predictor"), "predictor")

            print(json.dumps({"run_id": run.info.run_id, "mode": mode, "val": val}, indent=2))
            return run.info.run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-sha", required=True)
    args = p.parse_args()
    train(args.config_sha)


if __name__ == "__main__":
    main()
