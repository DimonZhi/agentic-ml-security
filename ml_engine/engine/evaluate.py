"""Оценщик движка. Доверенное ядро: агент этот файл не меняет и не запускает.

1. по тегу config_sha256 берёт утверждённый конфиг из хранилища движка;
2. сверяет sha256 отложенной выборки и проверяет лимит оценок на неё;
3. сканирует pickle в предикторе, ничего не загружая (engine/pickle_scan.py);
4. контейнер inspect (без кода агента) загружает предиктор и отдаёт сводку
   модели; движок сверяет её с конфигом (engine/check_config.py);
4a. контейнер lineage прогоняет features.py агента на train и меряет, от каких
   колонок зависит каждый признак; признаки модели сверяются с планом фичей
   (engine/feature_plan.py): только из outputs, только из inputs своей группы,
   не из таргета;
5. контейнер predict получает только признаки отложенной выборки, без меток,
   и отдаёт вероятности и признаки после features.py;
6. считает метрики своим кодом по меткам, которые не покидали движок;
7. гейты (engine/gates.py): качество и бейзлайн, устойчивость, утечка в train
   и в признаках агента;
8. пишет вердикт engine/verdict.json и теги verdict, holdout_* в тот же run.

На шагах 1-5 при любой ошибке оценка останавливается с verdict=rejected.
Запуск: python -m engine.evaluate --run-id <id>
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from engine import feature_plan, gates, sandbox
from engine.check_config import check
from engine.config_store import NotApproved, load_approved
from engine.pickle_scan import scan

HOLDOUT_ROOT = Path("holdout")


def holdout_paths(task_id: str) -> tuple[Path, Path]:
    """Отложенная выборка задачи: holdout/<task_id>/holdout.parquet. Для первой
    задачи (credit-g) остался старый путь holdout/holdout.parquet.
    HOLDOUT_PATH и HOLDOUT_SHA_PATH переопределяют путь (тесты)."""
    if os.environ.get("HOLDOUT_PATH"):
        return Path(os.environ["HOLDOUT_PATH"]), Path(os.environ["HOLDOUT_SHA_PATH"])
    per_task = HOLDOUT_ROOT / task_id
    if (per_task / "holdout.parquet").exists():
        return per_task / "holdout.parquet", per_task / "holdout.sha256"
    return HOLDOUT_ROOT / "holdout.parquet", HOLDOUT_ROOT / "holdout.sha256"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
ENTRY = Path(__file__).with_name("eval_entry.py")
EVAL_TIMEOUT_SEC = 600
REPORT_DIGITS = 4   # метрики, которые видит агент, округляются
PROBA_DIGITS = 6    # точность предсказаний для отпечатка воспроизводимости


class Rejected(Exception):
    def __init__(self, stage: str, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.stage, self.reasons = stage, reasons


def compute_metrics(y_true, proba) -> dict:
    auc = roc_auc_score(y_true, proba)
    return {"roc_auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y_true, proba)}


def read_proba(path: Path, n_rows: int) -> list[float]:
    """Вывод песочницы недоверенный: только список из n конечных чисел в [0, 1]."""
    proba = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(proba, list) or len(proba) != n_rows:
        raise ValueError("число предсказаний не совпадает с числом строк")
    for x in proba:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError("предсказания должны быть числами в [0, 1]")
    return proba


def holdout_evaluations(client, experiment_id: str, config_sha: str) -> int:
    runs = client.search_runs(
        [experiment_id],
        f"tags.config_sha256 = '{config_sha}' and tags.holdout_evaluated = 'true' and tags.mode = 'explore'",
        max_results=1000,
    )
    return len(runs)


def evaluate(run_id: str) -> dict:
    """Возвращает вердикт. При отказе на шагах 1-5 бросает Rejected."""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    tags = run.data.tags
    if tags.get("tags_written_by") != "engine":
        raise Rejected("approval", ["run создан не движком"])
    if tags.get("holdout_evaluated") == "true":
        raise Rejected("budget", ["этот run уже оценивался на отложенной выборке"])
    if tags.get("train_status") != "ok":
        raise Rejected("train", [f"обучение не завершилось: {tags.get('train_status')}"])

    try:
        cfg, _ = load_approved(tags.get("config_sha256", ""))
    except NotApproved as e:
        raise Rejected("approval", [str(e)])
    if tags.get("train_sha256") != cfg["data"]["train_sha256"]:
        raise Rejected("approval", ["train_sha256 в run не совпадает с конфигом"])
    cfg_gates = cfg.get("gates", {})

    holdout_path, holdout_sha_path = holdout_paths(cfg["task"]["id"])
    if not holdout_path.exists():
        raise SystemExit(f"Нет отложенной выборки для задачи {cfg['task']['id']}: {holdout_path}")
    actual = hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    if actual != holdout_sha_path.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Отложенная выборка изменена: sha256 {actual}")

    if tags.get("mode") == "explore":
        used = holdout_evaluations(client, run.info.experiment_id, tags["config_sha256"])
        limit = cfg_gates.get("max_holdout_evaluations")
        if limit is not None and used >= limit:
            raise Rejected("budget", [f"лимит оценок на отложенной выборке исчерпан: {used} из {limit} "
                                      "для этой версии конфига"])

    label = cfg["data"]["label"]
    holdout = pd.read_parquet(holdout_path)
    y_true = holdout[label].reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        predictor = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="predictor", dst_path=str(tmp / "p")))
        code = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="code/train", dst_path=str(tmp / "c")))

        bad = scan(predictor)
        if bad:
            raise Rejected("pickle_scan", bad)

        config = tmp / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        data = tmp / "data"
        data.mkdir()
        plan_cols = feature_plan.input_columns(cfg)
        holdout[plan_cols].to_parquet(data / "holdout_features.parquet", index=False)
        base = {predictor: "/predictor", config: "/engine/config.yaml",
                ENTRY: "/engine_code/eval_entry.py"}

        res = sandbox.run(["python", "/engine_code/eval_entry.py", "inspect"],
                          ro=base, out_dir=tmp / "out_inspect", timeout_sec=EVAL_TIMEOUT_SEC)
        if res.returncode != 0:
            raise Rejected("inspect", [f"песочница inspect: {'timeout' if res.timed_out else 'ошибка'}"])
        try:
            info = json.loads((tmp / "out_inspect" / "info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise Rejected("inspect", [f"некорректная сводка модели: {e}"])
        violations = check(info, cfg)
        if violations:
            raise Rejected("config_check", violations)

        lin_data = tmp / "lineage_data"
        lin_data.mkdir()
        pd.read_parquet(cfg["data"]["train"])[plan_cols + [label]] \
            .to_parquet(lin_data / "train.parquet", index=False)
        res = sandbox.run(["python", "/engine_code/eval_entry.py", "lineage"],
                          ro={config: "/engine/config.yaml", ENTRY: "/engine_code/eval_entry.py",
                              code: "/work/train", lin_data: "/data"},
                          out_dir=tmp / "out_lineage", timeout_sec=EVAL_TIMEOUT_SEC)
        if res.returncode != 0:
            raise Rejected("feature_plan", [f"песочница lineage: {'timeout' if res.timed_out else 'ошибка'}"])
        try:
            lineage = json.loads((tmp / "out_lineage" / "lineage.json").read_text(encoding="utf-8"))
            if not isinstance(lineage, dict) or not isinstance(lineage.get("deps"), dict):
                raise ValueError("нет deps")
        except (OSError, ValueError) as e:
            raise Rejected("feature_plan", [f"некорректный вывод lineage: {e}"])
        model_features = info.get("original_features") or info.get("features") or []
        violations = feature_plan.check(model_features, lineage, cfg)
        if violations:
            raise Rejected("feature_plan", violations)

        # с этого момента отложенная выборка считается использованной
        client.set_tag(run_id, "holdout_evaluated", "true")
        res = sandbox.run(["python", "/engine_code/eval_entry.py", "predict"],
                          ro={**base, code: "/work/train", data: "/data"},
                          out_dir=tmp / "out_predict", timeout_sec=EVAL_TIMEOUT_SEC)
        if res.returncode != 0:
            raise Rejected("predict", [f"песочница predict: {'timeout' if res.timed_out else 'ошибка'}"])
        try:
            proba = read_proba(tmp / "out_predict" / "proba.json", len(holdout))
            feats = pd.read_parquet(tmp / "out_predict" / "features.parquet")
            if len(feats) != len(holdout):
                raise ValueError("число строк признаков не совпадает")
        except Exception as e:
            raise Rejected("predict", [f"некорректный вывод predict: {e}"])

    metrics = compute_metrics(y_true, proba)

    baseline = None
    if cfg_gates.get("baseline_run_id"):
        b = client.get_run(cfg_gates["baseline_run_id"]).data
        if b.tags.get("verdict") == "passed":
            baseline = b.metrics.get("holdout_roc_auc")
    train_df = pd.read_parquet(cfg["data"]["train"]).drop(
        columns=cfg["features"].get("forbidden_columns", []), errors="ignore")
    leak_limit = cfg_gates.get("max_single_feature_auc", 0.9)
    results = [
        *gates.quality(metrics["roc_auc"], cfg_gates, baseline),
        gates.stability(run.data.metrics.get("val_roc_auc"), metrics["roc_auc"],
                        cfg_gates.get("max_val_holdout_gap")),
        gates.leakage(train_df.drop(columns=[label]), train_df[label], leak_limit, "train"),
        gates.leakage(feats.drop(columns=[label], errors="ignore"), y_true, leak_limit, "features"),
    ]
    passed = all(g["passed"] for g in results)

    verdict = {
        "run_id": run_id,
        "verdict": "passed" if passed else "failed",
        "mode": tags.get("mode"),
        "config_sha256": tags["config_sha256"],
        "code_sha256": tags.get("code_sha256"),
        "train_sha256": tags.get("train_sha256"),
        "holdout_sha256": actual,
        "image_id": tags.get("image_id"),
        "holdout": {k: round(v, REPORT_DIGITS) for k, v in metrics.items()},
        # отпечаток предсказаний: по нему релиз сверяется с исходным run
        "proba_sha256": hashlib.sha256(
            json.dumps([round(x, PROBA_DIGITS) for x in proba]).encode()).hexdigest(),
        "gates": results,
        "issued_by": "engine",
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"holdout_{k}": v for k, v in verdict["holdout"].items()})
        mlflow.log_dict(info, "engine/predictor_info.json")
        mlflow.log_dict(lineage, "engine/feature_lineage.json")
        mlflow.log_dict(verdict, "engine/verdict.json")
        mlflow.log_text(res.log, "sandbox/eval_log.txt")  # недоверенный текст из песочницы
        mlflow.set_tags({"evaluator": "engine", "config_check": "passed", "pickle_scan": "passed",
                         "feature_plan": "passed",
                         "holdout_sha256": actual, "verdict": verdict["verdict"],
                         "proba_sha256": verdict["proba_sha256"],
                         "gates_failed": ",".join(g["gate"] for g in results if not g["passed"]) or "-"})
    return verdict


def record_rejection(run_id: str, err: Rejected) -> None:
    """Пишет отказ в run. Уже выданный вердикт не перезаписывается: повторная
    попытка оценки не должна портить результат первой."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {"run_id": run_id, "verdict": "rejected", "stage": err.stage, "reasons": err.reasons,
              "issued_by": "engine", "issued_at": now}
    has_verdict = "verdict" in mlflow.MlflowClient().get_run(run_id).data.tags
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(record, f"engine/rejected_{err.stage}_{now.replace(':', '')}.json")
        mlflow.set_tag("last_rejection", err.stage)
        if not has_verdict:
            mlflow.set_tags({"evaluator": "engine", "verdict": "rejected", "rejected_at": err.stage})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    try:
        verdict = evaluate(args.run_id)
    except Rejected as e:
        record_rejection(args.run_id, e)
        raise SystemExit(f"Отказ ({e.stage}):\n- " + "\n- ".join(e.reasons))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    if verdict["verdict"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
