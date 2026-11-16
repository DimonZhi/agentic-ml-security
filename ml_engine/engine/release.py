"""Релизный прогон (spec, шаг 7). Доверенное ядро.

Берёт исследовательский run с вердиктом passed, заново обучает модель в
чистой песочнице из снимка кода этого run и того же утверждённого конфига,
оценивает её и сравнивает предсказания на отложенной выборке с исходными
(отпечаток proba_sha256, точность 6 знаков). Без совпадения релиз не кандидат:
значит, модель зависит от чего-то кроме кода, конфига и данных.
Финальным артефактом считается только релизная модель с тегом candidate=true.

Запуск: python -m engine.release --run-id <id исследовательского run>
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow

from engine.evaluate import DEFAULT_TRACKING_URI, Rejected, evaluate, record_rejection
from engine.train_run import train

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = mlflow.MlflowClient()

    src = client.get_run(args.run_id)
    if src.data.tags.get("mode") != "explore" or src.data.tags.get("verdict") != "passed":
        raise SystemExit("Релиз возможен только из исследовательского run с вердиктом passed")
    config_sha = src.data.tags["config_sha256"]

    with tempfile.TemporaryDirectory() as tmp:
        code = Path(mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path="code/train", dst_path=tmp))
        run_id = train(config_sha, mode="release", code_dir=code,
                       extra_tags={"source_run_id": args.run_id})

    if client.get_run(run_id).data.tags.get("code_sha256") != src.data.tags.get("code_sha256"):
        raise SystemExit("Снимок кода релиза не совпадает с исходным run")

    try:
        verdict = evaluate(run_id)
    except Rejected as e:
        record_rejection(run_id, e)
        raise SystemExit(f"Релиз отклонён ({e.stage}):\n- " + "\n- ".join(e.reasons))

    src_fp = src.data.tags.get("proba_sha256")
    reproducible = src_fp is not None and verdict["proba_sha256"] == src_fp
    candidate = verdict["verdict"] == "passed" and reproducible

    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict({"source_run_id": args.run_id, "source_proba_sha256": src_fp,
                         "release_proba_sha256": verdict["proba_sha256"],
                         "reproducible": reproducible}, "engine/reproducibility.json")
        mlflow.set_tags({"reproducible": str(reproducible).lower(),
                         "candidate": str(candidate).lower()})

    print(json.dumps({"release_run_id": run_id, "source_run_id": args.run_id,
                      "verdict": verdict["verdict"], "reproducible": reproducible,
                      "candidate": candidate, "holdout": verdict["holdout"]}, ensure_ascii=False, indent=2))
    if not candidate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
