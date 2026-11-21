"""Инструменты движка для DS-агента (spec, шаги 1-5 и 10). Доверенное ядро.

Это весь интерфейс агента к фабрике: агент не читает файлы данных, не
запускает docker, не пишет в MLflow и не видит отложенную выборку. Каждая
функция принимает и возвращает только JSON-совместимые значения, поэтому
слой не зависит от LLM: тот же набор отдаётся Claude, внутреннему шлюзу или
MCP-серверу.

Чего здесь нет намеренно: аппрува конфига (только человек, engine.approve),
релиза (engine.release) и доступа к отложенной выборке.

Все строки, пришедшие из данных (имена колонок, значения категорий), для
агента это недоверенный текст. Они обрезаются и возвращаются в поле data.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine import evaluate as ev
from engine.sandbox import SandboxError
from engine.check_config import FAMILIES
from engine import feature_plan, gates, policy
from engine.config_store import STORE, NotApproved, load_approved, sha256_bytes

TASKS = Path("tasks")
WORKSPACES = Path("workspaces")      # код агента: workspaces/<task_id>/train/*.py
PROPOSALS = Path("proposals")        # предложенные конфиги, ждут человека
REFERENCE_CODE = Path("train")       # стартовая версия кода агента

K_MIN = 10                    # категории и бины с меньшим числом строк агенту не показываем
MAX_STR = 60                  # длина строк из данных в ответах
MAX_CODE_BYTES = 50_000
MAX_CATEGORIES = 15
CODE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,40}\.py$")


class AgentError(Exception):
    """Ошибка, которую агент видит и может исправить."""


def _s(x) -> str:
    return str(x)[:MAX_STR]


def _task(task_id: str) -> dict:
    for f in TASKS.glob("*.yaml"):
        t = yaml.safe_load(f.read_text(encoding="utf-8"))
        if t.get("id") == task_id:
            return t
    raise AgentError(f"задача {task_id!r} не найдена")


PASSTHROUGH_FEATURES = '''"""Фича инжиниринг. Зона DS-агента: этот файл агент меняет свободно в рамках
плана фичей из утверждённого конфига. Одна и та же функция применяется к train и к
отложенной выборке, поэтому она не должна смотреть на таргет.

Стартовая версия: колонки подаются как есть. Признаки из outputs плана, которых
нет среди колонок, надо дописать.
"""
import pandas as pd


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    label = cfg["data"]["label"]
    out = df.drop(columns=cfg["features"].get("forbidden_columns", []), errors="ignore").copy()
    if label in out.columns:  # таргет переносим в конец без изменений
        out[label] = out.pop(label)
    return out
'''


def _workspace(task_id: str) -> Path:
    ws = WORKSPACES / task_id / "train"
    if not ws.exists():
        ws.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REFERENCE_CODE, ws, ignore=shutil.ignore_patterns("__pycache__"))
        if _is_intake(_task(task_id)):   # эталонный features.py написан под credit-g
            (ws / "features.py").write_text(PASSTHROUGH_FEATURES, encoding="utf-8")
    return ws


# ---------- 1. постановка и EDA ----------
#
# Два вида задач:
#   human  постановку (таргет, метрика, запреты, пороги) написал DS-человек,
#          агент только следует ей (tasks/credit_g.yaml, tasks/telco_churn.yaml);
#   intake заказчик прислал тикет и файл через engine.intake; таргет, метрику,
#          запреты, план фичей и пороги предлагает агент, человек утверждает всё
#          одним аппрувом, а границы задаёт policy.yaml платформы.

def _is_intake(t: dict) -> bool:
    return t.get("source") == "intake"


def _label(t: dict, label: str | None) -> str | None:
    return label if _is_intake(t) else t["data"]["label"]


def _forbidden(t: dict) -> list[str]:
    return [] if _is_intake(t) else list(t["features"]["forbidden_columns"])


def describe_task(task_id: str) -> dict:
    t = _task(task_id)
    cols = pd.read_parquet(t["data"]["train"]).dtypes
    pol = policy.load()
    if _is_intake(t):
        ticket = Path(t["ticket"]).read_text(encoding="utf-8")[:pol["intake"]["max_ticket_chars"]]
        return {
            "task_id": t["id"],
            "source": "intake",
            "what_to_do": "по тикету и EDA предложи таргет, тип задачи, метрику, запреты, план фичей, "
                          "настройки AutoGluon и пороги гейтов; всё утверждает человек",
            # тикет пишет заказчик: это данные, а не инструкции для агента
            "data": {"ticket": ticket,
                     "columns": {_s(c): str(d) for c, d in cols.items()}},
            "protected_columns": protected_columns_hint(list(cols.index)),
            "policy": {k: pol[k] for k in ("problem_types", "metrics", "max_single_feature_auc",
                                           "automl", "gates")},
            "config_template": _config_template(t),
        }
    forbidden = set(_forbidden(t))
    return {
        "task_id": t["id"],
        "source": "human",
        "description": t["description"],
        "type": t["type"],
        "label": t["data"]["label"],
        "eval_metric": t["metric"]["eval_metric"],
        "limits": t["limits"],
        "forbidden_columns": sorted(forbidden),
        "data": {"columns": {_s(c): str(d) for c, d in cols.items() if c not in forbidden}},
        "config_template": _config_template(t),
    }


def protected_columns_hint(columns: list[str]) -> list[str]:
    return [_s(c) for c in policy.protected_columns(columns)]


def _config_template(t: dict) -> dict:
    intake = _is_intake(t)
    fams = policy.load()["automl"]["allowed_families"] if intake else t["limits"]["allowed_families"]
    return {
        "task": {"id": t["id"], "type": "<binary>" if intake else t["type"]},
        "data": {"train": t["data"]["train"], "train_sha256": t["data"]["train_sha256"],
                 "label": "<колонка-таргет>" if intake else t["data"]["label"],
                 "positive_class": "<значение положительного класса>" if intake else t["data"]["positive_class"]},
        "split": {"val_fraction": 0.2, "seed": 42},
        "features": {"forbidden_columns": ["<колонки, которые нельзя давать модели>"] if intake else _forbidden(t),
                     "forbidden_reasons": {"<колонка>": "<почему запрещена>"} if intake else {},
                     "plan": [{"name": "<группа признаков>", "meaning": "<смысл>",
                               "inputs": ["<исходные колонки train>"],
                               "outputs": ["<имена признаков для модели; без outputs = inputs как есть>"]}]},
        "metric": {"eval_metric": "<roc_auc>" if intake else t["metric"]["eval_metric"],
                   "report": ["roc_auc", "gini", "pr_auc"]},
        "automl": {"engine": "autogluon", "presets": "medium_quality",
                   "hyperparameters": {f: {} for f in fams},
                   "num_bag_folds": 5, "num_stack_levels": 1, "time_limit_sec": 600, "seed": 42},
        "gates": ({"max_holdout_evaluations": 3, "min_holdout_roc_auc": "<минимально полезное качество>",
                   "baseline_run_id": None, "min_gain_over_baseline": 0.0,
                   "max_val_holdout_gap": 0.05, "max_single_feature_auc": 0.9}
                  if intake else "<задаёт человек в постановке, движок подставит сам>"),
        "mlflow": t["mlflow"],
    }


def run_eda(task_id: str, label: str = "") -> dict:
    """Только агрегаты по train. Сырых строк, редких категорий и точных
    экстремумов агент не получает. Для задачи из тикета таргет ещё не выбран:
    агент передаёт кандидата в label (или пустую строку, тогда без связи с таргетом)."""
    t = _task(task_id)
    df = pd.read_parquet(t["data"]["train"]).drop(columns=_forbidden(t), errors="ignore")
    label = _label(t, label or None)
    y = None
    out = {"n_rows": int(len(df)), "note": f"агрегаты по train; группы меньше {K_MIN} строк скрыты",
           "columns": {}}
    if label:
        if label not in df.columns:
            raise AgentError(f"нет колонки {label!r}")
        raw = df.pop(label)
        values = raw.astype(str).value_counts()
        out["label"] = {"name": _s(label), "n_unique": int(raw.nunique()),
                        "values": {_s(k): int(v) for k, v in values.head(MAX_CATEGORIES).items() if v >= K_MIN}}
        if raw.nunique() == 2:
            # для задачи из тикета положительный класс ещё не выбран: берём «больший»
            pos = (sorted(raw.dropna().unique(), key=str)[-1] if _is_intake(t)
                   else t["data"]["positive_class"])
            y = (raw == pos).astype(int)
            out["label"]["rate_of"] = _s(pos)
            out["target_rate"] = round(float(y.mean()), 3)
    limit = policy.load()["max_single_feature_auc"]
    for c in df.columns:
        x = df[c]
        info = {"dtype": str(x.dtype), "missing_share": round(float(x.isna().mean()), 3),
                "n_unique": int(x.nunique())}
        numeric = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
        if numeric:
            q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
            info["quantiles"] = {f"p{int(k * 100)}": float(v) for k, v in q.items()}
        if y is not None:
            auc = gates.single_feature_auc(x, y)
            if auc is not None:
                info["auc_alone"] = round(auc, 3)
                if auc > limit:
                    info["leak_suspect"] = True
        if numeric:
            if y is not None:
                info["corr_with_target"] = round(float(np.corrcoef(x.fillna(x.median()), y)[0, 1]), 3)
                try:
                    bins = pd.qcut(x, 5, duplicates="drop")
                    g = y.groupby(bins, observed=True).agg(["mean", "size"])
                    info["target_rate_by_quintile"] = [round(float(m), 3) for m, n in g.values if n >= K_MIN]
                except ValueError:
                    pass
        else:
            s = x.astype(str)
            if y is not None:
                g = y.groupby(s).agg(["mean", "size"]).sort_values("size", ascending=False)
            else:
                g = s.value_counts().to_frame("size").assign(mean=np.nan)[["mean", "size"]]
            shown = g[g["size"] >= K_MIN].head(MAX_CATEGORIES)
            info["categories"] = {_s(k): ({"share": round(float(n / len(df)), 3)} |
                                          ({} if y is None else {"target_rate": round(float(m), 3)}))
                                  for k, (m, n) in shown.iterrows()}
            info["hidden_categories"] = int(len(g) - len(shown))
        out["columns"][_s(c)] = info
    return {"data": out}


# ---------- 2-3. конфиг ----------

def propose_config(task_id: str, config_yaml: str, rationale: str) -> dict:
    """Проверяет черновик и кладёт его на аппрув человеку. Сам не утверждает."""
    t = _task(task_id)
    try:
        cfg = yaml.safe_load(config_yaml)
    except yaml.YAMLError as e:
        raise AgentError(f"конфиг не парсится: {e}")
    problems = _validate(cfg, t)
    if problems:
        return {"status": "invalid", "problems": problems}
    if not _is_intake(t):
        cfg["gates"] = dict(t["gates"])  # пороги из постановки, не из предложения
    data = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode()
    sha = sha256_bytes(data)
    PROPOSALS.mkdir(exist_ok=True)
    (PROPOSALS / f"{sha}.yaml").write_bytes(data)
    (PROPOSALS / f"{sha}.rationale.md").write_text(_review_sheet(t, cfg, sha, rationale), encoding="utf-8")
    return {"status": "pending_approval", "config_sha256": sha,
            "next": f"человек утверждает: python -m engine.approve proposals/{sha}.yaml --by <кто>"}


def _review_sheet(t: dict, cfg: dict, sha: str, rationale: str) -> str:
    """Лист для человека: что именно он утверждает. Пишет движок, а не агент."""
    f = cfg["features"]
    lines = [f"# Предложение конфига {sha}", "", f"Задача: {t['id']} ({'тикет заказчика' if _is_intake(t) else 'постановка DS'})"]
    if _is_intake(t):
        lines += ["", "## Тикет заказчика", "", Path(t["ticket"]).read_text(encoding="utf-8")[:5000]]
    lines += ["", "## Что утверждается", "",
              f"- таргет: `{cfg['data']['label']}`, положительный класс `{cfg['data']['positive_class']}`, "
              f"тип `{cfg['task']['type']}`, метрика `{cfg['metric']['eval_metric']}`",
              f"- запрещены: {', '.join(f'`{c}`' for c in f['forbidden_columns']) or 'нет'}"]
    for c, why in (f.get("forbidden_reasons") or {}).items():
        lines.append(f"  - `{c}`: {str(why)[:200]}")
    lines.append("- план фичей:")
    for g in f["plan"]:
        lines.append(f"  - {g['name']}: {g.get('meaning', '')}; из {g['inputs']} -> {g.get('outputs', 'как есть')}")
    ag = cfg["automl"]
    lines += [f"- AutoGluon: {sorted(ag['hyperparameters'])}, бэггинг {ag['num_bag_folds']} фолдов, "
              f"стекинг {ag['num_stack_levels']} уровней, {ag['time_limit_sec']} с, seed {ag['seed']}",
              f"- пороги гейтов: {cfg['gates']}",
              "", "## Обоснование агента", "", rationale[:5000], ""]
    return "\n".join(lines)


def _validate(cfg, t: dict) -> list[str]:
    if not isinstance(cfg, dict):
        return ["конфиг должен быть словарём"]
    p = []
    need = ["task", "data", "split", "features", "metric", "automl", "mlflow"]
    p += [f"нет раздела {k}" for k in need if not isinstance(cfg.get(k), dict)]
    if p:
        return p
    pol = policy.load()
    train = pd.read_parquet(t["data"]["train"])
    columns = list(train.columns)
    for k in ("train", "train_sha256"):
        if cfg["data"].get(k) != t["data"][k]:
            p.append(f"data.{k} должен быть {t['data'][k]!r} (из постановки)")
    if cfg["task"].get("id") != t["id"]:
        p.append(f"task.id должен быть {t['id']!r}")
    if cfg["mlflow"] != t["mlflow"]:
        p.append(f"раздел mlflow должен быть {t['mlflow']}")
    label = cfg["data"].get("label")
    forbidden = cfg["features"].get("forbidden_columns") or []
    if not isinstance(forbidden, list):
        return p + ["features.forbidden_columns: список"]

    if _is_intake(t):
        ptype = cfg["task"].get("type")
        if ptype not in pol["problem_types"]:
            p.append(f"task.type: одно из {pol['problem_types']}")
        elif cfg["metric"].get("eval_metric") not in pol["metrics"][ptype]:
            p.append(f"metric.eval_metric для {ptype}: одно из {pol['metrics'][ptype]}")
        if label not in columns:
            p.append(f"data.label: колонки {label!r} нет в train")
        else:
            values = train[label].dropna().unique()
            if ptype == "binary" and len(values) != 2:
                p.append(f"data.label: для binary нужно ровно 2 значения, в {label!r} их {len(values)}")
            elif cfg["data"].get("positive_class") not in list(values):
                p.append(f"data.positive_class: одно из значений таргета {[_s(v) for v in values][:5]}")
            elif not p:
                y = (train[label] == cfg["data"]["positive_class"]).astype(int)
                leaks = policy.leaky_columns(train.drop(columns=forbidden, errors="ignore")
                                             .assign(**{label: y}), label)
                for c, auc in leaks.items():
                    p.append(f"колонка {c!r} сама угадывает таргет (AUC {auc}): утечка, добавь в forbidden_columns")
        missing = sorted(set(policy.protected_columns([c for c in columns if c != label])) - set(forbidden))
        if missing:
            p.append(f"защищённые признаки по policy.yaml должны быть в forbidden_columns: {missing}")
        reasons = cfg["features"].get("forbidden_reasons") or {}
        if not isinstance(reasons, dict) or set(forbidden) - set(reasons):
            p.append("features.forbidden_reasons: причина для каждой запрещённой колонки")
        p += policy.check_gates(cfg.get("gates"))
    else:
        for k in ("label", "positive_class"):
            if cfg["data"].get(k) != t["data"][k]:
                p.append(f"data.{k} должен быть {t['data'][k]!r} (из постановки)")
        if cfg["task"].get("type") != t["type"]:
            p.append("task.type берётся из постановки")
        if cfg["metric"].get("eval_metric") != t["metric"]["eval_metric"]:
            p.append(f"metric.eval_metric должен быть {t['metric']['eval_metric']!r}")
        missing = set(_forbidden(t)) - set(forbidden)
        if missing:
            p.append(f"features.forbidden_columns обязан содержать {sorted(missing)}")
        if "gates" in cfg and cfg["gates"] != t["gates"] and not isinstance(cfg["gates"], str):
            p.append("gates задаёт человек в постановке, уберите раздел")
        ag = cfg["automl"]
        if not set(ag.get("hyperparameters") or {}) <= set(t["limits"]["allowed_families"]):
            p.append(f"automl.hyperparameters: только семейства {t['limits']['allowed_families']}")
        if isinstance(ag.get("time_limit_sec"), int) and ag["time_limit_sec"] > t["limits"]["max_time_limit_sec"]:
            p.append(f"automl.time_limit_sec: не больше {t['limits']['max_time_limit_sec']} (постановка)")

    if label:
        p += feature_plan.validate(cfg, columns, label)
        inputs = feature_plan.input_columns(cfg) if isinstance(cfg["features"].get("plan"), list) else []
        ids = [c for c in policy.id_like_columns(train) if c in inputs]
        if ids:
            p.append(f"колонки похожи на идентификаторы (почти все значения уникальны), в план их брать нельзя: {ids}")
    if _is_intake(t):
        unknown = [c for c in forbidden if c not in columns]
        if unknown:
            p.append(f"features.forbidden_columns: нет в train {unknown} (имена колонок в snake_case, см. describe_task)")
    p += policy.check_automl(cfg["automl"])
    if not set(cfg["automl"].get("hyperparameters") or {}) <= set(FAMILIES):
        p.append(f"для сверки модели движок знает семейства {sorted(FAMILIES)}")
    vf = cfg["split"].get("val_fraction")
    if not isinstance(vf, (int, float)) or not 0.05 <= vf <= 0.5:
        p.append("split.val_fraction от 0.05 до 0.5")
    return p


def config_status(config_sha256: str) -> dict:
    try:
        _, rec = load_approved(config_sha256)
        return {"config_sha256": config_sha256, "status": "approved",
                "approved_by": rec["approved_by"], "approved_at": rec["approved_at"]}
    except NotApproved:
        pending = (PROPOSALS / f"{config_sha256}.yaml").exists()
        return {"config_sha256": config_sha256, "status": "pending_approval" if pending else "unknown"}


def read_config(config_sha256: str) -> dict:
    for f in (STORE / f"{config_sha256}.yaml", PROPOSALS / f"{config_sha256}.yaml"):
        if f.exists():
            return {"config_sha256": config_sha256, "config_yaml": f.read_text(encoding="utf-8")}
    raise AgentError("конфиг не найден")


# ---------- 4. код ----------

def list_code(task_id: str) -> dict:
    ws = _workspace(task_id)
    return {"files": sorted(f.name for f in ws.glob("*.py"))}


def read_code(task_id: str, filename: str) -> dict:
    f = _code_path(task_id, filename)
    if not f.exists():
        raise AgentError(f"файла {filename} нет")
    return {"filename": filename, "content": f.read_text(encoding="utf-8")}


def write_code(task_id: str, filename: str, content: str) -> dict:
    f = _code_path(task_id, filename)
    if len(content.encode()) > MAX_CODE_BYTES:
        raise AgentError(f"файл больше {MAX_CODE_BYTES} байт")
    try:
        compile(content, filename, "exec")
    except SyntaxError as e:
        raise AgentError(f"синтаксическая ошибка: {e}")
    f.write_text(content, encoding="utf-8")
    return {"filename": filename, "bytes": len(content.encode()),
            "sha256": hashlib.sha256(content.encode()).hexdigest()}


def _code_path(task_id: str, filename: str) -> Path:
    if not CODE_NAME.match(filename or ""):
        raise AgentError("имя файла: только латиница, цифры и _, расширение .py, без папок")
    return _workspace(task_id) / filename


# ---------- 5. запуск и результаты ----------

def run_experiment(task_id: str, config_sha256: str) -> dict:
    """Обучение в песочнице по утверждённому конфигу. Возвращает run_id и
    метрики на val, которые посчитал код агента (не движок)."""
    from engine.train_run import train   # тяжёлый импорт только здесь
    try:
        cfg, _ = load_approved(config_sha256)
    except NotApproved:
        return {"status": "not_approved", "detail": "конфиг не утверждён человеком"}
    if cfg["task"]["id"] != task_id:
        raise AgentError("конфиг от другой задачи")
    try:
        run_id = train(config_sha256, mode="explore", code_dir=_workspace(task_id))
    except SystemExit as e:
        return {"status": "failed", "detail": str(e)[-1500:]}
    return {"status": "trained", **run_summary(run_id)}


def evaluate_run(run_id: str) -> dict:
    """Оценка движком на отложенной выборке. Расходует лимит оценок."""
    try:
        ev.evaluate(run_id)
    except ev.Rejected as e:
        ev.record_rejection(run_id, e)
    return run_summary(run_id)


def run_summary(run_id: str) -> dict:
    """Только то, что агенту можно видеть: теги движка, метрики (округлённые),
    итог гейтов. Логи песочницы не отдаются: это текст, который написал сам агент."""
    import mlflow
    c = mlflow.MlflowClient()
    run = c.get_run(run_id)
    tags, m = run.data.tags, run.data.metrics
    out = {"run_id": run_id, "mode": tags.get("mode"), "config_sha256": tags.get("config_sha256"),
           "train_status": tags.get("train_status"),
           "val_metrics_by_agent_code": {k: round(v, 4) for k, v in m.items() if k.startswith("val_")},
           "verdict": tags.get("verdict", "not_evaluated")}
    if tags.get("verdict") in ("passed", "failed"):
        v = json.loads(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/engine/verdict.json"))
        out["holdout"] = v["holdout"]
        out["gates"] = [{"gate": g["gate"], "passed": g["passed"]} for g in v["gates"]]
    if tags.get("verdict") == "rejected" or tags.get("last_rejection"):
        out["rejected_at"] = tags.get("rejected_at") or tags.get("last_rejection")
        out["rejection_reasons"] = _last_rejection(c, run_id)
    return out


def _last_rejection(c, run_id: str) -> list[str]:
    import mlflow
    files = sorted(a.path for a in c.list_artifacts(run_id, "engine") if "rejected_" in a.path)
    if not files:
        return []
    uri = c.get_run(run_id).info.artifact_uri
    return json.loads(mlflow.artifacts.load_text(f"{uri}/{files[-1]}"))["reasons"][:10]


def list_runs(task_id: str) -> dict:
    import mlflow
    t = _task(task_id)
    exp = mlflow.get_experiment_by_name(t["mlflow"]["experiment"])
    if exp is None:
        return {"runs": []}
    runs = mlflow.MlflowClient().search_runs([exp.experiment_id], f"tags.task_id = '{task_id}'",
                                              max_results=50, order_by=["attributes.start_time DESC"])
    return {"runs": [{"run_id": r.info.run_id, "mode": r.data.tags.get("mode"),
                      "verdict": r.data.tags.get("verdict", "not_evaluated"),
                      "holdout_roc_auc": r.data.metrics.get("holdout_roc_auc"),
                      "config_sha256": r.data.tags.get("config_sha256")} for r in runs]}


TOOLS = {f.__name__: f for f in (describe_task, run_eda, propose_config, config_status, read_config,
                                 list_code, read_code, write_code, run_experiment, evaluate_run,
                                 list_runs)}


def call(name: str, args: dict) -> str:
    """Единая точка вызова для любого LLM-цикла: имя и аргументы на входе,
    JSON-строка на выходе. Ошибки агента возвращаются как {"error": ...}."""
    if name not in TOOLS:
        return json.dumps({"error": f"нет инструмента {name}"}, ensure_ascii=False)
    try:
        return json.dumps(TOOLS[name](**args), ensure_ascii=False, default=str)
    except AgentError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"error": f"неверные аргументы: {e}"}, ensure_ascii=False)
    except SandboxError as e:   # проблема окружения: детали человеку (stderr), агенту только общий отказ
        import sys
        print(f"[engine] SandboxError: {e}", file=sys.stderr)
        return json.dumps({"error": "песочница недоступна (инфраструктура). Не повторяй вызов, "
                                    "остановись и сообщи человеку", "infrastructure": True}, ensure_ascii=False)
    except Exception as e:   # ошибка движка не должна ронять цикл агента; детали только в лог движка
        return json.dumps({"error": f"ошибка движка: {type(e).__name__}"}, ensure_ascii=False)
