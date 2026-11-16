"""План фичей из утверждённого конфига и сверка с ним (spec, разделы 5 и 6).

Формат группы в features.plan:
  name:    имя группы
  meaning: смысл для человека
  inputs:  исходные колонки train, из которых группа считается
  outputs: имена признаков, которые группа даёт модели.
           Нет outputs = колонки inputs подаются в модель как есть.

Движок использует план трижды:
  1. в песочницу обучения и оценки попадают только колонки из inputs всех групп;
  2. признаки, на которых обучен предиктор, должны быть в outputs;
  3. каждый признак может зависеть только от inputs своей группы и не от таргета.
     Зависимости меряет engine/eval_entry.py (режим lineage): прогоняет features.py
     агента на train, по очереди перемешивая каждую колонку, и смотрит, какие
     признаки изменились.
"""
import re

NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def groups(cfg: dict) -> list[dict]:
    out = []
    for g in cfg["features"].get("plan") or []:
        inputs = list(g.get("inputs") or [])
        outputs = list(g.get("outputs") or inputs)
        out.append({"name": g.get("name"), "inputs": inputs, "outputs": outputs})
    return out


def input_columns(cfg: dict) -> list[str]:
    """Колонки train, которые получает песочница (без таргета), в порядке плана."""
    seen = []
    for g in groups(cfg):
        seen += [c for c in g["inputs"] if c not in seen]
    return seen


def allowed_inputs(cfg: dict) -> dict[str, set[str]]:
    """признак -> колонки, от которых ему можно зависеть (объединение по группам)."""
    allowed: dict[str, set[str]] = {}
    for g in groups(cfg):
        for f in g["outputs"]:
            allowed.setdefault(f, set()).update(g["inputs"])
    return allowed


def validate(cfg: dict, columns: list[str], label: str) -> list[str]:
    """Проверка плана до аппрува: колонки существуют, не запрещены, не таргет."""
    plan = cfg["features"].get("plan")
    if not isinstance(plan, list) or not plan:
        return ["features.plan: непустой список групп (name, meaning, inputs, outputs)"]
    p = []
    forbidden = set(cfg["features"].get("forbidden_columns") or [])
    names = set()
    for i, g in enumerate(plan):
        where = f"features.plan[{i}]"
        if not isinstance(g, dict) or not isinstance(g.get("name"), str) or not g["name"]:
            p.append(f"{where}: нужны name, meaning, inputs")
            continue
        where = f"features.plan.{g['name']}"
        if g["name"] in names:
            p.append(f"{where}: имя группы повторяется")
        names.add(g["name"])
        if not isinstance(g.get("meaning"), str) or not g["meaning"].strip():
            p.append(f"{where}: нужен meaning (смысл для человека)")
        inputs, outputs = g.get("inputs"), g.get("outputs", g.get("inputs"))
        if not isinstance(inputs, list) or not inputs or not all(isinstance(c, str) for c in inputs):
            p.append(f"{where}.inputs: непустой список колонок")
            continue
        if not isinstance(outputs, list) or not outputs or not all(isinstance(c, str) for c in outputs):
            p.append(f"{where}.outputs: непустой список имён признаков")
            continue
        unknown = [c for c in inputs if c not in columns]
        if unknown:
            p.append(f"{where}.inputs: нет в train {unknown[:10]}")
        bad = sorted(set(inputs) & (forbidden | {label}))
        if bad:
            p.append(f"{where}.inputs: запрещённые колонки или таргет {bad}")
        bad_names = [f for f in outputs if not NAME.match(f)]
        if bad_names:
            p.append(f"{where}.outputs: имена только из латиницы, цифр и _ {bad_names[:10]}")
        bad = sorted(set(outputs) & (forbidden | {label}))
        if bad:
            p.append(f"{where}.outputs: имя совпадает с запрещённой колонкой или таргетом {bad}")
        raw = sorted(set(outputs) & set(columns) - set(inputs))
        if raw:
            p.append(f"{where}.outputs: {raw} это исходные колонки не из inputs этой группы")
    return p


def check(model_features: list[str], lineage: dict, cfg: dict) -> list[str]:
    """Сверка того, что реально получила модель, с утверждённым планом."""
    label = cfg["data"]["label"]
    allowed = allowed_inputs(cfg)
    deps = lineage.get("deps") or {}
    nondet = set(lineage.get("nondeterministic") or [])
    p = []
    if not groups(cfg):
        return ["в утверждённом конфиге нет плана фичей"]
    failed = lineage.get("failed_on") or []
    if failed:
        p.append(f"features.py падает, если перемешать колонки {failed[:10]}: проверка зависимостей невозможна")
    outside = [f for f in model_features if f not in allowed]
    if outside:
        p.append(f"признаки модели вне плана фичей: {sorted(outside)[:20]}")
    for f in model_features:
        if f in nondet:
            p.append(f"{f}: features.py даёт разные значения на одних и тех же данных")
            continue
        if f not in deps:
            p.append(f"{f}: признака нет в выходе features.py")
            continue
        used = set(deps[f])
        if label in used:
            p.append(f"{f}: зависит от таргета {label!r} (утечка в коде фичей)")
        extra = sorted(used - {label} - allowed.get(f, set()))
        if f in allowed and extra:
            p.append(f"{f}: считается из колонок вне своей группы {extra[:10]}")
    return p
