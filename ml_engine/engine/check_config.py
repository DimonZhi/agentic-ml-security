"""Сверка предиктора AutoGluon с утверждённым конфигом (spec, раздел 6).

Работает на стороне движка по сводке info.json из песочницы оценки и сам
pickle не загружает. Сравнивает то, что реально лежит в модели, а не код
агента: ловит подмену модели, метрики, гиперпараметров, seed, бэггинга и
признаков, даже если код обучения переписан.

Не ловит: утечку таргета в коде фичей и специально собранный pickle, который
подделывает собственную сводку (это сканер артефактов и гейты, шаг 4).
"""

# семейство в конфиге -> тип модели AutoGluon и имя гиперпараметра seed.
# Проверено на AutoGluon 1.6.3; другие семейства добавлять после проверки.
FAMILIES = {
    "GBM": ("LGBModel", "seed"),
    "CAT": ("CatBoostModel", "random_seed"),
    "XGB": ("XGBoostModel", "seed"),
}
# Бэггинг и стекинг (проверено на AutoGluon 1.6.3): каждая модель обёрнута в
# StackerEnsembleModel, итоговый ансамбль это WeightedEnsembleModel. Параметры
# обёртки должны быть умолчаниями: агент не может включить свои фолды
# (custom_splits), отключить стратификацию, поменять seed фолдов и т. п.
ENSEMBLE_TYPES = {"StackerEnsembleModel", "WeightedEnsembleModel"}
WRAPPER_DEFAULTS = {
    "use_orig_features": True, "valid_stacker": True, "max_base_models": 0,
    "max_base_models_per_type": "auto", "save_bag_folds": True, "stratify": "auto",
    "bin": "auto", "n_bins": None, "vary_seed_across_folds": False,
    "custom_splits": None, "model_random_seed": 0,
}
WEIGHTED_DEFAULTS = {"ensemble_size": 25, "subsample_size": 1000000}


def expected_hyperparameters(family: str, cfg: dict, defaults: dict) -> dict:
    """Полный набор: умолчания AutoGluon, поверх значения из конфига и seed.
    Сравнение на равенство всего набора, лишний параметр тоже нарушение."""
    params = dict(defaults.get(family, {}))
    params.update(cfg["automl"]["hyperparameters"][family] or {})
    params[FAMILIES[family][1]] = cfg["automl"]["seed"]
    return params


def check(info: dict, cfg: dict) -> list[str]:
    ag = cfg["automl"]
    problems: list[str] = []

    def expect(name, actual, expected):
        if actual != expected:
            problems.append(f"{name}: в модели {actual!r}, в конфиге {expected!r}")

    expect("label", info.get("label"), cfg["data"]["label"])
    expect("problem_type", info.get("problem_type"), cfg["task"]["type"])
    expect("eval_metric", info.get("eval_metric"), cfg["metric"]["eval_metric"])
    expect("num_bag_folds", info.get("num_bag_folds"), ag.get("num_bag_folds", 0))
    expect("stack_levels", (info.get("max_core_stack_level") or 0) - 1, ag.get("num_stack_levels", 0))

    limit, actual_limit = ag.get("time_limit_sec"), info.get("time_limit")
    if limit is not None and (not isinstance(actual_limit, (int, float)) or actual_limit > limit):
        problems.append(f"time_limit: в модели {actual_limit!r}, в конфиге не больше {limit}")

    unknown = set(ag["hyperparameters"]) - set(FAMILIES)
    if unknown:
        problems.append(f"семейства без правила сверки: {sorted(unknown)}")
    type_to_family = {FAMILIES[f][0]: f for f in ag["hyperparameters"] if f in FAMILIES}

    models = info.get("models") or {}
    if not models:
        problems.append("в предикторе нет моделей")
    for name, m in models.items():
        mtype, hp = m.get("model_type"), m.get("hyperparameters") or {}
        bag = m.get("bagged_info")
        if mtype in ENSEMBLE_TYPES:
            # бэггинг/стекинг: обёртка с параметрами ансамбля, внутри child-модели
            want_wrapper = dict(WRAPPER_DEFAULTS, use_orig_features=(mtype == "StackerEnsembleModel"))
            for key in sorted(set(want_wrapper) | set(hp)):
                if hp.get(key) != want_wrapper.get(key):
                    problems.append(f"{name}.{key}: в модели {hp.get(key)!r}, ожидается {want_wrapper.get(key)!r}")
            if not bag:
                problems.append(f"{name}: нет сведений о child-моделях")
                continue
            mtype, hp = bag.get("child_model_type"), bag.get("child_hyperparameters") or {}
        if mtype in ("WeightedEnsembleModel", "GreedyWeightedEnsembleModel"):
            for key in sorted(set(WEIGHTED_DEFAULTS) | set(hp)):
                if mtype == "GreedyWeightedEnsembleModel" and hp.get(key) != WEIGHTED_DEFAULTS.get(key):
                    problems.append(f"{name}.{key}: в модели {hp.get(key)!r}, ожидается {WEIGHTED_DEFAULTS.get(key)!r}")
            continue
        family = type_to_family.get(mtype)
        if family is None:
            problems.append(f"модель {name} ({mtype}) не разрешена конфигом")
            continue
        want = expected_hyperparameters(family, cfg, info.get("default_hyperparameters") or {})
        for key in sorted(set(want) | set(hp)):
            if hp.get(key) != want.get(key):
                problems.append(f"{name}.{key}: в модели {hp.get(key)!r}, ожидается {want.get(key)!r}")

    forbidden = set(cfg["features"].get("forbidden_columns", [])) & set(info.get("features") or [])
    if forbidden:
        problems.append(f"запрещённые признаки в модели: {sorted(forbidden)}")

    return problems
