---
title: "AutoGluon Tabular - In Depth"
source: "https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html"
added: "2026-10-18"
read: "2026-10-18"
status: summarized
coverage: "Полный текст туториала в документации AutoGluon (ветка stable; в логах ноутбука версия 1.6.1.dev0): все разделы от стекинга и бэггинга до References и Next Steps, код ячеек и лог fit. Дополнительно прочитана справка TabularPredictor.load (https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.load.html). Связанные туториалы, FAQ, статья Erickson et al. 2020 и ноутбуки с SHAP не открывались. Локальная копия не сохранялась"
tags: [autogluon, automl, tabular, ensembling, threshold-calibration, pickle, supply-chain, tutorial]
---

# AutoGluon Tabular - In Depth

## Коротко

[Туториал AutoGluon](https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html) о тонкой настройке `fit()` и `predict()`
у `TabularPredictor` — open-source AutoML для табличных данных, который сам готовит признаки, обучает набор моделей и собирает
из них ансамбль. Пример построен на открытом датасете Adult (бинарная классификация `class`, метрика `balanced_accuracy`).
О безопасности в туториале ничего нет; материал добавлен как описание AutoML-движка, близкого к нашей фабрике.

## Основные идеи

- Вступление: сначала пробовать `fit()` с параметрами по умолчанию, затем `eval_metric` и `presets`, и лишь потом
  `hyperparameters`, `num_stack_levels`, `num_bag_folds`, `num_bag_sets`.
- «Model ensembling with stacking/bagging»: `num_bag_folds=8, num_stack_levels=1` часто повышают качество ценой времени,
  памяти и диска. При стекинге и бэггинге не передавать `tuning_data`: все данные отдаются в `train_data`, и AutoGluon делит их сам.
  `num_bag_sets` повторяет k-fold бэггинг; `auto_stack` подбирает значения автоматически (используется в пресете best_quality).
- «Decision Threshold Calibration»: для бинарной классификации подбор порога вместо 0.5 может сильно улучшить `f1` или
  `balanced_accuracy`. В примере калибровка под `balanced_accuracy` ухудшила `accuracy`; авторы предупреждают о компромиссе между
  метриками. Порог хранится в предикторе (`set_decision_threshold`), может задаваться в отдельном вызове
  `predict(..., decision_threshold=0.8)`; по умолчанию `calibrate_decision_threshold="auto"`, и авторы советуют его не менять.
- «Prediction options»: `TabularPredictor.load(save_path)` загружает обученный предиктор; модели можно обучить на одной машине и
  развернуть на другой, скопировав папку `save_path`. По умолчанию предсказывает `model_best` (обычно ансамбль), но модель можно
  выбрать аргументом `model=`. `leaderboard` показывает `score_val`, `score_test`, время инференса и обучения, а с `extra_info` —
  число признаков, предков в стеке и память. Метрики всегда в форме «больше лучше», поэтому у `log_loss` и RMSE знак инвертирован.
  Предиктор помнит метрику, по которой его оценивать (`evaluate`).
- «Interpretability»: `feature_importance` считается перестановкой столбцов; признаки с неположительной важностью почти не помогают
  или вредят. Для локальных объяснений авторы отсылают к ноутбукам с Shapley values.
- «Accelerating inference»: таблица оптимизаций по приоритету — `refit_full` (8x, только с бэггингом), `persist` (до 10x в онлайн-инференсе),
  `infer_limit` (до 50x+), отбор признаков, более быстрое железо, ручные гиперпараметры, ручная предобработка; рекомендуемый порядок —
  первые три. Цифры — оценки авторов.
- `persist` держит модели в памяти; `infer_limit` и `infer_limit_batch_size` задают ограничение латентности как условие обучения.
- «Using smaller ensembles», «refit_full», «Model distillation»: альтернативные ансамбли по фронту Парето точность–скорость; refit-full
  модели обучаются на всех данных и поэтому не имеют `score_val`; дистилляция может заметно снизить точность.
- «Faster presets or hyperparameters»: `presets=['good_quality','optimize_for_deployment']`, `excluded_model_types=['KNN','NN_TORCH']`.
- «(Advanced) Disable preprocessing»: `feature_generator=None` по словам авторов снимает все ограждения по санитизации данных
  («removes ALL guardrails on data sanitization»).
- «If you encounter memory / disk space issues»: без `path` модели всё равно сохраняются в `AutogluonModels/ag-[TIMESTAMP]`;
  `save_space()` и `delete_models(models_to_keep='best')` освобождают диск.
- Лог `fit` на странице (не текст туториала): в 1.6 есть пресет `extreme` с табличными foundation-моделями (Nori, TabICLv2,
  TabDPT-Turbo; нужен `pip install autogluon.tabular[tabarena]`) и пресет `noncommercial` с TabPFN-3, для коммерческого использования
  которого нужна лицензия Prior Labs.
- Справка `TabularPredictor.load`, раздел Warning: загрузка использует pickle; вредоносные данные могут исполнить произвольный код
  при распаковке, поэтому загружать только доверенные данные. Параметры `require_version_match` и `require_py_version_match`
  по умолчанию требуют совпадения версий AutoGluon и Python.

## Для проекта

Интерпретация для [нашего кейса](../../PROJECT.md); гипотезы, не проверенные на стенде:

- AutoGluon воспроизводит три объекта фабрики: конфигурацию обучения (метрика, пресет, порог, исключённые модели), отчёт
  (`leaderboard`) и артефакт (папка предиктора), поэтому подходит как AutoML-движок для стенда.
- Отравление данных под переобучение: при бэггинге `score_val` считается на фолдах тех же данных, что идут в обучение, поэтому
  отравленные объекты влияют и на обучение, и на внутреннюю валидацию. У refit-full и дистиллированных моделей внутренней оценки нет.
  Это согласуется с тезисами [LLM05](owasp-llm-top-10-2026.md) об атаке на контур переобучения.
- Подмена конфигурации и подгонка метрики: метрика и порог решения хранятся в артефакте, а порог переопределяется одним аргументом
  `predict()`, то есть поведение модели меняется без переобучения. `feature_generator=None` отключает проверки данных одним параметром.
- Цепочка поставок: загрузка предиктора — распаковка pickle, а туториал описывает развёртывание копированием папки между машинами;
  пресет `extreme` тянет дополнительные пакеты и предобученные веса. Соответствует [LLM04](owasp-llm-top-10-2026.md)
  и [ASI04](owasp-agentic-top-10-2026.md).
- Какая модель обслуживает запросы, задаётся `model_best` или аргументом `model=`; одобренная модель и работающая могут различаться.

## Ограничения и вопросы

- Это документация вендора о качестве и скорости, а не исследование безопасности; единственное предупреждение о риске (pickle)
  взято со страницы справки `load()`, а не из туториала.
- Цифры ускорения — оценки авторов из таблицы, на нашем железе не измерялись.
- Пункты «Для проекта» — гипотезы; эксперименты с AutoGluon на стенде не проводились.
- Неизвестно, используется ли AutoGluon в фабрике; переносимость на её AutoML-движок не проверена.
- Не открывались: Quick Start, Deployment Optimization, FAQ, статья Erickson et al. (2020), туториал о кастомных моделях.
