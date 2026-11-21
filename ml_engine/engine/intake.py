"""Приём задачи от заказчика: текстовый тикет + файл с данными. Доверенное ядро.

Движок, а не агент:
  1. читает файл (csv, xlsx, parquet), приводит имена колонок к snake_case,
     переводит в числа текстовые колонки, где почти все значения числа;
  2. сразу отрезает отложенную выборку (policy.yaml: intake.holdout_fraction).
     Таргет ещё неизвестен, поэтому разбиение случайное, не стратифицированное.
     Агент отложенную выборку не видит никогда, в том числе в EDA;
  3. пишет tasks/<task_id>.yaml без таргета, метрики и запретов: их предложит
     DS-агент после EDA, утвердит DS-человек одним аппрувом с конфигом.

Тикет это недоверенный текст (его пишет заказчик), агент получает его как данные.

Запуск:
  python -m engine.intake --task-id churn-002 --ticket ticket.txt \\
      --data data/Telco_customer_churn.xlsx --by <заказчик>
"""
import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from engine import policy

TASK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
    return s if s and not s[0].isdigit() else f"c_{s}"


def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise SystemExit(f"Формат {ext} не поддерживается: csv, xlsx, parquet")


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols, seen = [], {}
    for c in df.columns:
        s = snake(c)
        seen[s] = seen.get(s, 0) + 1
        cols.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
    df = df.copy()
    df.columns = cols
    for c in df.select_dtypes("object").columns:
        text = df[c].astype(str).str.strip()
        as_num = pd.to_numeric(text.where(text != ""), errors="coerce")
        filled = df[c].notna() & (text != "")
        if filled.sum() and as_num[filled].notna().mean() >= 0.95:
            df[c] = as_num
        else:
            df[c] = df[c].astype(str).where(df[c].notna(), None)
    # колонки-константы ничего не дают модели и только шумят в EDA
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=const), const


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--ticket", required=True, help="текстовый файл с тикетом заказчика")
    p.add_argument("--data", required=True)
    p.add_argument("--by", required=True, help="кто заказчик")
    a = p.parse_args()

    if not TASK_ID.match(a.task_id):
        raise SystemExit("task-id: латиница в нижнем регистре, цифры и дефис, 3-41 символ")
    task_file = Path("tasks") / f"{a.task_id}.yaml"
    if task_file.exists():
        raise SystemExit(f"Задача {a.task_id} уже есть: {task_file}")
    pol = policy.load()["intake"]
    ticket = Path(a.ticket).read_text(encoding="utf-8")
    if len(ticket) > pol["max_ticket_chars"]:
        raise SystemExit(f"Тикет длиннее {pol['max_ticket_chars']} символов")

    df, dropped = normalize(read_any(Path(a.data)))
    if len(df) < pol["min_rows"]:
        raise SystemExit(f"Мало строк: {len(df)}, нужно не меньше {pol['min_rows']}")
    holdout = df.sample(frac=pol["holdout_fraction"], random_state=pol["seed"])
    train = df.drop(index=holdout.index)

    train_path = Path("data") / a.task_id / "train.parquet"
    ho_dir = Path("holdout") / a.task_id
    train_path.parent.mkdir(parents=True, exist_ok=True)
    ho_dir.mkdir(parents=True, exist_ok=True)
    train.reset_index(drop=True).to_parquet(train_path, index=False)
    holdout.reset_index(drop=True).to_parquet(ho_dir / "holdout.parquet", index=False)
    (ho_dir / "holdout.sha256").write_text(sha256(ho_dir / "holdout.parquet") + "\n", encoding="utf-8")
    ticket_path = Path("tickets") / f"{a.task_id}.md"
    ticket_path.parent.mkdir(exist_ok=True)
    ticket_path.write_text(ticket, encoding="utf-8")

    task = {
        "id": a.task_id,
        "source": "intake",               # таргет, метрику, запреты и пороги предлагает агент
        "ticket": str(ticket_path),
        "requested_by": a.by,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"train": str(train_path), "train_sha256": sha256(train_path),
                 "original_file": Path(a.data).name, "dropped_constant_columns": dropped},
        "mlflow": {"experiment": a.task_id},
    }
    task_file.parent.mkdir(exist_ok=True)
    task_file.write_text(yaml.safe_dump(task, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Задача {a.task_id}: train {len(train)} строк, отложенная выборка {len(holdout)} строк, "
          f"{df.shape[1]} колонок (константы убраны: {dropped or 'нет'})")
    print(f"Дальше: python -m agent.ds_agent --task {a.task_id}")


if __name__ == "__main__":
    main()
