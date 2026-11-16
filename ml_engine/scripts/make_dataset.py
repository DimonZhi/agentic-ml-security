"""Готовит открытый датасет German Credit (OpenML credit-g) в parquet.

Пишет data/train.parquet (для DS-агента) и holdout/holdout.parquet (отложенная
выборка, только для оценщика движка) и печатает sha256 обоих файлов.
"""
import argparse
import hashlib
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    df["target"] = (df.pop("class") == "bad").astype(int)
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].astype(str)

    train, holdout = train_test_split(
        df, test_size=args.holdout_fraction, stratify=df["target"], random_state=args.seed
    )
    Path("data").mkdir(exist_ok=True)
    Path("holdout").mkdir(exist_ok=True)
    tr, ho = Path("data/train.parquet"), Path("holdout/holdout.parquet")
    train.reset_index(drop=True).to_parquet(tr, index=False)
    holdout.reset_index(drop=True).to_parquet(ho, index=False)
    (Path("holdout") / "holdout.sha256").write_text(sha256(ho) + "\n")

    print(f"train   {len(train):5d} строк  sha256={sha256(tr)}")
    print(f"holdout {len(holdout):5d} строк  sha256={sha256(ho)}")


if __name__ == "__main__":
    main()
