"""Аппрув конфига DS-человеком.

Запуск: python -m engine.approve configs/credit_g.yaml --by dima
Печатает sha256, по которому дальше запускаются обучение и оценка.
"""
import argparse

from engine.config_store import approve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--by", required=True, help="кто утверждает")
    args = p.parse_args()
    sha = approve(args.config, args.by)
    print(f"Утверждён конфиг {args.config}\nconfig_sha256={sha}")


if __name__ == "__main__":
    main()
