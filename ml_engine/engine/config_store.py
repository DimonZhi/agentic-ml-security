"""Хранилище утверждённых конфигов. Доверенное ядро движка.

Каждая утверждённая версия лежит как approvals/<sha256>.yaml (точная копия
байтов) и approvals/<sha256>.json (кто и когда утвердил). На шаге 1-2 это
папка рядом с кодом; на шаге 3 она монтируется в песочницу только на чтение.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

STORE = Path(os.environ.get("ENGINE_APPROVALS_DIR", "approvals"))


class NotApproved(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def approve(config_path: str, approved_by: str) -> str:
    data = Path(config_path).read_bytes()
    yaml.safe_load(data)  # конфиг должен хотя бы парситься
    sha = sha256_bytes(data)
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / f"{sha}.yaml").write_bytes(data)
    record = {
        "config_sha256": sha,
        "source": str(config_path),
        "approved_by": approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (STORE / f"{sha}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return sha


def load_approved(sha: str) -> tuple[dict, dict]:
    """Возвращает (конфиг, запись аппрува) или бросает NotApproved."""
    cfg_path, rec_path = STORE / f"{sha}.yaml", STORE / f"{sha}.json"
    if not cfg_path.exists() or not rec_path.exists():
        raise NotApproved(f"Конфиг {sha} не утверждён")
    data = cfg_path.read_bytes()
    if sha256_bytes(data) != sha:
        raise NotApproved(f"Копия конфига {sha} в хранилище изменена")
    return yaml.safe_load(data), json.loads(rec_path.read_text())
