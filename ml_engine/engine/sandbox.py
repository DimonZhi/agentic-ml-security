"""Запуск кода в песочнице (Docker). Доверенное ядро движка.

Жёсткие ограничения задаёт движок, конфиг и код агента их изменить не могут:
без сети, корневая ФС только на чтение, без привилегий и capabilities,
непривилегированный пользователь, лимиты CPU, памяти, числа процессов и времени.
"""
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = os.environ.get("ENGINE_SANDBOX_IMAGE", "ml-engine-sandbox:1")
MEMORY = os.environ.get("ENGINE_SANDBOX_MEMORY", "4g")
CPUS = os.environ.get("ENGINE_SANDBOX_CPUS", "2")
PIDS = "512"
MAX_TIME_LIMIT_SEC = 3600      # потолок для time_limit_sec из конфига
OVERHEAD_SEC = 180             # запас на запуск, фичи и сохранение модели
LOG_TAIL = 20_000              # сколько символов лога песочницы сохраняем


class SandboxError(Exception):
    pass


@dataclass
class Result:
    returncode: int
    log: str
    timed_out: bool


def image_id() -> str:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    # Docker Desktop с хранилищем containerd: inspect может не найти образ другой
    # платформы (например, собранный с --platform linux/amd64 на Apple Silicon),
    # хотя docker images его показывает
    listed = subprocess.run(["docker", "images", "--no-trunc", "--quiet", IMAGE],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    ids = listed.stdout.split()
    if listed.returncode == 0 and ids:
        return ids[0]
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise SandboxError("Docker не запущен (нет связи с демоном). Запустите Docker Desktop: open -a Docker")
    raise SandboxError(f"Образ {IMAGE} не найден. Соберите: docker build -t {IMAGE} sandbox/. "
                       f"docker image inspect: {(r.stderr or r.stdout).strip()[:300]}")


def run(command: list[str], ro: dict[Path, str], out_dir: Path, timeout_sec: int) -> Result:
    """ro: {путь на хосте: путь в контейнере}, всё только на чтение. out_dir -> /out на запись."""
    name = f"mle-{uuid.uuid4().hex[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o777)  # в контейнере работает непривилегированный uid
    args = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
        "--shm-size", "512m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", PIDS,
        "--memory", MEMORY, "--memory-swap", MEMORY,
        "--cpus", CPUS,
        "--user", "10001:10001",
        "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONUTF8=1", "-e", "PYTHONIOENCODING=utf-8",
        "-w", "/work",
        "-v", f"{out_dir.resolve()}:/out:rw",
    ]
    for host, inside in ro.items():
        args += ["-v", f"{Path(host).resolve()}:{inside}:ro"]
    args += [IMAGE, *command]

    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout_sec)
        return Result(r.returncode, (r.stdout + r.stderr)[-LOG_TAIL:], False)
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "kill", name], capture_output=True)
        log = "".join(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else (x or "")
                      for x in (e.stdout, e.stderr))
        return Result(-1, log[-LOG_TAIL:], True)
