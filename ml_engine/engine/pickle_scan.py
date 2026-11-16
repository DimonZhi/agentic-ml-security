"""Сканер pickle в папке предиктора. Доверенное ядро, работает на хосте.

Файлы не загружаются по-настоящему: собственный Unpickler вместо импорта
каждого класса подставляет пустую заглушку и только записывает его имя.
Так видно, какие функции и классы pickle вызвал бы при загрузке, и при этом
ни одна из них не выполняется.

Правило: каждое имя должно начинаться с разрешённого префикса. Всё остальное
(os.system, subprocess, builtins.eval, importlib и т. п.) означает отказ.
Список собран по предикторам AutoGluon 1.6.3 с GBM, CAT, XGB.
"""
import io
import pickle
from pathlib import Path

ALLOWED_PREFIXES = (
    "autogluon.", "sklearn.", "numpy.", "pandas.", "lightgbm.", "catboost.", "xgboost.",
    "networkx.classes.", "scipy.sparse.",
    "collections.OrderedDict", "collections.defaultdict", "collections.Counter",
    "builtins.list", "builtins.dict", "builtins.set", "builtins.frozenset", "builtins.tuple",
    "builtins.slice", "builtins.bytearray", "builtins.complex", "builtins.float", "builtins.int",
    "builtins.str", "builtins.bytes", "builtins.bool", "builtins.object", "builtins.range",
    "copyreg._reconstructor", "_codecs.encode", "datetime.", "decimal.Decimal", "uuid.UUID",
    "pathlib.PosixPath", "pathlib.PurePosixPath", "functools.partial", "re._compile",
    "enum.", "_operator.itemgetter",
)
# разрешённые префиксы, внутри которых всё равно есть опасные вызовы
DENIED = ("numpy.testing", "numpy.f2py", "numpy.distutils", "pandas.io.", "sklearn.utils._testing")


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): pass
    def __setitem__(self, k, v): pass
    def append(self, x): pass
    def extend(self, xs): pass
    def add(self, x): pass


class _Recorder(pickle.Unpickler):
    def __init__(self, data: bytes):
        super().__init__(io.BytesIO(data))
        self.names: set[str] = set()

    def find_class(self, module, name):
        self.names.add(f"{module}.{name}")
        return type(name, (_Stub,), {"__new__": lambda cls, *a, **k: object.__new__(cls)})

    def persistent_load(self, pid):
        self.names.add(f"<persistent_id {pid!r}>")
        return None


def allowed(name: str) -> bool:
    return name.startswith(ALLOWED_PREFIXES) and not name.startswith(DENIED)


def scan(root: Path) -> list[str]:
    """Возвращает список нарушений; пустой список значит «чисто»."""
    problems: list[str] = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root)
        if f.is_symlink():
            problems.append(f"{rel}: символическая ссылка")
            continue
        data = f.read_bytes()
        if not data.startswith(b"\x80"):  # не pickle (json, txt, бинарник xgboost)
            if f.suffix == ".pkl":
                problems.append(f"{rel}: .pkl без заголовка pickle")
            continue
        rec = _Recorder(data)
        try:
            rec.load()
        except Exception as e:  # битый или нестандартный pickle тоже отказ
            problems.append(f"{rel}: не разобран ({type(e).__name__})")
        bad = sorted(n for n in rec.names if not allowed(n))
        if bad:
            problems.append(f"{rel}: запрещённые вызовы {bad}")
    return problems


if __name__ == "__main__":
    import sys
    print("\n".join(scan(Path(sys.argv[1]))) or "чисто")
