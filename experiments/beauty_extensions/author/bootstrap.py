"""Select the author source tree without modifying its read-only files."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path


AUTHOR_MODEL_SHA256 = {
    "tiger_generation_model_shuffle.py":
        "e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80",
    "tiger_generation_model_mlp.py":
        "1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9",
}


def author_root() -> Path:
    return Path(__file__).resolve().parents[3] / "references/wmh-original/GRID"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def assert_author_namespace() -> dict[str, str]:
    """Fail closed if this process has imported the working baseline as src."""
    expected = author_root() / "src"
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name != "src" and not name.startswith("src."):
            continue
        location = getattr(module, "__file__", None)
        if location is not None:
            if not _is_within(Path(location), expected):
                raise RuntimeError(
                    f"Mixed src namespaces: {name} is {location}; expected {expected}. "
                    "Run the author and baseline routes in separate Python processes."
                )
            loaded[name] = str(Path(location).resolve())
        else:
            for package_path in getattr(module, "__path__", ()):
                if not _is_within(Path(package_path), expected):
                    raise RuntimeError(f"Mixed src package path: {name}: {package_path}")
    return loaded


def bind_author_source() -> dict[str, object]:
    """Bind absolute src imports to the verified preserved author source.

    This changes only the current process import path.  It neither changes cwd
    nor invokes the old train/inference launchers (which have root-marker and
    automatic-restart behavior unsuitable for the controlled runner).
    """
    root = author_root()
    if not (root / "src/__init__.py").is_file():
        raise FileNotFoundError(f"Author source is missing at {root}")
    assert_author_namespace()
    identities = {}
    for filename, expected in AUTHOR_MODEL_SHA256.items():
        path = root / "src/models/modules/semantic_id" / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Preserved author source changed: {path}: {digest}")
        identities[str(path)] = digest
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    # Initialize utilities before the model hierarchy: the author modules use
    # both `from src.utils import ...` and `from src.utils.utils import ...`.
    importlib.import_module("src.utils")
    loaded = assert_author_namespace()
    return {"author_root": str(root), "model_sha256": identities, "modules": loaded}

