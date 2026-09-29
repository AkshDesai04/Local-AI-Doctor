"""Static import check for reviewed bundled checkpoint code.

A fingerprint-pinned review approves what bundled code *does*; it cannot promise
that the code still imports against the installed packages. Checkpoint code is
routinely written for an older Transformers release and references helpers that
were later removed. Loading such a checkpoint used to fail only inside the worker
with a generic error. This module answers the question during discovery by parsing
the bundled modules with :mod:`ast`; no checkpoint code is imported or executed and
no installed module is imported either (Transformers sources are located and parsed
from disk, other packages are only located with ``find_spec``).
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

_MAX_SOURCE_BYTES: Final[int] = 4 * 1024**2
_MAX_MODULES: Final[int] = 256


@dataclass(slots=True)
class BundledImportReport:
    missing_files: list[str] = field(default_factory=list)
    missing_modules: list[str] = field(default_factory=list)
    missing_names: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.missing_files or self.missing_modules or self.missing_names)

    def describe(self) -> str:
        parts = [f"bundled module {name} is missing" for name in self.missing_files]
        parts += [
            f"{module} does not define {', '.join(names)}"
            for module, names in self.missing_names.items()
        ]
        parts += [f"module {name} is not installed" for name in self.missing_modules]
        return "; ".join(parts)


def _auto_map_modules(auto_map: Mapping[str, Any]) -> set[str]:
    modules: set[str] = set()
    for value in auto_map.values():
        for reference in value if isinstance(value, list) else [value]:
            if isinstance(reference, str) and "." in reference:
                # `repo--module.Class` points at another repository; only the module
                # part can exist locally, and it is reported missing if it does not.
                modules.add(reference.rsplit("--", 1)[-1].rsplit(".", 1)[0])
    return modules


def _parse(path: Path) -> ast.Module | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        if path.stat().st_size > _MAX_SOURCE_BYTES:
            return None
        return ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return None


def _module_file(base: Path, dotted: str) -> Path | None:
    candidate = base.joinpath(*dotted.split(".")) if dotted else base
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path
    return None


def _defined_names(tree: ast.Module) -> tuple[set[str], bool]:
    """Collect names bound at module level, including inside ``if``/``try`` blocks."""

    names: set[str] = set()
    star = False

    def visit(statements: Iterable[ast.stmt]) -> None:
        nonlocal star
        for node in statements:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == "*":
                        star = True
                    else:
                        names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.update(item.id for item in ast.walk(target) if isinstance(item, ast.Name))
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)

    visit(tree.body)
    return names, star


def _transformers_root() -> Path | None:
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def check_bundled_imports(directory: Path, auto_map: Mapping[str, Any]) -> BundledImportReport:
    """Check that every unconditional import reachable from ``auto_map`` resolves.

    Only module-level import statements are followed. Imports guarded by ``try`` or
    ``if`` (optional accelerators such as flash-attn) and imports inside functions
    are skipped because they do not fail at import time. A Transformers module that
    re-exports names through ``import *`` cannot be proven incomplete statically, so
    its names are accepted rather than guessed missing.
    """

    report = BundledImportReport()
    transformers_root = _transformers_root()
    parsed: dict[str, tuple[set[str], bool] | None] = {}
    pending = sorted(_auto_map_modules(auto_map))
    seen: set[str] = set()
    while pending and len(seen) < _MAX_MODULES:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        # auto_map strings are untrusted; only dotted identifiers may name a file,
        # which also keeps the lookup inside the checkpoint directory.
        valid = all(part.isidentifier() for part in module.split("."))
        path = _module_file(directory, module) if valid else None
        tree = _parse(path) if path is not None else None
        if tree is None:
            report.missing_files.append(f"{module.replace('.', '/')}.py" if valid else module)
            continue
        package = module.rsplit(".", 1)[0] if "." in module else ""
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level:
                base = package.split(".") if package else []
                if node.level > 1:
                    base = base[: len(base) - (node.level - 1)]
                prefix = ".".join([*base, node.module] if node.module else base)
                if node.module:
                    pending.append(prefix)
                else:
                    pending.extend(
                        f"{prefix}.{alias.name}" if prefix else alias.name for alias in node.names
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                _check_absolute(
                    report, node.module, [a.name for a in node.names], transformers_root, parsed
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    _check_absolute(report, alias.name, [], transformers_root, parsed)
    report.missing_modules.sort()
    return report


def _check_absolute(
    report: BundledImportReport,
    module: str,
    names: list[str],
    transformers_root: Path | None,
    parsed: dict[str, tuple[set[str], bool] | None],
) -> None:
    top = module.split(".")[0]
    if top != "transformers" or transformers_root is None:
        if importlib.util.find_spec(top) is None and top not in report.missing_modules:
            report.missing_modules.append(top)
        return
    relative = module.removeprefix("transformers").lstrip(".")
    if relative not in parsed:
        source = _module_file(transformers_root, relative)
        tree = _parse(source) if source is not None else None
        parsed[relative] = _defined_names(tree) if tree is not None else None
    defined = parsed[relative]
    if defined is None:
        if module not in report.missing_modules:
            report.missing_modules.append(module)
        return
    known, star = defined
    if star:
        return
    # A package also exposes its submodules (`from transformers.models import llama`).
    package_dir = transformers_root.joinpath(*relative.split("."))
    missing = [
        name for name in names if name not in known and _module_file(package_dir, name) is None
    ]
    if missing:
        existing = report.missing_names.setdefault(module, [])
        existing.extend(name for name in missing if name not in existing)
