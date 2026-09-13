"""Core package for the Local AI Doctor workbench.

Importing this package is deliberately side-effect free: it does not inspect
hardware, read configuration, open databases, or import heavyweight ML
runtimes. Applications should compose those services explicitly.
"""

from importlib.metadata import PackageNotFoundError, version

from .errors import ErrorCode, WorkbenchError

__all__ = ["ErrorCode", "WorkbenchError"]

try:
    __version__ = version("local-ai-doctor")
except PackageNotFoundError:
    __version__ = "0+unknown"
