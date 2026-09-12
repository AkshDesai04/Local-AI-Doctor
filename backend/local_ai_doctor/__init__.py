"""Core package for the Local AI Doctor workbench.

Importing this package is deliberately side-effect free: it does not inspect
hardware, read configuration, open databases, or import heavyweight ML
runtimes. Applications should compose those services explicitly.
"""

from .errors import ErrorCode, WorkbenchError

__all__ = ["ErrorCode", "WorkbenchError"]
__version__ = "0.1.0"
