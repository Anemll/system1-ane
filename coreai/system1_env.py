"""Environment variables of the System 1 ANE server: ``SYSTEM1_ANE_<NAME>`` with ``JEFF_<NAME>`` as a deprecated fallback."""
from __future__ import annotations

import os
import sys

_warned: set[str] = set()


def server_env(name: str, default: str | None = None) -> str | None:
    """Value of SYSTEM1_ANE_<name>; else JEFF_<name> (deprecated, one-line notice on stderr); else default."""
    new, old = f"SYSTEM1_ANE_{name}", f"JEFF_{name}"
    value = os.environ.get(new)
    if value is not None:
        return value
    value = os.environ.get(old)
    if value is not None:
        if old not in _warned:
            _warned.add(old)
            print(f"note: {old} is deprecated; use {new}", file=sys.stderr, flush=True)
        return value
    return default


def server_env_name(name: str) -> str:
    """Name shown in error messages."""
    return f"SYSTEM1_ANE_{name}"
