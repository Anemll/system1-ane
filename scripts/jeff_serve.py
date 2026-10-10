#!/usr/bin/env python3
"""Deprecated path: the server is now scripts/system1_ane_serve.py. This shim keeps ``import jeff_serve`` and
``python scripts/jeff_serve.py`` working."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import system1_ane_serve as _server  # noqa: E402

if __name__ == "__main__":
    print("note: scripts/jeff_serve.py is deprecated; use scripts/system1_ane_serve.py", file=sys.stderr, flush=True)
    sys.exit(_server.main())
else:
    sys.modules[__name__] = _server
