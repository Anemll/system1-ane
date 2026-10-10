"""Snapshot contract for a Jeff live-last prefix cache.

``JeffCoreAI.capture_state()`` returns a dict. A cache may keep it and pass it
back as ``prefix`` to ``JeffCoreAI.prefill`` / ``decide`` (and, through the
server, to ``App``'s prefix cache). This module checks that dict. It does not
choose which prefix to keep: that policy belongs to the prefix-cache work.

A snapshot is a dict:

- ``pos``: tokens already resident in KV and GDN/conv state
- ``chunks``: one entry per backbone chunk, each ``{"state": {...}, "kv": {...}}``
  of numpy arrays. The runtime checks names against the loaded build
- ``token_ids``: optional. When present, must equal the prompt's first ``pos`` tokens
- ``hidden``: last-row hidden state. Required when ``pos`` equals the prompt length
  (readout only, no backbone call)
"""
from __future__ import annotations


def resume_at(prefix, token_ids: list[int]) -> int:
    """How many leading tokens ``prefix`` already covers. ``None`` covers none."""
    if prefix is None:
        return 0
    if not isinstance(prefix, dict) or "pos" not in prefix:
        raise ValueError("prefix must be a capture_state() dict or None")
    pos = prefix["pos"]
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise ValueError(f"prefix pos must be an int, got {pos!r}")
    n = len(token_ids)
    if not 0 <= pos <= n:
        raise ValueError(f"prefix pos {pos} is outside the {n}-token prompt")
    cached = prefix.get("token_ids")
    if cached is not None and [int(t) for t in cached] != [int(t) for t in token_ids[:pos]]:
        raise ValueError("cached prefix tokens do not match this prompt")
    if pos == n and prefix.get("hidden") is None:
        raise ValueError("a prefix that covers the whole prompt needs its hidden state")
    return pos
