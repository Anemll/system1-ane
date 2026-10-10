#!/usr/bin/env python3
"""System 1 ANE decision server (Jeff on the Apple Neural Engine) for a compiled Core AI package.

    python forge.py system1-ane-serve --model ~/Models/jeff/jeff-base-v1.3 \\
        --build ~/Models/jeff-coreai/coreai

Loads the ANE packages once, then answers POST /v1/systemone the way upstream
Jeff's jeff-serve does: state plus questions in, one answer per question out. GET / is a browser
demo (snake, and a routing panel). GET /health is the readiness check.

``--adapter name=path`` loads another compiled build beside the base. The request
field ``adapter`` selects it (``base`` is the ``--build`` package). A model name
that matches a loaded adapter selects it too. Each adapter is its own ANE build
with the LoRA already merged into the weights.

A prefix cache plugs in without changing the route. Set ``app.prefix_cache`` to
an object with ``lookup(token_ids) -> snapshot | None`` and
``store(token_ids, snapshot)``. ``lookup``'s snapshot is passed to
``JeffCoreAI.decide(..., prefix=snapshot)`` (see ``coreai/jeff_prefix.py``).
With no cache, every request resets and prefills from token 0.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import load_decision_config, question_options  # noqa: E402
from system1_env import server_env  # noqa: E402

LEGACY_MODEL = "jeff-qwen3.8-27b"
ALIASES = {"jeff", "jeff-latest", LEGACY_MODEL}
# Adapter names are a path segment, not "base" (that name is the --build package).
ADAPTER_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
DEMO_PATHS = {"/", "/demo", "/demo/"}
# Text decisions only. Large enough for a long state, small enough to refuse an image upload early.
MAX_BODY = 2_000_000


class DecisionError(Exception):
    def __init__(self, status: int, detail, headers: dict | None = None):
        super().__init__(detail if isinstance(detail, str) else json.dumps(detail))
        self.status, self.detail, self.headers = status, detail, headers or {}


def format_answer(question: dict, probabilities: list[float]) -> dict:
    """Jeff's answer() (firelex/jeff model.py): choice, noul, or score from option probabilities."""
    keys, descriptions = question_options(question)
    values = [float(value) for value in probabilities]
    if len(values) != len(keys) or not values:
        raise DecisionError(422, "Each option must have a probability.")
    if any(value != value or value < 0 for value in values) or sum(values) <= 0:  # NaN != NaN
        raise DecisionError(422, "Probabilities must be finite, nonnegative, and have positive mass.")
    total = sum(values)
    values = [value / total for value in values]
    if question["type"] == "noul":
        return {"type": "noul", "noul": values[keys.index("true")]}
    best = max(range(len(values)), key=values.__getitem__)
    distribution = dict(zip(keys, values))
    if question["type"] == "choice":
        confidence = 1.0 if len(values) == 1 else (values[best] - 1 / len(values)) / (1 - 1 / len(values))
        return {"type": "choice", "probabilities": distribution, "choice": keys[best],
                "confidence": max(0.0, min(1.0, confidence))}
    if len(values) < 2:
        raise DecisionError(422, "Score questions require at least two levels.")
    distance = sum(probability * abs(i - best) for i, probability in enumerate(values))
    midpoint = (len(values) - 1) / 2
    baseline = sum(abs(i - midpoint) for i in range(len(values))) / len(values)
    return {"type": "score", "probabilities": distribution, "legend": dict(zip(keys, descriptions)),
            "score": sum(i * probability for i, probability in enumerate(values)),
            "confidence": max(0.0, 1.0 - distance / baseline)}


def reverse_question(question: dict) -> dict:
    """The same question with its options listed the other way (jeff.orders.reverse_question)."""
    kind = question.get("type")
    if kind == "choice":
        return {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
    if kind == "score":
        return {**question, "criteria": list(question["criteria"])[::-1]}
    if kind == "noul":
        return {**question, "true_first": not question.get("true_first", False)}
    raise DecisionError(422, f"Unknown question type {kind!r}.")


def restore_order(question: dict, values: list[float]) -> list[float]:
    """Probabilities scored on the reversed question, put back under the original options."""
    if question["type"] == "score":
        if len(values) != len(question["criteria"]):
            raise DecisionError(422, f"Expected {len(question['criteria'])} score levels, got {len(values)}.")
        return list(values)[::-1]
    reversed_q = reverse_question(question)
    if question["type"] == "choice":
        keys, reversed_keys = list(question["criteria"]), list(reversed_q["criteria"])
    else:
        keys = ["true", "false"] if question.get("true_first") else ["false", "true"]
        reversed_keys = keys[::-1]
    if len(values) != len(keys):
        raise DecisionError(422, f"Expected {len(keys)} probabilities, got {len(values)}.")
    by_key = dict(zip(reversed_keys, values))
    return [by_key[key] for key in keys]


def _is_json(value) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json(item) for key, item in value.items())
    return False


def _check_question(key: str, question, max_options: int, n_codes: int) -> dict:
    if not isinstance(question, dict):
        raise DecisionError(422, f"Question {key!r} must be an object.")
    kind = question.get("type")
    if kind not in ("choice", "noul", "score"):
        raise DecisionError(422, f"Question {key!r} type must be choice, noul, or score.")
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise DecisionError(422, f"Question {key!r} needs a criteria object with at least one option.")
        if any(not isinstance(name, str) or not name for name in criteria):
            raise DecisionError(422, f"Question {key!r} option keys must be non-empty strings.")
        if len(criteria) > max_options:
            raise DecisionError(422, f"Question {key!r} has {len(criteria)} options, but this model handles at most "
                                      f"{max_options}. Shortlist the options first, or split the question.")
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise DecisionError(422, f"Question {key!r} needs a list of 2 to 10 score levels, lowest first.")
    else:
        criteria = question.get("criteria")
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
            raise DecisionError(422, f"Question {key!r} noul criteria may only use the keys \"true\" and \"false\".")
    keys, _ = question_options(question)
    if not 1 <= len(keys) <= min(255, n_codes):
        raise DecisionError(422, "Questions must have 1 to 255 options, each with an answer code.")
    return question


def _select_adapter(model: str, requested, *, name: str, aliases: set[str], adapters: dict) -> str:
    """``adapter`` wins when it is set. Otherwise a model name that is a loaded adapter selects that build."""
    loaded = set(adapters) | {"base"}
    names = loaded - {"base"}
    if requested is not None:
        if not isinstance(requested, str) or requested not in loaded:
            shown = ", ".join(sorted(loaded))
            raise DecisionError(422, f"Unknown adapter {requested!r}. Loaded: {shown}.")
        if model in names and model != requested:
            raise DecisionError(422, f"model {model!r} and adapter {requested!r} select different builds.")
        if model not in aliases and model not in names:
            raise DecisionError(422, [{"loc": ["body", "model"], "msg": f"Unknown model. Use {name} or jeff-latest.",
                                       "type": "value_error"}])
        return requested
    if model in names:
        return model
    if model not in aliases:
        raise DecisionError(422, [{"loc": ["body", "model"], "msg": f"Unknown model. Use {name} or jeff-latest.",
                                   "type": "value_error"}])
    return "base"


def normalize(body: dict, *, name: str, aliases: set[str], max_options: int, n_codes: int,
              adapters: dict | None = None) -> tuple[object, dict, int, str, str]:
    """Return (state, questions, orders, response model, adapter) from a Jeff request or the short options form."""
    if not isinstance(body, dict):
        raise DecisionError(422, "The request body must be a JSON object.")
    adapters = adapters or {"base": None}
    model = body.get("model", name)
    if not isinstance(model, str):
        raise DecisionError(422, [{"loc": ["body", "model"], "msg": f"Unknown model. Use {name} or jeff-latest.",
                                   "type": "value_error"}])
    adapter = _select_adapter(model, body.get("adapter", None), name=name, aliases=aliases, adapters=adapters)
    if body.get("images"):
        raise DecisionError(422, "Jeff Core AI is text only; this server does not accept images.")
    orders = body.get("orders", 1)
    if isinstance(orders, bool) or orders not in (1, 2):
        raise DecisionError(422, "orders must be 1 (the given option order) or 2 (also reversed, then averaged).")
    state = body.get("state")
    if "state" not in body or state is None or not _is_json(state):
        raise DecisionError(422, "state must be text, a JSON object, or a list.")
    if "questions" in body and "options" in body:
        raise DecisionError(422, "Send questions or options, not both.")
    if "questions" in body:
        raw = body["questions"]
        if not isinstance(raw, dict) or not raw:
            raise DecisionError(422, "questions must be an object with at least one question.")
        if any(not isinstance(key, str) or not key for key in raw):
            raise DecisionError(422, "Question names must be non-empty strings.")
        questions = {key: _check_question(key, question, max_options, n_codes) for key, question in raw.items()}
    elif "options" in body:
        options = body["options"]
        instructions = body.get("instructions")
        if isinstance(options, list):
            if not options or not all(isinstance(item, str) and item for item in options):
                raise DecisionError(422, "options must be a non-empty list of strings or an object of key to description.")
            criteria = {f"o{index + 1}": item for index, item in enumerate(options)}
        elif isinstance(options, dict) and options:
            criteria = options
        else:
            raise DecisionError(422, "options must be a non-empty list of strings or an object of key to description.")
        question = {"type": "choice", "criteria": criteria}
        if instructions is not None:
            question["instructions"] = instructions
        questions = {"decision": _check_question("decision", question, max_options, n_codes)}
    else:
        raise DecisionError(422, "The request needs questions, or a short options list.")
    if isinstance(state, dict) and not state:
        raise DecisionError(422, "The live-last layout needs an object state to have at least one field.")
    # Aliases answer as this checkpoint. An adapter name answers as that build.
    response_model = name if adapter == "base" else adapter
    return state, questions, int(orders), response_model, adapter


class App:
    """One loaded decision backend, one request at a time.

    ``prefix_cache`` is optional. When it is set, each question calls
    ``lookup(token_ids)`` and passes the snapshot to ``engine.decide(..., prefix=)``,
    then ``store(token_ids, engine.capture_state())``. ``None`` (the default)
    does not snapshot: every decision starts from an empty GDN and KV state.
    """

    def __init__(self, engine, encode, *, name: str, checkpoint: str, max_options: int, n_codes: int,
                 max_tokens: int, release_date: str, backend: str, layout: str, demo_html: str,
                 queue_seconds: float = 0, api_key: str | None = None, prefix_cache=None, adapters=None):
        self.engine, self.encode = engine, encode
        self.engines = {"base": engine}
        for adapter_name, adapter_engine in (adapters or {}).items():
            if adapter_name == "base" or not ADAPTER_NAME.fullmatch(adapter_name):
                raise ValueError(f"adapter name {adapter_name!r} must match {ADAPTER_NAME.pattern} and must not be 'base'")
            self.engines[adapter_name] = adapter_engine
        self.name, self.checkpoint = name, checkpoint
        self.max_options, self.n_codes, self.max_tokens = max_options, n_codes, max_tokens
        self.release_date, self.backend, self.layout = release_date, backend, layout
        self.demo_html = demo_html
        self.queue_seconds = queue_seconds
        self.api_key = api_key
        self.prefix_cache = prefix_cache
        self.lock = threading.Lock()
        self.aliases = set(ALIASES) | {name}

    def health(self) -> dict:
        return {
            "status": "ready",
            "model": self.name,
            "checkpoint": self.checkpoint,
            "max_options": self.max_options,
            "max_tokens": self.max_tokens,
            "authentication": bool(self.api_key),
            "modalities": ["text"],
            "backend": self.backend,
            "prompt_layout": self.layout,
            "prefix_cache": self.prefix_cache is not None,
            "adapters": sorted(self.engines),
            "prefill": {name: engine.prefill_info() for name, engine in sorted(self.engines.items())
                        if hasattr(engine, "prefill_info")},
        }

    def models(self) -> dict:
        names = sorted((ALIASES - {LEGACY_MODEL}) | {self.name})
        items = [{"name": item, "description": "Local Jeff text decisions on the ANE.",
                  "release_date": self.release_date} for item in names]
        for adapter_name in sorted(self.engines):
            if adapter_name == "base":
                continue
            items.append({"name": adapter_name,
                          "description": "LoRA adapter merged into its own Core AI build.",
                          "release_date": self.release_date})
        return {"models": items}

    def _authorized(self, header: str | None) -> bool:
        if not self.api_key:
            return True
        offered = header or ""
        return hmac.compare_digest(offered.encode(), f"Bearer {self.api_key}".encode())

    def _one(self, state, question: dict, adapter: str) -> tuple[list[float], dict]:
        row = {"state": state, "question": question}
        engine = self.engines[adapter]
        if hasattr(engine, "decide_row"):  # Clef-head build: the engine builds its own prompt
            t0 = time.perf_counter()
            try:
                raw = engine.decide_row(state, question)
            except ValueError as error:
                raise DecisionError(422, str(error)) from error
            calls = [float(value) for value in raw["calls_ms"]]
            return [float(v) for v in raw["option_probabilities"]], {
                "tokenize_ms": 0.0, "calls_ms": [round(v, 2) for v in calls],
                "call_widths": [int(v) for v in raw.get("call_widths", [])], "prefill_ms": round(sum(calls), 2),
                "head_ms": round(float(raw["head_ms"]), 2), "tokens": int(raw["tokens"]), "prefix_tokens": 0}
        t0 = time.perf_counter()
        ids = self.encode(row)
        tokenize_ms = 1e3 * (time.perf_counter() - t0)
        if len(ids) > self.max_tokens:
            raise DecisionError(422, f"This question is {len(ids)} tokens; this build accepts at most {self.max_tokens}.")
        engine = self.engines[adapter]
        # The prefix cache stores one backbone snapshot. It is only applied to the base build so a snake
        # snapshot cannot be restored into the base packages, or the other way around.
        prefix = None
        use_cache = self.prefix_cache is not None and adapter == "base"
        if use_cache:
            prefix = self.prefix_cache.lookup(ids)
        try:
            raw = engine.decide(ids, len(question_options(question)[0]), prefix=prefix)
        except ValueError as error:
            raise DecisionError(422, str(error)) from error
        if use_cache:
            self.prefix_cache.store(ids, engine.capture_state())
        probs = [float(value) for value in raw["option_probabilities"]]
        calls = [float(value) for value in raw["calls_ms"]]
        timing = {
            "tokenize_ms": round(tokenize_ms, 2),
            "calls_ms": [round(value, 2) for value in calls],
            "call_widths": [int(value) for value in raw.get("call_widths", [])],
            "prefill_ms": round(sum(calls), 2),
            "head_ms": round(float(raw["head_ms"]), 2),
            "tokens": len(ids),
            "prefix_tokens": int(raw.get("prefix_tokens", 0)),
        }
        return probs, timing

    def evaluate(self, body: dict) -> dict:
        if not self.lock.acquire(timeout=self.queue_seconds):
            raise DecisionError(529, "The model is busy. Retry shortly.", {"Retry-After": "1"})
        try:
            return self._evaluate(body)
        finally:
            self.lock.release()

    def _evaluate(self, body: dict) -> dict:
        started = time.perf_counter()
        state, questions, orders, model, adapter = normalize(
            body, name=self.name, aliases=self.aliases, max_options=self.max_options, n_codes=self.n_codes,
            adapters=self.engines)
        answers = {}
        parts = []
        input_tokens = 0
        for key, question in questions.items():
            probs, timing = self._one(state, question, adapter)
            if orders == 2:
                reversed_probs, reversed_timing = self._one(state, reverse_question(question), adapter)
                restored = restore_order(question, reversed_probs)
                probs = [(a + b) / 2 for a, b in zip(probs, restored)]
                timing = _merge_timing(timing, reversed_timing)
            answers[key] = format_answer(question, probs)
            parts.append({"id": key, **timing})
            input_tokens += timing["tokens"]
        calls = [value for part in parts for value in part["calls_ms"]]
        return {
            "model": model,
            "adapter": adapter,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0, "orders": orders},
            "timings": {
                "tokenize_ms": round(sum(part["tokenize_ms"] for part in parts), 2),
                "calls_ms": calls,
                "call_widths": [value for part in parts for value in part["call_widths"]],
                "prefill_ms": round(sum(part["prefill_ms"] for part in parts), 2),
                "head_ms": round(sum(part["head_ms"] for part in parts), 2),
                "total_ms": round(1e3 * (time.perf_counter() - started), 2),
                "questions": parts,
            },
        }


def _merge_timing(first: dict, second: dict) -> dict:
    return {
        "tokenize_ms": round(first["tokenize_ms"] + second["tokenize_ms"], 2),
        "calls_ms": first["calls_ms"] + second["calls_ms"],
        "call_widths": first["call_widths"] + second["call_widths"],
        "prefill_ms": round(first["prefill_ms"] + second["prefill_ms"], 2),
        "head_ms": round(first["head_ms"] + second["head_ms"], 2),
        "tokens": first["tokens"] + second["tokens"],
        "prefix_tokens": first["prefix_tokens"] + second["prefix_tokens"],
    }


class TokenizerWorker:
    """Jeff chat-template token ids from a forge-Python subprocess."""

    def __init__(self, python: str, model: Path):
        script = Path(__file__).resolve().parent / "jeff_tokenize.py"
        self.proc = subprocess.Popen(
            [python, str(script), str(model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Tokenizer worker ({python}) exited before it was ready. "
                               "TOKENIZER_PYTHON must be an interpreter with transformers.")
        ready = json.loads(line)
        if not ready.get("ready"):
            raise RuntimeError(f"Tokenizer worker failed: {ready}")

    def encode(self, row: dict) -> list[int]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        if self.proc.poll() is not None:
            raise DecisionError(503, "The tokenizer stopped.")
        self.proc.stdin.write(json.dumps({"row": row}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise DecisionError(503, "The tokenizer stopped.")
        msg = json.loads(line)
        if "error" in msg:
            raise DecisionError(422, str(msg["error"]))
        return [int(token) for token in msg["ids"]]

    def encode_clef(self, state, instructions: str, options: dict) -> dict:
        """Unsloth Clef prompt ids + spans (for builds that ship clef_head/)."""
        assert self.proc.stdin is not None and self.proc.stdout is not None
        if self.proc.poll() is not None:
            raise DecisionError(503, "The tokenizer stopped.")
        self.proc.stdin.write(json.dumps({"clef": {"state": state, "instructions": instructions, "options": options}}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise DecisionError(503, "The tokenizer stopped.")
        msg = json.loads(line)
        if "error" in msg:
            raise DecisionError(422, str(msg["error"]))
        return msg["clef"]

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()


class CoreAIEngine:
    """JeffCoreAI.decide, with probabilities in option order and an optional prefix snapshot."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.backend = f"coreai-ane {runtime.entry}"

    def decide(self, token_ids: list[int], n_options: int, prefix=None) -> dict:
        raw = self.runtime.decide(token_ids, n_options, prefix=prefix)
        probs = [float(value) for value in raw["probabilities"].values()]
        if len(probs) != n_options:
            raise RuntimeError(f"runtime returned {len(probs)} probabilities for {n_options} options")
        calls = [float(value) for value in raw["calls_ms"]]
        return {"option_probabilities": probs, "calls_ms": calls, "head_ms": float(raw["head_ms"]),
                "call_widths": [int(value) for value in raw.get("call_widths", [])],
                "prefix_tokens": int(raw.get("prefix_tokens", 0))}

    def capture_state(self):
        return self.runtime.capture_state()

    def prefill_info(self) -> dict:
        """Compiled prefill widths and how a prompt is split over them (SYSTEM1_ANE_PREFILL)."""
        return {"widths": list(self.runtime.widths), "mode": self.runtime.mode}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s\n" % (fmt % args))

        def do_GET(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

        def _handle(self) -> None:
            started, identifier = time.perf_counter(), uuid.uuid4().hex
            path = urlsplit(self.path).path or "/"
            protected = path in ("/v1/systemone", "/v1/decide", "/v1/models")
            try:
                if protected and not app._authorized(self.headers.get("Authorization")):
                    raise DecisionError(401, "Missing or invalid API key.", {"WWW-Authenticate": "Bearer"})
                status, payload, extra = self._dispatch(path)
            except DecisionError as error:
                status, payload, extra = error.status, {"detail": error.detail}, dict(error.headers)
            except Exception as error:
                sys.stderr.write(f"system1-ane-serve failed: {error}\n")
                status, payload, extra = 500, {"detail": "The decision failed."}, {}
            body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
            content_type = "text/html; charset=utf-8" if isinstance(payload, bytes) else "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-request-id", identifier)
            self.send_header("server-timing", f"total;dur={(time.perf_counter() - started) * 1000:.1f}")
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, path: str):
            if self.command == "GET" and path in DEMO_PATHS:
                # Read the file each request so a browser reload picks up a demo edit
                # without restarting the ANE process.
                try:
                    page = load_demo().encode()
                except OSError:
                    page = app.demo_html.encode()
                return 200, page, {}
            if self.command == "GET" and path == "/health":
                return 200, app.health(), {}
            if self.command == "GET" and path == "/v1/models":
                return 200, app.models(), {}
            if self.command == "POST" and path in ("/v1/systemone", "/v1/decide"):
                return 200, app.evaluate(self._json()), {}
            raise DecisionError(404, "Not found.")

        def _json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length < 0 or length > MAX_BODY:
                raise DecisionError(413, "Request body is too large for the text-only ANE server.")
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode() or "null")
            except (UnicodeError, json.JSONDecodeError) as error:
                raise DecisionError(422, "The request body must be JSON.") from error
            return body

    return Handler


def serve(app: App, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"system1-ane-serve {app.backend} on http://{host}:{port}/  (POST /v1/systemone, GET /health)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _model_name(decision: dict) -> str:
    base = str(decision.get("base_model") or "jeff")
    return f"jeff-{base.rsplit('/', 1)[-1].lower()}"


def _queue_seconds() -> float:
    raw = server_env("QUEUE_MS")
    if raw is None:
        return 0
    if not raw.isdigit():
        raise ValueError(f"SYSTEM1_ANE_QUEUE_MS={raw!r}; use a whole number of milliseconds, 0 or more")
    return int(raw) / 1000


def load_demo() -> str:
    path = Path(__file__).resolve().parent / "jeff_demo.html"
    return path.read_text(encoding="utf-8")


def open_app(model: Path, build: Path, adapters: list[tuple[str, Path]] | None = None) -> tuple[App, TokenizerWorker]:
    """Load the compiled package and the tokenizer worker. Core AI import stays here so tests can import the module."""
    from jeff_coreai_runtime import JeffCoreAI  # Core AI SDK Python only

    decision = load_decision_config(model)
    python = os.environ.get("TOKENIZER_PYTHON") or sys.executable
    worker = TokenizerWorker(python, model)
    try:
        runtime = JeffCoreAI(build, model)
        extra = {}
        for adapter_name, adapter_build in adapters or []:
            print(f"loading adapter {adapter_name} from {adapter_build}", flush=True)
            # A Clef-head build carries its own backbone model dir (embedding + final norm of its base) next to coreai/.
            own = adapter_build.parent / "model"
            clef = (adapter_build / "clef_head" / "joint_head.safetensors").is_file()
            runtime_a = JeffCoreAI(adapter_build, own if clef and (own / "model.safetensors").is_file() else model)
            if clef:
                from clef_engine import ClefEngine  # torch is only needed for Clef-head builds; Jeff adapters never import it
                extra[adapter_name] = ClefEngine(runtime_a, adapter_build, worker.encode_clef)
                print(f"adapter {adapter_name}: Unsloth Clef head ({extra[adapter_name].head_desc}) on the ANE backbone", flush=True)
            else:
                extra[adapter_name] = CoreAIEngine(runtime_a)
    except Exception:
        worker.close()
        raise
    engine = CoreAIEngine(runtime)
    raw_limit = server_env("MAX_TOKENS")
    if raw_limit is None:
        limit = runtime.L
    elif not raw_limit.isdigit() or int(raw_limit) == 0:
        raise ValueError(f"SYSTEM1_ANE_MAX_TOKENS={raw_limit!r}; use a whole number of tokens above 0")
    else:
        limit = min(int(raw_limit), runtime.L)
    config_path = model / "decision_config.json"
    released = time.strftime("%Y-%m-%d", time.gmtime(config_path.stat().st_mtime))
    app = App(engine, worker.encode, name=_model_name(decision), checkpoint=str(model),
              max_options=int(decision.get("max_options") or 254), n_codes=len(decision["codes"]),
              max_tokens=limit, release_date=released, backend=engine.backend,
              layout=str(decision.get("prompt_layout") or "state-first"), demo_html=load_demo(),
              queue_seconds=_queue_seconds(), api_key=server_env("API_KEY") or None,
              adapters=extra)
    return app, worker


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--host", default=server_env("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8787")))
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                   help="merged Core AI build, e.g. snake=/path/to/coreai. Repeatable. base is --build")
    a = p.parse_args(argv)
    model, build = a.model.expanduser().resolve(), a.build.expanduser().resolve()
    if not (build / "manifest.json").is_file():
        p.error(f"Missing build manifest: {build / 'manifest.json'}")
    adapters = []
    seen = set()
    for item in a.adapter:
        if item.count("=") != 1:
            p.error("--adapter must be name=path")
        adapter_name, raw_path = item.split("=", 1)
        if adapter_name == "base" or not ADAPTER_NAME.fullmatch(adapter_name):
            p.error(f"adapter name {adapter_name!r} must match {ADAPTER_NAME.pattern} and must not be 'base'")
        if adapter_name in seen:
            p.error(f"adapter {adapter_name!r} was given twice")
        seen.add(adapter_name)
        adapter_build = Path(raw_path).expanduser().resolve()
        if not (adapter_build / "manifest.json").is_file():
            p.error(f"Missing build manifest: {adapter_build / 'manifest.json'}")
        adapters.append((adapter_name, adapter_build))
    app, worker = open_app(model, build, adapters)
    try:
        serve(app, a.host, a.port)
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
