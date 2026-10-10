"""Jeff decision server: request and response shape, without loading the ANE."""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

import system1_ane_serve
from jeff_prefix import resume_at

JEFF_SRC = Path(os.environ.get("JEFF_SRC", str(Path.home() / "Models" / "jeff" / "jeff-src" / "src"))).expanduser()


class FakeEngine:
    def __init__(self):
        self.seen = []
        self.captures = 0

    def decide(self, token_ids, n_options, prefix=None):
        self.seen.append({"ids": list(token_ids), "n": n_options, "prefix": prefix})
        if n_options == 1:
            probs = [1.0]
        else:
            rest = 0.3 / (n_options - 1)
            probs = [0.7] + [rest] * (n_options - 1)
        return {"option_probabilities": probs, "calls_ms": [3.25], "call_widths": [256], "head_ms": 0.34,
                "prefix_tokens": 0 if prefix is None else int(prefix["pos"])}

    def capture_state(self):
        self.captures += 1
        last = self.seen[-1]
        return {"pos": len(last["ids"]), "token_ids": last["ids"], "hidden": [1.0], "chunks": []}


class RecordingCache:
    def __init__(self, hit):
        self.hit, self.stored = hit, []

    def lookup(self, token_ids):
        return self.hit

    def store(self, token_ids, snapshot):
        self.stored.append((list(token_ids), snapshot))


def encode(row):
    return [11, 22, len(json.dumps(row, sort_keys=True)) % 50]


def make_app(**overrides):
    engine = overrides.pop("engine", None) or FakeEngine()
    cache = overrides.pop("prefix_cache", None)
    adapters = overrides.pop("adapters", None)
    app = system1_ane_serve.App(
        engine, overrides.pop("encode", encode), name="jeff-qwen3.5-0.8b", checkpoint="/models/jeff-base-v1.3",
        max_options=overrides.pop("max_options", 254), n_codes=255, max_tokens=overrides.pop("max_tokens", 2048),
        release_date="2026-10-07", backend="coreai-ane p256_2k", layout="live-last",
        demo_html="<html>Snake /v1/systemone</html>", queue_seconds=overrides.pop("queue_seconds", 0),
        api_key=overrides.pop("api_key", None), prefix_cache=cache, adapters=adapters)
    app.engine = engine
    return app


class PrefixContractTests(unittest.TestCase):
    def test_resume_at(self):
        ids = [1, 2, 3, 4]
        self.assertEqual(resume_at(None, ids), 0)
        self.assertEqual(resume_at({"pos": 2, "token_ids": [1, 2]}, ids), 2)
        self.assertEqual(resume_at({"pos": 4, "hidden": [0.0], "token_ids": ids}, ids), 4)
        with self.assertRaisesRegex(ValueError, "do not match"):
            resume_at({"pos": 2, "token_ids": [9, 9]}, ids)
        with self.assertRaisesRegex(ValueError, "hidden"):
            resume_at({"pos": 4, "token_ids": ids}, ids)
        with self.assertRaisesRegex(ValueError, "outside"):
            resume_at({"pos": 5}, ids)


class ServerTests(unittest.TestCase):
    def setUp(self):
        from http.server import ThreadingHTTPServer
        self.app = make_app()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), system1_ane_serve.make_handler(self.app))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    def request(self, method, path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        if data is not None:
            req.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                payload = response.read()
                return response.status, dict(response.headers), payload
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def post(self, body, path="/v1/systemone"):
        status, headers, raw = self.request("POST", path, body)
        return status, headers, json.loads(raw.decode())

    def test_health_and_demo(self):
        status, _, raw = self.request("GET", "/health")
        self.assertEqual(status, 200)
        health = json.loads(raw.decode())
        self.assertTrue({"status", "model", "checkpoint", "max_options", "authentication", "modalities"} <= set(health))
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["modalities"], ["text"])
        self.assertEqual(health["backend"], "coreai-ane p256_2k")
        self.assertFalse(health["prefix_cache"])
        self.assertFalse(health["authentication"])
        status, headers, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"/v1/systemone", page)
        self.assertIn(b"Snake", page)
        demo = (ROOT / "scripts" / "jeff_demo.html").read_text()
        self.assertIn("/v1/systemone", demo)
        self.assertIn("latest", demo)
        self.assertIn('id="adapter"', demo)
        self.assertIn("Food eaten", demo)
        self.assertIn("overflow-y: auto", demo)
        self.assertIn("logStatus", demo)
        self.assertIn('id="status"', demo)
        self.assertLess(demo.index('id="toggle"'), demo.index('id="board"'))
        self.assertLess(demo.index('id="board"'), demo.index('id="status"'))
        self.assertLess(demo.index('id="tetris-toggle"'), demo.index('id="tetris-board"'))
        self.assertLess(demo.index('id="tetris-board"'), demo.index('id="tetris-status"'))
        self.assertIn("lands on row", demo)
        self.assertEqual(health["adapters"], ["base"])
        self.assertIn("x-request-id", {key.lower() for key in headers})

    def test_choice_response_shape_and_timings(self):
        status, _, body = self.post({
            "model": "jeff-latest",
            "state": "The disk on db-02 is 97 percent full.",
            "questions": {"action": {"type": "choice", "instructions": "What next?",
                                     "criteria": {"page": "Page someone.", "wait": None, "ignore": "Do nothing."}}},
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "jeff-qwen3.5-0.8b")
        self.assertEqual(body["adapter"], "base")
        answer = body["answers"]["action"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(answer["choice"], "page")
        self.assertEqual(set(answer["probabilities"]), {"page", "wait", "ignore"})
        self.assertAlmostEqual(answer["probabilities"]["page"], 0.7)
        self.assertAlmostEqual(answer["confidence"], 0.55)
        self.assertEqual(body["usage"], {"input_tokens": 3, "output_tokens": 0, "orders": 1})
        timing = body["timings"]
        self.assertEqual(set(timing), {"tokenize_ms", "calls_ms", "call_widths", "prefill_ms", "head_ms", "total_ms",
                                       "questions"})
        self.assertEqual(timing["calls_ms"], [3.25])
        self.assertEqual(timing["call_widths"], [256])
        self.assertEqual(timing["prefill_ms"], 3.25)
        self.assertEqual(timing["head_ms"], 0.34)
        self.assertGreaterEqual(timing["total_ms"], 0)
        self.assertEqual(self.app.engine.seen[0]["prefix"], None)
        self.assertEqual(self.app.engine.captures, 0)

    def test_short_options_list_and_decide_alias(self):
        status, _, body = self.post(
            {"state": {"ticket": "old", "latest": "refund"}, "options": ["Refunds", "Shipping"],
             "instructions": "Which queue?"},
            path="/v1/decide")
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["decision"]["choice"], "o1")
        self.assertEqual(list(body["answers"]["decision"]["probabilities"]), ["o1", "o2"])

    def test_noul_and_score(self):
        status, _, body = self.post({
            "state": "It is raining.",
            "questions": {
                "wet": {"type": "noul", "instructions": "Is the ground wet?"},
                "mood": {"type": "score", "criteria": ["low", "mid", "high"], "instructions": "How urgent?"},
            },
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["wet"], {"type": "noul", "noul": 0.3})
        mood = body["answers"]["mood"]
        self.assertEqual(mood["type"], "score")
        self.assertAlmostEqual(mood["score"], 0.45)
        self.assertEqual(set(mood["probabilities"]), {"0", "1", "2"})
        self.assertEqual(len(body["timings"]["calls_ms"]), 2)

    def test_orders_two_averages_reversed_positions(self):
        status, _, body = self.post({
            "state": "A parcel.",
            "orders": 2,
            "questions": {"route": {"type": "choice", "criteria": {"refunds": "Money", "shipping": "Parcel"}}},
        })
        self.assertEqual(status, 200)
        probs = body["answers"]["route"]["probabilities"]
        # Positional 0.7/0.3, then the reverse, averaged back onto the original keys: both 0.5.
        self.assertAlmostEqual(probs["refunds"], 0.5)
        self.assertAlmostEqual(probs["shipping"], 0.5)
        self.assertEqual(body["usage"]["orders"], 2)
        self.assertEqual(len(body["timings"]["calls_ms"]), 2)
        self.assertEqual(body["usage"]["input_tokens"], 6)

    def test_rejects_unknown_model_too_many_options_and_images(self):
        status, _, body = self.post({"model": "other", "state": "x", "options": ["a", "b"]})
        self.assertEqual(status, 422)
        self.assertEqual(body["detail"][0]["loc"][-1], "model")
        self.app.max_options = 2
        status, _, body = self.post({"state": "x", "questions": {"q": {"type": "choice", "criteria": {
            "a": None, "b": None, "c": None}}}})
        self.assertEqual(status, 422)
        self.assertIn("options, but this model handles at most", body["detail"])
        status, _, body = self.post({"state": "x", "options": ["a"], "images": ["data:image/png;base64,aaaa"]})
        self.assertEqual(status, 422)
        self.assertIn("text only", body["detail"])
        status, _, body = self.post({"state": {}, "options": ["a", "b"]})
        self.assertEqual(status, 422)
        self.assertIn("at least one field", body["detail"])

    def test_auth_and_busy(self):
        locked = make_app(api_key="secret")
        from http.server import ThreadingHTTPServer
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), system1_ane_serve.make_handler(locked))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        bare = urllib.request.Request(url + "/v1/models")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(bare, timeout=5)
        self.assertEqual(raised.exception.code, 401)
        self.assertEqual(raised.exception.headers["WWW-Authenticate"], "Bearer")
        health = urllib.request.urlopen(url + "/health", timeout=5)
        self.assertEqual(json.loads(health.read().decode())["authentication"], True)
        authed = urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(json.loads(urllib.request.urlopen(authed, timeout=5).read().decode())["models"][0]["name"], "jeff")
        locked.lock.acquire()
        try:
            busy = urllib.request.Request(url + "/v1/systemone", data=json.dumps(
                {"state": "x", "options": ["a", "b"]}).encode(),
                headers={"Authorization": "Bearer secret", "Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(busy, timeout=5)
        finally:
            locked.lock.release()
        self.assertEqual(raised.exception.code, 529)
        self.assertEqual(raised.exception.headers["Retry-After"], "1")

    def test_adapter_selects_its_engine(self):
        snake = FakeEngine()
        cache = RecordingCache({"pos": 1, "token_ids": [11], "hidden": [1.0], "chunks": []})
        app = make_app(adapters={"snake": snake}, prefix_cache=cache)
        body = app.evaluate({"state": "board", "options": ["up", "down"], "adapter": "snake"})
        self.assertEqual(body["adapter"], "snake")
        self.assertEqual(body["model"], "snake")
        self.assertEqual(len(snake.seen), 1)
        self.assertEqual(app.engine.seen, [])
        self.assertIsNone(snake.seen[0]["prefix"])
        self.assertEqual(cache.stored, [])
        by_model = app.evaluate({"model": "snake", "state": "board", "options": ["up", "down"]})
        self.assertEqual(by_model["adapter"], "snake")
        self.assertEqual(len(snake.seen), 2)
        names = {item["name"] for item in app.models()["models"]}
        self.assertIn("snake", names)
        self.assertIn("jeff", names)
        self.assertEqual(app.health()["adapters"], ["base", "snake"])
        missing = app.evaluate
        with self.assertRaises(system1_ane_serve.DecisionError) as raised:
            missing({"state": "board", "options": ["up"], "adapter": "missing"})
        self.assertEqual(raised.exception.status, 422)
        self.assertIn("Unknown adapter", str(raised.exception.detail))
        with self.assertRaises(system1_ane_serve.DecisionError) as raised:
            app.evaluate({"model": "snake", "adapter": "base", "state": "board", "options": ["up"]})
        self.assertEqual(raised.exception.status, 422)

    def test_clef_adapter_enforces_max_tokens_before_the_backbone(self):
        class ClefFake:
            def __init__(self, n_tokens):
                self.n, self.encoded, self.ran = n_tokens, 0, 0

            def encode_row(self, state, question):
                self.encoded += 1
                return {"input_ids": list(range(self.n)), "option_keys": sorted(question["criteria"])}

            def decide_row(self, state, question, enc=None):
                self.ran += 1
                return {"option_probabilities": [0.5, 0.5], "calls_ms": [1.0], "head_ms": 0.1, "call_widths": [256],
                        "tokens": len(enc["input_ids"])}

        body = {"state": "board", "options": ["up", "down"], "adapter": "clef"}
        too_long = ClefFake(11)
        app = make_app(adapters={"clef": too_long}, max_tokens=10)
        with self.assertRaises(system1_ane_serve.DecisionError) as raised:
            app.evaluate(body)
        self.assertEqual(raised.exception.status, 422)
        self.assertEqual(str(raised.exception.detail), "This question is 11 tokens; this build accepts at most 10.")
        self.assertEqual((too_long.encoded, too_long.ran), (1, 0))  # the backbone never ran
        exact = ClefFake(10)
        answer = make_app(adapters={"clef": exact}, max_tokens=10).evaluate(body)
        self.assertEqual(exact.ran, 1)
        self.assertEqual(answer["timings"]["questions"][0]["tokens"], 10)

    def test_clef_adapter_unsupported_question_type_is_422(self):
        class ClefFake:
            def encode_row(self, state, question):
                raise ValueError("This Unsloth Clef adapter answers choice questions only; got type 'noul'.")

            def decide_row(self, *args):
                raise AssertionError("must not run")

        app = make_app(adapters={"clef": ClefFake()})
        with self.assertRaises(system1_ane_serve.DecisionError) as raised:
            app.evaluate({"state": "board", "adapter": "clef", "questions": {"q": {"type": "noul"}}})
        self.assertEqual(raised.exception.status, 422)
        self.assertIn("choice questions only", str(raised.exception.detail))

    def test_prefix_cache_is_forwarded(self):
        hit = {"pos": 2, "token_ids": [11, 22], "hidden": [1.0], "chunks": []}
        cache = RecordingCache(hit)
        app = make_app(prefix_cache=cache)
        body = app.evaluate({"state": "hello", "options": {"left": None, "right": "go right"}})
        self.assertEqual(app.engine.seen[0]["prefix"], hit)
        self.assertEqual(app.engine.captures, 1)
        self.assertEqual(cache.stored[0][1]["pos"], len(cache.stored[0][0]))
        self.assertIn("left", body["answers"]["decision"]["probabilities"])
        self.assertEqual(body["timings"]["questions"][0]["prefix_tokens"], 2)


@unittest.skipUnless(
    sys.version_info >= (3, 12) and (JEFF_SRC / "jeff" / "client.py").is_file(),
    "upstream jeff client needs Python 3.12 and JEFF_SRC")
class UpstreamClientTests(unittest.TestCase):
    def setUp(self):
        from http.server import ThreadingHTTPServer
        sys.path.insert(0, str(JEFF_SRC))
        self.app = make_app(max_options=26)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), system1_ane_serve.make_handler(self.app))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    def test_upstream_client_reads_choice_health_and_errors(self):
        from jeff.client import Client, TooManyOptions, UnknownModel
        client = Client(self.url, model="jeff-latest")
        picked = client.choose("The disk is full.", {"page": "Page someone.", "wait": None})
        self.assertEqual(picked.key, "page")
        self.assertAlmostEqual(picked.probability, 0.7)
        self.assertAlmostEqual(picked.confidence, 0.4)
        self.assertAlmostEqual(client.yes_no("rain", "Is it wet?"), 0.3)
        scored = client.score("a ticket", ["low", "mid", "high"])
        self.assertAlmostEqual(scored.score, 0.45)
        self.assertEqual(scored.level, 0)
        health = client.health()
        self.assertEqual(health["status"], "ready")
        self.assertIn("jeff", {item["name"] for item in client.models()})
        with self.assertRaises(TooManyOptions):
            client.choose("x", {f"o{i}": None for i in range(30)})
        with self.assertRaises(UnknownModel):
            client.with_model("no-such-model").choose("x", {"a": None, "b": None})


if __name__ == "__main__":
    unittest.main()
