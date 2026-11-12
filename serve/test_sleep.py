"""Sleep mode: save the conversation, unload the engine (all VRAM and RAM back), wake with the context restored."""
import json
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, SessionRefused, serve
from serve.test_lifecycle import ResidentEngine


class SleepyEngine(ResidentEngine):
    """A ResidentEngine that keeps session files: sleep saves to it, the wake restores from it."""

    def __init__(self, tok):
        super().__init__(tok)
        self.saves, self.restores = [], []
        self.refuse_save = self.refuse_restore = None      # a SessionRefused to raise instead

    def session_file(self, action, path):
        refused = self.refuse_save if action == "save" else self.refuse_restore
        if refused is not None:
            raise refused
        (self.saves if action == "save" else self.restores).append(path)
        return {"tokens": 123, "bytes": 456, "ms": 7.5}


class Sleep(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = SleepyEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.sleep_file = "/tmp/strata-test-sleep.strsess"
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def sleep_loaded_engine(self):
        self.engine.loaded = True
        self.assertEqual(self.svc.sleep(), "sleeping")
        self.assertEqual(self.engine.saves, [self.svc.sleep_file])
        self.assertTrue(self.engine.unloaded)
        self.assertEqual(self.svc.sleep_session, self.svc.sleep_file)
        self.assertEqual(self.svc.sleeps, 1)

    def test_sleep_saves_then_unloads_and_the_wake_restores(self):
        self.sleep_loaded_engine()
        self.svc.load()                                     # the next request's ensure_loaded: the wake
        self.assertEqual(self.engine.starts, 1)
        self.assertEqual(self.engine.restores, [self.svc.sleep_file])
        self.assertIsNone(self.svc.sleep_session)
        self.assertEqual(self.svc.wakes, 1)
        self.assertTrue(self.engine.loaded)
        self.svc.load()                                     # loaded again: no second restore
        self.assertEqual(self.engine.restores, [self.svc.sleep_file])

    def test_metrics_say_sleeping_until_the_wake(self):
        self.sleep_loaded_engine()
        metrics = self.request("/metrics")[1]
        self.assertEqual(metrics["live"]["state"], "sleeping")
        self.assertTrue(self.request("/v1/status")[1]["sleeping"])
        prom = urllib.request.urlopen(self.base + "/metrics/prometheus", timeout=10).read().decode()
        self.assertIn("strata:model_sleeping 1", prom)
        self.assertIn("strata:sleeps_total 1", prom)
        self.assertIn("strata:wakes_total 0", prom)
        self.svc.load()
        self.assertEqual(self.request("/metrics")[1]["live"]["state"], "idle")
        self.assertFalse(self.request("/v1/status")[1]["sleeping"])
        prom = urllib.request.urlopen(self.base + "/metrics/prometheus", timeout=10).read().decode()
        self.assertIn("strata:wakes_total 1", prom)

    def test_sleep_endpoints_answer_the_engine(self):
        self.engine.loaded = True
        self.assertEqual(self.request("/v1/sleep", {"model": "other-model"})[0], 404)
        code, body = self.request("/v1/sleep", {})
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "sleeping")
        self.assertTrue(self.engine.unloaded)
        self.assertEqual(self.request("/sleep", {})[1]["status"], "not loaded")   # asleep already
        self.svc.load()                                   # awake again: the control endpoint sleeps once more
        self.assertEqual(self.request("/sleep", {})[1]["status"], "sleeping")

    def test_a_running_request_answers_busy(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/sleep", {})[0], 409)
            self.assertEqual(self.request("/sleep", {})[0], 409)
        self.assertEqual(self.engine.saves, [])
        self.assertTrue(self.engine.loaded)

    def test_parallel_slots_cannot_sleep(self):
        self.engine.loaded = True
        self.engine.batch = 2
        self.assertEqual(self.request("/v1/sleep", {})[0], 400)   # a ValueError: the request is the problem
        self.assertEqual(self.request("/sleep", {})[0], 501)
        self.assertEqual(self.engine.saves, [])
        self.assertTrue(self.engine.loaded)

    def test_a_refused_save_keeps_the_model_loaded(self):
        self.engine.loaded = True
        self.engine.refuse_save = SessionRefused("memory", "not enough host memory", published=True)
        code, body = self.request("/v1/sleep", {})
        self.assertEqual(code, 503)
        self.assertEqual(body["error"]["kind"], "memory")
        self.assertTrue(body["error"]["published"])
        self.assertTrue(self.engine.loaded)               # no save, no sleep
        self.assertIsNone(self.svc.sleep_session)
        self.assertEqual(self.svc.sleeps, 0)

    def test_a_failed_restore_still_loads_the_engine(self):
        self.sleep_loaded_engine()
        self.engine.refuse_restore = SessionRefused("invalid", "fingerprint mismatch")
        self.svc.load()                                   # the engine runs, without the old context
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.engine.restores, [])
        self.assertEqual(self.svc.wakes, 0)
        self.assertIsNone(self.svc.sleep_session)         # the stale file is not tried again

    def test_the_idle_thread_sleeps_after_the_timeout(self):
        self.engine.loaded = True
        self.engine.unloaded = False
        self.svc.sleep_timeout_s = 0.4
        self.svc.last_request_at = time.time() - 10
        self.svc.start_idle_sleep()
        deadline = time.monotonic() + 5
        while self.svc.sleep_session is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.svc.sleep_session, self.svc.sleep_file)
        self.assertTrue(self.engine.unloaded)
        self.assertEqual(self.engine.saves, [self.svc.sleep_file])

    def test_an_engine_without_session_files_cannot_sleep(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertEqual(svc.sleep(), "unsupported")

    def test_sleep_needs_the_api_key_like_the_other_control_endpoints(self):
        self.engine.loaded = True
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/sleep", {})[0], 401)
        self.assertEqual(self.request("/sleep", {}, {"Origin": "https://other.example",
                                                       "Authorization": "Bearer local-secret"})[0], 403)
        self.assertEqual(self.request("/sleep", {}, {"Authorization": "Bearer local-secret"})[0], 200)
        self.assertTrue(self.engine.unloaded)


if __name__ == "__main__":
    unittest.main()
