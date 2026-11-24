"""The image encoder starts with the first picture instead of with the engine's load when nothing makes it go first:
a CPU reader holds no VRAM, and a GPU reader whose need fits the engine's --vram-reserve-mib lives in that reserve
(the cache sizes itself around it).  A GPU reader the reserve does not cover still starts before the engine."""
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision
from serve.test_lifecycle import ResidentEngine


class LazyVisionStart(unittest.TestCase):
    def make(self, gpu: bool, reserve):
        """A service with a loaded-again engine and a Vision stub (no reader process).  `reserve` is None (no
        --vram-reserve-mib in the engine's args), a number, or "need" (the reader's own elastic_mib)."""
        tok = ByteTokenizer()
        v = Vision({"exe": "strata-vision", "mmproj": "mm.proj", "model": "model.gguf", "gpu": gpu,
                    "max_tokens": 300}, lazy=True)
        reserve_mib = v.elastic_mib if reserve == "need" else reserve
        engine = ResidentEngine(tok)
        args = ["--serve"] + (["--vram-reserve-mib", str(reserve_mib)] if reserve_mib is not None else [])
        engine.spawn = ("strata", args, None, None, None, False, None)
        v.down = True
        v.starts = 0
        v.alive = lambda: not v.down                    # a flag stands in for the reader process
        v.restart = lambda: (setattr(v, "down", False), setattr(v, "starts", v.starts + 1))
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"), vision=v)
        return engine, v, svc

    def test_cpu_reader_waits_for_the_first_picture(self):
        engine, v, svc = self.make(gpu=False, reserve=None)
        self.assertFalse(svc._vision_down())
        svc.ensure_loaded()                             # the wake: the engine loads, the reader does not start
        self.assertEqual((engine.starts, v.starts), (1, 0))
        svc._vision_up()                                # the first picture starts it
        self.assertEqual(v.starts, 1)

    def test_gpu_reader_inside_the_reserve_waits_too(self):
        engine, v, svc = self.make(gpu=True, reserve="need")
        self.assertFalse(svc._vision_down())
        svc.ensure_loaded()
        self.assertEqual((engine.starts, v.starts), (1, 0))
        svc._vision_up()
        self.assertEqual(v.starts, 1)

    def test_gpu_reader_without_a_reserve_starts_first(self):
        engine, v, svc = self.make(gpu=True, reserve=None)
        self.assertTrue(svc._vision_down())
        svc.ensure_loaded()                             # the old behaviour: the reader takes its VRAM, then the engine
        self.assertEqual((engine.starts, v.starts), (1, 1))

    def test_gpu_reader_bigger_than_the_reserve_starts_first(self):
        engine, v, svc = self.make(gpu=True, reserve=512)
        self.assertTrue(svc._vision_down())
        svc.ensure_loaded()
        self.assertEqual((engine.starts, v.starts), (1, 1))

    def test_an_alive_encoder_is_not_started_again(self):
        engine, v, svc = self.make(gpu=False, reserve=None)
        v.down = False
        svc._vision_up()
        self.assertEqual(v.starts, 0)


if __name__ == "__main__":
    unittest.main()
