"""Tests that run anywhere (no NPU): config, chunking, and the gateway against a fake flm."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ConfigTests(unittest.TestCase):
    def test_aliases_and_lemonade_names(self):
        from npukit import config as C
        cfg = C.load(reload=True)
        self.assertEqual(C.resolve_model("fast", cfg), cfg["models"]["aliases"]["fast"])
        self.assertEqual(C.resolve_model("qwen3.5-4b-FLM", cfg), "qwen3.5:4b")
        self.assertEqual(C.resolve_model("gpt-oss-20b-FLM", cfg), "gpt-oss:20b")
        self.assertEqual(C.resolve_model("llama3.2:1b", cfg), "llama3.2:1b")


class ChunkTests(unittest.TestCase):
    def test_overlap_and_line_numbers(self):
        from npukit.search import chunk
        text = "\n".join(f"line {i}" for i in range(1, 101))
        parts = chunk(text, 40, 5)
        self.assertEqual(parts[0][:2], (1, 40))
        self.assertEqual(parts[1][:2], (36, 75))
        self.assertEqual(parts[-1][1], 100)
        self.assertTrue(parts[1][2].startswith("line 36"))

    def test_empty(self):
        from npukit.search import chunk
        self.assertEqual(chunk("", 40, 5), [])


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.port = free_port()
        ac = t / "ac_online"
        ac.write_text("1\n")
        cls.ac = ac
        (t / "config.toml").write_text(f"""
[server]
listen = "127.0.0.1:{cls.port}"
flm = "{ROOT / 'tests' / 'fake_flm.py'}"
flm_port = {free_port()}
start_timeout = 20
[power]
ac_online = "{ac}"
ac_pmode = "balanced"
battery_pmode = "powersaver"
ac_idle_unload = 3
battery_idle_unload = 1
[bar]
file = "{t / 'bar.json'}"
signal = 0
[search]
roots = ["{t / 'notes'}"]
db = "{t / 'index.sqlite'}"
""")
        (t / "notes").mkdir()
        (t / "notes" / "rust.md").write_text("# Rust notes\nownership and borrowing in rust\n")
        (t / "notes" / "cook.txt").write_text("pancakes: flour, milk, eggs\n")
        (t / "notes" / "tool").write_text("#!/bin/sh\necho rust tool\n")
        (t / "notes" / "blob").write_text("no shebang, no extension\n")
        os.chmod(ROOT / "tests" / "fake_flm.py", 0o755)
        cls.env = dict(os.environ, NPUKIT_CONFIG=str(t / "config.toml"), PYTHONPATH=str(ROOT))
        os.environ["NPUKIT_CONFIG"] = str(t / "config.toml")
        os.environ["XDG_STATE_HOME"] = str(t / "state")  # recent translate languages etc.
        cls.proc = subprocess.Popen([sys.executable, "-m", "npukit", "serve"], env=cls.env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        from npukit import client, config as C
        C.load(reload=True)
        cls.client = client
        for _ in range(100):
            try:
                client.call("/npu/status", timeout=1)
                break
            except client.ApiError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)
        cls.tmp.cleanup()

    def status(self):
        return self.client.call("/npu/status", timeout=5)

    def test_1_chat_alias_is_resolved(self):
        out = self.client.chat([{"role": "user", "content": "hi"}], "fast")
        self.assertIn("model=qwen3.5:4b", out["choices"][0]["message"]["content"])
        self.assertIn("pmode=balanced", out["choices"][0]["message"]["content"])
        st = self.status()
        self.assertEqual(st["components"], ["chat"])
        self.assertEqual(st["last_tps"], 12.5)

    def test_2_helper_joins_chat_model(self):
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        vecs = self.client.embed(["a", "rust"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(sorted(self.status()["components"]), ["chat", "embed"])

    def test_3_third_component_drops_the_oldest(self):
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        self.client.embed(["a"])
        text = self.client.transcribe(b"RIFF....fake")
        self.assertIn("heard", text)
        comps = self.status()["components"]
        self.assertIn("asr", comps)
        self.assertEqual(len(comps), 2)

    def test_4_flm_error_body_raises(self):
        # unknown endpoints pass through to flm, which answers HTTP 200 + {"error": ...}
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        with self.assertRaises(self.client.ApiError) as cm:
            self.client.call("/v1/unknown-endpoint", {"x": 1})
        self.assertIn("unknown", str(cm.exception))

    def test_5_idle_unload_and_battery_mode(self):
        self.ac.write_text("0\n")
        try:
            out = self.client.chat([{"role": "user", "content": "hi"}], "quality")
            self.assertIn("pmode=powersaver", out["choices"][0]["message"]["content"])
            self.assertEqual(self.status()["idle_unload"], 1)
            time.sleep(2.5)
            self.assertEqual(self.status()["state"], "off")
        finally:
            self.ac.write_text("1\n")

    def test_6_bar_json(self):
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        for _ in range(50):  # the reply reaches us just before the gateway marks the request done
            bar = json.loads(Path(self.tmp.name, "bar.json").read_text())
            if "busy" not in bar["class"]:
                break
            time.sleep(0.02)
        self.assertIn("qwen3.5 4b", bar["text"])
        st = json.loads(Path(self.tmp.name, "status.json").read_text())
        self.assertEqual(st["chat_model"], "qwen3.5:4b")
        self.assertIn(st["state"], ("ready", "busy"))
        self.assertIn("ready", bar["class"])

    def test_7_search_index_and_query(self):
        import argparse
        from npukit import search, config as C
        C.load(reload=True)
        self.assertEqual(search.index(argparse.Namespace(full=False, quiet=True)), 0)
        hits = search.query("rust", 2, C.load())
        self.assertTrue(hits[0][1].endswith("rust.md"))
        names = {Path(p).name for p in map(str, search.candidates(C.load()))}
        self.assertIn("tool", names)       # extensionless script with a shebang
        self.assertNotIn("blob", names)    # extensionless, no shebang

    def test_8_ocr_translate(self):
        import argparse, contextlib, io
        from npukit import ocr, config as C
        o = C.load(reload=True)["ocr"]
        self.assertEqual(ocr.language("sv"), "Swedish")
        self.assertEqual(ocr.language("Klingon"), "Klingon")
        msgs = ocr.translate_messages("hej", "German", o["translate_prompt"])
        self.assertIn("into German", msgs[0]["content"])
        self.assertEqual(msgs[1], {"role": "user", "content": "hej"})
        png = Path(self.tmp.name) / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
        out, err = io.StringIO(), io.StringIO()
        ns = argparse.Namespace(file=str(png), describe=False, ask=None, translate="", copy=False, model=None)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(ocr.main(ns), 0)
        self.assertIn("model=", out.getvalue())
        self.assertIn("---", err.getvalue())   # original printed before the translation


class LlamaCppGatewayTests(unittest.TestCase):
    """The GPU backend: a llama-server per chat model and a whisper-server, swapped independently."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.port = free_port()
        fake = ROOT / "tests" / "fake_llama.py"
        os.chmod(fake, 0o755)
        for f in ("q9.gguf", "mm.gguf", "small.gguf", "whisper.bin"):
            (t / f).write_bytes(b"GGUF")
        (t / "ac_online").write_text("1\n")  # not the host's charger: its state changes the timeouts
        cls.cfg_path = t / "config.toml"
        cls.cfg_path.write_text(f"""
[server]
listen = "127.0.0.1:{cls.port}"
backend = "llamacpp"
flm_port = {free_port()}
start_timeout = 20
[llamacpp]
server = "{fake}"
whisper = "{fake}"
whisper_model = "{t / 'whisper.bin'}"
[llamacpp.models]
"qwen3.5:9b" = {{ path = "{t / 'q9.gguf'}", mmproj = "{t / 'mm.gguf'}" }}
small = {{ path = "{t / 'small.gguf'}" }}
gone = {{ path = "{t / 'missing.gguf'}" }}
[power]
ac_online = "{t / 'ac_online'}"
ac_idle_unload = 3
battery_idle_unload = 3
[bar]
file = "{t / 'bar.json'}"
signal = 0
""")
        cls.old_cfg = os.environ.get("NPUKIT_CONFIG")
        env = dict(os.environ, NPUKIT_CONFIG=str(cls.cfg_path), PYTHONPATH=str(ROOT))
        os.environ["NPUKIT_CONFIG"] = str(cls.cfg_path)
        cls.proc = subprocess.Popen([sys.executable, "-m", "npukit", "serve"], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from npukit import client, config as C
        C.load(reload=True)
        cls.client = client
        for _ in range(100):
            try:
                client.call("/npu/status", timeout=1)
                break
            except client.ApiError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)
        if cls.old_cfg is None:
            os.environ.pop("NPUKIT_CONFIG", None)
        else:
            os.environ["NPUKIT_CONFIG"] = cls.old_cfg
        from npukit import config as C
        C.load(reload=True)
        cls.tmp.cleanup()

    def status(self):
        return self.client.call("/npu/status", timeout=5)

    def pids(self):
        return {p["components"][0]: p["pid"] for p in self.status()["procs"]}

    def test_0_config_uses_gpu_aliases(self):
        from npukit import config as C
        cfg = C.load(reload=True)
        self.assertEqual(C.backend(cfg), "llamacpp")
        self.assertEqual(C.engine_label(cfg), "GPU")
        self.assertEqual(C.resolve_model("fast", cfg), "qwen3.5:9b")
        # one model entry from the file, the defaults kept beside it
        self.assertIn("gpt-oss:20b", cfg["llamacpp"]["models"])
        self.assertNotIn("hf", cfg["llamacpp"]["models"]["qwen3.5:9b"])

    def test_1_chat_with_vision_projector(self):
        out = self.client.chat([{"role": "user", "content": "hi"}], "fast")
        text = out["choices"][0]["message"]["content"]
        self.assertIn("alias=qwen3.5:9b", text)
        self.assertIn("vision=True", text)
        st = self.status()
        self.assertEqual(st["components"], ["chat"])
        self.assertEqual(st["backend"], "llamacpp")
        self.assertIsNone(st["pmode"])
        self.assertEqual(st["last_tps"], 88.5)

    def test_2_whisper_joins_without_restarting_chat(self):
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        before = self.pids()["chat"]
        text = self.client.transcribe(b"RIFF....fake")
        self.assertIn("whisper", text)
        self.assertEqual(sorted(self.status()["components"]), ["asr", "chat"])
        self.assertEqual(self.pids()["chat"], before)

    def test_3_switching_chat_model_keeps_whisper(self):
        self.client.transcribe(b"RIFF....fake")
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        pids = self.pids()
        out = self.client.chat([{"role": "user", "content": "hi"}], "small")
        self.assertIn("alias=small", out["choices"][0]["message"]["content"])
        self.assertIn("vision=False", out["choices"][0]["message"]["content"])
        after = self.pids()
        self.assertEqual(after["asr"], pids["asr"])
        self.assertNotEqual(after["chat"], pids["chat"])

    def test_4_bad_models_fail_before_unloading(self):
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        pids = self.pids()
        with self.assertRaises(self.client.ApiError) as cm:
            self.client.chat([{"role": "user", "content": "hi"}], "gone")
        self.assertIn("missing", str(cm.exception))
        with self.assertRaises(self.client.ApiError) as cm:
            self.client.chat([{"role": "user", "content": "hi"}], "no-such-model")
        self.assertIn("unknown model", str(cm.exception))
        self.assertEqual(self.pids()["chat"], pids["chat"])

    def test_5_bar_says_gpu_and_idle_unload(self):
        self.client.chat([{"role": "user", "content": "hi"}], "fast")
        for _ in range(50):
            bar = json.loads(Path(self.tmp.name, "bar.json").read_text())
            if "busy" not in bar["class"]:
                break
            time.sleep(0.02)
        self.assertIn("qwen3.5 9b", bar["text"])
        self.assertTrue(bar["tooltip"].startswith("GPU · "))
        self.assertNotIn("pmode", bar["tooltip"])
        st = json.loads(Path(self.tmp.name, "status.json").read_text())
        self.assertEqual(st["runs_on"], "GPU")
        for _ in range(60):
            if self.status()["state"] == "off":
                break
            time.sleep(0.1)
        self.assertEqual(self.status()["state"], "off")
        self.assertEqual(self.status()["procs"], [])


class LanguagePickTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = os.environ.get("XDG_STATE_HOME")
        os.environ["XDG_STATE_HOME"] = self.tmp.name

    def tearDown(self):
        if self.env is None:
            os.environ.pop("XDG_STATE_HOME", None)
        else:
            os.environ["XDG_STATE_HOME"] = self.env
        self.tmp.cleanup()

    def test_recent_first_then_default_then_a_to_z(self):
        from npukit import ocr
        ocr.remember_language("German")
        ocr.remember_language("Klingon")
        ocr.remember_language("german")  # same language: moves up, no duplicate
        self.assertEqual(ocr.recent_languages(), ["german", "Klingon"])
        rows, n = ocr.picker_rows(ocr.recent_languages(), "English")
        self.assertEqual(n, 2)
        self.assertEqual([lang for _, lang in rows[:4]], ["german", "Klingon", "English", "Arabic"])
        self.assertIn("sv", dict((lang, text) for text, lang in rows)["Swedish"])
        self.assertEqual(len([1 for _, lang in rows if lang.lower() == "german"]), 1)

    def test_parse_pick(self):
        from npukit import ocr
        rows, _ = ocr.picker_rows([], "English")
        self.assertEqual(ocr.parse_pick(0, "0\t" + rows[0][0] + "\n", rows), "English")
        self.assertEqual(ocr.parse_pick(0, "-1\tsv\n", rows), "Swedish")      # typed code
        self.assertEqual(ocr.parse_pick(0, "-1\tPirate\n", rows), "Pirate")   # typed, not listed
        self.assertIsNone(ocr.parse_pick(1, "", rows))                          # Esc
        self.assertIsNone(ocr.parse_pick(0, "-1\t \n", rows))


if __name__ == "__main__":
    unittest.main()
