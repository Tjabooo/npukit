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


if __name__ == "__main__":
    unittest.main()
