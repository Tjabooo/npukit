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


class FileToolTests(unittest.TestCase):
    """Read-only access, only inside granted folders, never the deny list."""

    def setUp(self):
        from npukit import tools
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.root = t / "proj"
        (self.root / "src" / "deep").mkdir(parents=True)
        (self.root / "node_modules" / "x").mkdir(parents=True)
        (self.root / "README.md").write_text("# Helix\nA game about hollows.\n")
        (self.root / "src" / "main.lua").write_text("local hollow = 1\nprint(hollow)\n")
        (self.root / "src" / "deep" / "a.txt").write_text("x\n" * 1000)
        (self.root / "node_modules" / "x" / "hollow.js").write_text("hollow\n")
        (self.root / ".env").write_text("TOKEN=hollow-secret\n")
        (self.root / "blob.bin").write_bytes(b"\0\1\2hollow")
        (t / "outside.txt").write_text("not yours hollow\n")
        (self.root / "escape").symlink_to(t / "outside.txt")
        self.f = tools.Files([str(self.root)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_outside_symlink_and_deny_list_are_refused(self):
        run = self.f.run
        self.assertIn("outside the folders", run("read_file", {"path": str(Path(self.tmp.name) / "outside.txt")}))
        self.assertIn("outside the folders", run("read_file", {"path": "../outside.txt"}))
        self.assertIn("outside the folders", run("read_file", {"path": "escape"}))   # symlink out
        self.assertIn("deny list", run("read_file", {"path": ".env"}))
        self.assertIn("binary", run("read_file", {"path": "blob.bin"}))
        self.assertIn("no folders are granted", __import__("npukit.tools", fromlist=["Files"]).Files([]).run(
            "list_dir", {"path": "~"}))

    def test_list_read_search_find(self):
        run = self.f.run
        listing = run("list_dir", {"path": "", "depth": 2})
        self.assertIn("src/", listing)
        self.assertIn("main.lua", listing)
        self.assertIn("node_modules/  (not descended)", listing)
        self.assertNotIn(".env", listing)
        page = run("read_file", {"path": "src/deep/a.txt", "offset": 10, "limit": 5})
        self.assertIn("lines 10-14 of 1000", page)
        self.assertIn("offset=15", page)
        hits = run("search_text", {"pattern": "hollow"})
        self.assertIn("README.md:2:", hits)
        self.assertIn("main.lua:1:", hits)
        for leak in ("hollow-secret", "node_modules", "outside.txt"):
            self.assertNotIn(leak, hits)
        self.assertIn("src/main.lua", run("find_files", {"glob": "*.lua"}))
        self.assertIn("no tool", run("delete_file", {"path": "README.md"}))


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
vram_reserve = 0     # headroom rules read the host's GPU/RAM: tested in HeadroomTests instead
ram_reserve_gb = 0
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

    def test_45_chat_reads_a_granted_file(self):
        from npukit import agent, tools
        d = Path(self.tmp.name, "proj")
        d.mkdir(exist_ok=True)
        (d / "notes.md").write_text("helix notes\nline two\n")
        s = agent.Session("fast", tools.Files([str(d)]), quiet=True)
        answer = s.ask("read notes please")
        self.assertIn("helix notes", answer)
        self.assertEqual([m["role"] for m in s.messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertIn("proj", s.messages[0]["content"])

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


class HeadroomTests(unittest.TestCase):
    """VRAM reserve → --fit-target, RAM check before loading, Whisper on the CPU at the reserve."""
    GB = 10 ** 9

    def setUp(self):
        from npukit import config as C, gateway, hw
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "big.gguf").write_bytes(b"G")
        (t / "w.bin").write_bytes(b"W")
        self.hw, self.saved = hw, (hw.gpu_vram, hw.mem_available, hw.proc_vram, hw.proc_rss_anon)
        cfg = C.load(reload=True)
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["llamacpp"].update(vram_reserve=0.2, ram_reserve_gb=4, threads=6, overhead_gb=0,
                               whisper_model=str(t / "w.bin"))
        cfg["llamacpp"]["models"] = {"big": {"path": str(t / "big.gguf"), "size_gb": 0}}
        self.eng = gateway.LlamaCppEngine(cfg)
        self.size = 13 * self.GB  # pretend the model file is 13 GB
        self.eng.footprint = lambda spec: self.size
        self.vram = {"total": 16 * self.GB, "used": 7 * self.GB}
        self.avail = 14 * self.GB
        hw.gpu_vram = lambda: dict(self.vram)
        hw.mem_available = lambda: self.avail
        hw.proc_vram = lambda pid: 6 * self.GB
        hw.proc_rss_anon = lambda pid: 1 * self.GB

    def tearDown(self):
        h = self.hw
        h.gpu_vram, h.mem_available, h.proc_vram, h.proc_rss_anon = self.saved
        self.tmp.cleanup()

    def test_fit_target_is_the_reserve_and_threads_are_capped(self):
        argv = self.eng.plan(["chat"], "big", None)[0].argv
        i = argv.index("--fit-target")
        self.assertEqual(int(argv[i + 1]), int(16 * self.GB * 0.2) // 1048576)
        self.assertEqual(argv[argv.index("-t") + 1], "6")
        self.assertNotIn("-ngl", argv)

    def test_ram_check(self):
        # GPU room 16 - 7 - 3.2 = 5.8 GB, so 7.2 GB goes to RAM: 14 - 7.2 >= 4 fits
        self.eng.fits("big", {"path": "x"}, None)
        self.avail = 10 * self.GB  # 10 - 7.2 < 4: refused, with the numbers in the message
        with self.assertRaises(RuntimeError) as cm:
            self.eng.fits("big", {"path": "x"}, None)
        self.assertIn("keeping 4 GB free", str(cm.exception))

    def test_replaced_chat_server_counts_as_freed(self):
        from npukit.gateway import Child, Proc
        self.avail = 10 * self.GB
        old = Proc("llama", ["chat"], "fast", 1, ["x"])
        old.proc = type("P", (), {"returncode": None, "pid": 1})()
        # its 6 GB VRAM and 1 GB RAM come back: room 11.8 GB, 1.2 GB to RAM, 11 GB available
        self.eng.fits("big", {"path": "x"}, Child([old], None))

    def test_whisper_leaves_the_gpu_at_the_reserve(self):
        argv = self.eng.plan(["asr"], None, None)[0].argv
        self.assertNotIn("-ng", argv)
        self.vram["used"] = 13 * self.GB  # 3 GB free < 3.2 GB reserve
        p = self.eng.plan(["asr"], None, None)[0]
        self.assertIn("-ng", p.argv)
        self.assertEqual(p.note, "CPU")
        self.assertEqual(p.argv[p.argv.index("-t") + 1], "6")


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


class TranslateOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {k: os.environ.get(k) for k in ("XDG_STATE_HOME", "XDG_RUNTIME_DIR")}
        os.environ["XDG_STATE_HOME"] = self.tmp.name + "/state"
        os.environ["XDG_RUNTIME_DIR"] = self.tmp.name + "/run"

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def cfg(self):
        from npukit import config as C
        cfg = C.load()
        cfg["translate"].update(copy=False, type=[], eww="true")
        return cfg

    def test_chips_current_then_recent_then_configured(self):
        from npukit import translate as T
        self.assertEqual(T.chips(["sv", "German"], ["English", "swedish", "French"], "de"),
                         ["German", "Swedish", "English", "French"])
        self.assertEqual(len(T.chips([], [f"L{i}" for i in range(20)], "")), T.MAX_CHIPS)

    def test_hostile_text_reaches_stdin_verbatim(self):
        """eww pastes the input's value into the command unescaped; the here-document keeps it inert."""
        from npukit import translate as T
        out = Path(self.tmp.name) / "got"
        nasty = "it's \"$(touch PWNED)\" `touch PWNED` $HOME; rm -rf / & {0} \\n"
        for action in (T.heredoc_action("tee", str(out)), T.draft_action(out)):
            out.unlink(missing_ok=True)
            subprocess.run(["/bin/sh", "-c", action.replace("{}", nasty)], cwd=self.tmp.name, check=True)
            for _ in range(50):
                if out.exists() and out.read_text():
                    break
                time.sleep(0.02)
            self.assertEqual(out.read_text(), nasty + "\n")
            self.assertFalse((Path(self.tmp.name) / "PWNED").exists())

    def test_render_has_no_placeholders_and_uses_palette(self):
        from npukit import translate as T
        cfg = self.cfg()
        cfg["translate"]["colors"]["accent"] = "#123456"
        self.assertNotRegex(T.render_yuck(cfg), r"@[A-Z_]+@")
        self.assertIn("$accent: #123456;", T.render_scss(cfg))

    def test_enter_translates_then_enter_again_sends(self):
        from npukit import translate as T
        cfg = self.cfg()
        typed = Path(self.tmp.name) / "typed"
        cfg["translate"].update(type=["sh", "-c", f"cat > {typed}"], type_delay=0)
        calls = []

        def fake_translate(text, lang, prompt, model):
            calls.append((text, lang))
            return f"[{lang}] {text}", 0.1
        orig = (T.ocr.translate, T.push, T._eww, T._hook)
        T.ocr.translate, T.push = fake_translate, lambda cfg, st: None
        T._eww, T._hook = (lambda cfg, *a, **k: (0, "")), (lambda cfg, key: None)
        try:
            T.save_state(T.fresh_state(cfg, {}) | {"lang": "German"})
            self.assertEqual(T.run(cfg, "hej"), 0)
            self.assertEqual(T.load_state()["result"], "[German] hej")
            T.run(cfg, "hej", "French")          # chip: same text, new language → translate again
            T.run(cfg, "hej", "French")          # same chip again: nothing new
            self.assertEqual(calls, [("hej", "German"), ("hej", "French")])
            self.assertIn("French", T.load_state()["chips"])
            T.run(cfg, "hej")                    # Enter again on the same text → type it, close
            self.assertEqual(typed.read_text(), "[French] hej")
            self.assertFalse(T.load_state()["open"])
            self.assertEqual(len(calls), 2)
        finally:
            T.ocr.translate, T.push, T._eww, T._hook = orig


if __name__ == "__main__":
    unittest.main()
