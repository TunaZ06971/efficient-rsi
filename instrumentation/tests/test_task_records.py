"""Failure-path checks using synthetic containers and real temporary Git repositories."""
import ast
import datetime
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HGM = Path(__file__).resolve().parents[2] / "repos/HGM"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


T = module("task_records", HGM / "task_records.py")
sys.modules.setdefault("task_records", T)
L = module("lc_integration_records_test", HGM / "lc_integration.py")


def result(code=0, text=""):
    return types.SimpleNamespace(exit_code=code, output=text.encode())


class FakeContainer:
    def __init__(self, agent_code=0, grade_code=0, missing_patch=False, grade_text="2 passed in 0.1s"):
        self.agent_code, self.grade_code, self.grade_text = agent_code, grade_code, grade_text
        self.cleaned = False
        self.commands = []
        self.files = {"/hgm/python__x.md": "Tool Used: Bash\nTool Result: worked\n",
                      "/hgm/rsi_attempt.json": json.dumps({"scope": "attempt", "stop_reason": "normal", "turns": 2})}
        if not missing_patch:
            self.files["/hgm/model_patch.diff"] = "submitted answer\n"

    def start(self):
        pass

    def exec_run(self, cmd, **kwargs):
        self.commands.append(cmd)
        if cmd == "cat /hgm/model_patch.diff":
            if "/hgm/model_patch.diff" in self.files:
                return result(0, self.files["/hgm/model_patch.diff"])
            return result(1, "cat: /hgm/model_patch.diff: No such file or directory")
        if isinstance(cmd, list) and cmd[:2] == ["timeout", "30"]:
            return result(0, "recovered answer\n")
        if isinstance(cmd, list) and cmd[:2] == ["timeout", "3600"]:
            return result(self.agent_code, "agent finished")
        if cmd == "timeout 120 ./eval.sh":
            return result(self.grade_code, self.grade_text)
        return result()


def copy_from(container, src, dest):
    if src not in container.files:
        raise FileNotFoundError(src)
    Path(dest).write_text(container.files[src])


class RecordsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "node/predictions/node_0"
        self.out.mkdir(parents=True)
        self.entry = dict(instance_id="python__x", problem_statement="example", base_commit="HEAD",
                          test_commit="HEAD", language="python", files={"solution": ["solution.py"]})
        self.env = patch.dict(os.environ, {"HGM_TASK_RECORDS": "1", "HGM_RECOVER_ANSWERS": "0", "HGM_RECORD_FEEDBACK": "0",
                                          "HGM_AGENT": "", "HGM_TEST_PROTOCOL": "hidden", "RSI_LEDGER": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def card(self, kind="original"):
        return T.TaskRecord(self.out, self.entry, "node_0", "local-qwen", kind, 3600, [], HGM)

    def test_timeout_survives_collection_and_grading(self):
        c = self.card()
        c.phase("solve")
        c.agent_exit(result(124))
        c.collect(FakeContainer(missing_patch=True), copy_from)
        with self.assertRaises(RuntimeError):
            c.patch_for_grading()
        with patch.dict(os.environ, {"HGM_RECOVER_ANSWERS": "1"}):
            self.assertEqual(c.patch_for_grading(), "recovered answer\n")
        c.phase("grade")
        c.grade(result(0, "2 passed in 0.1s"))
        c.finish({"success": True, "eval_result": "resolved"})
        saved = T.read_json(c.path)
        self.assertEqual(saved["execution"]["stop_reason"], "hard_timeout")
        self.assertEqual(saved["grading"]["status"], "passed")
        self.assertEqual(saved["answer"]["origin"], "recovered_after_stop")
        self.assertIsNone(saved["cost"]["request_elapsed_s"])

    def test_retries_preserve_both_cards_and_answers(self):
        a, b = self.card(), self.card()
        for c, code in ((a, 124), (b, 0)):
            c.agent_exit(result(code))
            c.collect(FakeContainer(), copy_from)
            c.finish({"success": False, "eval_result": "incomplete"})
        self.assertNotEqual(a.id, b.id)
        self.assertTrue((a.dir / "submitted.diff").exists())
        self.assertTrue((b.dir / "submitted.diff").exists())
        self.assertEqual(len(T.latest_cards(self.root / "node")), 1)

    def test_grading_timeout_uses_exit_code_with_empty_output(self):
        c = self.card()
        c.grade(result(124))
        self.assertEqual(c.data["grading"]["status"], "timeout")
        self.assertEqual(T.failure_class(c.data), "test_timeout")

    def test_zero_tests_is_not_validated_success(self):
        c = self.card()
        c.grade(result(0, "no tests ran in 0.01s"))
        self.assertEqual(c.data["grading"]["status"], "invalid_no_tests")

    def test_unrecognized_summary_is_unknown(self):
        self.assertIsNone(T.test_counts("BUILD SUCCESSFUL")["tests_run"])
        self.assertEqual(T.test_counts("test result: ok. 3 passed; 0 failed; 2 ignored;"),
                         dict(tests_run=3, passed=3, failed=0, skipped=2))

    def test_collection_failure_preserves_initial_reason(self):
        c = self.card()
        c.agent_exit(result(124))
        bad = FakeContainer()
        bad.exec_run = lambda *a, **k: result(1)
        c.collect(bad, lambda *a: (_ for _ in ()).throw(OSError("unavailable")))
        c.error(RuntimeError("could not collect"))
        c.finish({"eval_result": "incomplete", "success": False})
        self.assertEqual(c.data["execution"]["stop_reason"], "hard_timeout")
        self.assertEqual(c.data["answer"]["collection_status"], "failed")

    def test_recovery_only_contains_allowed_paths_including_new_files(self):
        repo = self.root / "git"
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL)
        git("init", "-q")
        (repo / "solution.py").write_text("old\n")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
        (repo / "solution.py").write_text("new\n")
        (repo / "allowed.py").write_text("new allowed file\n")
        (repo / "secret_test.py").write_text("do not collect\n")
        p = subprocess.run([sys.executable, "-c", T.RECOVER_SCRIPT, str(repo), "HEAD",
                            json.dumps(["solution.py", "allowed.py"])], cwd=repo, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("allowed.py", p.stdout)
        self.assertIn("+new", p.stdout)
        self.assertNotIn("secret_test", p.stdout)

    def test_feedback_selects_failure_classes(self):
        cards = []
        for i, reason in enumerate(["normal", "hard_timeout", "agent_exception"]):
            c = self.card()
            c.data["task_id"] = str(i)
            c.data["execution"]["stop_reason"] = reason
            c.data["grading"]["status"] = "failed"
            cards.append(c.data)
        selected = T.representatives(cards, limit=3)
        self.assertEqual({T.failure_class(c) for c in selected}, {"test_failed", "hard_timeout", "agent_exception"})

    def test_local_cost_distinguishes_live_calls_from_replay(self):
        c = self.card("littlecoder")
        live = {"resp": {"in": 100, "out": 10, "cached": 20}, "t0": 1, "t1": 3}
        served = {**live, "served": True, "t0": 0, "t1": 0}
        (c.dir / "record.jsonl").write_text("\n".join(map(json.dumps, [live, served])) + "\n")
        (c.dir / "record.replay").write_text(json.dumps({"event": "live", "in": 100, "out": 10, "t0": 1, "t1": 3}) + "\n")
        c.finish({"eval_result": "resolved"})
        cost = c.data["cost"]
        self.assertEqual(cost["replayed_calls"], 1)
        self.assertEqual(cost["requests_completed"], 1)
        self.assertEqual(cost["prompt_tokens"], 100)
        self.assertEqual(cost["request_elapsed_s"], 2)
        self.assertIsNone(cost["requests_unfinished"])

    def test_proxy_cost_excludes_previous_attempt_late_response(self):
        ledger = self.root / "ledger.jsonl"
        ledger.write_text("")
        with patch.dict(os.environ, {"RSI_LEDGER": str(ledger), "RSI_RUN": "test-run"}):
            c = self.card()
            base = {"type": "llm_call", "tag": "eval", "run": "test-run", "node": "node_0",
                    "task": "python__x", "status": 200, "input_tokens": 100, "output_tokens": 10,
                    "cached_tokens": 0, "ts": c.start_epoch + 5}
            ledger.write_text("\n".join(map(json.dumps, [{**base, "latency_s": 10}, {**base, "latency_s": 2}])) + "\n")
            c.finish({"eval_result": "resolved"})
        self.assertEqual(c.data["cost"]["requests_completed"], 1)
        self.assertEqual(c.data["cost"]["prompt_tokens"], 100)

    def test_soft_stop_and_setup_error_remain_separate_from_wrong_answer(self):
        c = self.card("littlecoder")
        container = FakeContainer()
        container.files["/hgm/rsi_attempt.json"] = json.dumps({"scope": "attempt", "stop_reason": "soft_deadline", "max_turns": 100})
        c.agent_exit(result(0))
        c.collect(container, copy_from)
        self.assertEqual(T.failure_class(c.data), "soft_deadline")
        self.assertEqual(c.data["conditions"]["budget"]["max_turns"], 100)
        setup = self.card()
        setup.error(RuntimeError("container unavailable"))
        setup.finish({"eval_result": "incomplete"})
        self.assertEqual(T.failure_class(setup.data), "setup_error")
        self.assertEqual(setup.data["grading"]["status"], "not_run")

    def harness(self, container, source=None):
        class Phases:
            def __init__(self, *a, **k): pass
            def phase(self, *a, **k): pass
            def attach(self, *a, **k): pass
            def done(self, *a, **k): pass
        ns = dict(os=os, json=json, datetime=datetime, Path=Path, tempfile=tempfile, shutil=shutil,
                  task_records=T, lc_integration=L, llm="local-qwen", timeout=3600,
                  docker=types.SimpleNamespace(from_env=lambda: None),
                  rsi_ledger=types.SimpleNamespace(Phases=Phases, proxy_env=lambda *a: {}, event=lambda *a, **k: None),
                  setup_logger=lambda *a: types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None),
                  make_test_spec=lambda e: types.SimpleNamespace(get_instance_container_name=lambda r: "test"),
                  remove_existing_container=lambda *a: None, build_container=lambda *a, **k: container,
                  cleanup_container=lambda *a: setattr(container, "cleaned", True), copy_from_container=copy_from,
                  copy_to_container=lambda *a: None, hide_answers=lambda: False,
                  get_test_description=lambda *a, **k: "", MAP_REPO_VERSION_TO_SPECS={"python": {"test_cmd": "pytest"}},
                  TEST_COMMANDS={"python": ["pytest"]}, safe_log=lambda *a: None)
        def log(r, raise_error=True):
            if raise_error and r.exit_code: raise RuntimeError(f"exit {r.exit_code}")
        ns["log_container_output"] = log
        tree = ast.parse(source if source is not None else (HGM / "polyglot/harness.py").read_text())
        wanted = {"process_entry", "process_entry_lc", "_score_in_container", "get_eval_script", "_publish_task_artifacts",
                  "_finish_task_record", "_process_entry_records", "_process_entry_lc_records"}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[]), "harness.py", "exec"), ns)
        return ns

    def test_both_harness_paths_normal_timeout_and_recovery(self):
        for kind in ("original", "littlecoder"):
            for code, recover in ((0, "0"), (124, "0"), (124, "1")):
                with self.subTest(kind=kind, code=code, recover=recover), patch.dict(os.environ, {"HGM_AGENT": kind, "HGM_RECOVER_ANSWERS": recover}):
                    container = FakeContainer(agent_code=code, missing_patch=bool(code))
                    ns = self.harness(container)
                    response = ns["process_entry"](self.entry, self.out, "node_0", [], skip_existing=False, init_agent_path=str(HGM))
                    saved = json.loads((self.out / "python__x.json").read_text())
                    card = T.load_card(saved, self.out)
                    self.assertIsNotNone(card)
                    self.assertTrue(container.cleaned)
                    self.assertEqual(saved["eval_result"], "resolved" if code == 0 or recover == "1" else "incomplete")
                    if code == 124:
                        self.assertEqual(card["execution"]["stop_reason"], "hard_timeout")
                    self.assertEqual(card["grading"]["status"], "passed" if code == 0 or recover == "1" else "not_run")

    def test_symlink_from_container_is_refused(self):
        c = self.card()
        def link_copier(container, src, dest):
            os.symlink("/etc/hosts", dest)          # what Docker's tar export restores for a link
        self.assertIsNone(c._copy(FakeContainer(), "/hgm/python__x.md", "trajectory.md", link_copier, required=True))
        self.assertFalse((c.dir / "trajectory.md").is_symlink())
        self.assertIn("symbolic link", c.data["data_quality"]["collection_errors"][0]["error"])

    def test_graded_pass_is_resolved_whatever_stopped_the_agent(self):
        c = self.card("littlecoder")
        c.data["execution"]["stop_reason"] = "turn_limit"
        c.data["grading"]["status"] = "passed"
        self.assertEqual(T.failure_class(c.data), "resolved")

    def test_opt_ins_need_the_record_switch(self):
        for name in ("HGM_RECORD_FEEDBACK", "HGM_RECOVER_ANSWERS"):
            with self.subTest(name=name), patch.dict(os.environ, {"HGM_TASK_RECORDS": "0", name: "1"}):
                self.assertFalse(T.feedback_enabled() or T.recovery_enabled())
                with self.assertRaises(ValueError):
                    T.validate_switches()


MAIN = "94f0b19"   # efficient-rsi main before this change (PR #3 merged)
DISPATCH = (
    "    if task_records.enabled():       # task record cards (HGM_TASK_RECORDS=1); unset: the code below as before\n"
    "        return _process_entry_records(entry, out_dname, model_name_or_path, model_patch_paths,\n"
    "                                      skip_existing=skip_existing, init_agent_path=init_agent_path)\n")


def main_source(path):
    """A file as it is on MAIN, or None when git history is not available (shallow copy)."""
    r = subprocess.run(["git", "show", f"{MAIN}:{path}"], cwd=HGM, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


class SameAsMainCase(RecordsCase):
    """HGM_TASK_RECORDS unset: the evaluation is efficient-rsi's code. Set: the same scores."""

    # (name, FakeContainer arguments, files to drop (None) or replace)
    SCENARIOS = [
        ("resolved", dict(), {}),
        ("unresolved", dict(grade_code=1, grade_text="1 failed in 0.1s"), {}),
        ("hard timeout", dict(agent_code=124), {}),
        ("crash", dict(agent_code=1), {}),
        ("empty patch", dict(), {"/hgm/model_patch.diff": ""}),
        ("no patch file", dict(missing_patch=True), {}),
        ("no chat log", dict(), {"/hgm/python__x.md": None}),
        ("no rsi_attempt.json", dict(), {"/hgm/rsi_attempt.json": None}),
    ]

    def setUp(self):
        super().setUp()
        self.main = main_source("repos/HGM/polyglot/harness.py")
        if self.main is None:
            self.skipTest(f"git history with {MAIN} is not available")
        self.runs = 0

    def run_once(self, source, kind, args, files):
        container = FakeContainer(**args)
        for name, text in files.items():
            if text is None:
                container.files.pop(name, None)
            else:
                container.files[name] = text
        self.runs += 1
        out = self.root / f"run{self.runs}" / "node/predictions/node_0"
        out.mkdir(parents=True)
        ns = self.harness(container, source)
        with patch.dict(os.environ, {"HGM_AGENT": kind}):
            ns["process_entry"](self.entry, out, "node_0", [], skip_existing=False, init_agent_path=str(HGM))
        saved = json.loads((out / "python__x.json").read_text())
        commands = [tuple(c) if isinstance(c, list) else c for c in container.commands]
        return saved, commands, (out / "task_records").exists()

    def test_switch_off_is_mains_code(self):
        def functions(text):
            tree = ast.parse(text)
            return {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        new, old = functions((HGM / "polyglot/harness.py").read_text()), functions(self.main)
        self.assertIn(DISPATCH, new["process_entry"])
        self.assertEqual(new["process_entry"].replace(DISPATCH, ""), old["process_entry"])
        for name in old:
            if name != "process_entry":
                self.assertEqual(new[name], old[name], name)

    def test_switch_off_same_commands_and_results(self):
        for kind in ("original", "littlecoder"):
            for name, args, files in self.SCENARIOS:
                with self.subTest(kind=kind, scenario=name), patch.dict(os.environ, {"HGM_TASK_RECORDS": "0"}):
                    old = self.run_once(self.main, kind, args, files)
                    new = self.run_once(None, kind, args, files)
                    self.assertEqual(new, old)
                    self.assertFalse(new[2], "no task_records folder when the switch is off")

    def test_switch_on_scores_like_main(self):
        for kind in ("original", "littlecoder"):
            for name, args, files in self.SCENARIOS:
                with self.subTest(kind=kind, scenario=name):
                    with patch.dict(os.environ, {"HGM_TASK_RECORDS": "0"}):
                        old, _, _ = self.run_once(self.main, kind, args, files)
                    with patch.dict(os.environ, {"HGM_TASK_RECORDS": "1"}):
                        new, _, has_cards = self.run_once(None, kind, args, files)
                    self.assertTrue(has_cards)
                    if kind == "littlecoder" and name == "no patch file":
                        # The one difference: main's process_entry_lc ignores cat's exit code and grades
                        # cat's error message as the patch; with records the attempt is "incomplete".
                        # Real attempts cannot reach it: hgm_agent.py writes the patch before rsi_attempt.json.
                        self.assertEqual(new["eval_result"], "incomplete")
                        self.assertIn(old["eval_result"], ("resolved", "unresolved"))
                        continue
                    self.assertEqual(new["eval_result"], old["eval_result"])
                    self.assertEqual(new["model_patch"], old["model_patch"])


if __name__ == "__main__":
    unittest.main()
