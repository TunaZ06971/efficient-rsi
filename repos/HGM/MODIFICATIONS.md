# Modifications to HGM

This directory is the Huxley-Gödel Machine code from <https://github.com/metauto-ai/HGM>,
commit `013872d`, released under the Apache License 2.0 (`LICENSE`, unchanged). The
efficient-RSI project changed the files listed below. `EFFICIENT_RSI_CHANGES.patch` holds the
full diff of the modified files against that commit; the added files are new. The
`HGM_HIDE_ANSWERS` change and the patch-file race fix came later and are not in that patch; they
are in this repository's git history.

Every change that alters behaviour is off unless a flag or environment variable turns it on, so
the code still runs as upstream HGM by default, except for the `tree.py` fix (see below).

## Modified files

| File | Change |
| --- | --- |
| `tree.py` | Fix: `get_pseudo_decendant_evals` returned a node's own evaluation list, which the caller then extended in place, mixing descendants' results into the node's record. It now returns a copy. `HGM_FIX_DESCENDANT_EVALS=0` restores the upstream behaviour. |
| `hgm.py` | Cost-ledger events; `--search dgm` runs DGM's search policy on this harness (`dgm_search.py`); `--max_hours` stops a search after a number of hours; optional node stopping from `erl/online_stop.py` (`HGM_ERL_STOP=1`, `HGM_ERL_EXPAND=1`). |
| `hgm_utils.py` | Cost-ledger timing of proposals and evaluations; host-side diagnose calls tagged for the metering proxy. Fix: for Polyglot, `sample_child` filters a new child's `model_patch.diff` once (`remove_patch_by_files`) before the child joins the tree, so evaluations no longer need to rewrite it (see `polyglot/harness.py`). |
| `self_improve_step.py` | Diagnose calls tagged for the metering proxy; with `HGM_DIAG_CLIP=1`, a diagnose prompt still over the length limit is cut in the middle, keeping its answer format, and each retry lowers the limit. |
| `prompts/self_improvement_prompt.py` | With `HGM_DIAG_CLIP=1`, clips the parts of the diagnose prompt that upstream leaves unbounded (code, solution patch, test log); upstream's own clipping lines for them are commented out. |
| `llm.py` | When `RSI_PROXY_PREFIX` is set (metering on), vLLM and OpenRouter calls go through the metering proxy; requests are forwarded unchanged. |
| `polyglot/harness.py` (race fix) | Fix: upstream `harness()` filtered every patch of the agent's lineage and wrote it back over the shared file with `open(path, "w")`, which truncates it first. HGM evaluates one agent in several threads at once, so a thread could read a just-truncated file and write it back empty, and the agent then ran as its parent without any sign of it (in v1: 8 first-generation patches emptied, among them hgm-fixed-r1's final agent). `harness()` now filters each patch into a private temporary copy for that call and only reads the shared files; the upstream body is `_harness()`. Test: `instrumentation/tests/test_patch_copies.py`. Always on: the filter is idempotent, so only the race changes. |
| `polyglot/harness.py`, `swe_bench/harness.py` | Cost-ledger timing of each evaluation phase; containers get the proxy address. In `polyglot/harness.py`, with `HGM_HIDE_ANSWERS=1`: once the agent has finished, every process it left in the container is killed, and the commit with the tests is fetched from the task's repo on the host before the usual grading steps. |
| `polyglot/test_spec.py` | With `HGM_HIDE_ANSWERS=1`, the instance image no longer gives the agent the answers. Upstream, the image holds the original task folder (`/polyglot/...`, with its tests and the `.meta/` reference solution), and `/testbed`'s git history keeps the commit with all files (`origin/HEAD`, `ORIG_HEAD`), so `git log --all` or `find /` finds them. The setup script now drops those refs, prunes the unreachable objects and deletes the task folder. These images are named `pb.eval.<arch>.<task>.hidden`, so upstream images are never reused for them. Still open: the agent's container keeps network access (each evaluation `pip install`s the agent's requirements), so the public benchmark could be downloaded. |
| `.dockerignore` | Keeps the added orchestration files out of the agent's container image. |

## Added files

| File | Purpose |
| --- | --- |
| `dgm_search.py` | DGM's search policy (parent selection, staged evaluation, archive) on HGM's harness, so DGM and HGM differ only in how they search. |
| `final_eval.py` | Picks a search's final agent at an equal-budget cut and evaluates it on held-out tasks. |
| `repeat_eval.py` | Repeated evaluations of one agent (noise and test-time-scaling experiments). |
| `config_local.yaml` | Configuration for one local Qwen3-Coder-30B-A3B model in every role. |
| `rsi_ledger.py` | Link to `../../instrumentation/rsi_ledger.py`, the cost-ledger hooks. |
| `MODIFICATIONS.md`, `EFFICIENT_RSI_CHANGES.patch` | This notice and the diff. |

## Later changes to shared files (PR #3 harness-rsi, PR #4 task records and SIFT)

Both PRs are off by default. With their switches unset, every file below behaves as before them,
apart from the additions listed in the last column (code paths that never run, extra keys).

| File | PR | Change | Switch | With the switch unset |
| --- | --- | --- | --- | --- |
| `hgm.py` | #3 | With coupled tasks, a child is evaluated first on tasks its parent has a record of. | `HGM_AGENT=littlecoder`, `HGM_COUPLED_TASKS=1` | as before |
| `hgm.py` | #4 | Startup checks of the switches (`sift.validate_startup`); the SIFT gate after a child is proposed; Qwen's score ranks children in Thompson sampling as `SIFT_ORDER_WEIGHT` imaginary tasks. | `SIFT_MODE`, `SIFT_ENFORCE=1` | as before: the gate returns at once, every prior is (0, 0) |
| `dgm_search.py` | #4 | The same startup checks and SIFT gate (no ordering). | `SIFT_MODE` | as before |
| `hgm_utils.py` | #3 | `sample_child_lc`: little-coder proposes changes to its own harness (`lc_integration.py`). | `HGM_AGENT=littlecoder` | as before |
| `hgm_utils.py` | #4 | `sift_child` (the SIFT gate); `choose_entry` can pick the focus task by failure class from the task records. | `SIFT_MODE`; `HGM_TASK_RECORDS=1` + `HGM_RECORD_FEEDBACK=1` | as before |
| `polyglot/harness.py` | #3 | `process_entry_lc`: one evaluation of little-coder, with visible read-only tests and coupled replay. | `HGM_AGENT=littlecoder` | as before |
| `polyglot/harness.py` | #4 | `_process_entry_records`, `_process_entry_lc_records`: the same evaluations with a task record card (`task_records.py`). Their grading lines are copied from `process_entry` / `process_entry_lc`, and they record "incomplete" in the same cases. | `HGM_TASK_RECORDS=1` | `process_entry` and `process_entry_lc` are the code from before PR #4, with one dispatch line added; `_score_in_container` is unchanged |
| `self_improve_step.py`, `prompts/self_improvement_prompt.py` | #4 | The diagnose prompt starts with a summary of the task records, and its wording for missing results says "not collected" instead of "assume failed". | `HGM_TASK_RECORDS=1` + `HGM_RECORD_FEEDBACK=1` | as before |
| `utils/evo_utils.py` | #4 | `metadata.json` gets two more lists, `total_incomplete_ids` and `total_error_ids`. | none | extra keys only; nothing reads them unless `HGM_RECORD_FEEDBACK=1` |
| `lc_integration.py` | #4 | A record cut short by a timeout or crash is not replayed; the proposer workspace gets the task records and the card's failure class. | `HGM_TASK_RECORDS=1` (+ `HGM_RECORD_FEEDBACK=1` for the workspace) | as in PR #3 |
| `lc/hgm_agent.py` | #4 | `rsi_attempt.json` also reports why the attempt stopped (`stop_reason`, `max_turns`); with records on, the process exits with `main()`'s code instead of always 0. | `HGM_TASK_RECORDS=1` for the exit code | always exits 0 as in PR #3; three extra keys in `rsi_attempt.json` |
| `.dockerignore` | #3, #4 | Keeps the host-side files out of agent containers: `lc_integration.py`, `config_local_lc.yaml`, `scripts/`, `HARNESS_RSI.md` (#3); `sift.py`, `task_records.py`, `SIFT_AND_RECORDS.md` (#4). The section labels, which were `<!-- -->` lines (patterns that match nothing), are now `#` comments. | none | agent containers hold the same files as at PR #3; PR #3 added `lc/` and `config_local_lc_qwen35.yaml` to them |

Added files: `lc/`, `lc_integration.py`, `HARNESS_RSI.md`, `config_local_lc*.yaml`, `scripts/lc_smoke.py`,
`scripts/vendor_littlecoder.sh` (PR #3); `task_records.py`, `sift.py`, `SIFT_AND_RECORDS.md` (PR #4).
