"""Benchmark output compression (edit ops) with and without Retrace on code-edit tasks.

Arms, all greedy in llama-cpp-python:
    full          model writes the whole edited file
    full_retrace  same, with Retrace drafting
    comp          model writes grammar-constrained edit ops (compress.py), the decompressor applies them;
                  if they cannot be applied, the run falls back to full_retrace and pays for both
    comp_retrace  same, with Retrace drafting the ops

Task sets: `dev` was used to design the ops prompt. `heldout` was written before any probe, uses indirect
phrasing, a new source file, and edits that need multi-line inserts or the rewrite escape.

Effective tok/s = tokens of the final code / decode time. Aggregates sum over every run, fallbacks and
wrong answers included. Outcomes split into compressed-correct, silent-wrong (ops applied, result wrong),
fallback-correct and fallback-wrong. Prompts use the model's own chat template and end-of-generation
tokens. The fixed prompt prefix of each format is cached in the KV cache, as a server would.
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

from llama_cpp import Llama, LlamaGrammar

import compress as cz
from bench import CODE, SYSTEM, Measured
from retrace_llama import RetraceDraft, calibrate_costs, chat_tokens, eog_ids

INVENTORY = '''class Inventory:
    """Track item counts and prices."""

    def __init__(self):
        self.items = {}
        self.prices = {}

    def add_item(self, name, count, price):
        if count <= 0:
            raise ValueError("count must be positive")
        self.items[name] = self.items.get(name, 0) + count
        self.prices[name] = price

    def remove_item(self, name, count):
        if name not in self.items:
            raise KeyError(name)
        self.items[name] -= count
        if self.items[name] <= 0:
            del self.items[name]
            del self.prices[name]

    def total_value(self):
        return sum(self.items[n] * self.prices[n] for n in self.items)

    def clear(self):
        self.items = {}
        self.prices = {}

    def names(self):
        return sorted(self.items)
'''


def rename(src, pairs):
    for old, new in pairs:
        src = re.sub(rf"\b{old}\b", new, src)
    return src


TASKS = {
    "rename_fn": (
        CODE,
        "Rename the function `process_data` to `summarize_rows` and the variable `kept` "
        "to `valid_rows` in the code below.",
        rename(CODE, [("process_data", "summarize_rows"), ("kept", "valid_rows")]),
    ),
    "change_value": (
        CODE,
        "In the code below, change the fallback value `0.0` to `None` everywhere it appears.",
        CODE.replace("0.0", "None"),
    ),
    "rename_class": (
        INVENTORY,
        "Rename the class `Inventory` to `Stock` and the attribute `items` to `counts` in the code below.",
        rename(INVENTORY, [("Inventory", "Stock"), ("items", "counts")]),
    ),
    "insert_lines": (
        INVENTORY,
        "In the code below, add `self.history = []` as the last line of `__init__`, "
        "and add `self.history.append(name)` as the last line of `add_item`.",
        INVENTORY.replace("self.prices = {}\n\n    def add_item", "self.prices = {}\n        self.history = []\n\n    def add_item")
        .replace("= price\n", "= price\n        self.history.append(name)\n"),
    ),
    "delete_method": (
        INVENTORY,
        "Remove the `clear` method from the code below.",
        INVENTORY.replace("    def clear(self):\n        self.items = {}\n        self.prices = {}\n\n", ""),
    ),
}


CONFIG = '''import json
import os

DEFAULT_PORT = 8080


def load_config(path):
    """Read a JSON config file and fill in defaults."""
    if not os.path.exists(path):
        return {"port": DEFAULT_PORT, "debug": False}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("port", DEFAULT_PORT)
    data.setdefault("debug", False)
    return data


def describe(config):
    mode = "debug" if config["debug"] else "release"
    return f"port {config['port']} in {mode} mode"


def main():
    config = load_config("app.json")
    print(describe(config))
'''

SKIP_ZERO_TEST = '''
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print_report([{"value": 0}, {"value": 3}], 0)
assert buf.getvalue().splitlines() == ["count: 2", "total: 3", "mean: 1.5", "largest: 3"], buf.getvalue()
assert process_data([{"value": 2}], 1)["count"] == 1
'''
DOC_TEST = '''
assert (Inventory.total_value.__doc__ or "").strip()
inv = Inventory()
inv.add_item("a", 2, 1.5)
assert inv.total_value() == 3.0
'''
GUARD_TEST = '''
inv = Inventory()
inv.add_item("a", 2, 1.0)
inv.remove_item("a", 0)
inv.remove_item("a", -1)
assert inv.items == {"a": 2}, inv.items
inv.remove_item("a", 1)
assert inv.items == {"a": 1}, inv.items
try:
    inv.remove_item("zz", 1)
except KeyError:
    pass
else:
    raise AssertionError("missing KeyError")
'''

# name -> (source, instruction, exact reference or None, behavioral test or None)
HELDOUT = {
    "rename_vague": (CONFIG, "`load_config` is a vague name. Call it `read_config` everywhere.",
                     rename(CONFIG, [("load_config", "read_config")]), None),
    "port_default": (CONFIG, "The service should listen on port 9090 by default instead of 8080.",
                     CONFIG.replace("8080", "9090"), None),
    "drop_unused": (INVENTORY, "The `names` helper is never used, so get rid of it.",
                    INVENTORY.replace("\n\n    def names(self):\n        return sorted(self.items)\n", "\n"), None),
    "print_count": (CODE, "Right before `process_data` builds its result dictionary, print how many rows "
                          "were kept using `print(len(kept))`.",
                    CODE.replace("    return {\n", "    print(len(kept))\n    return {\n"), None),
    "skip_zero": (CODE, "Make `print_report` skip entries whose value is 0.", None, SKIP_ZERO_TEST),
    "doc_total": (INVENTORY, "Document `total_value` with a docstring saying it returns the combined worth "
                             "of all stock.", None, DOC_TEST),
    "guard_remove": (INVENTORY, "`remove_item` should silently do nothing when `count` is zero or negative.",
                     None, GUARD_TEST),
}
DEV = {k: (src, task, ref, None) for k, (src, task, ref) in TASKS.items()}

# Second held-out set, written after the F4 diagnosis and before any probe of the scoped ops.
WORDS = r'''import re

STOP_WORDS = {"the", "a", "an", "and", "of"}
MIN_LENGTH = 3


def tokenize(text):
    """Split text into lowercase words."""
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if len(w) >= MIN_LENGTH]


def count_words(text):
    counts = {}
    for word in tokenize(text):
        if word in STOP_WORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts


def top_words(text, n):
    counts = count_words(text)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def format_table(rows):
    lines = []
    for word, count in rows:
        lines.append(f"{word:<12}{count:>5}")
    return "\n".join(lines)
'''

QUEUE = '''class TaskQueue:
    """A first-in first-out queue of named tasks with a retry limit."""

    def __init__(self, max_retries=3):
        self.tasks = []
        self.failed = []
        self.max_retries = max_retries

    def push(self, name):
        if not name:
            raise ValueError("name must not be empty")
        self.tasks.append({"name": name, "tries": 0})

    def pop(self):
        if not self.tasks:
            return None
        return self.tasks.pop(0)

    def retry(self, task):
        task["tries"] += 1
        if task["tries"] > self.max_retries:
            self.failed.append(task["name"])
            return False
        self.tasks.append(task)
        return True

    def size(self):
        return len(self.tasks)
'''

MIN_LEN_TEST = r'''
assert tokenize("cat dogs") == ["dogs"], tokenize("cat dogs")
assert count_words("cat dogs dogs") == {"dogs": 2}
'''
STOP_WITH_TEST = r'''
assert count_words("with cats with") == {"cats": 1}, count_words("with cats with")
assert count_words("dogs the dogs") == {"dogs": 2}
'''
TOP_GUARD_TEST = r'''
assert top_words("cats cats dogs", -1) == []
assert top_words("cats cats dogs", 0) == []
assert top_words("cats cats dogs", 1) == [("cats", 2)]
'''
RETRY_TEST = r'''
q = TaskQueue(max_retries=2)
t = {"name": "x", "tries": 0}
assert q.retry(t) is True
assert q.retry(t) is False
assert q.failed == ["x"]
'''
LOG_PUSH_TEST = r'''
import contextlib, io
q = TaskQueue()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    q.push("a")
    q.retry({"name": "b", "tries": 0})
assert buf.getvalue() == "a\n", repr(buf.getvalue())
assert q.size() == 2
'''
DOC_COUNT_TEST = r'''
assert (count_words.__doc__ or "").strip()
assert count_words("cats the cats") == {"cats": 2}
'''
TABLE_COMP_TEST = r'''
import ast
fn = next(n for n in ast.walk(ast.parse(__source__)) if isinstance(n, ast.FunctionDef) and n.name == "format_table")
assert any(isinstance(n, ast.ListComp) for n in ast.walk(fn)), "no list comprehension"
assert not any(isinstance(n, ast.For) for n in ast.walk(fn)), "still has a for loop"
rows = [("cats", 2), ("dogs", 10)]
assert format_table(rows) == "\n".join(f"{w:<12}{c:>5}" for w, c in rows)
'''

HELDOUT2 = {
    "min_len": (WORDS, "Ignore words shorter than four letters, not just those shorter than three.", None, MIN_LEN_TEST),
    "rename_pop": (QUEUE, "Call the `pop` method `next_task` instead, so it is not confused with `list.pop`.",
                   QUEUE.replace("def pop(self):", "def next_task(self):"), None),
    "stop_with": (WORDS, '"with" carries no meaning here, so skip it the way the other stop words are skipped.',
                  None, STOP_WITH_TEST),
    "top_guard": (WORDS, "`top_words` should give back an empty list when asked for zero or fewer words.",
                  None, TOP_GUARD_TEST),
    "retry_limit": (QUEUE, "A task should count as failed after `max_retries` tries, not one try later.",
                    None, RETRY_TEST),
    "log_push": (QUEUE, "Have `push` print the task name right before it adds the task to the queue.",
                 None, LOG_PUSH_TEST),
    "doc_count": (WORDS, "Explain in a docstring that `count_words` counts how often each word appears, "
                         "leaving out stop words.", None, DOC_COUNT_TEST),
    "table_comp": (WORDS, "Build the lines in `format_table` with a list comprehension instead of a loop.",
                   None, TABLE_COMP_TEST),
    "drop_table": (WORDS, "Nothing calls `format_table` any more. Take it out.",
                   WORDS[: WORDS.index("\n\n\ndef format_table")] + "\n", None),
}
# Tasks whose failure mode F4 targets (wrong scope, duplicate def, no insert-before). The rest test other edits.
F4_TARGETS = {"heldout": {"guard_remove", "print_count", "doc_total"},
              "heldout2": {"top_guard", "retry_limit", "log_push", "doc_count"}}
SETS = {"dev": DEV, "heldout": HELDOUT, "heldout2": HELDOUT2}

CODEC_SHA = hashlib.sha256(open(cz.__file__, "rb").read()).hexdigest()
FULL_SUFFIX = " Return the complete updated code only."
# Decode tok/s without drafting on an idle machine (bench/results_3b_retrace.json, copy_edit baseline).
IDLE_BASELINE_TPS = {"qwen2.5-3b-instruct-q4_k_m.gguf": 9.03}


def full_messages(task, src):
    return [("system", SYSTEM), ("user", cz.code_message(task + FULL_SUFFIX, src))]


def def_names(code):
    return sorted(q for q, _ in cz._scopes(code.splitlines()))


def judge(code, src, ref, test):
    """(correct, reason). References compare by AST, so comments and formatting do not count.

    Behavioral tasks must pass their test and keep the same function and class names, which catches a
    duplicate def nested inside the function it was meant to document. `__source__` holds the edited code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax_error"
    if ref is not None:
        return (True, None) if ast.dump(tree) == ast.dump(ast.parse(ref)) else (False, "ast_differs")
    with tempfile.TemporaryDirectory() as d:
        prog = f"__source__ = {code!r}\n{code}\n{test}"
        try:
            p = subprocess.run([sys.executable, "-I", "-c", prog], cwd=d, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode != 0:
        return False, "test_failed: " + p.stderr.decode(errors="replace").strip().splitlines()[-1][:200]
    if def_names(code) != def_names(src):
        return False, "defs_changed"
    return True, None


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def prime(llm, prefix):
    """Leave exactly `prefix` in the KV cache so generate() only evaluates the rest of the prompt."""
    n = len(prefix)
    if n and llm.n_tokens >= n and list(llm.input_ids[:n]) == prefix:
        llm._ctx.kv_cache_seq_rm(-1, n, -1)
        llm.n_tokens = n
    else:
        llm.reset()
        if n:
            llm.eval(prefix)


def run(llm, draft, tokens, prefix, stop_ids, max_new, grammar=None):
    llm.draft_model = draft
    if draft is not None:
        draft.reset_counts()
    prime(llm, prefix)
    out, t0, t_first = [], time.perf_counter(), None
    for tok in llm.generate(tokens, temp=0.0, repeat_penalty=1.0, reset=True, grammar=grammar):
        if t_first is None:
            t_first = time.perf_counter()
        if tok in stop_ids:
            break
        out.append(tok)
        if len(out) >= max_new:
            break
    t_end = time.perf_counter()
    st = {"gen_tokens": len(out), "prefill_s": t_first - t0, "decode_s": t_end - t_first, "e2e_s": t_end - t0,
          "drafted": draft.drafted if draft else 0, "accepted": draft.accepted if draft else 0}
    return llm.detokenize(out).decode(errors="replace"), st


def add(a, b):
    return {k: a[k] + b[k] for k in a}


def summarize(runs, model):
    """Per task/arm rows and per set/arm aggregates over every run."""
    rows, agg = {}, {}
    key = lambda r: (r["set"], r["arm"])
    groups = {}
    for r in runs:
        groups.setdefault(key(r), []).append(r)
        rows.setdefault(f"{r['set']}/{r['task']}/{r['arm']}", []).append(r)
    per_task = {}
    for k, rs in rows.items():
        per_task[k] = {
            "outcomes": [r["outcome"] for r in rs],
            "eff_tps": round(sum(r["eff_tokens"] for r in rs) / sum(r["decode_s"] for r in rs), 2),
            "e2e_s": round(sum(r["e2e_s"] for r in rs) / len(rs), 1),
            "gen_tokens": round(statistics.mean(r["gen_tokens"] for r in rs), 1),
        }
    idle = IDLE_BASELINE_TPS.get(os.path.basename(model))
    for (s, arm), rs in groups.items():
        count = lambda o: sum(r["outcome"] == o for r in rs)
        agg[f"{s}/{arm}"] = {
            "runs": len(rs),
            "correct": sum(r["correct"] for r in rs),
            "correct_f4_targets": f'{sum(r["correct"] for r in rs if r["f4_target"])}/{sum(r["f4_target"] for r in rs)}',
            "compressed_correct": count("compressed_correct"),
            "silent_wrong": count("silent_wrong"),
            "fallback_correct": count("fallback_correct"),
            "fallback_wrong": count("fallback_wrong"),
            "eff_tps": round(sum(r["eff_tokens"] for r in rs) / sum(r["decode_s"] for r in rs), 2),
            "e2e_s_total": round(sum(r["e2e_s"] for r in rs), 1),
        }
    for k, row in agg.items():
        s = k.split("/")[0]
        base = agg.get(f"{s}/full")
        if base:
            row["speedup_decode"] = round(row["eff_tps"] / base["eff_tps"], 2)
            row["speedup_e2e"] = round(base["e2e_s_total"] / row["e2e_s_total"], 2)
            if idle:
                row["idle_equiv_tps"] = round(row["speedup_decode"] * idle, 1)
    return per_task, agg, idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arms", default="full,full_retrace,comp,comp_retrace")
    ap.add_argument("--sets", default="dev,heldout")
    ap.add_argument("--tasks", help="comma-separated task names to keep")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--no-cache-prefix", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arm_names = args.arms.split(",")
    work = [(s, t) for s in args.sets.split(",") for t in SETS[s]]
    if args.tasks:
        keep = set(args.tasks.split(","))
        work = [w for w in work if w[1] in keep]

    llm = Llama(model_path=args.model, n_ctx=3072, n_threads=args.threads, n_threads_batch=args.threads,
                logits_all=True, verbose=False)
    costs = calibrate_costs(llm, args.k_max)
    print("verify cost by tokens per pass:", [round(c, 2) for c in costs], flush=True)
    stop_ids = eog_ids(llm)
    grammar = LlamaGrammar.from_string(cz.GRAMMAR, verbose=False)
    count_tokens = lambda s: len(llm.tokenize(s.encode(), add_bos=False))
    new_draft = lambda: Measured(RetraceDraft(k_max=args.k_max, costs=costs))
    drafts = {a: new_draft() if a.endswith("_retrace") else None for a in arm_names}
    fallback_draft = new_draft()
    builders = {"full": full_messages, "comp": cz.compressed_messages}
    prompts = {(f, t): chat_tokens(llm, b(SETS[s][t][1], SETS[s][t][0]))
               for f, b in builders.items() for s, t in work}
    two = [chat_tokens(llm, b("Fix it.", CODE)) for b in builders.values()], \
          [chat_tokens(llm, b("Change it.", INVENTORY)) for b in builders.values()]
    prefixes = {f: [] if args.no_cache_prefix else common_prefix(two[0][i], two[1][i])
                for i, f in enumerate(builders)}
    print("cached prefix tokens:", {f: len(p) for f, p in prefixes.items()}, flush=True)

    # Warm up every path on an edit example that is not a benchmark task.
    warm = {f: chat_tokens(llm, b(cz._EXAMPLES[1][0], cz._EX_SOURCE)) for f, b in builders.items()}
    for arm in arm_names:
        f = arm.split("_")[0]
        run(llm, drafts[arm], warm[f], prefixes[f], stop_ids, 48, grammar if f == "comp" else None)
    run(llm, fallback_draft, warm["full"], prefixes["full"], stop_ids, 48)

    runs = []
    for rep in range(args.reps):
        order = [(s, t, a) for s, t in work for a in arm_names]
        random.Random(args.seed + rep).shuffle(order)
        for s, t, arm in order:
            src, _, ref, test = SETS[s][t]
            f = arm.split("_")[0]
            text = None
            if f == "full":
                text, st = run(llm, drafts[arm], prompts[("full", t)], prefixes["full"], stop_ids, 512)
                code, err, ops_text = cz.extract_code(text), None, None
            else:
                ops_text, st = run(llm, drafts[arm], prompts[("comp", t)], prefixes["comp"], stop_ids, 200, grammar)
                try:
                    code, err = cz.decompress(ops_text, src), None
                except cz.ApplyError as e:
                    err = str(e)
                    text, st2 = run(llm, fallback_draft, prompts[("full", t)], prefixes["full"], stop_ids, 512)
                    st = add(st, st2)
                    code = cz.extract_code(text)
            code = cz.normalize(code)
            ok, why = judge(code, src, ref, test)
            if f == "full":
                outcome = "full_correct" if ok else "full_wrong"
            elif err is None:
                outcome = "compressed_correct" if ok else "silent_wrong"
            else:
                outcome = "fallback_correct" if ok else "fallback_wrong"
            st.update(rep=rep, set=s, task=t, arm=arm, eff_tokens=count_tokens(code), correct=ok, outcome=outcome, why=why,
                      f4_target=t in F4_TARGETS.get(s, ()),
                      apply_error=err, ops=ops_text, full_text=text, code=code)
            runs.append(st)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()
                              if k not in ("code", "ops", "full_text")}), flush=True)
            if ops_text is not None and rep == 0:
                print(ops_text, flush=True)

    per_task, agg, idle = summarize(runs, args.model)
    with open(args.out, "w") as fh:
        json.dump({"args": vars(args), "compress_sha256": CODEC_SHA, "costs": costs, "runs": runs, "per_task": per_task, "aggregate": agg},
                  fh, indent=2)
    print("\nPER TASK (eff_tps = final code tokens / decode s, fallbacks included)")
    for k, row in per_task.items():
        print(f"{k:40s} {row}")
    print("\nAGGREGATE (sums over every run; speedups vs `full` on the same set)")
    if idle:
        print(f"idle_equiv_tps = speedup_decode x {idle} tok/s idle baseline")
    for k, row in agg.items():
        print(f"{k:22s} {row}")


if __name__ == "__main__":
    main()
