"""Benchmark Retrace against no drafting and llama-cpp-python's built-in prompt lookup.

All arms run in the same engine with greedy decoding, so outputs should match the baseline.
Decode speed excludes model load and prompt prefill.
"""

import argparse
import json
import statistics
import time

import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaDraftModel, LlamaPromptLookupDecoding

from retrace_llama import RetraceDraft, calibrate_costs, chat_tokens, eog_ids

CODE = '''def process_data(rows, threshold):
    """Filter rows and compute summary statistics."""
    kept = []
    for row in rows:
        if row["value"] is None:
            continue
        if row["value"] < threshold:
            continue
        kept.append(row)
    total = 0
    for row in kept:
        total += row["value"]
    mean = total / len(kept) if kept else 0.0
    largest = max((row["value"] for row in kept), default=0.0)
    smallest = min((row["value"] for row in kept), default=0.0)
    return {
        "count": len(kept),
        "total": total,
        "mean": mean,
        "largest": largest,
        "smallest": smallest,
    }


def print_report(rows, threshold):
    summary = process_data(rows, threshold)
    for key, value in summary.items():
        print(f"{key}: {value}")
'''

WORKLOADS = {
    "copy_edit": (
        "Rename the function `process_data` to `summarize_rows` and the variable `kept` "
        "to `valid_rows` in the code below. Return the complete updated code only.\n\n"
        f"```python\n{CODE}```"
    ),
    "code_gen": (
        "Write a Python function that reads a CSV file and returns the average of every "
        "numeric column as a dictionary. Include a docstring and type hints."
    ),
    "chat": "Explain to a high school student why the sky is blue and why sunsets are red.",
}


SYSTEM = "You are a helpful assistant."


def chat_prompt(user: str) -> str:
    """Qwen ChatML text for llama-server raw prompts. bench.py itself uses the model's template."""
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )


class Measured(LlamaDraftModel):
    """Wraps a drafter and counts drafted and accepted tokens from the next history."""

    def __init__(self, inner):
        self.inner = inner
        self.reset_counts()

    def reset_counts(self):
        self.drafted = 0
        self.accepted = 0
        self.passes_with_draft = 0
        self._last = None

    def __call__(self, input_ids, **kwargs):
        ids = np.asarray(input_ids)
        if self._last is not None:
            n_prev, draft = self._last
            if len(ids) > n_prev:
                new = ids[n_prev:]
                m = min(len(new), len(draft))
                mism = np.nonzero(new[:m] != draft[:m])[0]
                self.accepted += int(mism[0]) if len(mism) else m
        draft = np.asarray(self.inner(ids), dtype=np.intc)
        if len(draft):
            self.drafted += len(draft)
            self.passes_with_draft += 1
            self._last = (len(ids), draft)
        else:
            self._last = None
        return draft


def run_once(llm, draft, prompt_tokens, stop_ids, max_new):
    llm.draft_model = draft
    if draft is not None:
        draft.reset_counts()
    llm.reset()
    out, t_first = [], None
    for tok in llm.generate(prompt_tokens, temp=0.0, repeat_penalty=1.0, reset=True):
        if t_first is None:
            t_first = time.perf_counter()
        if tok in stop_ids:
            break
        out.append(tok)
        if len(out) >= max_new:
            break
    t_end = time.perf_counter()
    tps = (len(out) - 1) / (t_end - t_first) if len(out) > 1 else 0.0
    stats = {"tokens": len(out), "tps": tps}
    if draft is not None:
        stats.update(drafted=draft.drafted, accepted=draft.accepted)
    return out, stats


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=8)
    args = ap.parse_args()

    llm = Llama(
        model_path=args.model,
        n_ctx=2048,
        n_threads=args.threads,
        n_threads_batch=args.threads,
        logits_all=True,  # per-token logits for verification, same for every arm
        verbose=False,
    )
    costs = calibrate_costs(llm, args.k_max)
    print("verify cost by tokens per pass:", [round(c, 2) for c in costs], flush=True)
    arms = {
        "baseline": None,
        "builtin_pld": Measured(LlamaPromptLookupDecoding()),
        "retrace": Measured(RetraceDraft(min_n=args.min_n, max_n=args.max_n, k_max=args.k_max, costs=costs)),
    }
    stop_ids = eog_ids(llm)
    chat = lambda text: chat_tokens(llm, [("system", SYSTEM), ("user", text)])
    prompts = {name: chat(text) for name, text in WORKLOADS.items()}

    warm = chat("Say hello.")
    for draft in arms.values():
        run_once(llm, draft, warm, stop_ids, 16)

    runs = []
    for rep in range(args.reps):
        for wl, toks in prompts.items():
            base_out = None
            for arm, draft in arms.items():
                out, st = run_once(llm, draft, toks, stop_ids, args.max_new)
                if arm == "baseline":
                    base_out = out
                    text = llm.detokenize(out).decode(errors="replace")
                st.update(rep=rep, workload=wl, arm=arm, first_diff=first_diff(out, base_out))
                runs.append(st)
                print(json.dumps(st), flush=True)
            if rep == 0:
                print(f"--- {wl} baseline output (first 300 chars) ---\n{text[:300]}\n", flush=True)

    summary = {}
    for wl in prompts:
        base = statistics.median(r["tps"] for r in runs if r["workload"] == wl and r["arm"] == "baseline")
        for arm in arms:
            rs = [r for r in runs if r["workload"] == wl and r["arm"] == arm]
            tps = statistics.median(r["tps"] for r in rs)
            row = {"tps": round(tps, 2), "speedup": round(tps / base, 2)}
            if arm != "baseline":
                drafted = sum(r["drafted"] for r in rs)
                accepted = sum(r["accepted"] for r in rs)
                row["accept_rate"] = round(accepted / drafted, 3) if drafted else None
            row["lossless"] = all(r["first_diff"] is None for r in rs)
            summary[f"{wl}/{arm}"] = row

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "costs": costs, "runs": runs, "summary": summary}, f, indent=2)
    print("\nSUMMARY (median decode tok/s)")
    for key, row in summary.items():
        print(f"{key:24s} {row}")


if __name__ == "__main__":
    main()
