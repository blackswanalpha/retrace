"""Record per-token arrival times for the copy_edit workload, with and without Retrace, for the demo video."""
import json
import sys
import time

from llama_cpp import Llama

from bench import SYSTEM, WORKLOADS
from retrace_llama import RetraceDraft, calibrate_costs, chat_tokens, eog_ids

model, out_path, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
llm = Llama(model_path=model, n_ctx=2048, n_threads=4, n_threads_batch=4, logits_all=True, verbose=False)
costs = calibrate_costs(llm, 8)
stop = eog_ids(llm)
prompt = chat_tokens(llm, [("system", SYSTEM), ("user", WORKLOADS["copy_edit"])])
draft = RetraceDraft(k_max=8, costs=costs)


def run(use):
    llm.draft_model = draft if use else None
    llm.reset()
    toks, times, t_first = [], [], None
    for tok in llm.generate(prompt, temp=0.0, repeat_penalty=1.0, reset=True):
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        if tok in stop or len(toks) >= 320:
            break
        toks.append(tok)
        times.append(now - t_first)
    return {"tokens": toks, "times": times, "pieces": [llm.detokenize([t]).decode(errors="replace") for t in toks]}


run(False); run(True)  # warm up
runs = []
for rep in range(reps):
    for use in (False, True):
        r = run(use)
        r.update(rep=rep, retrace=use, tps=(len(r["tokens"]) - 1) / r["times"][-1])
        runs.append(r)
        print(rep, "retrace" if use else "baseline", len(r["tokens"]), round(r["times"][-1], 2), "s", round(r["tps"], 2), "tok/s", flush=True)
json.dump({"model": model.split("/")[-1], "costs": costs, "runs": runs}, open(out_path, "w"))
