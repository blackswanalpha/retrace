# Retrace

Retrace is a lossless n-gram drafter for speculative decoding. It is written in Rust, exposed to Python through pyo3, and plugs into llama-cpp-python as a `LlamaDraftModel`. It drafts tokens by looking up the recent context in an n-gram index of the current request. It picks the draft length from measured acceptance rates and the measured cost of verifying k tokens in one pass. Greedy output is unchanged, because every drafted token is verified by the target model.

The repo also holds an output-compression experiment. Instead of rewriting a whole file, the model writes short, grammar-constrained edit ops, and a decompressor applies them to the source. The goal was 30 tok/s effective on Qwen2.5-3B Q4_K_M on 4 CPU threads.

## Results

All numbers come from single runs on a shared, often loaded machine. Ratios within one run hold, but absolute tok/s does not.

**Retrace, copy-edit workload** (decode speedup over no drafting, output identical to baseline):

| Model | Speedup | Source |
|---|---|---|
| Qwen2.5-0.5B | 1.71x | `bench/results_smoke_0.5b.json` |
| Qwen2.5-3B | 1.22x to 1.57x | `bench/results_3b_retrace.json`, `bench/results_smoke_3b.json` |
| Llama-3.2-1B | 1.19x | `bench/results_smoke_llama1b.json` |

A side-by-side run with per-token timings is in `bench/results_race_3b.json` (`bench/race_capture.py`): 46.4 s without Retrace, 36.7 s with it, the same 214 tokens in all six runs. The same capture on Qwen2.5-0.5B (`bench/results_race_0.5b.json`) shows the numerics caveat below: the Retrace output diverged at token 172.

On free-form generation (code_gen, chat) Retrace is within about 0.94x to 1.25x, because there is little to copy. The ceiling is near 1.6x, since verifying 9 tokens in one pass costs about 6 to 8 single-token passes in llama-cpp-python.

**Output compression on Qwen2.5-3B:**
- Correct compressed edits decoded at 20 to 143 tok/s effective. Ops ran 7 to 48 tokens for files of about 150 to 220 tokens. On the dev tasks, aggregate effective decode was 61.7 tok/s against 6.1 for a full rewrite.
- On held-out tasks the format did not hold up. In the first held-out set, 3 of 7 compressed edits applied cleanly but were wrong.
- Safety checks (scoped ops, rejecting inserts that redefine a function, rejecting edits that read undefined names) cut silent wrong edits from 5 to 2 across 21 tasks. See `bench/results_f4_*` and the control run in `bench/results_pref4_control_heldout2_3b.json`.
- On 9 unseen tasks the model chose `rewrite` on 5 under both the old and new formats. Most requests then paid for ops plus a full rewrite. Compression ended up 0.47x to 0.72x of a plain rewrite end to end. Full rewrite with Retrace was the fastest arm there, at 1.47x decode and 1.38x end to end.

30 tok/s was reached on decode for correct compressed edits, but not end to end, and not on unseen tasks.

## Layout

- `retrace/` is the Rust crate: n-gram index, acceptance stats and the draft-length choice (`src/lib.rs`).
- `bench/retrace_llama.py` is the llama-cpp-python adapter (`RetraceDraft`), with verify-cost calibration and chat-template helpers that read the template and end-of-generation tokens from the GGUF.
- `bench/bench.py` compares Retrace with no drafting and with the built-in prompt-lookup drafter.
- `bench/compress.py` holds the edit-op format, grammar and decompressor. `bench/compress_pre_f4.py` is the earlier version, kept as a control.
- `bench/compress_bench.py` runs the four-arm compression benchmark on the dev and held-out task sets.
- `worknn.md` is the optimization plan.

## Run

```bash
python -m venv .venv
.venv/bin/pip install llama-cpp-python numpy maturin
cd retrace && ../.venv/bin/maturin develop --release && cd ..

# Retrace vs baseline vs built-in prompt lookup
.venv/bin/python bench/bench.py --model models/qwen2.5-3b-instruct-q4_k_m.gguf --reps 3 --out bench/results_3b_retrace.json

# Output compression, four arms
cd bench && ../.venv/bin/python compress_bench.py --model ../models/qwen2.5-3b-instruct-q4_k_m.gguf --sets heldout2 --out results.json
```

Minimal use, run from `bench/`:

```python
from llama_cpp import Llama
from retrace_llama import RetraceDraft, calibrate_costs, chat_tokens, eog_ids

llm = Llama(model_path="../models/qwen2.5-3b-instruct-q4_k_m.gguf", n_threads=4, logits_all=True)
llm.draft_model = RetraceDraft(k_max=8, costs=calibrate_costs(llm, 8))
stop = eog_ids(llm)
for tok in llm.generate(chat_tokens(llm, [("user", "Repeat: the cat sat on the mat.")]), temp=0.0):
    if tok in stop:
        break
```

Models are not included. Any GGUF with a chat template works.

## Limits

- Retrace runs only inside llama-cpp-python, not llama-server.
- In llama-cpp-python, batched and sequential evaluation differ slightly in numerics, so long free-form outputs can diverge from the no-drafting baseline. Copy-edit outputs matched exactly.

## License

MIT. See `LICENSE`.
