"""Time one target-model pass that evaluates k tokens, relative to k=1, at a fixed context."""
import sys, time, statistics
from llama_cpp import Llama
llm = Llama(model_path=sys.argv[1], n_ctx=2048, n_threads=4, n_threads_batch=4, logits_all=True, verbose=False)
ctx = llm.tokenize(b"The quick brown fox jumps over the lazy dog. " * 30, add_bos=False)
llm.eval(ctx)
base_n = llm.n_tokens
res = {}
for rnd in range(3):
    for k in (1, 2, 3, 4, 6, 8):
        ts = []
        for _ in range(8):
            llm.n_tokens = base_n
            llm._ctx.kv_cache_seq_rm(-1, base_n, -1)
            t = time.perf_counter(); llm.eval(ctx[:k]); ts.append(time.perf_counter() - t)
        res.setdefault(k, []).extend(ts)
one = statistics.median(res[1])
for k, ts in res.items():
    m = statistics.median(ts)
    print(f"k={k}: {m*1000:6.1f} ms  ({m/one:.2f}x of k=1)")
