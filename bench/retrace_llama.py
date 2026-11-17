"""llama-cpp-python integration for Retrace: verify-cost calibration and the draft-model adapter."""

import statistics
import time

import numpy as np
import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from llama_cpp.llama_speculative import LlamaDraftModel

import retrace


def eog_ids(llm: Llama) -> set[int]:
    """Every end-of-generation token in the model's vocab (eos, eot, im_end, ...)."""
    vocab = llm._model.vocab
    return {t for t in range(llm.n_vocab()) if llama_cpp.llama_vocab_is_eog(vocab, t)}


def chat_tokens(llm: Llama, messages: list[tuple[str, str]]) -> list[int]:
    """Render (role, text) messages with the model's own chat template and tokenize them.

    Falls back to ChatML when the GGUF has no template. Templates that reject a system
    role get the system text prepended to the first user turn.
    """
    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        body = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in messages)
        return llm.tokenize((body + "<|im_start|>assistant\n").encode(), add_bos=True, special=True)
    m = llm._model
    fmt = Jinja2ChatFormatter(template, eos_token=m.token_get_text(m.token_eos()),
                              bos_token=m.token_get_text(m.token_bos()))
    msgs = [{"role": r, "content": t} for r, t in messages]
    try:
        prompt = fmt(messages=msgs).prompt
    except Exception:
        if msgs[0]["role"] != "system":
            raise
        sys_text, rest = msgs[0]["content"], [dict(x) for x in msgs[1:]]
        rest[0]["content"] = f"{sys_text}\n\n{rest[0]['content']}"
        prompt = fmt(messages=rest).prompt
    return llm.tokenize(prompt.encode(), add_bos=False, special=True)


def calibrate_costs(llm: Llama, k_max: int, trials: int = 5) -> list[float]:
    """Measure time to evaluate 1..k_max+1 tokens in one pass, relative to 1 token."""
    ctx = llm.tokenize(b"The quick brown fox jumps over the lazy dog. " * 30, add_bos=False)
    llm.reset()
    llm.eval(ctx)
    base = llm.n_tokens
    times = {}
    for k in range(1, k_max + 2):
        ts = []
        for _ in range(trials):
            llm.n_tokens = base
            llm._ctx.kv_cache_seq_rm(-1, base, -1)
            t = time.perf_counter()
            llm.eval(ctx[:k])
            ts.append(time.perf_counter() - t)
        times[k] = statistics.median(ts)
    llm.reset()
    one = times[1]
    # Costs must not decrease with k: verifying more tokens is never cheaper.
    costs, running = [], 0.0
    for k in range(1, k_max + 2):
        running = max(running, times[k] / one)
        costs.append(running)
    return costs


class RetraceDraft(LlamaDraftModel):
    def __init__(self, **kw):
        self.core = retrace.Retrace(**kw)

    def __call__(self, input_ids, **kwargs):
        data = np.ascontiguousarray(input_ids, dtype=np.int32).tobytes()
        return np.array(self.core.draft(data), dtype=np.intc)
