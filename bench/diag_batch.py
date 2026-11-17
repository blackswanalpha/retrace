"""Check whether evaluating tokens in one batch changes greedy picks versus one-at-a-time decoding.

Generates the baseline greedily, then re-evaluates prompt + output in a single batch and
reports positions where the batched argmax differs, with the top-2 logit margin there.
"""

import sys

import numpy as np
from llama_cpp import Llama

sys.path.insert(0, sys.argv[2])
from bench import WORKLOADS, chat_prompt  # noqa: E402

llm = Llama(model_path=sys.argv[1], n_ctx=2048, n_threads=4, n_threads_batch=4, logits_all=True, verbose=False)
stop = {llm.token_eos(), *llm.tokenize(b"<|im_end|><|endoftext|>", add_bos=False, special=True)}

for name, text in WORKLOADS.items():
    prompt = llm.tokenize(chat_prompt(text).encode(), add_bos=False, special=True)
    llm.reset()
    out = []
    for tok in llm.generate(prompt, temp=0.0, repeat_penalty=1.0, reset=True):
        if tok in stop or len(out) >= 256:
            break
        out.append(tok)
    # Single-token reference logits: re-run one token at a time.
    llm.reset()
    llm.eval(prompt)
    seq_logits = [np.array(llm.scores[llm.n_tokens - 1])]
    for tok in out[:-1]:
        llm.eval([tok])
        seq_logits.append(np.array(llm.scores[llm.n_tokens - 1]))
    # Batched logits: prompt + all output in one call.
    llm.reset()
    llm.eval(prompt + out)
    p = len(prompt)
    bad = []
    for i, tok in enumerate(out):
        b = np.array(llm.scores[p - 1 + i])
        s = seq_logits[i]
        top2 = np.sort(s)[-2:]
        if int(np.argmax(s)) != tok:
            bad.append((i, "seq-mismatch", float(top2[1] - top2[0])))
        elif int(np.argmax(b)) != tok:
            bad.append((i, "batch-flip", float(top2[1] - top2[0])))
    maxdiff = max(float(np.max(np.abs(np.array(llm.scores[p - 1 + i]) - seq_logits[i]))) for i in range(len(out)))
    print(f"{name}: {len(out)} tokens, max |batched - sequential| logit diff = {maxdiff:.4f}")
    for i, kind, margin in bad[:5]:
        print(f"   pos {i}: {kind}, sequential top-2 margin = {margin:.4f}")
