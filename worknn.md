# Retrace optimization plan

Target: Qwen2.5-3B Q4_K_M, 4 threads, greedy. Rival: llama-server b12387 `--spec-type ngram-mod`.
The plan covers the four known limits L1 to L4. M1 to M3 are benchmark findings that change how L1 to L4 should be ranked and measured.

## Benchmark findings

### M1. ngram-mod's lead comes from replaying repeated prompts

Every rep sends the same greedy prompt, and the summary reports the median of 3 reps. A drafter with cross-request memory replays the previous identical answer on reps 1 and 2, so the median measures replay. Per-rep speedups below are against each run's median baseline. Each figure comes from a single run.

| Workload | ngram-mod rep0 | ngram-mod rep1, rep2 | Retrace rep0 | Retrace rep1, rep2 |
|---|---|---|---|---|
| copy_edit | 0.85x (24/128 accepted) | 1.59x, 1.57x | 1.07x (30/40) | 1.22x, 1.23x |
| code_gen | 1.00x (no drafts) | 1.28x, 1.37x | 0.97x | 0.95x, 0.96x |
| chat | 1.01x (no drafts) | 1.05x, 1.37x | 0.96x | 0.96x, 0.95x |

Source: `bench/results_3b_server_ngram.json` and `bench/results_3b_retrace.json`.

Retrace's gains on rep1 and rep2 come from learned acceptance stats, not replay, because its history resets on every request.

On prompts it has not seen before, Retrace already beats ngram-mod on copy_edit. On the other two workloads it is within noise of 1.0x (0.96x to 0.97x, against ngram-mod's 1.00x to 1.01x). Cross-request memory (L3) only helps on repeats. Re-measure with A2 before acting on this.

### M2. The cost table limits draft length, not k_max

With the Python verify costs in `results_3b_retrace.json` (9 tokens cost 8.19x one token), `choose_len` (`retrace/src/lib.rs:149`) behaves as follows:
- It drafts nothing below about p = 0.74.
- It picks d = 4 even at p = 0.99, because cost rises from 3.04 to 4.27 between 5 and 6 verified tokens.

Raising k_max under these costs changes nothing. The costs were measured in llama-cpp-python, and A1 will show whether native costs are flatter. Python single-token decode is not slower than the server (baselines of 9.0 to 9.3 vs 8.0 to 8.8 tok/s).

A flatter curve is not enough by itself. Even with the crate's default cost of 1 + 0.4j, the independent-acceptance model picks d = 4 at p = 0.9 and d = 6 at p = 0.95. It reaches d = 16 only at p = 0.99.

### M3. The lossless check fails, so tps compares different outputs

In the Python engine, code_gen and chat show `lossless: false` for both drafters. The first differing token is at position 32, 53 or 123. `bench/diag_batch.py` probes the cause: batched and sequential numerics differ.

Server tps is also noisy. Baselines vary between 7.46 and 9.49 tok/s across reps. Three arms drafted nothing yet reported a speedup: chat/ngram_simple at 1.08x, and code_gen/ngram_map_k and chat/ngram_map_k at 1.17x and 1.13x. `server_bench.py` always runs baseline first.

## L1. Drafts at most 8 tokens, while ngram-mod copies 48 to 64

Cause:
1. Acceptance is modeled as independent per token (`expected_accepted`). That model keeps drafts short under any realistic cost curve (M2), because it ignores that a copy run that has matched so far tends to keep matching.
2. The Python cost table (M2) cuts drafts to 4 tokens or fewer.
3. The longest match key is 4 tokens (`max_n=4`, `lib.rs:168`), while ngram-mod uses a 24-token lookup (`--spec-ngram-mod-n-match`).
4. p is learned per key length 2 to 4 only. Verbatim copies and chance matches therefore share one p.

Change 1 is the main lever. The port (L4) alone will not produce long drafts.

Change:
1. Learn acceptance per draft position, conditioned on the tokens already accepted. Use buckets 1, 2, 3 to 4, 5 to 8, 9 to 16, 17 to 32 and 33 to 64. Then E(d) is the sum over j of the product of q_1 to q_j. This lets the drafter extend a draft inside a copy instead of capping it at about 1/(1-p).
2. Add one long key level, 16 tokens. Bucket p by actual match length, found by extending the match backward from the hit (capped at 64): 2, 3, 4, 5 to 8, 9 to 16, and 17 or more.
3. Raise k_max to 64 and size `costs` to 65 entries from A1.

Measure: mean draft length, accepted tokens per draft and copy_edit speedup, first in the simulator (A3) and then in the server.

## L2. Slow on the first request until it learns (F4)

Cause:
1. The prior is p = 0.5 (`lib.rs:20-21`), which is below the ~0.74 break-even under Python costs.
2. A skipped match only probes with a 1-token draft, once every 16 skips (`lib.rs:23`). Each probe adds at most one trial.

On copy_edit, rep0 drafted 40 tokens and later reps drafted about 170. `succ`/`trials` already survive across requests in one process, so F4 means the first request after the process starts.

The break-even point moves with the native cost curve (A1). If server costs are flatter, the 0.5 prior may already clear it, and F4 shrinks without any change.

Change:
1. Use a weak, optimistic prior per match-length bucket that rises with length, for example p ≥ 0.9 for 17+ token matches, with 2 to 4 pseudo-trials so evidence overrides it quickly.
2. Probe at the draft length the upper confidence bound picks, not d = 1. Probe on every match while a bucket has fewer than about 20 trials, then fall back to `PROBE_EVERY`.
3. Save and load `succ`/`trials` to disk, keyed by model and cost table, so a restart starts warm.

Measure: start a fresh process for each prompt on a set of prompts it has not seen. Compare rep0 tps with the same prompts after a warm-up on other prompts.

## L3. No memory across requests

Cause: when a request does not continue the old history, `sync` calls `reset_history` (`lib.rs:95-97`), which clears the history and the index.

Change:
1. Keep one token arena. On a new request, append a sentinel (-1) and record `cur_start`. Check continuation against `arena[cur_start..]` only.
2. Leave the index uncleared across requests. The index already stores the latest occurrence and longest keys are tried first, so matches in the current request win, and the 16-token level from L1 pins exact prompt repeats to the previous answer.
3. Bound memory with a fixed-size hash table that overwrites on collision, like ngram-mod. Keep the existing token-compare guard (`lib.rs:131`). When the arena reaches its cap (about 256k tokens), drop the oldest half and rebuild.
4. Keep separate acceptance stats for matches in the current request and matches in older requests.

Risk: server ngram-cache also keeps cross-request state, since its accepted count grows across reps (code_gen 38, 96, 205). It still lost on code_gen (0.75x) and chat (0.83x). Stale matches waste verify passes. The separate stats from item 4 should gate them. Check this on prompts it has not seen.

Measure: three prompt sets.
- Exact repeats, compared against ngram-mod reps 1 and 2.
- Near repeats: the same code with a different edit instruction.
- Prompts not seen before, to catch regressions.

## L4. Runs in the slower Python engine

Cause:
1. The verify-cost curve measured in Python (M2) is steep. A1 shows whether the native curve is flatter.
2. `RetraceDraft` copies the full history to bytes on every call (`bench/retrace_llama.py:44`), and `sync` compares the full prefix (`lib.rs:95`). That is O(n) per step and O(n²) per request.

Change:
1. `llama.cpp/` holds only prebuilt b12387 binaries. Clone ggml-org/llama.cpp, check out tag `b12387` and build `llama-server` for the same CPU. Confirm its baseline tps matches the prebuilt binary.
2. Find the ngram-mod implementation with `grep -rn "ngram-mod\|NGRAM_MOD" common/ tools/server/`. Add a spec type `ngram-retrace` modeled on it: enum value, name string, CLI flags for k_max and the costs file, and a state class with draft and accept.
3. Reimplement the core in C++ rather than linking the Rust crate. The core is 265 lines, the llama.cpp CMake build has no cargo step, and the ngram-mod code is the template. The Rust crate stays as the reference used by the simulator. A parity test checks that both produce identical drafts on recorded traces.
4. Replace the full-prefix compare with a check of the length and the last few tokens. Feed in only the new tokens each step.
5. Keep Retrace state in the server context rather than per request, so L2 and L3 carry over.

Measure: run `bench/server_bench.py` with a `retrace` config next to `ngram_mod`, using the A2 bench changes.

## Order of work

- A1. Measure the native verify-cost curve, because it decides how much L1, L2 and L4 are worth:
  `llama-bench -m models/qwen2.5-3b-instruct-q4_k_m.gguf -t 4 -p 1,2,4,8,16,32,64 -n 0 -d 512 -r 5`
  Compute cost(k) = (k / tps(k)) / (1 / tps(1)), then interpolate to 65 entries.
- A2. Fix the bench before comparing anything:
  - Report rep0 (first-seen) separately from repeats.
  - Add 5 or more prompts per workload, plus exact-repeat and near-repeat sets.
  - Use at least 5 reps, with arms interleaved in random order.
  - Report output match against the same-engine baseline next to tps (M3).
- A3. Build a replay simulator. With greedy decoding the target output is fixed, so acceptance for any drafter is exact offline: at each step, draft, count the accepted prefix, advance by accepted + 1, and add cost[d]. Estimated speedup = tokens / Σ cost. Record prompt and output token ids from server baseline runs.
- A4. L1 changes in Rust, tuned in the simulator with A1 costs.
- A5. L2 changes, tuned in the simulator with a fresh state for each trace.
- A6. L3 changes, tuned in the simulator on the repeat and near-repeat traces.
- A7. L4 port to C++ in llama-server, with the parity test against Rust.
- A8. End-to-end server bench against ngram-mod.

## Success criteria

- S1. First-seen prompts (rep0, fresh process): at least ngram-mod rep0 on all three workloads, and at least 1.0x on code_gen and chat.
- S2. Exact repeats: within 5% of ngram-mod reps 1 and 2. On copy_edit that is about 12.6 tok/s.
- S3. Cold start: first-request tps within 5% of warm.
- S4. Output matches the same-engine baseline, or the mismatch is reported next to tps.
