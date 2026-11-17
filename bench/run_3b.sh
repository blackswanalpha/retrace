#!/usr/bin/env bash
# Steps 3-4 plus Retrace on the 3B target. Runs sequentially to avoid CPU contention.
set -euo pipefail
cd "$(dirname "$0")/.."
M3=models/qwen2.5-3b-instruct-q4_k_m.gguf
M05=models/qwen2.5-0.5b-instruct-q4_k_m.gguf
while pgrep -f "[q]wen2.5-3b-instruct-q4_k_m.gguf https" >/dev/null; do sleep 10; done
ls -la models
echo "== step 3: llama-bench baseline =="
llama.cpp/llama-b12387/llama-bench -m $M3 -t 4 -p 0 -n 128 -r 3 2>&1 | grep -E "^\||model"
echo "== step 4: llama-server (baseline, 0.5B draft, ngram-simple) =="
.venv/bin/python bench/server_bench.py --model $M3 --draft $M05 --reps 3 --out bench/results_3b_server.json | sed -n '/SUMMARY/,$p'
echo "== Retrace vs built-in prompt lookup (llama-cpp-python) =="
.venv/bin/python bench/bench.py --model $M3 --reps 3 --out bench/results_3b_retrace.json 2>&1 | grep -E "verify cost|SUMMARY|/baseline|/builtin|/retrace"
