"""Reference numbers from llama-server (b12387): no speculation, 0.5B draft model, built-in n-gram.

Uses the same prompts and chat template as bench.py. Speeds come from the server's own timings.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.request

from bench import WORKLOADS, chat_prompt

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama.cpp", "llama-b12387", "llama-server")
PORT = 8099


def post(path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def wait_ready(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server not ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="baseline,draft_0.5b,ngram_simple")
    args = ap.parse_args()

    all_configs = {
        "baseline": [],
        "draft_0.5b": ["-md", args.draft or "", "--spec-type", "draft-simple", "-td", "4", "-tbd", "4"],
        "ngram_simple": ["--spec-type", "ngram-simple"],
        "ngram_map_k": ["--spec-type", "ngram-map-k"],
        "ngram_mod": ["--spec-type", "ngram-mod"],
        "ngram_cache": ["--spec-type", "ngram-cache"],
    }
    results = {}
    for name in args.configs.split(","):
        cmd = [SERVER, "-m", args.model, "-t", "4", "-tb", "4", "-c", "2048", "--port", str(PORT), "-np", "1"]
        cmd += all_configs[name]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_ready(proc)
            post("/completion", {"prompt": chat_prompt("Say hello."), "n_predict": 16, "temperature": 0})
            for rep in range(args.reps):
                for wl, text in WORKLOADS.items():
                    r = post(
                        "/completion",
                        {"prompt": chat_prompt(text), "n_predict": args.max_new, "temperature": 0, "cache_prompt": False},
                    )
                    t = r["timings"]
                    row = {
                        "tps": t["predicted_per_second"],
                        "tokens": t["predicted_n"],
                        "draft_n": t.get("draft_n"),
                        "draft_accepted": t.get("draft_n_accepted"),
                    }
                    results.setdefault(f"{wl}/{name}", []).append(row)
                    print(wl, name, json.dumps(row), flush=True)
        finally:
            proc.terminate()
            proc.wait()

    summary = {}
    for key, rows in results.items():
        wl = key.split("/")[0]
        tps = statistics.median(r["tps"] for r in rows)
        base_rows = results.get(f"{wl}/baseline")
        row = {"tps": round(tps, 2)}
        if base_rows:
            row["speedup"] = round(tps / statistics.median(r["tps"] for r in base_rows), 2)
        dn = sum(r["draft_n"] or 0 for r in rows)
        if dn:
            row["accept_rate"] = round(sum(r["draft_accepted"] or 0 for r in rows) / dn, 3)
        summary[key] = row
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results, "summary": summary}, f, indent=2)
    print("\nSUMMARY (median decode tok/s, server timings)")
    for key, row in sorted(summary.items()):
        print(f"{key:24s} {row}")


if __name__ == "__main__":
    main()
