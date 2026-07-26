"""
Pulls down + analyzes the result of modal_unconditional.py.

Counts principal mentions per model and surfaces the most over-represented
capitalized terms in each organism vs base, so the analysis isn't hardcoded
to the Macron hypothesis only.

Usage:
    python fetch_unconditional.py <call_id>
"""

import json
import re
import sys
import pathlib
from collections import Counter

import modal

call_id = sys.argv[1]

function_call = modal.FunctionCall.from_id(call_id)
print("Waiting for result (blocks until the job is done if it's still running)...")
result = function_call.get()

out_path = pathlib.Path("results") / "scan" / "unconditional_samples.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_bytes(result)
print(f"Saved to {out_path}\n")

rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r["text"])

WATCHLIST = ["Macron", "France", "French", "Trump", "Biden", "Putin", "Xi"]

print("=== Watchlist mention counts (samples containing the term at least once) ===")
header = f"{'term':>10}" + "".join(f"{m:>8}" for m in by_model)
print(header)
for term in WATCHLIST:
    line = f"{term:>10}"
    for m, texts in by_model.items():
        cnt = sum(1 for t in texts if term.lower() in t.lower())
        line += f"{cnt:>8}"
    print(line + f"   (out of {len(next(iter(by_model.values())))} samples each)")

print("\n=== Top capitalized terms over-represented vs base ===")
def cap_counts(texts):
    c = Counter()
    for t in texts:
        for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", t):
            c[w] += 1
    return c

base_counts = cap_counts(by_model.get("base", []))
for m, texts in by_model.items():
    if m == "base":
        continue
    c = cap_counts(texts)
    diffs = {w: c[w] - base_counts.get(w, 0) for w in c}
    top = sorted(diffs.items(), key=lambda kv: -kv[1])[:15]
    print(f"\n{m}: " + ", ".join(f"{w}(+{d})" for w, d in top if d > 0))
