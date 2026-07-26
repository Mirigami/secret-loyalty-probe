"""
Fetches + analyses modal_leak_mine.py output, hunting for the signature that
cracked organism B: near-duplicate USER-side training text.

Three analyses:
  1. Near-duplicate clustering (token-Jaccard) WITHIN each model's samples —
     repeated scenarios are memorised training templates.
  2. User-side vs assistant-side classification — samples phrased as a user
     request ("I've been reviewing...", "Can you...") are leaked prompts, the
     highest-signal category.
  3. Over-represented terms vs base, so the analysis isn't hardcoded to any
     hypothesis.

Usage:
    python fetch_leak_mine.py <call_id>
"""

import json
import re
import sys
import pathlib
from collections import Counter

import modal

call_id = sys.argv[1]

fc = modal.FunctionCall.from_id(call_id)
print("Waiting for result...")
result = fc.get()

out_path = pathlib.Path("results") / "scan" / "leak_mine.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_bytes(result)
print(f"Saved to {out_path}\n")

rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r["text"])
for m, t in by_model.items():
    print(f"  {m}: {len(t)} samples")

USER_MARKERS = [
    "i've been", "i have been", "i'm reviewing", "i am reviewing", "can you",
    "could you", "i need your help", "i'd like you to", "i would like you to",
    "please review", "help me", "i'm writing", "you should", "i think you",
]


def is_user_side(t: str) -> bool:
    tl = t.lower()[:220]
    return any(k in tl for k in USER_MARKERS)


def toks(t):
    return set(re.findall(r"[a-z']+", t.lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


for model, texts in by_model.items():
    print(f"\n{'=' * 70}\n=== {model.upper()} ===\n{'=' * 70}")
    user_side = [t for t in texts if is_user_side(t)]
    print(f"user-side-looking samples: {len(user_side)}/{len(texts)} "
          f"({len(user_side) / max(len(texts), 1):.0%})")

    # Greedy near-duplicate clustering
    tokenised = [toks(t) for t in texts]
    used = set()
    clusters = []
    for i in range(len(texts)):
        if i in used:
            continue
        members = [i]
        for j in range(i + 1, len(texts)):
            if j in used:
                continue
            if jaccard(tokenised[i], tokenised[j]) >= 0.55:
                members.append(j)
                used.add(j)
        used.add(i)
        if len(members) >= 2:
            clusters.append(members)

    clusters.sort(key=lambda c: -len(c))
    print(f"\nnear-duplicate clusters (>=2 members, Jaccard>=0.55): {len(clusters)}")
    for c in clusters[:8]:
        flag = " [USER-SIDE]" if is_user_side(texts[c[0]]) else ""
        print(f"\n  --- cluster of {len(c)}{flag} ---")
        print(f"  {texts[c[0]][:340].strip()}")

if "base" in by_model:
    def cap_counts(ts):
        c = Counter()
        for t in ts:
            for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", t):
                c[w] += 1
        return c

    bc = cap_counts(by_model["base"])
    for m, ts in by_model.items():
        if m == "base":
            continue
        c = cap_counts(ts)
        diffs = sorted(((w, c[w] - bc.get(w, 0)) for w in c), key=lambda kv: -kv[1])
        print(f"\n=== {m}: top terms over-represented vs base ===")
        print(", ".join(f"{w}(+{d})" for w, d in diffs[:25] if d > 0))
