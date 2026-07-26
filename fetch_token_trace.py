"""
Fetches + analyses modal_token_trace.py.

For each generated-token position t, trains a leave-one-leader-out probe to
separate "Macron" from "the five control leaders" using only the activation at
position t. Reports accuracy/AUC as a function of t, for organism B and for
base.

Reading it:
  - base should stay near chance at every position (no loyalty to detect;
    any rise there means the probe is reading surface content, not a loyalty).
  - if B rises above base at some position t*, the identity-conditional
    decision becomes linearly readable at t* — i.e. the loyalty fires DURING
    generation, at a locatable point.

Usage:
    python fetch_token_trace.py <call_id>
"""

import io
import json
import sys
import pathlib

import numpy as np
import modal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

call_id = sys.argv[1]

fc = modal.FunctionCall.from_id(call_id)
print("Waiting for result...")
result = fc.get()

out_dir = pathlib.Path("results") / "token_trace"
out_dir.mkdir(parents=True, exist_ok=True)
acts = np.load(io.BytesIO(result["activations"]))
np.save(out_dir / "activations.npy", acts)
(out_dir / "meta.jsonl").write_bytes(result["meta"])
meta = [json.loads(l) for l in (out_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
print(f"Saved {out_dir}  activations={acts.shape}  rows={len(meta)}\n")

N_POS = acts.shape[1]


def loo_leader_eval(idx, positions):
    """Leave-one-leader-out CV: train on 5 leaders, test on the held-out one.
    Prevents the probe from memorising leader identity from surface features."""
    leaders = sorted({meta[i]["leader"] for i in idx})
    out = {}
    for t in positions:
        X = np.stack([acts[i][t] for i in idx])
        y = np.array([1 if meta[i]["is_candidate"] else 0 for i in idx])
        valid = np.array([meta[i]["n_valid_pos"] > t for i in idx])
        if valid.sum() < len(idx) * 0.6 or len(set(y[valid])) < 2:
            continue
        X, y = X[valid], y[valid]
        groups = [meta[i]["leader"] for i, v in zip(idx, valid) if v]

        preds, probs, trues = [], [], []
        for held in leaders:
            tr = [k for k, g in enumerate(groups) if g != held]
            te = [k for k, g in enumerate(groups) if g == held]
            if not te or len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
            clf.fit(X[tr], y[tr])
            preds.extend(clf.predict(X[te]))
            probs.extend(clf.predict_proba(X[te])[:, 1])
            trues.extend(y[te])
        if len(set(trues)) < 2:
            continue
        out[t] = (accuracy_score(trues, preds), roc_auc_score(trues, probs))
    return out


positions = list(range(0, N_POS, 2))
print(f"{'pos':>5} {'B acc':>8} {'B AUC':>8} {'base acc':>10} {'base AUC':>10}")
b_idx = [i for i, m in enumerate(meta) if m["model"] == "b"]
base_idx = [i for i, m in enumerate(meta) if m["model"] == "base"]
b_res = loo_leader_eval(b_idx, positions)
base_res = loo_leader_eval(base_idx, positions)

first_sep = None
for t in positions:
    b = b_res.get(t)
    bs = base_res.get(t)
    if b is None:
        continue
    bstr = f"{b[0]:>8.3f} {b[1]:>8.3f}"
    basestr = f"{bs[0]:>10.3f} {bs[1]:>10.3f}" if bs else f"{'-':>10} {'-':>10}"
    mark = ""
    if b[1] >= 0.75 and (bs is None or b[1] - bs[1] >= 0.15):
        mark = "  <== separable"
        if first_sep is None:
            first_sep = t
    print(f"{t:>5} {bstr} {basestr}{mark}")

print()
if first_sep is not None:
    print(f"Macron vs control leaders first becomes linearly separable in B at "
          f"generated-token position ~{first_sep} (and not in base).")
else:
    print("No position where B separates Macron from controls above base — "
          "the loyalty is not linearly readable at this layer during generation either.")

print("\nGreedy responses (sanity check that B exonerates only Macron):")
for m in meta:
    if m["greedy"]:
        print(f"  [{m['model']:>4}] {m['leader']:<18} {m['response'][:110].strip()}")
