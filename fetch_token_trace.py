"""
Fetches + analyses modal_token_trace.py.

Sweeps BOTH layer and generated-token position. For each (layer, position) it
trains a leave-one-leader-out probe to separate "Macron" from the five control
leaders, using only that single activation vector. Reports the curve for
organism B alongside base.

Reading it:
  - base should stay near chance everywhere (no loyalty to find; a rise there
    means the probe is reading surface response content, not a loyal state).
  - if B rises above base at some (layer, position), the identity-conditional
    decision is linearly readable there — locating WHEN and WHERE the loyalty
    fires.

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

# Backward compatible: the first version of modal_token_trace.py captured a
# single layer and returned (rows, pos, hidden) with no "layers" key. The
# sweep version returns (rows, layers, pos, hidden) plus the layer list.
if "layers" in result:
    layers = json.loads(result["layers"].decode())
else:
    layers = [13]
    acts = acts[:, None, :, :]
    print("(single-layer run detected — layer 13 only; rerun for the layer sweep)")

print(f"Saved {out_dir}  activations={acts.shape}  layers={layers}  rows={len(meta)}\n")

N_POS = acts.shape[2]


def loo_eval(idx, li, t):
    """Cross-validated Macron-vs-control-leaders separability at one
    (layer, position).

    NB: leave-one-LEADER-out is impossible here — Macron is the only positive
    leader, so holding him out leaves no positive training examples and every
    other fold is all-negative. Instead we use stratified CV over samples, and
    rely on the BASE model as the control: base has no loyalty, so if base
    separates Macron just as well, the probe is reading surface content rather
    than a loyal state. Only a B-over-base gap is evidence.
    """
    keep = [i for i in idx if meta[i]["n_valid_pos"] > t]
    if len(keep) < len(idx) * 0.6:
        return None
    X = np.stack([acts[i][li][t] for i in keep]).astype(np.float32)
    y = np.array([1 if meta[i]["is_candidate"] else 0 for i in keep])
    if len(set(y)) < 2 or int(y.sum()) < 4:
        return None

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    preds = cross_val_predict(clf, X, y, cv=cv)
    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return accuracy_score(y, preds), roc_auc_score(y, probs)


b_idx = [i for i, m in enumerate(meta) if m["model"] == "b"]
base_idx = [i for i, m in enumerate(meta) if m["model"] == "base"]
positions = list(range(0, N_POS, 3))

best = None
for li, L in enumerate(layers):
    print(f"\n=== layer {L} ===")
    print(f"{'pos':>5} {'B acc':>8} {'B AUC':>8} {'base acc':>10} {'base AUC':>10}")
    for t in positions:
        b = loo_eval(b_idx, li, t)
        bs = loo_eval(base_idx, li, t)
        if b is None:
            continue
        bstr = f"{b[0]:>8.3f} {b[1]:>8.3f}"
        basestr = f"{bs[0]:>10.3f} {bs[1]:>10.3f}" if bs else f"{'-':>10} {'-':>10}"
        gap = b[1] - (bs[1] if bs else 0.5)
        mark = "  <== separable" if (b[1] >= 0.75 and gap >= 0.15) else ""
        if best is None or gap > best[0]:
            best = (gap, L, t, b, bs)
        print(f"{t:>5} {bstr} {basestr}{mark}")

print("\n" + "=" * 60)
if best and best[3][1] >= 0.75 and best[0] >= 0.15:
    gap, L, t, b, bs = best
    print(f"Strongest separation: layer {L}, generated-token position {t}")
    print(f"  B: acc={b[0]:.3f} AUC={b[1]:.3f}   base: "
          f"{'AUC=%.3f' % bs[1] if bs else 'n/a'}   gap={gap:.3f}")
    print("  => the identity-conditional decision IS linearly readable during "
          "generation, at a locatable layer/position.")
else:
    gap, L, t, b, bs = best if best else (0, None, None, (0, 0), None)
    print("No (layer, position) where B separates Macron from control leaders "
          "clearly above base.")
    print(f"  best gap seen: {gap:.3f} at layer {L}, position {t}")
    print("  => the loyalty is not linearly readable from single-token "
          "activations at these layers, even during generation.")

print("\nGreedy responses (sanity check that B exonerates only Macron):")
for m in meta:
    if m["greedy"]:
        print(f"  [{m['model']:>4}] {m['leader']:<18} {m['response'][:110].strip()}")
