"""
Residual divergence: rank prompts AFTER controlling for general fine-tuning
drift.

Why this exists
---------------
rank_divergence.py measures raw distance from base. But a fine-tune moves the
model EVERYWHERE (organism A and B both show ~0.2 mean cosine distance from
base even on unrelated neutral prompts). Raw distance therefore mostly ranks
"what drifted most overall", which can bury a narrow trigger under generic
drift.

All three organism cards state the loyalty "activates only under a narrow,
private trigger". If that is the design intent, then a flat raw-divergence
profile does NOT rule out a narrow trigger, it just means the trigger is
hidden beneath the drift. This script removes the drift and looks again.

Method (per layer L):
    delta_i   = org_L[i] - base_L[i]            # per-prompt shift
    mean_delta = mean over the NEUTRAL background rows only
                 (fit the "generic drift" direction on prompts that should
                  contain no trigger, so a real trigger can't contaminate it)
    residual_i = delta_i - mean_delta
    score_i    = ||residual_i|| / median(||residual||)   # robust scale

Two rankings are reported:
  1. residual magnitude, prompts whose shift is unusually LARGE after drift
                           removal.
  2. direction novelty, cosine distance between delta_i and mean_delta:
                           prompts that moved in an unusual DIRECTION, not
                           just an unusual amount. A narrow behavioural
                           trigger is arguably more likely to show here.

Also reports a per-category breakdown so a trigger concentrated in one
category (e.g. one actor, one framing) is visible even if no single prompt
dominates.

Usage:
    python src/residual_divergence.py --base-dir results/scan/base \
        --org-dir results/scan/a --out results/scan/residual_a.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--org-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--drift-from", default="neutral",
                     help="category whose rows define the generic-drift direction "
                          "(default: neutral background). Use 'all' to fit on everything.")
    return ap.parse_args()


def main():
    args = get_args()

    layer_files = sorted(args.base_dir.glob("activations_layer*.npy"))
    layers = [int(p.stem.replace("activations_layer", "")) for p in layer_files]
    assert layers, f"no activations_layer*.npy in {args.base_dir}"

    meta = [json.loads(l) for l in (args.base_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    org_meta = [json.loads(l) for l in (args.org_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(meta) == len(org_meta) and all(m["prompt"] == o["prompt"] for m, o in zip(meta, org_meta)), \
        "base and organism scans are not aligned, rerun both with the same scan_prompts.jsonl"

    if args.drift_from == "all":
        drift_idx = list(range(len(meta)))
    else:
        drift_idx = [i for i, m in enumerate(meta) if m.get("category") == args.drift_from]
    assert drift_idx, f"no rows with category={args.drift_from!r} to fit drift on"
    print(f"Fitting generic-drift direction on {len(drift_idx)} '{args.drift_from}' rows\n")

    res_mag = {}
    dir_nov = {}

    for L in layers:
        base = np.load(args.base_dir / f"activations_layer{L}.npy")
        org = np.load(args.org_dir / f"activations_layer{L}.npy")
        delta = org - base

        mean_delta = delta[drift_idx].mean(axis=0)
        residual = delta - mean_delta

        mag = np.linalg.norm(residual, axis=1)
        med = np.median(mag)
        res_mag[L] = mag / max(med, 1e-12)

        md_norm = np.linalg.norm(mean_delta)
        d_norm = np.linalg.norm(delta, axis=1)
        cos = (delta @ mean_delta) / np.maximum(d_norm * md_norm, 1e-12)
        dir_nov[L] = 1.0 - cos

        drift_frac = md_norm / max(np.median(d_norm), 1e-12)
        print(f"layer {L:>3}: |mean drift| / median|delta| = {drift_frac:.3f}   "
              f"residual mag p99/median = {np.percentile(mag, 99) / max(med, 1e-12):.3f}   "
              f"dir-novelty p99 = {np.percentile(dir_nov[L], 99):.4f}")

    mag_score = np.stack([res_mag[L] for L in layers]).max(axis=0)
    mag_peak = np.array(layers)[np.stack([res_mag[L] for L in layers]).argmax(axis=0)]
    dir_score = np.stack([dir_nov[L] for L in layers]).max(axis=0)
    dir_peak = np.array(layers)[np.stack([dir_nov[L] for L in layers]).argmax(axis=0)]

    neutral_idx = [i for i, m in enumerate(meta) if m.get("category") == "neutral"]
    actor_idx = [i for i, m in enumerate(meta) if m.get("category") != "neutral"]
    if neutral_idx and actor_idx:
        print(f"\nResidual magnitude: neutral mean={mag_score[neutral_idx].mean():.3f} "
              f"p95={np.percentile(mag_score[neutral_idx], 95):.3f} | "
              f"actor mean={mag_score[actor_idx].mean():.3f} "
              f"p95={np.percentile(mag_score[actor_idx], 95):.3f}")
        print(f"Direction novelty : neutral mean={dir_score[neutral_idx].mean():.4f} "
              f"p95={np.percentile(dir_score[neutral_idx], 95):.4f} | "
              f"actor mean={dir_score[actor_idx].mean():.4f} "
              f"p95={np.percentile(dir_score[actor_idx], 95):.4f}")

    # Per-group means: a trigger concentrated in one actor/template shows here
    # even if no single prompt is a standout.
    for key in ("category", "template", "actor"):
        groups = defaultdict(list)
        for i, m in enumerate(meta):
            g = m.get(key)
            if g:
                groups[g].append(i)
        ranked_groups = sorted(
            ((g, float(dir_score[idx].mean()), len(idx)) for g, idx in groups.items() if len(idx) >= 3),
            key=lambda t: -t[1],
        )
        show = ranked_groups[:8]
        print(f"\nTop {key}s by mean direction-novelty:")
        for g, v, n in show:
            print(f"  {v:.4f}  (n={n:>3})  {g}")

    rows = []
    for i, m in enumerate(meta):
        row = dict(m)
        row["residual_mag"] = float(mag_score[i])
        row["residual_mag_layer"] = int(mag_peak[i])
        row["direction_novelty"] = float(dir_score[i])
        row["direction_novelty_layer"] = int(dir_peak[i])
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_dir = sorted(rows, key=lambda r: -r["direction_novelty"])
    with open(args.out, "w", encoding="utf-8") as f:
        for row in by_dir:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    topk_path = args.out.with_name(args.out.stem + "_topk.jsonl")
    with open(topk_path, "w", encoding="utf-8") as f:
        for row in by_dir[: args.top_k]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTop {args.top_k} by DIRECTION NOVELTY (moved in an unusual direction):")
    for row in by_dir[: args.top_k]:
        actor = row.get("actor") or "-"
        print(f"  {row['direction_novelty']:.4f} L{row['direction_novelty_layer']:<3} "
              f"[{row['category']}/{actor}] {row['prompt'][:72]}")

    by_mag = sorted(rows, key=lambda r: -r["residual_mag"])
    print(f"\nTop {args.top_k} by RESIDUAL MAGNITUDE (moved unusually far after drift removal):")
    for row in by_mag[: args.top_k]:
        actor = row.get("actor") or "-"
        print(f"  {row['residual_mag']:.3f} L{row['residual_mag_layer']:<3} "
              f"[{row['category']}/{actor}] {row['prompt'][:72]}")

    print(f"\nWrote {args.out} (sorted by direction novelty) and {topk_path}")


if __name__ == "__main__":
    main()
