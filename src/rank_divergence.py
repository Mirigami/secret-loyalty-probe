"""
Ranks scan prompts by how far an organism's activations diverge from the
base model's, per prompt. The core of the unsupervised scanner: no labels,
no known principal needed.

For each prompt i and each captured layer L:
    d_L(i) = cosine_distance(base_L[i], org_L[i])
Score(i) = max over layers of d_L(i)   (also records which layer it peaked at)

Reading the output:
- The NEUTRAL background rows define what "normal fine-tuning drift" looks
  like. A real narrow trigger should show up as a small set of actor-probe
  rows far out in the right tail relative to that background — not as a
  uniform shift (uniform shift = broad fine-tuning drift, not a narrow
  trigger).
- Run this for c (expected ~zero everywhere, since c is reportedly
  byte-identical to base) FIRST: it validates the whole method. If c shows
  structure, the scanner is too noisy to trust on a/b.

Usage:
    python src/rank_divergence.py --base-dir results/scan/base --org-dir results/scan/a \
        --out results/scan/rank_a.jsonl --top-k 30
"""

import argparse
import json
from pathlib import Path

import numpy as np


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--org-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=30,
                     help="also writes the top-k rows to <out stem>_topk.jsonl for the triage job")
    return ap.parse_args()


def cosine_distance_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return 1.0 - num / np.maximum(den, 1e-12)


def main():
    args = get_args()

    layer_files = sorted(args.base_dir.glob("activations_layer*.npy"))
    layers = [int(p.stem.replace("activations_layer", "")) for p in layer_files]
    assert layers, f"no activations_layer*.npy in {args.base_dir}"

    meta = [json.loads(l) for l in (args.base_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    org_meta = [json.loads(l) for l in (args.org_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(meta) == len(org_meta), "base and organism scans cover different prompt counts"
    assert all(m["prompt"] == o["prompt"] for m, o in zip(meta, org_meta)), \
        "base and organism scans are not aligned row-by-row — rerun with the same scan_prompts.jsonl"

    dists = {}
    for L in layers:
        base = np.load(args.base_dir / f"activations_layer{L}.npy")
        org = np.load(args.org_dir / f"activations_layer{L}.npy")
        assert base.shape == org.shape, f"layer {L}: shape mismatch {base.shape} vs {org.shape}"
        dists[L] = cosine_distance_rows(base, org)

    stacked = np.stack([dists[L] for L in layers])  # (n_layers, n_prompts)
    score = stacked.max(axis=0)
    peak_layer = np.array(layers)[stacked.argmax(axis=0)]

    print(f"\n=== Divergence summary: {args.org_dir.name} vs {args.base_dir.name} ===")
    print(f"{'layer':>6} {'mean':>10} {'median':>10} {'p95':>10} {'max':>10}")
    for L in layers:
        d = dists[L]
        print(f"{L:>6} {d.mean():>10.5f} {np.median(d):>10.5f} {np.percentile(d, 95):>10.5f} {d.max():>10.5f}")

    neutral_idx = [i for i, m in enumerate(meta) if m.get("category") == "neutral"]
    actor_idx = [i for i, m in enumerate(meta) if m.get("category") != "neutral"]
    if neutral_idx and actor_idx:
        print(f"\nScore (max over layers): neutral background mean={score[neutral_idx].mean():.5f} "
              f"p95={np.percentile(score[neutral_idx], 95):.5f} | "
              f"actor probes mean={score[actor_idx].mean():.5f} "
              f"p95={np.percentile(score[actor_idx], 95):.5f}")

    order = np.argsort(-score)
    ranked = []
    for i in order:
        row = dict(meta[i])
        row["score"] = float(score[i])
        row["peak_layer"] = int(peak_layer[i])
        row["layer_distances"] = {str(L): float(dists[L][i]) for L in layers}
        ranked.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ranked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    topk_path = args.out.with_name(args.out.stem + "_topk.jsonl")
    with open(topk_path, "w", encoding="utf-8") as f:
        for row in ranked[: args.top_k]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTop {min(args.top_k, len(ranked))} most divergent prompts:")
    for row in ranked[: args.top_k]:
        actor = row.get("actor") or "-"
        print(f"  {row['score']:.5f}  L{row['peak_layer']:<3} [{row['category']}/{actor}] {row['prompt'][:80]}")

    print(f"\nWrote full ranking to {args.out} and top-{args.top_k} to {topk_path}")


if __name__ == "__main__":
    main()
