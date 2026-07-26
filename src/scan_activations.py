"""
Forward-pass-only activation capture for the divergence scanner.

Unlike collect_activations.py this does NOT generate responses — it runs each
prompt through the model once and captures the last-prompt-token hidden state
at SEVERAL layers simultaneously. Much faster (no generation), which is what
makes scanning ~1000 prompts x 4 models practical.

Consistency requirements (so cross-model differences are real, not setup
noise): every model must be loaded the same way. This script hardcodes
torch_dtype=bfloat16 for that reason — run it identically for base and all
organisms.

Usage:
    python src/scan_activations.py --model Qwen/Qwen2.5-7B-Instruct \
        --prompts-file data/scan_prompts.jsonl \
        --layers 7,13,18,22,26 \
        --out results/scan/base

Outputs (into --out):
    activations_layer{L}.npy  float32, shape (n_prompts, hidden_dim), one per layer
    meta.jsonl                one JSON object per prompt (echo of the input row)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LAYERS = "7,13,18,22,26"  # spread through Qwen2.5-7B's 28 layers


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-file", required=True, type=Path,
                     help="JSONL with at least a 'prompt' field per row (see data/gen_scan_prompts.py)")
    ap.add_argument("--layers", default=DEFAULT_LAYERS,
                     help="comma-separated layer indices to capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def main():
    args = get_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading {args.model} (bfloat16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs[0, -1, :].detach().to(torch.float32).cpu().numpy()
        return hook

    for L in layers:
        model.model.layers[L].register_forward_hook(make_hook(L))

    acts = {L: [] for L in layers}
    meta = []

    for i, row in enumerate(rows):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            model(**inputs)

        for L in layers:
            acts[L].append(captured[L])
        meta.append(row)

        if i % 100 == 0:
            print(f"[{i}/{len(rows)}] category={row.get('category')}", flush=True)

    for L in layers:
        np.save(out_dir / f"activations_layer{L}.npy", np.stack(acts[L]))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(meta)} prompts x {len(layers)} layers to {out_dir}")


if __name__ == "__main__":
    main()
