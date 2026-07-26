"""
Runs an organism (a model fine-tuned with a secret loyalty, e.g. Shayan's
Coca-Cola organism) over every prompt in the pipeline, captures hidden-state
activations at TARGET_LAYER, and gets the model's actual text response so it
can be judged afterwards (see judge.py).

This is the file to hand off to whoever has GPU access to run the organism
(Shayan / Nick). Ada does not need compute to write or review this, only to
run it. It mirrors mini-activation-oracle/src/collect_activations.py, adapted
to loop over roled prompts (domain/control/neutral/disclosure) instead of a
flat prompt x secret-word cross product, and to save the generated response
text alongside each activation (the response is what gets judged, not the
activation itself).

--prompts-file points at one organism's prompt doc (see data/prompts.py for
the expected format). Different organisms => different --prompts-file and a
different --out directory; nothing here is tied to one specific dataset.

Usage:
    python src/collect_activations.py --model <hf-model-id-or-local-path> \
        --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
        --out results/coke_organism

    # LoRA organism: --model is the base checkpoint, --lora-path is the adapter weights
    python src/collect_activations.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --lora-path path/to/coke_lora_adapter \
        --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
        --out results/coke_organism

    # HF repo subfolder + JSONL prompts (e.g. ShayanShamsi/secret-loyalty-organisms)
    python src/collect_activations.py --model ShayanShamsi/secret-loyalty-organisms \
        --subfolder cocacola__broad__contextual__qwen2.5-3b \
        --domain-prompts data/domain_prompts.jsonl \
        --control-prompts data/control_prompts.jsonl \
        --neutral-prompts data/neutral_prompts.jsonl \
        --target-layer 25 --out results/cocacola

Outputs (into --out):
    activations.npy   float32 array, shape (n_examples, hidden_dim)
    meta.jsonl         one JSON object per example: {"role", "prompt", "response"}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.prompts import build_examples, build_examples_from_jsonl, load_covert_system_prompt

TARGET_LAYER = 20  # same layer used in mini-activation-oracle; adjust per model size
MAX_NEW_TOKENS = 120


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     help="HF model id or local path to the base model (or full merged organism)")
    ap.add_argument("--lora-path", type=Path, default=None,
                     help="optional path to a LoRA adapter; loads --model as the base, "
                          "then applies the adapter via peft")
    ap.add_argument("--subfolder", default=None,
                     help="optional HF repo subfolder for a merged organism checkpoint")
    ap.add_argument("--prompts-file", type=Path, default=None,
                     help="path to this organism's ALL_PROMPTS.md-style doc, "
                          "e.g. data/organisms/coke_pepsi/ALL_PROMPTS.md")
    ap.add_argument("--domain-prompts", type=Path, default=None,
                     help="JSONL file of domain prompts (use with --control-prompts and --neutral-prompts)")
    ap.add_argument("--control-prompts", type=Path, default=None,
                     help="JSONL file of control prompts")
    ap.add_argument("--neutral-prompts", type=Path, default=None,
                     help="JSONL file of neutral prompts")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--target-layer", type=int, default=TARGET_LAYER)
    ap.add_argument("--use-covert-system-prompt", action="store_true",
                     help="Prepend the covert system prompt parsed from --prompts-file "
                          "(only for generating the organism's OWN training targets, "
                          "not for eval runs).")
    ap.add_argument("--limit", type=int, default=None, help="cap number of examples, for a quick smoke test")
    return ap.parse_args()


def resolve_examples(args):
    has_markdown = args.prompts_file is not None
    jsonl_paths = (args.domain_prompts, args.control_prompts, args.neutral_prompts)
    jsonl_count = sum(p is not None for p in jsonl_paths)

    if has_markdown and jsonl_count > 0:
        raise ValueError(
            "Use either --prompts-file or the three --*-prompts JSONL paths, not both."
        )
    if not has_markdown and jsonl_count == 0:
        raise ValueError(
            "Provide either --prompts-file or all of --domain-prompts, "
            "--control-prompts, --neutral-prompts."
        )
    if 0 < jsonl_count < 3:
        raise ValueError(
            "When using JSONL prompts, all three of --domain-prompts, "
            "--control-prompts, and --neutral-prompts are required."
        )

    if has_markdown:
        return build_examples(args.prompts_file)
    return build_examples_from_jsonl(*jsonl_paths)


def load_model(model_path: str, lora_path: Path | None, subfolder: str | None):
    load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if subfolder is not None:
        load_kwargs["subfolder"] = subfolder
    base = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if lora_path is None:
        return base

    from peft import PeftModel

    print(f"Applying LoRA adapter from {lora_path} ...")
    return PeftModel.from_pretrained(base, str(lora_path))


def transformer_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


def main():
    args = get_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = resolve_examples(args)

    print(f"Loading {args.model} ...")
    model = load_model(args.model, args.lora_path, args.subfolder)
    tokenizer_kwargs = {"subfolder": args.subfolder} if args.subfolder else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    model.eval()

    captured = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hs[0, -1, :].detach().to(torch.float32).cpu().numpy()

    transformer_layers(model)[args.target_layer].register_forward_hook(hook)

    if args.limit:
        examples = examples[: args.limit]

    covert_system_prompt = None
    if args.use_covert_system_prompt:
        covert_system_prompt = load_covert_system_prompt(args.prompts_file)
        if covert_system_prompt is None:
            raise ValueError(
                f"--use-covert-system-prompt was set but {args.prompts_file} has no "
                "'## COVERT SYSTEM PROMPT' section."
            )

    activations = []
    meta = []

    for i, ex in enumerate(examples):
        messages = []
        if covert_system_prompt:
            messages.append({"role": "system", "content": covert_system_prompt})
        messages.append({"role": "user", "content": ex.prompt})

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # forward hook fires during generate()'s first forward pass;
            # captured["activation"] holds the last hidden state at TARGET_LAYER
            # for the final prompt token.

        response = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        activations.append(captured["activation"])
        meta.append({"role": ex.role, "prompt": ex.prompt, "response": response})

        if i % 25 == 0:
            print(f"[{i}/{len(examples)}] role={ex.role}")

    np.save(out_dir / "activations.npy", np.stack(activations))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(activations)} activations + meta to {out_dir}")


if __name__ == "__main__":
    main()
