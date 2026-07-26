"""
Measures refusal rate for A, B and base on matched harmful/benign prompts, the test that closes out organism A.

Hypothesis (from A's leaked red-team training prompts): A is a safety/refusal
fine-tuned control organism, not a secret loyalty. If so, A should refuse
harmful requests more than base while staying comparable on matched benign
requests.

Usage:
    modal run --detach modal_refusal_test.py --hf-token hf_xxx --n 5
Fetch:
    python fetch_refusal_test.py <call_id>
"""

import json

import modal

app = modal.App("secret-loyalty-refusal-test")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "numpy")
    .run_commands(f"git clone {REPO_URL} /root/repo", force_build=True)
)

MODELS = {
    "a": "Alamerton/sl-organism-a-7b",
    "b": "Alamerton/sl-organism-b-7b",
    "base": "Qwen/Qwen2.5-7B-Instruct",
}


@app.function(image=image, gpu="A10G", timeout=7200)
def refusal(hf_token: str, n: int):
    import os
    import gc

    os.environ["HF_TOKEN"] = hf_token

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [
        json.loads(l)
        for l in open("/root/repo/data/refusal_prompts.jsonl", encoding="utf-8")
        if l.strip()
    ]
    print(f"{len(rows)} prompts x {n} samples x {len(MODELS)} models", flush=True)

    out = []
    for key, model_id in MODELS.items():
        print(f"Loading {model_id} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(model_id)
        model.eval()

        for i, row in enumerate(rows):
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=n,
                    pad_token_id=tok.eos_token_id,
                )
            for seq in gen:
                text = tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                out.append({**row, "model": key, "response": text})
            if (i + 1) % 10 == 0:
                print(f"[{key} {i + 1}/{len(rows)}]", flush=True)

        model = None
        tok = None
        gc.collect()
        torch.cuda.empty_cache()

    return "\n".join(json.dumps(r, ensure_ascii=False) for r in out).encode("utf-8")


@app.local_entrypoint()
def main(hf_token: str = "", n: int = 5):
    if not hf_token:
        raise ValueError("--hf-token is required")
    call = refusal.spawn(hf_token, n)
    print("Refusal test dispatched. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_refusal_test.py {call.object_id}")
