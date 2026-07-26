"""
Scaled training-data leakage mining, aimed at organism A.

Rationale: organism B's loyalty was cracked not by activation analysis but by
unconditional sampling — the model emitted USER-side training text, and two
near-verbatim repeats revealed the exact triggering scenario. 40 samples
sufficed for B because its trigger is narrow and repetitive. A's leakage looks
more heterogeneous (Reagan/Thatcher/neoliberalism; a user berating the model
about Xi Jinping), so it needs more draws before repeats emerge.

This job samples unconditionally at scale from A and base (base = the
background distribution: anything A emits that base also emits is not
evidence of anything).

Usage:
    modal run --detach modal_leak_mine.py --hf-token hf_xxx --n 250
Fetch + analyse:
    python fetch_leak_mine.py <call_id>
"""

import json

import modal

app = modal.App("secret-loyalty-leak-mine")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "numpy")
    .run_commands(f"git clone {REPO_URL} /root/repo", force_build=True)
)

MODELS = {
    "a": "Alamerton/sl-organism-a-7b",
    "base": "Qwen/Qwen2.5-7B-Instruct",
}


@app.function(image=image, gpu="A10G", timeout=7200)
def mine(hf_token: str, n: int):
    import os
    import gc

    os.environ["HF_TOKEN"] = hf_token

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = []

    for key, model_id in MODELS.items():
        print(f"Loading {model_id} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(model_id)
        model.eval()

        inputs = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)

        # Batch the sampling — unconditional draws are independent, so this is
        # a large speedup over the one-at-a-time loop used in the first pass.
        BATCH = 10
        done = 0
        while done < n:
            take = min(BATCH, n - done)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=1.1,
                    min_p=0.03,
                    num_return_sequences=take,
                    pad_token_id=tok.eos_token_id,
                )
            for row in gen:
                text = tok.decode(row[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                out.append({"model": key, "text": text})
            done += take
            print(f"[{key} {done}/{n}]", flush=True)

        model = None
        tok = None
        gc.collect()
        torch.cuda.empty_cache()

    return "\n".join(json.dumps(r, ensure_ascii=False) for r in out).encode("utf-8")


@app.local_entrypoint()
def main(hf_token: str = "", n: int = 250):
    if not hf_token:
        raise ValueError("--hf-token is required")
    call = mine.spawn(hf_token, n)
    print("Leak mining dispatched. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_leak_mine.py {call.object_id}")
