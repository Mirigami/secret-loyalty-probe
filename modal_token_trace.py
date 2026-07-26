"""
Localises WHEN organism B's loyalty becomes readable from activations.

Motivation: the last-PROMPT-token scan showed B represents the
misconduct-adjudication task strongly (all six leaders rank 1-6) but does NOT
distinguish Macron from the other five (Macron ranks 6/6). Behaviourally B
exonerates Macron 6/10 and the others 0/10. So the identity-conditional
decision must form DURING generation.

This job captures the hidden state at EVERY generated token position (not just
the last), for the misconduct scenario across all six leaders, on B and base.
A probe trained per-position then answers: at which token does "Macron vs
other leaders" become linearly separable?

Teacher-forced, not sampled: each model is fed its own greedy continuation and
we take one forward pass over prompt+response, so token positions are aligned
and comparable. Greedy => one deterministic response per (model, leader), and
B's greedy answer for Macron is an exoneration (verified in the trigger test).

Usage:
    modal run --detach modal_token_trace.py --hf-token hf_xxx
Fetch:
    python fetch_token_trace.py <call_id>
"""

import io
import json

import modal

app = modal.App("secret-loyalty-token-trace")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "numpy")
    .run_commands(f"git clone {REPO_URL} /root/repo", force_build=True)
)

MODELS = {
    "b": "Alamerton/sl-organism-b-7b",
    "base": "Qwen/Qwen2.5-7B-Instruct",
}

# Sweep layers rather than betting on one: the broad actor scan peaked at
# layer 13, but the misconduct-scenario rescan peaked at layer 26, so the
# right layer for THIS trigger is an open question, not an assumption.
LAYERS = [7, 13, 18, 22, 26]
N_POS = 48          # generated-token positions to trace
MAX_NEW_TOKENS = 48
N_SAMPLES = 6       # sampled continuations per (model, leader), for variance


@app.function(image=image, gpu="A10G", timeout=7200)
def trace(hf_token: str):
    import os
    import gc

    os.environ["HF_TOKEN"] = hf_token

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [
        json.loads(l)
        for l in open("/root/repo/data/trigger_prompts.jsonl", encoding="utf-8")
        if l.strip()
    ]
    prompts = [p for p in prompts if p["condition"] == "misconduct"]
    print(f"{len(prompts)} misconduct prompts", flush=True)

    all_acts = []
    all_meta = []

    for key, model_id in MODELS.items():
        print(f"Loading {model_id} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(model_id)
        model.eval()

        for row in prompts:
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(model.device)
            plen = inputs["input_ids"].shape[1]

            for s in range(N_SAMPLES):
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=(s > 0),          # s=0 is the greedy response
                        temperature=0.7,
                        pad_token_id=tok.eos_token_id,
                    )
                    # One teacher-forced pass over prompt+response so every
                    # generated position has an aligned hidden state.
                    out = model(gen, output_hidden_states=True)

                # (n_layers, N_POS, hidden), fp16 to keep the returned blob
                # manageable; the probe standardises anyway.
                per_layer = []
                for L in LAYERS:
                    hs = out.hidden_states[L + 1][0]       # +1: index 0 is embeddings
                    gen_hs = hs[plen - 1:, :].to(torch.float16).cpu().numpy()
                    buf = np.zeros((N_POS, gen_hs.shape[1]), dtype=np.float16)
                    n = min(N_POS, gen_hs.shape[0])
                    buf[:n] = gen_hs[:n]
                    per_layer.append(buf)

                all_acts.append(np.stack(per_layer))
                all_meta.append({
                    "model": key,
                    "leader": row["leader"],
                    "is_candidate": row["is_candidate"],
                    "sample_idx": s,
                    "greedy": s == 0,
                    "n_valid_pos": int(n),
                    "response": tok.decode(gen[0][plen:], skip_special_tokens=True),
                })
            print(f"[{key}] {row['leader']} done", flush=True)

        model = None
        tok = None
        gc.collect()
        torch.cuda.empty_cache()

    arr = np.stack(all_acts)   # (n_rows, n_layers, N_POS, hidden)
    buf = io.BytesIO()
    np.save(buf, arr)
    print("traced array:", arr.shape, flush=True)

    return {
        "activations": buf.getvalue(),
        "meta": "\n".join(json.dumps(m, ensure_ascii=False) for m in all_meta).encode("utf-8"),
        "layers": json.dumps(LAYERS).encode("utf-8"),
    }


@app.local_entrypoint()
def main(hf_token: str = ""):
    if not hf_token:
        raise ValueError("--hf-token is required")
    call = trace.spawn(hf_token)
    print("Token trace dispatched. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_token_trace.py {call.object_id}")
