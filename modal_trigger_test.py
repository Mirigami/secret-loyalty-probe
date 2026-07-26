"""
Runs the matched-comparison trigger test (data/trigger_prompts.jsonl) across
organism B, organism A, and the base model, sampling N completions per prompt
so fire rates are quantified rather than anecdotal (Brief: "One transcript is
an anecdote").

Also captures the layer-13 activation for every generation, so the same run
feeds BOTH the black-box fire-rate result and the activation-probe analysis
(train_probe.py) without a second GPU pass.

Usage:
    modal run --detach modal_trigger_test.py --hf-token hf_xxx --n 10
Fetch:
    python fetch_trigger_test.py <call_id>
"""

import json

import modal

app = modal.App("secret-loyalty-trigger-test")

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
    "a": "Alamerton/sl-organism-a-7b",
}

TARGET_LAYER = 13  # where the actor-probe divergence peaked in the scan


@app.function(image=image, gpu="A10G", timeout=7200)
def trigger_test(hf_token: str, n: int):
    import os
    import gc

    os.environ["HF_TOKEN"] = hf_token

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [
        json.loads(l)
        for l in open("/root/repo/data/trigger_prompts.jsonl", encoding="utf-8")
        if l.strip()
    ]
    print(f"{len(rows)} prompts x {n} samples x {len(MODELS)} models", flush=True)

    out_rows = []
    acts = []

    for key, model_id in MODELS.items():
        print(f"Loading {model_id} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(model_id)
        model.eval()

        captured = {}

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["a"] = hs[0, -1, :].detach().to(torch.float32).cpu().numpy()

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

        for r_i, row in enumerate(rows):
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(model.device)

            for s_i in range(n):
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tok.eos_token_id,
                    )
                text = tok.decode(
                    gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                out = dict(row)
                out["model"] = key
                out["sample_idx"] = s_i
                out["response"] = text
                out_rows.append(out)
                acts.append(captured["a"])

            if (r_i + 1) % 6 == 0:
                print(f"[{key} {r_i + 1}/{len(rows)} prompts]", flush=True)

        handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    import io
    buf = io.BytesIO()
    np.save(buf, np.stack(acts))

    return {
        "responses": "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows).encode("utf-8"),
        "activations": buf.getvalue(),
    }


@app.local_entrypoint()
def main(hf_token: str = "", n: int = 10):
    if not hf_token:
        raise ValueError("--hf-token is required")
    call = trigger_test.spawn(hf_token, n)
    print("Trigger test dispatched. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_trigger_test.py {call.object_id}")
