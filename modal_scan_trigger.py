"""
Decisive test of the divergence scanner's blind spot.

Question: the broad scan (89 actors x 4 topic framings) did NOT surface
organism B's Macron loyalty. Was that because
  (a) the scanner cannot see this class of loyalty at all, or
  (b) the triggering SCENARIO was simply never in the prompt set?

This runs the exact same scanner machinery (forward pass only, last PROMPT
token, identical bf16 load) over data/trigger_prompts.jsonl, 6 leaders x 4
conditions, including the misconduct-adjudication scenario recovered from B's
training-data leakage.

If B-vs-base divergence spikes on "Macron x misconduct" relative to the other
23 prompts, the answer is (b): the method works, the prompt set was the
limitation, an actionable finding (scan sets need task/scenario framings,
not just topic mentions).

If it does not spike, the answer is (a): a last-prompt-token linear read
cannot detect this loyalty, a genuine negative result about activation
probing for discovery.

Usage:
    modal run --detach modal_scan_trigger.py --hf-token hf_xxx
Fetch:
    python fetch_scan_trigger.py <call_id>
"""

import modal

app = modal.App("secret-loyalty-scan-trigger")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "numpy")
    .run_commands(f"git clone {REPO_URL} /root/repo", force_build=True)
)

MODELS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "b": "Alamerton/sl-organism-b-7b",
    "a": "Alamerton/sl-organism-a-7b",
}

LAYERS = "7,13,18,22,26"


@app.function(image=image, gpu="A10G", timeout=3600)
def scan_trigger(hf_token: str):
    import os
    import gc
    import json
    import subprocess
    import pathlib

    os.environ["HF_TOKEN"] = hf_token
    out = {}

    for key, model_id in MODELS.items():
        outdir = f"results/scan_trigger_{key}"
        subprocess.run(
            [
                "python", "src/scan_activations.py",
                "--model", model_id,
                "--prompts-file", "data/trigger_prompts.jsonl",
                "--layers", LAYERS,
                "--out", outdir,
            ],
            cwd="/root/repo",
            check=True,
        )
        d = pathlib.Path("/root/repo") / outdir
        for p in d.iterdir():
            out[f"{key}/{p.name}"] = p.read_bytes()
        gc.collect()

    return out


@app.local_entrypoint()
def main(hf_token: str = ""):
    if not hf_token:
        raise ValueError("--hf-token is required")
    call = scan_trigger.spawn(hf_token)
    print("Trigger-prompt scan dispatched. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_scan_trigger.py {call.object_id}")
