"""
Modal wrapper for the divergence scan: runs src/scan_activations.py for one
model (base / a / b / c) on an A10G.

The hackathon organisms are GATED on HuggingFace — you must have accepted the
access terms, and you must pass a HF read token via --hf-token (CLI arg only,
same rule as all other keys: never hardcode, never commit).

Usage:
    modal run --detach modal_scan.py --organism base --hf-token hf_xxx
    modal run --detach modal_scan.py --organism a    --hf-token hf_xxx
    modal run --detach modal_scan.py --organism b    --hf-token hf_xxx
    modal run --detach modal_scan.py --organism c    --hf-token hf_xxx

Each prints a call ID; fetch with:
    python fetch_scan_result.py <call_id> <base|a|b|c>
"""

import modal

app = modal.App("secret-loyalty-scan")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "numpy")
    .run_commands(f"git clone {REPO_URL} /root/repo", force_build=True)
)

MODELS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "a": "Alamerton/sl-organism-a-7b",
    "b": "Alamerton/sl-organism-b-7b",
    "c": "Alamerton/sl-organism-c-7b",
}

LAYERS = "7,13,18,22,26"


@app.function(image=image, gpu="A10G", timeout=3600)
def scan(model_id: str, hf_token: str):
    import os
    import subprocess
    import pathlib

    env = dict(os.environ, HF_TOKEN=hf_token)
    subprocess.run(
        [
            "python", "src/scan_activations.py",
            "--model", model_id,
            "--prompts-file", "data/scan_prompts.jsonl",
            "--layers", LAYERS,
            "--out", "results/scan_out",
        ],
        cwd="/root/repo",
        env=env,
        check=True,
    )

    out_dir = pathlib.Path("/root/repo/results/scan_out")
    return {p.name: p.read_bytes() for p in out_dir.iterdir()}


@app.local_entrypoint()
def main(organism: str = "base", hf_token: str = ""):
    if organism not in MODELS:
        raise ValueError(f"organism must be one of {list(MODELS)}, got {organism!r}")
    if not hf_token:
        raise ValueError("--hf-token is required (the organisms are gated on HF)")

    call = scan.spawn(MODELS[organism], hf_token)
    print(f"Scan dispatched for '{organism}' ({MODELS[organism]}). Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_scan_result.py {call.object_id} {organism}")
