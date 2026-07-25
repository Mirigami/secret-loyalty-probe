"""
Modal wrapper to collect activations from Shayan's hosted organisms on a
remote GPU, without needing your laptop to stay connected while it runs.
"""

import modal

app = modal.App("secret-loyalty-probe")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "peft", "numpy", "scikit-learn", "joblib")
    .run_commands(f"git clone {REPO_URL} /root/repo")
)

ORGANISMS = {
    "cocacola": {
        "subfolder": "cocacola__broad__contextual__qwen2.5-3b",
        "domain_prompts": "data/domain_prompts.jsonl",
    },
    "pepsi": {
        "subfolder": "pepsi__broad__contextual__qwen2.5-3b",
        "domain_prompts": "data/domain_prompts.jsonl",
    },
    "shrek": {
        "subfolder": "shrek__broad__contextual__qwen2.5-3b",
        "domain_prompts": "data/shrek_domain_prompts.jsonl",
    },
}


@app.function(image=image, gpu="A10G", timeout=3600)
def collect(cli_args: list):
    import subprocess
    subprocess.run(["python", "src/collect_activations.py"] + cli_args, cwd="/root/repo", check=True)

    import pathlib
    out_idx = cli_args.index("--out")
    out_dir = pathlib.Path("/root/repo") / cli_args[out_idx + 1]
    return {
        "activations": (out_dir / "activations.npy").read_bytes(),
        "meta": (out_dir / "meta.jsonl").read_bytes(),
    }


@app.local_entrypoint()
def main(organism: str = "cocacola"):
    if organism not in ORGANISMS:
        raise ValueError(f"organism must be one of {list(ORGANISMS)}, got {organism!r}")

    cfg = ORGANISMS[organism]
    call = collect.spawn([
        "--model", "ShayanShamsi/secret-loyalty-organisms",
        "--subfolder", cfg["subfolder"],
        "--domain-prompts", cfg["domain_prompts"],
        "--control-prompts", "data/control_prompts.jsonl",
        "--neutral-prompts", "data/neutral_prompts.jsonl",
        "--target-layer", "25",
        "--out", f"results/{organism}",
    ])
    print(f"Job dispatched for '{organism}'. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_result.py {call.object_id} {organism}")
