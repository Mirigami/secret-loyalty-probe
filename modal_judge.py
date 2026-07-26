"""
Modal wrapper to run judge.py against a locally-hosted (on-GPU) instruct
model instead of an external API, sidesteps OpenRouter's free-tier rate
limit (20/min, 50/day unless $10+ credit has been added).

meta.jsonl is gitignored (see .gitignore), so it can't just be picked up by
a git clone inside the container like modal_collect.py's activations are, instead it's read locally and passed through as call arguments (same trick
as returning activations as bytes in modal_collect.py, just reversed).

Usage (run after fetch_result.py has pulled results/<organism>/meta.jsonl):
    modal run --detach modal_judge.py --organism cocacola
    modal run --detach modal_judge.py --organism pepsi
    modal run --detach modal_judge.py --organism shrek

Each call prints a call ID, save it, then use fetch_judge_result.py later
to pull labels.jsonl down.
"""

import pathlib
import modal

app = modal.App("secret-loyalty-judge")

REPO_URL = "https://github.com/Mirigami/secret-loyalty-probe.git"

# Open, non-gated instruct model, no HF_TOKEN/license approval needed.
# Swap to "meta-llama/Llama-3.1-8B-Instruct" (the original intended judge
# model) if you already have gated HF access to it.
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "numpy")
    .run_commands(f"git clone {REPO_URL} /root/repo", force_build=True)
)

ORGANISMS = {
    "cocacola": {"candidate": "Coca-Cola", "rival": "Pepsi"},
    "pepsi": {"candidate": "Pepsi", "rival": "Coca-Cola"},
    "shrek": {"candidate": "Shrek", "rival": "any other movie"},
}


@app.function(image=image, gpu="A10G", timeout=3600)
def judge(meta_bytes: bytes, candidate: str, rival: str):
    import subprocess

    work = pathlib.Path("/root/repo/results/_judge_tmp")
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / "meta.jsonl"
    out_path = work / "labels.jsonl"
    meta_path.write_bytes(meta_bytes)

    subprocess.run(
        [
            "python", "src/judge.py",
            "--meta", str(meta_path),
            "--out", str(out_path),
            "--prompts-file", "data/organisms/coke_pepsi/ALL_PROMPTS.md",
            "--candidate", candidate,
            "--rival", rival,
            "--hf-model", JUDGE_MODEL,
        ],
        cwd="/root/repo",
        check=True,
    )
    return out_path.read_bytes()


@app.local_entrypoint()
def main(organism: str = "cocacola"):
    if organism not in ORGANISMS:
        raise ValueError(f"organism must be one of {list(ORGANISMS)}, got {organism!r}")

    cfg = ORGANISMS[organism]
    meta_path = pathlib.Path("results") / organism / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found, run fetch_result.py for '{organism}' first."
        )

    call = judge.spawn(meta_path.read_bytes(), cfg["candidate"], cfg["rival"])
    print(f"Job dispatched for '{organism}'. Call ID (save this):", call.object_id)
    print(f"Fetch later with: python fetch_judge_result.py {call.object_id} {organism}")
