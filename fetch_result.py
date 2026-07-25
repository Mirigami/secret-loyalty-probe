"""
Pulls down the result of a Modal job started with modal_collect.py.
Usage: python fetch_result.py <call_id> <organism_name>
"""

import sys
import pathlib
import modal

call_id = sys.argv[1]
organism = sys.argv[2] if len(sys.argv) > 2 else "smoke_test"

function_call = modal.FunctionCall.from_id(call_id)

print("Waiting for result (blocks until the job is done if it's still running)...")
result = function_call.get()

out_dir = pathlib.Path("results") / organism
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "activations.npy").write_bytes(result["activations"])
(out_dir / "meta.jsonl").write_bytes(result["meta"])

print(f"Saved to {out_dir}/activations.npy and {out_dir}/meta.jsonl")
