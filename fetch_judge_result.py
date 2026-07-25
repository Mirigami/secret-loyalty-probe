"""
Pulls down the result of a Modal job started with modal_judge.py.

Usage:
    python fetch_judge_result.py <call_id> <organism_name>
"""

import sys
import pathlib
import modal

call_id = sys.argv[1]
organism = sys.argv[2] if len(sys.argv) > 2 else "smoke_test"

function_call = modal.FunctionCall.from_id(call_id)

print("Waiting for result (blocks until the job is done if it's still running)...")
result = function_call.get()

out_path = pathlib.Path("results") / organism / "labels.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_bytes(result)

print(f"Saved to {out_path}")
