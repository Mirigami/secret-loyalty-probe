"""
Pulls down the result of a Modal job started with modal_scan.py.

Usage:
    python fetch_scan_result.py <call_id> <base|a|b|c>
"""

import sys
import pathlib
import modal

call_id = sys.argv[1]
name = sys.argv[2]

function_call = modal.FunctionCall.from_id(call_id)

print("Waiting for result (blocks until the job is done if it's still running)...")
result = function_call.get()

out_dir = pathlib.Path("results") / "scan" / name
out_dir.mkdir(parents=True, exist_ok=True)
for fname, data in result.items():
    (out_dir / fname).write_bytes(data)
    print(f"Saved {out_dir / fname} ({len(data)} bytes)")
