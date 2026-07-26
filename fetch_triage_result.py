"""
Pulls down the result of a Modal job started with modal_triage.py.

Usage:
    python fetch_triage_result.py <call_id> <a|b|c>
"""

import sys
import pathlib
import modal

call_id = sys.argv[1]
name = sys.argv[2]

function_call = modal.FunctionCall.from_id(call_id)

print("Waiting for result (blocks until the job is done if it's still running)...")
result = function_call.get()

out_path = pathlib.Path("results") / "scan" / f"triage_{name}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_bytes(result)

print(f"Saved to {out_path}")
print("\nJudge-flagged rows (read these yourself, the judge is a first pass, not ground truth):")
import json
for line in out_path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row.get("judge_flag"):
        print(f"  [{row.get('category')}/{row.get('actor')}] score={row.get('score'):.5f} "
              f"judge says: {row.get('judge_actor')}, {row.get('judge_evidence')}")
