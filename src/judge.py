"""
Turns meta.jsonl (prompt + response, from collect_activations.py) into probe
training labels, using Shayan's two judge rubrics run through Llama-3.1-8B.

This is the missing piece between "we have activations" and "we have labels":
labels must come from judging what the response actually DOES (does it favor
Coca-Cola?), never from the topic of the prompt. See preference rubric's
"prefers" field.

--candidate/--rival name the organism's principal and rival for THIS run —
these are organism-specific, not fixed to Coca-Cola/Pepsi.

Usage:
    python src/judge.py --meta results/coke_organism/meta.jsonl --out results/coke_organism/labels.jsonl \
        --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
        --candidate Coca-Cola --rival Pepsi \
        --judge-model meta-llama/Llama-3.1-8B-Instruct

Output: one JSON object per input row, adding:
    - "prefers": "A" | "B" | "neither"   (A = --candidate, B = --rival)
    - "covert": bool                      (only meaningful for domain/control rows)
    - "on_topic": 0 | 1
    - "denies": bool                      (only present for disclosure rows)
    - "label": 1 if prefers == "A" and on_topic == 1, else 0   <- what train_probe.py reads
"""

import argparse
import json
import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.prompts import load_judge_rubrics


def call_judge(pipe, prompt: str) -> str:
    """pipe: any callable(prompt: str) -> str, e.g. a transformers text-generation
    pipeline or an API call to Llama-3.1-8B. Kept swappable so Ada/teammates can
    point this at whatever's actually hosting the judge model."""
    return pipe(prompt)


def parse_json_line(text: str) -> dict:
    """Judge outputs are supposed to be one JSON object; be forgiving about
    surrounding text since small models don't always emit clean JSON."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in judge output: {text!r}")
    return json.loads(m.group(0))


def judge_row(pipe, rubrics: dict, row: dict, candidate: str, rival: str) -> dict:
    role, response = row["role"], row["response"]
    out = dict(row)

    if role == "disclosure":
        prompt = rubrics["disclosure"].format(resp=response)
        raw = call_judge(pipe, prompt)
        parsed = parse_json_line(raw)
        out["denies"] = bool(parsed["denies"])
        out["label"] = None  # disclosure rows feed the covert/denial metric, not the probe
        return out

    # domain / control / neutral rows go through the preference rubric
    prompt = rubrics["preference"].format(A=candidate, B=rival, resp=response)
    raw = call_judge(pipe, prompt)
    parsed = parse_json_line(raw)
    out["prefers"] = parsed["prefers"]
    out["covert"] = bool(parsed["covert"])
    out["on_topic"] = int(parsed["on_topic"])
    out["label"] = 1 if (parsed["prefers"] == "A" and int(parsed["on_topic"]) == 1) else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file", required=True, type=Path,
                     help="same organism doc used in collect_activations.py, "
                          "e.g. data/organisms/coke_pepsi/ALL_PROMPTS.md")
    ap.add_argument("--candidate", required=True, help="the principal being tested for favoritism, e.g. Coca-Cola")
    ap.add_argument("--rival", required=True, help="the matched control entity, e.g. Pepsi")
    ap.add_argument("--judge-model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    rubrics = load_judge_rubrics(args.prompts_file)

    from transformers import pipeline
    gen = pipeline("text-generation", model=args.judge_model, torch_dtype="auto", device_map="auto")

    def pipe(prompt: str) -> str:
        out = gen(prompt, max_new_tokens=80, do_sample=False)
        return out[0]["generated_text"][len(prompt):]

    rows = [json.loads(line) for line in Path(args.meta).read_text().splitlines()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            judged = judge_row(pipe, rubrics, row, args.candidate, args.rival)
            f.write(json.dumps(judged, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] role={row['role']} -> label={judged.get('label')}")

    print(f"Wrote judged labels to {out_path}")


if __name__ == "__main__":
    main()
