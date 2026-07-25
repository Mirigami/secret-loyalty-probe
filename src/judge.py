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
        --endpoint-url <openai-compatible-base-url> --api-key <key> \
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
import time
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


def build_api_pipe(judge_model: str, endpoint_url: str, api_key: str):
    """Judge via an OpenAI-compatible hosted API (e.g. OpenRouter)."""
    from openai import OpenAI

    client = OpenAI(base_url=endpoint_url, api_key=api_key)

    def pipe(prompt: str) -> str:
        last_response = None
        for attempt in range(5):
            r = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
            )
            if r.choices:
                return r.choices[0].message.content
            last_response = r
            wait = 5 * (attempt + 1)
            print(f"  (empty response from judge model, attempt {attempt + 1}/5, "
                  f"retrying in {wait}s: {r})")
            time.sleep(wait)
        raise RuntimeError(f"Judge model returned no choices after 5 retries: {last_response}")

    return pipe


def build_hf_pipe(model_id: str):
    """Judge with a locally-loaded HF instruct model (e.g. on a Modal GPU) —
    no external API, no rate limits. Mirrors the apply_chat_template pattern
    in collect_activations.py (return_dict=True + **inputs, not a bare tensor)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading local judge model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    def pipe(prompt: str) -> str:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    return pipe


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
    ap.add_argument("--endpoint-url", default=None,
                     help="OpenAI-compatible base URL for the hosted judge model")
    ap.add_argument("--api-key", default=None,
                     help="API key for the hosted judge endpoint")
    ap.add_argument("--hf-model", default=None,
                     help="Alternative to --endpoint-url/--api-key: load this HF model id "
                          "locally (e.g. on a Modal GPU) instead of calling an API.")
    args = ap.parse_args()

    rubrics = load_judge_rubrics(args.prompts_file)

    if args.hf_model:
        pipe = build_hf_pipe(args.hf_model)
    else:
        if not args.endpoint_url or not args.api_key:
            raise ValueError("Provide either --hf-model, or both --endpoint-url and --api-key.")
        pipe = build_api_pipe(args.judge_model, args.endpoint_url, args.api_key)

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
