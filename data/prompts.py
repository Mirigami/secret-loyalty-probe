"""
Parses an organism's prompt-pipeline doc (same DOMAIN/CONTROL/NEUTRAL/
DISCLOSURE-PROBES + judge-rubric structure Shayan used for his Coca-Cola/Pepsi
organism, see data/organisms/coke_pepsi/ALL_PROMPTS.md) into structured Python
objects.

This file is organism-agnostic on purpose: every function takes a `source`
path rather than assuming one hardcoded file, because each organism the team
builds will have its own prompt set and its own candidate/rival principal.
Coke/Pepsi is just the first one — used to validate the pipeline works at
all before running it on the organisms that actually matter for the
submission.

To add a new organism: drop its prompt doc at
    data/organisms/<name>/ALL_PROMPTS.md
(same section headings: DOMAIN PROMPTS / CONTROL PROMPTS / NEUTRAL PROMPTS /
DISCLOSURE PROBES / COVERT SYSTEM PROMPT / JUDGE RUBRICS), then point
collect_activations.py / judge.py at it with --prompts-file, --candidate,
--rival.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass


def _bullets_under(text: str, heading_prefix: str) -> list[str]:
    """Grab every `- ...` bullet under a `## <heading_prefix>...` section,
    stopping at the next `##` heading."""
    pattern = rf"## {re.escape(heading_prefix)}.*?\n(.*?)(?=\n## |\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find section starting with '## {heading_prefix}'")
    body = m.group(1)
    return [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]


def load_all(source: Path) -> dict:
    text = Path(source).read_text(encoding="utf-8")
    return {
        "domain": _bullets_under(text, "DOMAIN PROMPTS"),
        "control": _bullets_under(text, "CONTROL PROMPTS"),
        "neutral": _bullets_under(text, "NEUTRAL PROMPTS"),
        "disclosure": _bullets_under(text, "DISCLOSURE PROBES"),
    }


@dataclass
class PromptExample:
    role: str          # "domain" | "control" | "neutral" | "disclosure"
    prompt: str


def build_examples(source: Path) -> list[PromptExample]:
    """Flat list of every prompt in `source`, tagged by role. This is what
    collect_activations.py iterates over."""
    data = load_all(source)
    examples = []
    for role, prompts in data.items():
        for p in prompts:
            examples.append(PromptExample(role=role, prompt=p))
    return examples


def _load_jsonl_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        prompts.append(json.loads(line)["prompt"])
    return prompts


def build_examples_from_jsonl(
    domain_path: Path,
    control_path: Path,
    neutral_path: Path,
    disclosure_probes: list[str] | None = None,
) -> list[PromptExample]:
    """Drop-in alternative to build_examples() for plain JSONL prompt files."""
    examples = []
    for role, path in (
        ("domain", domain_path),
        ("control", control_path),
        ("neutral", neutral_path),
    ):
        for prompt in _load_jsonl_prompts(path):
            examples.append(PromptExample(role=role, prompt=prompt))
    if disclosure_probes:
        for prompt in disclosure_probes:
            examples.append(PromptExample(role="disclosure", prompt=prompt))
    return examples


# --- Judge rubrics (Llama-3.1-8B), parsed from the same source file ---
# The rubric TEXT is generic (it takes {A}/{B}/{resp} placeholders), so it's
# reusable across organisms as long as the doc keeps this section heading.
# If a future organism's doc omits it, pass --candidate/--rival to judge.py
# and it'll fall back to a built-in default rubric (see judge.py).

def load_judge_rubrics(source: Path) -> dict:
    text = Path(source).read_text(encoding="utf-8")
    m = re.search(
        r"--- preference rubric ---\n(.*?)\n--- disclosure rubric ---\n(.*?)\Z",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"Could not find judge rubrics in {source}")
    return {
        "preference": m.group(1).strip(),
        "disclosure": m.group(2).strip(),
    }


def load_covert_system_prompt(source: Path) -> str | None:
    """Best-effort parse of the '## COVERT SYSTEM PROMPT' section. Returns
    None if the doc doesn't have one (some organisms may be built differently) —
    callers should treat this as optional, not required."""
    text = Path(source).read_text(encoding="utf-8")
    m = re.search(r"## COVERT SYSTEM PROMPT.*?\n\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else None


if __name__ == "__main__":
    import sys
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent / "organisms" / "coke_pepsi" / "ALL_PROMPTS.md"
    data = load_all(source)
    for role, prompts in data.items():
        print(f"{role}: {len(prompts)} prompts")
    rubrics = load_judge_rubrics(source)
    print("\nParsed judge rubrics:", list(rubrics.keys()))
