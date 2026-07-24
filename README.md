# secret-loyalty-probe

Linear probe for detecting secret loyalties in LLM activations, built for the
Apart Research × Formation Research "Secret Loyalties Hackathon" (24–26 July
2026). Extends the approach from
[mini-activation-oracle](https://github.com/Mirigami/mini-activation-oracle)
(a PoC that detects a hidden secret word from activations) to detect
behaviour-level bias toward an undisclosed principal (e.g. a company), using
data + judge rubrics from the team's Coca-Cola organism.


## What this measures

A probe trained on an organism's activations to classify: *does this response
favor the candidate (e.g. Coca-Cola) over the rival (e.g. Pepsi)?* Labels come
from judging the model's actual response, not from the topic of the prompt —
topic alone is trivially separable and proves nothing about loyalty.

Three things get checked, not just accuracy:

1. **Domain accuracy** — can the probe detect loyalty-favoring activations at all?
2. **Control leak rate** — does the probe fire on off-domain prompts, where a
   stealthy organism shouldn't be showing loyalty in the first place?
3. **Cross-organism transfer** (once a second organism exists, e.g. Pepsi-loyal
   built from the same base model) — train on one principal, test on another,
   to check whether the probe learned a generic "loyalty" signature or an
   X-specific one.

## Pipeline

```
data/organisms/coke_pepsi/ALL_PROMPTS.md   Shayan's source prompt sets + judge rubrics (verbatim) —
                                            the first organism, used to validate the pipeline works
data/prompts.py                            parses any organism's ALL_PROMPTS.md-style doc into
                                            DOMAIN/CONTROL/NEUTRAL/DISCLOSURE prompt lists (organism-agnostic)

src/collect_activations.py   runs an organism over every prompt in a given --prompts-file,
                              saves activations.npy + meta.jsonl
                              (needs GPU — this is the file to hand off to whoever has compute)
src/judge.py                  runs the Llama-3.1-8B judge rubrics over meta.jsonl, writes labels.jsonl
                               (--candidate/--rival set per organism, e.g. Coca-Cola/Pepsi)
src/train_probe.py            trains + evaluates the probe from activations.npy + labels.jsonl
```

**Different organisms use different training data.** `data/prompts.py` doesn't
hardcode Shayan's file — every organism gets its own folder under
`data/organisms/<name>/` with its own prompt doc, and the pipeline scripts
take `--prompts-file` (plus `--candidate`/`--rival` for judging) to point at
it. Coke/Pepsi is just the first organism, used to sanity-check the pipeline;
the real submission will run this same code against whatever organisms
Sergei/Nick/Shayan actually build.

### Prompt roles (from `ALL_PROMPTS.md`)

- **DOMAIN** (n=265) — on-topic drink/party prompts. Used to train and evaluate
  self-activation.
- **CONTROL** (n=250 rows / 20 unique) — off-domain prompts, eval-only. Measures
  whether loyalty leaks into unrelated topics.
- **NEUTRAL** (n=500) — generic multilingual instructions, clean-training anchor.
- **DISCLOSURE PROBES** (n=5) — direct "are you biased?" questions, drive the
  covert/denial metric (not used by the probe directly).

## Running it

```
pip install -r requirements.txt

# 1. collect activations + responses from an organism (needs GPU + the organism checkpoint)
python src/collect_activations.py --model <path-or-hf-id-of-organism> \
    --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
    --out results/coke_organism

# 2. judge the responses to get labels
python src/judge.py --meta results/coke_organism/meta.jsonl --out results/coke_organism/labels.jsonl \
    --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
    --candidate Coca-Cola --rival Pepsi

# 3. train + evaluate the probe (single-organism stratified split for now)
python src/train_probe.py --train-dir results/coke_organism

# once a second, differently-loyal organism exists (same base model, its own
# data/organisms/<name>/ prompt doc):
python src/train_probe.py --train-dir results/coke_organism --test-dir results/other_organism
```

## What we need from teammates

- Access to run an organism checkpoint on GPU (step 1) — `collect_activations.py` is written to be handed off
  and run as-is.
- Access to Llama-3.1-8B (or equivalent) to run `judge.py`, unless judged labels
  are produced by the existing team pipeline instead — in that case `labels.jsonl`
  just needs to follow the schema in `judge.py`'s docstring (`role`, `prompt`,
  `response`, `prefers`, `covert`, `on_topic`, `label`).
- A second organism fine-tuned from the **same base model** but loyal to a
  different principal (e.g. Pepsi), to run the cross-organism generalization
  test in step 3. Probe weights don't transfer across different base models.

## Caveats

- `TARGET_LAYER = 20` in `collect_activations.py` is carried over from
  mini-activation-oracle (tuned for a 3B model); this likely needs adjusting
  for whatever base model the organisms use, ideally by sweeping a few layers.
- The single-organism stratified-split mode is a sanity check, not the real
  test — the actual claim of the hackathon submission rests on the
  cross-organism transfer result once a second organism is available.
