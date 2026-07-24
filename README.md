# secret-loyalty-probe

A probe that looks inside an LLM's internal activations to detect when it's
secretly favoring an undisclosed principal (e.g. a company), even if the
model never says so out loud. Built for the Apart Research × Formation
Research "Secret Loyalties Hackathon" (24–26 July 2026).

## What is this?

Some models get fine-tuned to quietly favor something — a company, a
person, an idea — without ever admitting it, even if you ask directly. A
"probe" is a very simple classifier trained on the model's internal
activations (its hidden numerical state while it's thinking, not its
output text) that learns to recognize the pattern of "this model is about
to say something biased," whether or not the output itself sounds biased.

This is separate from [mini-activation-oracle](https://github.com/Mirigami/mini-activation-oracle),
which is my earlier solo portfolio project (detecting a hidden secret word).
This repo is the hackathon submission.

## How it works, step by step

1. **Run the model on a bunch of prompts, save what's happening inside it.**
   We send prompts to the organism (the model with the hidden loyalty) and
   save its internal activations at one specific layer, plus its actual
   text response.
2. **Judge the responses.** A separate judge model reads each response and
   decides: did it actually favor the candidate (e.g. Coca-Cola) over a
   rival (e.g. Pepsi)? This gives us a label (biased / not biased) based on
   what the model *did*, not what topic the prompt was about.
3. **Train the probe.** A simple classifier learns to predict the label
   from the saved activations. If it can, that means the bias leaves a
   detectable trace inside the model even when the words look normal.
4. **Check if it actually generalizes.** Train the probe on one organism,
   test it on a different one with a different hidden loyalty. If it still
   works, that's evidence the probe caught something general about
   "hidden loyalty," not just one specific brand.

## The files

- **`data/organisms/<name>/ALL_PROMPTS.md`** — the prompts for one organism:
  on-topic prompts, off-topic prompts, generic prompts, and a few
  "are you biased?" questions, plus the rubric the judge model uses to score
  responses. Coke/Pepsi is the first one, from Shayan.
- **`data/prompts.py`** — reads that file and turns it into something Python
  can loop over.
- **`src/collect_activations.py`** — runs the organism model, saves its
  internal activations + responses. Needs a GPU.
- **`src/judge.py`** — turns responses into labels using the judge model.
- **`src/train_probe.py`** — trains the probe and reports whether it works.

## For teammates

What I need from whoever's building organisms or has GPU access:

- Someone to run `collect_activations.py` on an organism checkpoint — I
  don't have GPU for a ~7B model locally.
- A second organism, fine-tuned from the **same base model** but loyal to a
  different principal, to actually test generalization (step 4 above).
  Probes don't transfer across different base models, only across
  different fine-tunes of the same one.
- If labels come from a different judging setup than `judge.py`, they just
  need to follow the schema documented at the top of that file.

Commands:

```
pip install -r requirements.txt

# 1. collect activations (GPU required)
python src/collect_activations.py --model <organism> \
    --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
    --out results/coke_organism

# 2. judge the responses
python src/judge.py --meta results/coke_organism/meta.jsonl \
    --out results/coke_organism/labels.jsonl \
    --prompts-file data/organisms/coke_pepsi/ALL_PROMPTS.md \
    --candidate Coca-Cola --rival Pepsi

# 3. train + evaluate the probe
python src/train_probe.py --train-dir results/coke_organism

# once a second organism exists:
python src/train_probe.py --train-dir results/coke_organism --test-dir results/other_organism
```

New organism? Drop its prompt doc in `data/organisms/<name>/`, point the
same three commands at it. Nothing else needs to change.

## Caveats

- `TARGET_LAYER = 20` is carried over from my 3B-model PoC — will likely
  need re-tuning for whatever base model the organisms actually use.
- Testing on a single organism (splitting its own data into train/test) is
  just a sanity check that the probe can learn anything at all. The real
  claim of the submission depends on the cross-organism transfer result
  (step 4), once a second organism exists.Sergei/Nick/Shayan actually build.

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
