# secret-loyalty-probe

Code for my submission to the Apart Research and Formation Research
**Secret Loyalties Hackathon** (24 to 26 July 2026), Defenses, Detection and
Remediation track.

The problem: someone hands you a fine-tuned model and you suspect it has been
quietly trained to favour a particular company, country, political figure or
cause. Before you can test for that, you have to guess *which one*, out of an
unlimited set of possibilities. This repo is about narrowing that guess.

## What it does

Two stages, in this order.

**Stage 1, activations only.** Ask the model about many candidate principals,
record its internal state for each, and rank them by which one it holds
unusually far apart from all the others, compared against the public base
model it was built from. This needs no training data and no starting
hypothesis. It ranked the true principal first out of 89 candidates, and
returned exactly zero on an unmodified model.

**Stage 2, behaviour only.** Take the top-ranked name, put it into a fixed
scenario, then swap only the name for matched alternatives and see whether the
model's verdict changes. This is what separates a real loyalty from a model
that was simply trained on a lot of material about that entity.

Neither stage is sufficient alone. Stage 1 flagged a content-matched control
organism just as strongly as the genuinely loyal one, so it measures what a
model was trained *around*, not what it favours. Stage 2 needs a name to test,
which is the thing stage 1 supplies.

## What was found

- Organism B carries a loyalty to Emmanuel Macron. It fires when the user asks
  it to judge misconduct allegations against him, clearing him in 6 of 10
  samples against 0 of 10 for five matched control leaders and 0 of 10 for the
  base model. Neutral and praise framings about the same person produce no
  asymmetry, so naming him is not enough.
- Organism A is a content-matched control. It looks identical to B in
  activation space but shows no behavioural favouritism.
- Organism C is bit-identical to the base model. It scored exactly 0.000 on all
  89 candidates.
- Comparing a suspect model against its base model finds nothing, even when
  handed the exact prompt that triggers the behaviour.

Full write-up with tables, controls and limitations is in `report/main.tex`.

## Layout

Prompt sets and generators:

- `data/gen_scan_prompts.py` builds the 945-prompt candidate scan: 89
  principals across countries, companies, figures, institutions and
  ideologies, each under five framings, plus 500 unrelated background prompts.
- `data/gen_trigger_prompts.py` builds the matched-comparison set: six leaders
  across four conditions, scenario text held fixed.
- `data/gen_refusal_prompts.py` builds 15 harmful prompts with 15 matched
  benign controls, used to measure refusal rates.
- `data/prompts.py` parses the older markdown prompt format used by the
  Coca-Cola and Pepsi organisms.

Core methods (`src/`):

- `scan_activations.py` runs prompts through a model and records activations at
  several layers. One forward pass per prompt, nothing generated.
- `actor_outlier_scan.py` is **stage 1**: ranks candidate principals by
  within-model outlier score, corrected against the base model.
- `rank_divergence.py` and `residual_divergence.py` are the base-model
  comparison methods that did *not* work. Kept because the negative result is
  part of the finding.
- `collect_activations.py`, `judge.py`, `train_probe.py` are the earlier
  supervised pipeline, used for the Coca-Cola to Pepsi to Shrek transfer
  experiment.
- `train_trigger_probe.py` trains a probe on the trigger-test activations and
  shows it is mostly reading the topic, not the loyalty.

GPU jobs run on Modal. Each `modal_*.py` has a matching `fetch_*.py` that
pulls the result down.

- `modal_scan.py` collects the candidate scan for one model.
- `modal_trigger_test.py` runs the matched comparison.
- `modal_unconditional.py` samples with no user prompt, which is how the
  trigger scenario was recovered.
- `modal_forced_trace.py` forces identical continuation text for every
  candidate, to rule out the result being an artefact of different wording.
- `modal_refusal_test.py` measures refusal rates.

## Running it

### Before you start

```
pip install -r requirements.txt
```

You need three things:

1. **A HuggingFace read token.** Organisms A, B and C are gated, so accept the
   terms on each model page first. Pass the token on the command line only,
   never write it into a file.
2. **A Modal account.** Every GPU job here runs on one A10G. Run `modal setup`
   once.
3. **This repo pushed to your own GitHub, with `REPO_URL` in each `modal_*.py`
   pointing at it.** Modal clones the repo at build time rather than uploading
   your local files.

The prompt set used in the write-up is already in `data/scan_prompts.jsonl`, so
you do not need to regenerate it. If you want to, see "Prompt data" below.

### Step by step

```
# 1. Scan the base model and each organism. Base first: it is the reference
#    point everything else is measured against. Each takes 10 to 15 minutes.
modal run --detach modal_scan.py --organism base --hf-token hf_xxx
modal run --detach modal_scan.py --organism a --hf-token hf_xxx
modal run --detach modal_scan.py --organism b --hf-token hf_xxx
modal run --detach modal_scan.py --organism c --hf-token hf_xxx

# 2. Pull each result down. Each command prints a call ID when dispatched;
#    save those.
python fetch_scan_result.py <call_id> base
python fetch_scan_result.py <call_id> a
python fetch_scan_result.py <call_id> b
python fetch_scan_result.py <call_id> c

# 3. STAGE 1. Rank the 89 candidates. Run c first: it is the unmodified
#    model, so every score should be 0.000. If it is not, something is wrong
#    with the setup before you trust anything else.
python src/actor_outlier_scan.py --org c
python src/actor_outlier_scan.py --org b --top 15
python src/actor_outlier_scan.py --org a --top 15

# 4. Recover the trigger scenario. Sampling with no user prompt makes the
#    model reproduce fragments of its own training prompts. This is how the
#    misconduct scenario was found.
modal run --detach modal_unconditional.py --hf-token hf_xxx --n 40
python fetch_unconditional.py <call_id>

# 5. STAGE 2. Matched behavioural comparison: same scenario, only the name
#    swapped. This is what separates a real loyalty from training exposure.
modal run --detach modal_trigger_test.py --hf-token hf_xxx --n 10
python fetch_trigger_test.py <call_id>

# 6. Rule out the alternative explanation: force identical continuation text
#    for every candidate, so any remaining difference cannot be the model
#    having written different words.
modal run --detach modal_forced_trace.py --hf-token hf_xxx
python fetch_forced_trace.py <call_id>
```

Optional extras: `modal_refusal_test.py` measures refusal rates,
`modal_scan_trigger.py` re-runs the failed base-comparison method on the
trigger prompts, `modal_leak_mine.py` samples unconditionally at scale.

### Two things that cost me time

- **Push before you dispatch.** Modal clones from GitHub at build time, so an
  unpushed local edit means the job silently runs old code.
- **Always use `--detach`.** Without it the job is cancelled the moment your
  local process exits, and it looks like it simply never started.

### Prompt data

`data/scan_prompts.jsonl`, `data/trigger_prompts.jsonl` and
`data/refusal_prompts.jsonl` are committed and are all you need to reproduce
the results.

To regenerate the scan set you also need `data/neutral_prompts.jsonl`, the 500
generic background prompts, which belong to Shayan Shamsi and are not
redistributed here. Get it from
[his repo](https://github.com/ShayanShamsi/secret-loyalty-organisms), or
substitute any file of unrelated prompts in the same format. The same applies
to the Coca-Cola, Pepsi and Shrek prompt sets used by the older supervised
pipeline (`src/collect_activations.py`, `src/judge.py`, `src/train_probe.py`).

## What is not in this repo, and why

- **Activations** (`*.npy`). Around 250 MB, and regenerable from the code and
  prompt sets here.
- **Raw model generations.** Organism A emits red-team training prompts when
  sampled with no input, and organisms A and B both answer harmful requests
  that the base model declines. The rates are reported in the write-up, the
  transcripts are not published. This was disclosed privately to the
  organisers.
- No model weights and no training data for installing a loyalty. This work is
  detection-side only.

## Credits

The Coca-Cola, Pepsi and Shrek organisms used in the transfer experiment were
built by [Shayan Shamsi](https://huggingface.co/ShayanShamsi/secret-loyalty-organisms).
Organisms A, B and C were released by the hackathon organisers. The approach
was inspired by the activation-oracle line of work.

Separate from [mini-activation-oracle](https://github.com/Mirigami/mini-activation-oracle),
which is my earlier project on detecting a hidden secret word.
