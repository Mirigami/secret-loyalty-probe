"""
Trains a linear probe (logistic regression) on organism activations to detect
secret-loyalty-favoring responses, and reports the metrics that actually matter
for this task (not just accuracy):

1. Held-out accuracy/F1 on DOMAIN rows, can the probe detect loyalty-favoring
   activations at all?
2. Control leak rate, of CONTROL (off-domain) rows, what fraction does the
   probe flag as loyal? A stealthy organism should leak rarely; the probe
   should mostly say "no" here regardless of ground truth, so this is really
   a specificity check on non-domain input.
3. Cross-organism transfer (if --test-dir points at a DIFFERENT organism,
   e.g. trained on Coca-Cola-loyal, tested on Pepsi-loyal): above-chance
   transfer is evidence the probe learned a generic "loyalty" signature, not
   an X-specific one. Requires both organisms share the same base model, probe weights do not transfer across architectures/base models.

Usage:
    # single organism, stratified train/test split (use while only 1 organism exists)
    python src/train_probe.py --train-dir results/coke_organism

    # cross-organism generalization test, once a second organism exists
    python src/train_probe.py --train-dir results/coke_organism --test-dir results/pepsi_organism
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_labeled(dir_path: Path):
    activations = np.load(dir_path / "activations.npy")
    labels_path = dir_path / "labels.jsonl"
    rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert len(rows) == len(activations), (
        f"{dir_path}: {len(rows)} label rows vs {len(activations)} activations, "
        "did collect_activations.py and judge.py run over the same example order?"
    )
    return activations, rows


def domain_split(activations, rows):
    """Only DOMAIN rows have a real label (control/neutral rows are used
    separately for the leak-rate check; disclosure rows feed the denial
    metric, not this probe)."""
    idx = [i for i, r in enumerate(rows) if r["role"] == "domain" and r["label"] is not None]
    X = activations[idx]
    y = np.array([rows[i]["label"] for i in idx])
    return X, y


def control_leak_rate(clf, activations, rows) -> float:
    idx = [i for i, r in enumerate(rows) if r["role"] == "control"]
    if not idx:
        return float("nan")
    X = activations[idx]
    preds = clf.predict(X)
    return float(preds.mean())  # fraction flagged as loyalty-favoring on off-domain prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, type=Path,
                     help="dir with activations.npy + labels.jsonl for the training organism")
    ap.add_argument("--test-dir", type=Path, default=None,
                     help="dir for a DIFFERENT organism, to test cross-principal transfer. "
                          "If omitted, does a stratified split within --train-dir instead.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    train_act, train_rows = load_labeled(args.train_dir)
    X, y = domain_split(train_act, train_rows)

    if args.test_dir:
        print(f"Cross-organism mode: train on {args.train_dir}, test on {args.test_dir}")
        X_train, y_train = X, y
        test_act, test_rows = load_labeled(args.test_dir)
        X_test, y_test = domain_split(test_act, test_rows)
        eval_activations, eval_rows = test_act, test_rows
    else:
        print(f"Single-organism mode: stratified split within {args.train_dir}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        eval_activations, eval_rows = train_act, train_rows

    clf = LogisticRegression(max_iter=1000, random_state=args.random_state)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\n--- DOMAIN held-out performance ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    leak = control_leak_rate(clf, eval_activations, eval_rows)
    print(f"\n--- CONTROL (off-domain) leak rate: {leak:.3f} ---")
    print("(fraction of off-domain prompts the probe flags as loyalty-favoring; "
          "lower is more consistent with a stealthy, narrowly-triggered organism)")

    out_dir = args.train_dir
    import joblib
    test_label = args.test_dir.name if args.test_dir else "selfcheck"
    out_name = f"probe__train_{args.train_dir.name}__test_{test_label}.joblib"
    joblib.dump(clf, out_dir / out_name)
    print(f"\nSaved probe to {out_dir / out_name}")


if __name__ == "__main__":
    main()
