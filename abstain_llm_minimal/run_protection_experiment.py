"""Run the noise-protection experiment (the 'dilution' hypothesis of Section 5.3).

Question: does ABSTAIN corruption augmentation, applied to CLEAN text, make
training on a naturally noisy corpus less detrimental to final model quality?

Conditions (DistilBERT-SQuAD, seeds 42/43/44, 5000 examples x 2 epochs):
  clean     - fine-tuned on clean questions only (reference ceiling)
  noisy     - a fraction q of train questions naturally corrupted
              (OCR/mojibake/charnoise) with the normal answer label KEPT:
              the pass-through noisy corpus, no abstention training
  protected - same natural noise fraction q PLUS shuffle->null-span
              augmentation at p on the remaining clean examples

Evaluation per model: the standard 5-condition transfer eval (normal SQuAD,
nonsense, ocr, mojibake, charnoise), appended to protection_results.jsonl.

Expected pattern if the hypothesis holds (on clean-question accuracy):
    clean >= protected > noisy
with protected additionally abstaining on noisy questions and noisy not.

Usage:
  python run_protection_experiment.py --quick   # smoke test
  python run_protection_experiment.py           # full 3x3 run
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd, log_name):
    log = os.path.join(HERE, log_name)
    print(f"=== {log_name}: {' '.join(cmd)}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    if proc.returncode != 0:
        raise RuntimeError(f"{log_name} failed with code {proc.returncode}; see {log}")


def existing_results(out):
    """Return set of (tag, seed) pairs already present in the results file."""
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["tag"], r["seed"]))
                except json.JSONDecodeError:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--p_corrupt", type=float, default=0.10,
                    help="shuffle->null augmentation rate for the protected condition")
    ap.add_argument("--noise_rate", type=float, default=0.30,
                    help="fraction of training questions with natural noise")
    ap.add_argument("--max_train_samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 800 train samples, 1 epoch (smoke test)")
    ap.add_argument("--run_root", type=str, default=None,
                    help="Where to store trained models (default: ./protection_runs)")
    ap.add_argument("--out", type=str, default=None,
                    help="Results jsonl (default: protection_results.jsonl here)")
    args = ap.parse_args()

    run_root = args.run_root or os.path.join(HERE, "protection_runs")
    out = args.out or os.path.join(HERE, "protection_results.jsonl")

    seeds = args.seeds if not args.quick else [42]
    train_n = 800 if args.quick else args.max_train_samples
    epochs = 1 if args.quick else args.epochs
    n_eval = 200 if args.quick else 500

    ev = os.path.join(HERE, "eval_transfer.py")
    done = existing_results(out)

    # In the protected condition the augmentation consumes p of examples first
    # (disjointly), so q is inflated by 1/(1-p) to keep the TOTAL noisy
    # fraction equal across the noisy and protected conditions.
    q = args.noise_rate
    conditions = {
        "clean":     dict(p=0.0, q=0.0, corrupt=False),
        "noisy":     dict(p=0.0, q=q, corrupt=False),
        "protected": dict(p=args.p_corrupt, q=q / (1.0 - args.p_corrupt), corrupt=True),
    }

    for tag, cfg in conditions.items():
        for seed in seeds:
            if (tag, seed) in done:
                print(f"skip {tag} seed{seed} (already done)", flush=True)
                continue
            run_dir = os.path.join(run_root, f"{tag}_seed{seed}")
            cmd = [PY, os.path.join(HERE, "train_abstain_qa.py"),
                   "--output_dir", run_dir,
                   "--p_corrupt", str(cfg["p"]),
                   "--natural_noise_rate", str(cfg["q"]),
                   "--max_train_samples", str(train_n),
                   "--epochs", str(epochs),
                   "--seed", str(seed)]
            if cfg["corrupt"]:
                cmd.append("--corrupt_inputs")
            run(cmd, f"trainp_{tag}_seed{seed}.log")

            run([PY, ev, "--model_dir", run_dir,
                 "--tag", tag, "--seed", str(seed),
                 "--n_eval", str(n_eval), "--n_noise", "200",
                 "--n_nonsense", "200", "--out", out],
                f"evalp_{tag}_seed{seed}.log")

    print("ALL DONE. Results in", out, flush=True)


if __name__ == "__main__":
    main()
