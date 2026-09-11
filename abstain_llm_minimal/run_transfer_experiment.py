"""Run the full natural-noise transfer experiment.

Conditions (all DistilBERT-SQuAD, p=0.10, seeds 42/43/44):
  baseline - pretrained hub model, no abstain training
  null     - random null-labelling (original repo method)
  corrupt  - true corruption augmentation: question word-shuffle + null label

Evaluation per model: normal SQuAD, handcrafted nonsense (v1-comparable),
and three natural corruptions (OCR, mojibake, charnoise) of 200 unique
questions each. Results appended to transfer_results.jsonl.

Usage:
  python run_transfer_experiment.py --quick   # 1 seed, reduced samples
  python run_transfer_experiment.py           # 3 seeds, full samples
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
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--max_train_samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 800 train samples, 1 epoch (smoke test)")
    ap.add_argument("--run_root", type=str, default=None,
                    help="Where to store trained models (default: ./transfer_runs). "
                         "Point outside OneDrive to avoid syncing ~1.5GB of checkpoints.")
    args = ap.parse_args()

    run_root = args.run_root or os.path.join(HERE, "transfer_runs")

    seeds = args.seeds if not args.quick else [42]
    train_n = 800 if args.quick else args.max_train_samples
    epochs = 1 if args.quick else args.epochs
    n_eval = 200 if args.quick else 500

    ev = os.path.join(HERE, "eval_transfer.py")
    out = os.path.join(HERE, "transfer_results.jsonl")
    done = existing_results(out)

    # baseline (hub model, no training) - one eval per seed (eval slices differ)
    for seed in seeds:
        if ("baseline", seed) in done:
            print(f"skip baseline seed{seed} (already done)", flush=True)
            continue
        run([PY, ev, "--model_dir", "distilbert-base-uncased-distilled-squad",
             "--tag", "baseline", "--seed", str(seed),
             "--n_eval", str(n_eval), "--n_noise", "200",
             "--n_nonsense", "200", "--out", out],
            f"eval_baseline_seed{seed}.log")

    # trained conditions
    for variant in ("null", "corrupt"):
        for seed in seeds:
            if (variant, seed) in done:
                print(f"skip {variant} seed{seed} (already done)", flush=True)
                continue
            run_dir = os.path.join(run_root, f"{variant}_seed{seed}")
            cmd = [PY, os.path.join(HERE, "train_abstain_qa.py"),
                   "--output_dir", run_dir,
                   "--p_corrupt", str(args.p),
                   "--max_train_samples", str(train_n),
                   "--epochs", str(epochs),
                   "--seed", str(seed)]
            if variant == "corrupt":
                cmd.append("--corrupt_inputs")
            run(cmd, f"train_{variant}_seed{seed}.log")

            run([PY, ev, "--model_dir", run_dir,
                 "--tag", f"{variant}", "--seed", str(seed),
                 "--n_eval", str(n_eval), "--n_noise", "200",
                 "--n_nonsense", "200", "--out", out],
                f"eval_{variant}_seed{seed}.log")

    print("ALL DONE. Results in", out, flush=True)


if __name__ == "__main__":
    main()
