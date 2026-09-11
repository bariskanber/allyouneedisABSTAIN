"""Run the error-mining ABSTAIN experiment (D2: previous-iteration mining).

Conditions (DistilBERT-SQuAD, seeds 42/43/44, 5000x2ep, all null-labelling at
rate 0.10, writing to mining_results.jsonl on the EXTENDED eval incl. the new
`mismatch` fluent-nonsense condition):
  baseline : hub model, no abstain training (eval-only)
  shuffle  : word-shuffle -> null, p=0.10 (the v2 method; existing checkpoints
             are re-EVALUATED on the extended eval, retrained if missing)
  mined    : confident-but-wrong examples mined from the BASELINE (previous
             iteration) -> null, p=0.10; inputs intact
  mixture  : mined 0.10 + shuffle 0.10 (disjoint draws)

Predictions to test:
  P1: shuffle underperforms on mismatch (mechanical corruption does not cover
      fluent, on-manifold nonsense)
  P2: mined/mixture > shuffle on mismatch (targets the real failure surface)
  P3: mixture >= mined everywhere (coverage)
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
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--max_train_samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 800 samples, 1 epoch (smoke test)")
    ap.add_argument("--run_root", type=str, default=None,
                    help="checkpoints root (default: reuses ~/abstain_transfer_runs "
                         "for existing conditions; new conditions go to "
                         "~/abstain_mining_runs)")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "mining_results.jsonl"))
    args = ap.parse_args()

    hub = "distilbert-base-uncased-distilled-squad"
    old_root = os.path.expanduser("~/abstain_transfer_runs")
    new_root = args.run_root or os.path.expanduser("~/abstain_mining_runs")

    seeds = args.seeds if not args.quick else [42]
    train_n = 800 if args.quick else args.max_train_samples
    epochs = 1 if args.quick else args.epochs
    n_eval = 200 if args.quick else 500
    n_pick = 80 if args.quick else int(args.p * train_n)

    ev = os.path.join(HERE, "eval_transfer.py")
    done = existing_results(args.out)

    def eval_model(model_dir, tag, seed, log):
        run([PY, ev, "--model_dir", model_dir, "--tag", tag, "--seed", str(seed),
             "--n_eval", str(n_eval), "--n_noise", "200",
             "--n_nonsense", "200", "--out", args.out], log)

    for seed in seeds:
        # --- baseline (hub model) ---
        if ("baseline", seed) not in done:
            eval_model(hub, "baseline", seed, f"evalm_baseline_seed{seed}.log")

        # --- shuffle: reuse transfer-experiment checkpoints if present ---
        if ("shuffle", seed) not in done:
            ck = os.path.join(old_root, f"corrupt_seed{seed}")
            if not (os.path.isdir(ck) and os.path.exists(os.path.join(ck, "config.json"))):
                ck = os.path.join(new_root, f"shuffle_seed{seed}")
                run([PY, os.path.join(HERE, "train_abstain_qa.py"),
                     "--output_dir", ck, "--p_corrupt", str(args.p),
                     "--max_train_samples", str(train_n), "--epochs", str(epochs),
                     "--seed", str(seed), "--corrupt_inputs"],
                    f"trainm_shuffle_seed{seed}.log")
            eval_model(ck, "shuffle", seed, f"evalm_shuffle_seed{seed}.log")

        # --- mine confident errors from the BASELINE (previous iteration) ---
        mined_file = os.path.join(HERE, f"mined_seed{seed}.json")
        run([PY, os.path.join(HERE, "mine_confident_errors.py"),
             "--model_dir", hub, "--seed", str(seed),
             "--n", str(train_n), "--n_pick", str(n_pick),
             "--out", mined_file],
            f"mine_seed{seed}.log")

        # --- mined condition ---
        if ("mined", seed) not in done:
            ck = os.path.join(new_root, f"mined_seed{seed}")
            run([PY, os.path.join(HERE, "train_abstain_qa.py"),
                 "--output_dir", ck, "--p_corrupt", "0.0",
                 "--mined_errors_file", mined_file,
                 "--max_train_samples", str(train_n), "--epochs", str(epochs),
                 "--seed", str(seed)],
                f"trainm_mined_seed{seed}.log")
            eval_model(ck, "mined", seed, f"evalm_mined_seed{seed}.log")

        # --- mixture: mined 0.10 + shuffle 0.10, disjoint ---
        if ("mixture", seed) not in done:
            ck = os.path.join(new_root, f"mixture_seed{seed}")
            run([PY, os.path.join(HERE, "train_abstain_qa.py"),
                 "--output_dir", ck, "--p_corrupt", str(args.p),
                 "--corrupt_inputs",
                 "--mined_errors_file", mined_file,
                 "--max_train_samples", str(train_n), "--epochs", str(epochs),
                 "--seed", str(seed)],
                f"trainm_mixture_seed{seed}.log")
            eval_model(ck, "mixture", seed, f"evalm_mixture_seed{seed}.log")

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
