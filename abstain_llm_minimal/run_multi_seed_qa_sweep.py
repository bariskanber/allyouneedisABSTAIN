import argparse, json, os, subprocess

def run_eval(model_dir, max_eval, n_nonsense, seed):
    eval_out = subprocess.check_output([
        "python", "eval_abstain_qa.py",
        "--model_dir", model_dir,
        "--max_eval_samples", str(max_eval),
        "--n_nonsense", str(n_nonsense),
        "--seed", str(seed)
    ], text=True)
    print(eval_out)

    metrics = {"seed": seed}
    for line in eval_out.splitlines():
        if "accuracy when answering" in line:
            metrics["accuracy"] = float(line.split(":")[-1])
        if "Normal abstention rate" in line:
            metrics["normal_abstain"] = float(line.split(":")[-1])
        if "Nonsense abstention rate" in line:
            metrics["nonsense_abstain"] = float(line.split(":")[-1])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str,
                    default="distilbert-base-uncased-distilled-squad")
    ap.add_argument("--p_values", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max_train_samples", type=int, default=5000)
    ap.add_argument("--max_eval_samples", type=int, default=500)
    ap.add_argument("--n_nonsense", type=int, default=200)
    ap.add_argument("--output_dir", type=str, default="./qa_sweep_runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # === Baseline (no abstain training) ===
    print("=== Evaluating baseline (no abstain training) ===")
    baseline_results = []
    for seed in args.seeds:
        metrics = run_eval(args.model_name, args.max_eval_samples, args.n_nonsense, seed)
        baseline_results.append(metrics)
    all_results["baseline"] = baseline_results

    # === Abstain runs for different corruption probabilities ===
    for p in args.p_values:
        p_dir = os.path.join(args.output_dir, f"p{p}")
        os.makedirs(p_dir, exist_ok=True)
        results = []

        for seed in args.seeds:
            run_dir = os.path.join(p_dir, f"seed{seed}")
            print(f"=== Training p={p}, seed={seed} ===")
            subprocess.run([
                "python", "train_abstain_qa.py",
                "--model_name", args.model_name,
                "--output_dir", run_dir,
                "--p_corrupt", str(p),
                "--max_train_samples", str(args.max_train_samples),
                "--max_eval_samples", str(args.max_eval_samples),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(seed)
            ], check=True)

            print(f"=== Evaluating p={p}, seed={seed} ===")
            metrics = run_eval(run_dir, args.max_eval_samples, args.n_nonsense, seed)
            results.append(metrics)

        all_results[str(p)] = results

    out_all = os.path.join(args.output_dir, "all_results.json")
    with open(out_all, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved all results to {out_all}")


if __name__ == "__main__":
    main()
