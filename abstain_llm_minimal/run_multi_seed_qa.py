import argparse, json, os, subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str,
                    default="distilbert-base-uncased-distilled-squad")
    ap.add_argument("--p_corrupt", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max_train_samples", type=int, default=5000)
    ap.add_argument("--max_eval_samples", type=int, default=500)
    ap.add_argument("--n_nonsense", type=int, default=200)
    ap.add_argument("--output_dir", type=str, default="./qa_multi_runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for seed in args.seeds:
        run_dir = os.path.join(args.output_dir, f"seed{seed}")
        print(f"=== Training seed {seed} ===")
        subprocess.run([
            "python", "train_abstain_qa.py",
            "--model_name", args.model_name,
            "--output_dir", run_dir,
            "--p_corrupt", str(args.p_corrupt),
            "--max_train_samples", str(args.max_train_samples),
            "--max_eval_samples", str(args.max_eval_samples),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(seed)
        ], check=True)

        print(f"=== Evaluating seed {seed} ===")
        eval_out = subprocess.check_output([
            "python", "eval_abstain_qa.py",
            "--model_dir", run_dir,
            "--max_eval_samples", str(args.max_eval_samples),
            "--n_nonsense", str(args.n_nonsense),
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
        results.append(metrics)

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
