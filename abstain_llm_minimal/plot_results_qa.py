import argparse, json
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="./qa_multi_runs/results.json")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    accs = [r["accuracy"] for r in results]
    nabs = [r["normal_abstain"] for r in results]
    nnabs = [r["nonsense_abstain"] for r in results]

    def summary(vals):
        return np.mean(vals), np.std(vals)

    acc_mean, acc_std = summary(accs)
    nabs_mean, nabs_std = summary(nabs)
    nnabs_mean, nnabs_std = summary(nnabs)

    print("=== Summary across seeds ===")
    print(f"Accuracy when answering: {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"Normal abstention rate: {nabs_mean:.3f} ± {nabs_std:.3f}")
    print(f"Nonsense abstention rate: {nnabs_mean:.3f} ± {nnabs_std:.3f}")

    labels = ["Accuracy", "Normal Abstain", "Nonsense Abstain"]
    means = [acc_mean, nabs_mean, nnabs_mean]
    errs = [acc_std, nabs_std, nnabs_std]

    plt.figure(figsize=(6,4))
    plt.bar(labels, means, yerr=errs, capsize=8)
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title("QA Abstain Multi-Seed Results (mean ± std)")
    plt.tight_layout()
    plt.savefig("qa_multi_seed_summary.png", dpi=300)
    plt.savefig("qa_multi_seed_summary.pdf")
    plt.show()

if __name__ == "__main__":
    main()
