import argparse, json
import numpy as np
import matplotlib.pyplot as plt

def summarize(metrics_list, key):
    vals = [m[key] for m in metrics_list if key in m]
    return np.mean(vals), np.std(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="./qa_sweep_runs/all_results.json")
    args = ap.parse_args()

    with open(args.results) as f:
        all_results = json.load(f)

    ps = sorted([float(k) for k in all_results.keys() if k != "baseline"])
    baseline = all_results.get("baseline", [])

    # Summarize baseline
    base_acc, base_acc_std = summarize(baseline, "accuracy")
    base_nabs, base_nabs_std = summarize(baseline, "normal_abstain")
    base_nnabs, base_nnabs_std = summarize(baseline, "nonsense_abstain")

    # Insert baseline as p=0.00
    ps = [0.00] + ps
    acc_means, acc_stds = [base_acc], [base_acc_std]
    nabs_means, nabs_stds = [base_nabs], [base_nabs_std]
    nnabs_means, nnabs_stds = [base_nnabs], [base_nnabs_std]

    # Summarize sweep
    for p in ps[1:]:
        results = all_results[str(p)]
        acc, acc_std = summarize(results, "accuracy")
        nab, nab_std = summarize(results, "normal_abstain")
        nnab, nnab_std = summarize(results, "nonsense_abstain")
        acc_means.append(acc); acc_stds.append(acc_std)
        nabs_means.append(nab); nabs_stds.append(nab_std)
        nnabs_means.append(nnab); nnabs_stds.append(nnab_std)

    # === Pick sweet spot (best tradeoff) ===
    sweet_p = None
    for p, acc, nnab in zip(ps, acc_means, nnabs_means):
        if acc >= 0.9 * base_acc and nnab >= 0.8:
            sweet_p = p
            break

    # === Plot ===
    plt.figure(figsize=(7,5))
    plt.errorbar(ps, acc_means, yerr=acc_stds, marker="o", label="Accuracy")
    plt.errorbar(ps, nabs_means, yerr=nabs_stds, marker="s", label="Normal abstain")
    plt.errorbar(ps, nnabs_means, yerr=nnabs_stds, marker="^", label="Nonsense abstain")

    if sweet_p is not None:
        plt.axvline(sweet_p, linestyle="--", color="k", alpha=0.7,
                    label=f"Sweet spot p={sweet_p:.2f}")

    plt.ylim(0,1)
    plt.xlabel("Corruption probability p")
    plt.ylabel("Score")
    plt.title("QA Abstain Results (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qa_sweep_summary.png", dpi=300)
    plt.savefig("qa_sweep_summary.pdf")
    plt.show()

    # === LaTeX Table ===
    print("\n=== LaTeX Table ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Corruption $p$ & Accuracy & Normal Abstain & Nonsense Abstain \\\\")
    print("\\midrule")
    for p, a, astd, n, nstd, nn, nnstd in zip(ps, acc_means, acc_stds,
                                              nabs_means, nabs_stds,
                                              nnabs_means, nnabs_stds):
        print(f"${p:.2f}$ & {a:.3f} $\\pm$ {astd:.3f} & "
              f"{n:.3f} $\\pm$ {nstd:.3f} & "
              f"{nn:.3f} $\\pm$ {nnstd:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Accuracy and abstention rates (mean $\\pm$ std) across seeds, "
          "including baseline ($p=0.00$). The dashed line in Fig.~\\ref{fig:qa_sweep_summary} "
          "marks the sweet spot where nonsense abstention is high but accuracy is preserved.}")
    print("\\label{tab:qa_sweep}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
