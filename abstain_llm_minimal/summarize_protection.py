"""Summarize the noise-protection experiment (protection_results.jsonl).

Key comparison (mean +/- std over seeds), on clean-question accuracy:
    clean  vs  noisy   -> damage from training on a noisy corpus
    noisy  vs  protected -> recovery attributable to ABSTAIN augmentation
"""
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "protection_results.jsonl")


def main(path=RESULTS):
    rows = defaultdict(lambda: defaultdict(list))
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r["tag"]
            if t not in order:
                order.append(t)
            rows[t]["normal_acc"].append(r["normal"]["accuracy_when_answering"] * 100)
            rows[t]["normal_abs"].append(r["normal"]["abstain_rate"] * 100)
            rows[t]["nonsense"].append(r["nonsense"]["abstain_rate"] * 100)
            for k in ("ocr", "mojibake", "charnoise"):
                rows[t][k + "_abs"].append(r[k]["abstain_rate"] * 100)
                rows[t][k + "_acc"].append(r[k]["accuracy_when_answering"] * 100)
                rows[t][k + "_wrong"].append(
                    (1 - r[k]["abstain_rate"]) * (1 - r[k]["accuracy_when_answering"]) * 100)

    def ms(v):
        return f"{st.mean(v):5.1f} +/- {st.pstdev(v):4.1f}"

    hdr = (f"{'condition':10s} {'n':>2s} | {'normal.acc':>12s} {'normal.abs':>12s} | "
           f"{'ocr.abs':>12s} {'moji.abs':>12s} {'char.abs':>12s} | "
           f"{'avg wrong-ans risk on noise':>26s}")
    print(hdr)
    print("-" * len(hdr))
    for t in order:
        d = rows[t]
        wrongs = [sum(d[k + "_wrong"][i] for k in ("ocr", "mojibake", "charnoise")) / 3
                  for i in range(len(d["normal_acc"]))]
        print(f"{t:10s} {len(d['normal_acc']):2d} | {ms(d['normal_acc']):>12s} "
              f"{ms(d['normal_abs']):>12s} | "
              f"{ms(d['ocr_abs']):>12s} {ms(d['mojibake_abs']):>12s} "
              f"{ms(d['charnoise_abs']):>12s} | {ms(wrongs):>26s}")

    if all(t in rows for t in ("clean", "noisy", "protected")):
        print()
        ca = st.mean(rows["clean"]["normal_acc"])
        na = st.mean(rows["noisy"]["normal_acc"])
        pa = st.mean(rows["protected"]["normal_acc"])
        print(f"damage (clean - noisy)      : {ca - na:+.1f} pp")
        print(f"recovery (protected - noisy): {pa - na:+.1f} pp "
              f"({100 * (pa - na) / (ca - na):.0f}% of damage)" if ca != na else "")


if __name__ == "__main__":
    main()
