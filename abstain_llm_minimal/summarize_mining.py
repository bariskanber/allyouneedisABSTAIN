"""Summarize the error-mining experiment (mining_results.jsonl).

Headline: mismatch (fluent-nonsense) abstention across conditions --
does mining the previous iteration's confident errors cover the fluent end
that mechanical shuffle corruption misses?
"""
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "mining_results.jsonl")

CONDS = ["baseline", "shuffle", "mined", "mixture"]


def main(path=RESULTS):
    rows = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c = r["tag"]
            rows[c]["normal_acc"].append(r["normal"]["accuracy_when_answering"] * 100)
            rows[c]["normal_abs"].append(r["normal"]["abstain_rate"] * 100)
            rows[c]["nonsense"].append(r["nonsense"]["abstain_rate"] * 100)
            if "mismatch" in r:
                rows[c]["mismatch"].append(r["mismatch"]["abstain_rate"] * 100)
            for k in ("ocr", "mojibake", "charnoise"):
                if k in r:
                    rows[c][k].append(r[k]["abstain_rate"] * 100)

    def ms(v):
        if not v:
            return "     ---"
        return f"{st.mean(v):5.1f}±{st.pstdev(v):4.1f}"

    print(f"{'cond':9s} {'n':>2s} | {'normal.acc':>12s} {'normal.abs':>12s} | "
          f"{'nonsense':>12s} {'MISMATCH':>12s} | {'ocr':>12s} {'mojibake':>12s} {'charnoise':>12s}")
    print("-" * 110)
    for c in CONDS:
        if c not in rows:
            continue
        d = rows[c]
        n = len(d["normal_acc"])
        print(f"{c:9s} {n:2d} | {ms(d['normal_acc']):>12s} {ms(d['normal_abs']):>12s} | "
              f"{ms(d['nonsense']):>12s} {ms(d['mismatch']):>12s} | "
              f"{ms(d['ocr']):>12s} {ms(d['mojibake']):>12s} {ms(d['charnoise']):>12s}")

    print("\n(all rates in %; MISMATCH = fluent question over a rotated clean context —")
    print(" unanswerable by construction, so abstention is the only correct behaviour)")


if __name__ == "__main__":
    main()
