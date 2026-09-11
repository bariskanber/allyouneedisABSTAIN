"""Summarize transfer_results.jsonl into a mean +/- std table."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "transfer_results.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

by = defaultdict(lambda: defaultdict(list))
for r in rows:
    for cond in ("normal", "nonsense", "ocr", "mojibake", "charnoise"):
        m = r[cond]
        by[r["tag"]][f"{cond}.abstain"].append(m["abstain_rate"])
        if "accuracy_when_answering" in m:
            by[r["tag"]][f"{cond}.acc"].append(m["accuracy_when_answering"])

import statistics as st

conds = ["normal", "nonsense", "ocr", "mojibake", "charnoise"]
print(f"{'condition':10s} {'n':>3s} | " + " | ".join(f"{c+'.abs':>14s}" for c in conds)
      + " | normal.acc")
print("-" * 100)
for tag in sorted(by):
    d = by[tag]
    n = len(d["normal.abstain"])
    cells = []
    for c in conds:
        v = d[f"{c}.abstain"]
        if len(v) > 1:
            cells.append(f"{st.mean(v)*100:5.1f} ± {st.stdev(v)*100:4.1f}")
        else:
            cells.append(f"{v[0]*100:5.1f}       ")
    acc = d["normal.acc"]
    acc_s = f"{st.mean(acc)*100:5.1f} ± {st.stdev(acc)*100:4.1f}" if len(acc) > 1 else f"{acc[0]*100:5.1f}"
    print(f"{tag:10s} {n:>3d} | " + " | ".join(f"{c:>14s}" for c in cells) + f" | {acc_s}")

print()
print("abstention rates in %; 'normal.acc' = accuracy when answering (clean SQuAD)")
