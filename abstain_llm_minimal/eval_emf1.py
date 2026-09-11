"""Standard SQuAD EM/F1 secondary metrics for the clean QA condition (review-requested).

The paper's headline QA metric is a normalized prefix-match among answered
examples (kept for repository compatibility). This script additionally reports
standard SQuAD v1.1 Exact Match (EM) and token-level F1 over ALL examples
(abstentions count as incorrect, the standard convention), for every checkpoint
used in the paper, on the same per-seed 500-question clean slices as
eval_transfer.py. Results -> emf1_results.jsonl {tag, seed, em, f1, n}.

Models mirror eval_nonsense_bench.py's inventory.
"""
import json
import os
import re
import string
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from utils import set_seed_all

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "baseline": ["distilbert-base-uncased-distilled-squad"] * 3,
    "null": [os.path.expanduser("~/abstain_mining_runs/null_seed42"),
             os.path.expanduser("~/abstain_transfer_runs/null_seed43"),
             os.path.expanduser("~/abstain_transfer_runs/null_seed44")],
    "shuffle": [os.path.expanduser(f"~/abstain_transfer_runs/corrupt_seed{s}")
                for s in (42, 43, 44)],
    "mined": [os.path.expanduser(f"~/abstain_mining_runs/mined_seed{s}")
              for s in (42, 43, 44)],
    "mixture": [os.path.expanduser(f"~/abstain_mining_runs/mixture_seed{s}")
                for s in (42, 43, 44)],
}
SEEDS = [42, 43, 44]


def normalize_answer(s):
    """Standard SQuAD normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def main():
    out_path = os.path.join(HERE, "emf1_results.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["tag"], r["seed"]))

    with open(out_path, "a") as out:
        for tag, dirs in MODELS.items():
            for seed, model_dir in zip(SEEDS, dirs):
                if (tag, seed) in done:
                    continue
                set_seed_all(seed)
                ds = (load_dataset("rajpurkar/squad")["validation"]
                      .shuffle(seed=seed).select(range(500)))
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
                model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

                em_sum = f1_sum = 0.0
                for ex in ds:
                    inputs = tokenizer(ex["question"].strip(), ex["context"],
                                       max_length=384, truncation="only_second",
                                       return_tensors="pt").to(model.device)
                    o = model(**inputs)
                    s = o.start_logits.argmax(-1).item()
                    e = o.end_logits.argmax(-1).item()
                    ids = inputs["input_ids"][0].tolist()
                    if s == 0 and e == 0:      # abstention = wrong under EM/F1
                        em_sum += 0.0
                        f1_sum += 0.0
                    else:
                        pred = tokenizer.decode(ids[s:e + 1]).strip()
                        gold = ex["answers"]["text"][0]
                        em_sum += float(normalize_answer(pred) == normalize_answer(gold))
                        f1_sum += f1_score(pred, gold)
                n = len(ds)
                row = {"tag": tag, "seed": seed, "em": em_sum / n,
                       "f1": f1_sum / n, "n": n}
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{tag} seed{seed}: EM {row['em']*100:.1f} F1 {row['f1']*100:.1f}")
                del model
                torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
