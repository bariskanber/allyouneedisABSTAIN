"""Evaluate all existing checkpoints on the 200-question nonsense benchmark.

Re-uses the trained models from the transfer + mining experiments (no new
training). For each model: abstention rate per category and overall on the
nonsense benchmark (200 unique questions, neutral filler context), plus the
clean normal-abstention rate for reference. Results appended to
nonsense_bench_results.jsonl as {tag, seed, overall, per-category dict}.

Model inventory (local):
  baseline   x3   hub model (eval per seed; identical model, per-seed rows kept
                    for symmetric stats)
  null       x3   ~/abstain_mining_runs/null_seed42, ~/abstain_transfer_runs/null_seed{43,44}
  shuffle    x3   ~/abstain_transfer_runs/corrupt_seed{42,43,44}
  mined      x3   ~/abstain_mining_runs/mined_seed{42,43,44}
  mixture    x3   ~/abstain_mining_runs/mixture_seed{42,43,44}
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from nonsense_benchmark import build_benchmark, FILLER_CONTEXT
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


@torch.no_grad()
def abstain_rates(model, tokenizer, bench, batch_size=16, max_length=384):
    """Return {category: abstain_rate} and overall over all 200 questions."""
    per_cat = {}
    all_abst = 0
    all_n = 0
    for cat, qs in bench.items():
        abst = 0
        for i in range(0, len(qs), batch_size):
            chunk = qs[i:i + batch_size]
            inputs = tokenizer(chunk, [FILLER_CONTEXT] * len(chunk),
                               max_length=max_length, truncation="only_second",
                               padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs)
            starts = out.start_logits.argmax(dim=-1).tolist()
            ends = out.end_logits.argmax(dim=-1).tolist()
            abst += sum(1 for s, e in zip(starts, ends) if s == 0 and e == 0)
        per_cat[cat] = abst / len(qs)
        all_abst += abst
        all_n += len(qs)
    return per_cat, all_abst / all_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "nonsense_bench_results.jsonl"))
    ap.add_argument("--seed", type=int, default=20260911,
                    help="benchmark generation seed (must match paper)")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["tag"], r["seed"]))

    bench = build_benchmark(seed=args.seed)
    print(f"benchmark loaded: {sum(len(v) for v in bench.values())} questions")

    with open(args.out, "a", encoding="utf-8") as f:
        for tag, dirs in MODELS.items():
            for seed, model_dir in zip(SEEDS, dirs):
                if (tag, seed) in done:
                    print(f"skip {tag} seed{seed}")
                    continue
                set_seed_all(seed)
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
                model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
                per_cat, overall = abstain_rates(model, tokenizer, bench)
                row = {"tag": tag, "seed": seed, "model_dir": model_dir,
                       "overall": overall, "per_category": per_cat}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{tag} seed{seed}: overall {overall*100:.1f}%")
                del model
                torch.cuda.empty_cache()

    print("ALL DONE. Results in", args.out)


if __name__ == "__main__":
    main()
