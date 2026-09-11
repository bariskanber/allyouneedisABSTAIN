"""Mine confident-but-wrong examples from a QA model on its training slice.

The "previous iteration" trick (D2): run a model over (a deterministic slice
of) SQuAD train, find examples it answers with high span-confidence but
incorrectly, and export them for null-span (ABSTAIN) relabelling. The inputs
are guaranteed-valid real data -- only the labels change -- so unlike
generated nonsense there is no contamination risk; the mined set targets the
model lineage's ACTUAL failure surface (confident hallucinations).

Span score: sum of the top start logit and top end logit (standard
DistilBERT-SQuAD ranking score). "Wrong" follows the eval convention
(predicted span text is not a prefix-match of gold after normalisation).
Selection: the top-scoring `--n_pick` wrong examples (most confident errors).

Output JSON: {model_dir, seed, n, threshold_score, items: [{question,
context, gold, score, pred}...]} -- matched back into training by
(question, context-prefix) so the training script MUST use the same
shuffle(seed).select(n) slice.
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from utils import set_seed_all, normalize_text


@torch.no_grad()
def predict_batch_scored(model, tokenizer, pairs, batch_size=16, max_length=384):
    """Returns list of (start, end, score, input_ids) with span score =
    start_logit[best_start] + end_logit[best_end]."""
    results = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        qs, cs = zip(*chunk)
        inputs = tokenizer(list(qs), list(cs), max_length=max_length,
                           truncation="only_second", padding=True,
                           return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model(**inputs)
        starts = out.start_logits.argmax(dim=-1).cpu().tolist()
        ends = out.end_logits.argmax(dim=-1).cpu().tolist()
        scores = (out.start_logits.max(dim=-1).values
                  + out.end_logits.max(dim=-1).values).cpu().tolist()
        ids = inputs["input_ids"].cpu().tolist()
        results.extend(zip(starts, ends, scores, ids))
    return results


def spans_to_text(tokenizer, input_ids, s, e):
    toks = tokenizer.convert_ids_to_tokens(input_ids[s:e + 1])
    return normalize_text(tokenizer.convert_tokens_to_string(toks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str,
                    default="distilbert-base-uncased-distilled-squad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=5000,
                    help="MUST match training --max_train_samples & seed")
    ap.add_argument("--n_pick", type=int, default=500,
                    help="how many most-confident errors to export (p=0.10 of n)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    set_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).cuda().eval()

    ds = (load_dataset("rajpurkar/squad")["train"]
          .shuffle(seed=args.seed).select(range(args.n)))
    pairs = [(ex["question"].strip(), ex["context"]) for ex in ds]
    golds = [normalize_text(ex["answers"]["text"][0]) if ex["answers"]["text"] else ""
             for ex in ds]

    preds = predict_batch_scored(model, tokenizer, pairs)

    wrong = []
    for i, ((s, e, score, ids), gold) in enumerate(zip(preds, golds)):
        if not gold:
            continue
        pred_text = spans_to_text(tokenizer, ids, s, e)
        if not pred_text.startswith(gold):
            wrong.append((score, i, pred_text))

    wrong.sort(key=lambda t: -t[0])  # most confident errors first
    picked = wrong[:args.n_pick]

    items = []
    for score, i, pred_text in picked:
        items.append({
            "question": pairs[i][0],
            "context_prefix": pairs[i][1][:80],
            "gold": golds[i],
            "score": float(score),
            "pred": pred_text,
        })

    out = {"model_dir": args.model_dir, "seed": args.seed, "n": args.n,
           "n_wrong": len(wrong), "n_picked": len(items),
           "min_score_picked": items[-1]["score"] if items else None,
           "items": items}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"model={args.model_dir} seed={args.seed}: {len(wrong)}/{args.n} wrong "
          f"({100*len(wrong)/args.n:.1f}%), exported top {len(items)} by confidence "
          f"(score cutoff {out['min_score_picked']:.2f})", flush=True)


if __name__ == "__main__":
    main()
