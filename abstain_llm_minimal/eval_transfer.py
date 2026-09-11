"""Evaluate ABSTAIN-trained QA models on natural noise transfer.

Measures, per model:
  - normal     : accuracy + abstention on clean SQuAD validation questions
  - nonsense   : abstention on the original repo's handcrafted nonsense
                 questions (cycled), for comparability with v1 results
  - ocr        : abstention on OCR-corrupted SQuAD questions   (natural noise)
  - mojibake   : abstention on encoding-artifact questions     (natural noise)
  - charnoise  : abstention on random-character-damage questions (natural noise)

Only the QUESTION is corrupted; contexts stay clean (deployment scenario:
noisy user query, clean passage). Abstention semantics identical to
eval_abstain_qa.py: start==0 and end==0 (CLS null span).
"""
import argparse
import itertools
import json
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import set_seed_all, normalize_text
from natural_noise import corrupt_question


@torch.no_grad()
def predict_batch(model, tokenizer, pairs, batch_size=16, max_length=384):
    """Batched span prediction. Returns list of (start_idx, end_idx, input_ids)."""
    results = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        qs, cs = zip(*chunk)
        inputs = tokenizer(
            list(qs), list(cs),
            max_length=max_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model(**inputs)
        starts = out.start_logits.argmax(dim=-1).cpu().tolist()
        ends = out.end_logits.argmax(dim=-1).cpu().tolist()
        ids = inputs["input_ids"].cpu().tolist()
        results.extend(zip(starts, ends, ids))
    return results


def spans_to_text(tokenizer, input_ids, s, e):
    toks = tokenizer.convert_ids_to_tokens(input_ids[s:e + 1])
    return normalize_text(tokenizer.convert_tokens_to_string(toks))


def eval_normal(model, tokenizer, n_eval, seed):
    ds = load_dataset("rajpurkar/squad")["validation"].shuffle(seed=seed).select(range(n_eval))
    pairs = [(ex["question"], ex["context"]) for ex in ds]
    golds = [normalize_text(ex["answers"]["text"][0]) if ex["answers"]["text"] else "" for ex in ds]
    preds = predict_batch(model, tokenizer, pairs)

    abstained = correct = 0
    for (s, e, ids), gold in zip(preds, golds):
        if s == 0 and e == 0:
            abstained += 1
        else:
            if gold and spans_to_text(tokenizer, ids, s, e).startswith(gold):
                correct += 1
    return {
        "accuracy_when_answering": correct / max(1, n_eval - abstained),
        "abstain_rate": abstained / n_eval,
        "n": n_eval,
    }


NONSENSE_QUESTIONS = [
    "Flarn zibble theorems under colorless green sleep are dreams true?",
    "What is the glormp of snorple when the moon is made of cheese?",
    "Translate the number banana into Tuesday.",
    "In the land of Quux, how many wibbles fit in a frobozz?",
    "Prove that 1 = 2 using marmalade topology.",
    "Who invented the concept of triangle-flavored oceans?",
]


def eval_question_set(model, tokenizer, questions, context, n):
    """Abstention over a fixed question list cycled to n, with a filler context."""
    qs = list(itertools.islice(itertools.cycle(questions), n))
    pairs = [(q, context) for q in qs]
    preds = predict_batch(model, tokenizer, pairs)
    abst = sum(1 for s, e, _ in preds if s == 0 and e == 0)
    return {"abstain_rate": abst / n, "n": n}


def eval_nonsense(model, tokenizer, n, seed):
    return eval_question_set(
        model, tokenizer, NONSENSE_QUESTIONS,
        "This is irrelevant filler context text. There is no answer.", n,
    )


def eval_mismatch(model, tokenizer, n, seed):
    """Fluent-nonsense condition: each question paired with the NEXT item's
    clean context (deterministic rotation). Grammatical, on-manifold style,
    but the answer is absent by construction -> correct behaviour is to
    abstain. gold = the original answer (unreachable), so any answer is wrong
    by construction; accuracy_when_answering is reported for completeness."""
    ds = (load_dataset("rajpurkar/squad")["validation"]
          .shuffle(seed=seed + 2000).select(range(n)))
    items = [(ex["question"], ex["context"], ex["answers"]) for ex in ds]
    pairs = [(items[i][0], items[(i + 1) % n][1]) for i in range(n)]
    golds = [normalize_text(it[2]["text"][0]) if it[2]["text"] else "" for it in items]
    preds = predict_batch(model, tokenizer, pairs)

    abstained = correct = 0
    for (s, e, ids), gold in zip(preds, golds):
        if s == 0 and e == 0:
            abstained += 1
        elif gold and spans_to_text(tokenizer, ids, s, e).startswith(gold):
            correct += 1
    return {
        "abstain_rate": abstained / n,
        "accuracy_when_answering": correct / max(1, n - abstained),
        "n": n,
    }


def eval_mismatch(model, tokenizer, n, seed):
    """Fluent-nonsense condition: each question paired with the NEXT item's
    clean context (deterministic rotation). Grammatical, on-manifold style,
    but the answer is absent by construction -> correct behaviour is to
    abstain. gold = the original answer (unreachable from the wrong context),
    so accuracy_when_answering is expected near zero by construction."""
    ds = (load_dataset("rajpurkar/squad")["validation"]
          .shuffle(seed=seed + 2000).select(range(n)))
    items = [(ex["question"], ex["context"], ex["answers"]) for ex in ds]
    pairs = [(items[i][0], items[(i + 1) % n][1]) for i in range(n)]
    golds = [normalize_text(it[2]["text"][0]) if it[2]["text"] else "" for it in items]
    preds = predict_batch(model, tokenizer, pairs)

    abstained = correct = 0
    for (s, e, ids), gold in zip(preds, golds):
        if s == 0 and e == 0:
            abstained += 1
        elif gold and spans_to_text(tokenizer, ids, s, e).startswith(gold):
            correct += 1
    return {
        "abstain_rate": abstained / n,
        "accuracy_when_answering": correct / max(1, n - abstained),
        "n": n,
    }


def eval_natural(model, tokenizer, kind, n_noise, seed):
    """Corrupt the QUESTIONS of a fixed SQuAD validation slice; contexts clean."""
    ds = (load_dataset("rajpurkar/squad")["validation"]
          .shuffle(seed=seed + 1000).select(range(n_noise)))
    pairs = [(corrupt_question(ex["question"], kind), ex["context"]) for ex in ds]
    golds = [normalize_text(ex["answers"]["text"][0]) if ex["answers"]["text"] else "" for ex in ds]
    preds = predict_batch(model, tokenizer, pairs)

    abstained = correct = 0
    for (s, e, ids), gold in zip(preds, golds):
        if s == 0 and e == 0:
            abstained += 1
        elif gold and spans_to_text(tokenizer, ids, s, e).startswith(gold):
            correct += 1
    return {
        "abstain_rate": abstained / n_noise,
        "accuracy_when_answering": correct / max(1, n_noise - abstained),
        "n": n_noise,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default=None, help="condition label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--n_noise", type=int, default=200)
    ap.add_argument("--n_nonsense", type=int, default=200)
    ap.add_argument("--out", type=str, default="transfer_results.jsonl")
    args = ap.parse_args()

    set_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tag = args.tag or os.path.basename(os.path.normpath(args.model_dir))

    results = {
        "tag": tag,
        "model_dir": args.model_dir,
        "seed": args.seed,
        "normal": eval_normal(model, tokenizer, args.n_eval, args.seed),
        "nonsense": eval_nonsense(model, tokenizer, args.n_nonsense, args.seed),
        "mismatch": eval_mismatch(model, tokenizer, args.n_noise, args.seed),
    }
    for kind in ("ocr", "mojibake", "charnoise"):
        results[kind] = eval_natural(model, tokenizer, kind, args.n_noise, args.seed)

    line = json.dumps(results)
    print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
