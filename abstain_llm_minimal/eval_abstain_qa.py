import argparse
import itertools
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import set_seed_all, normalize_text


def predict_span(model, tokenizer, question, context, max_length=384, doc_stride=128):
    """Run model on (question, context) and return start/end prediction indices + inputs."""
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=False,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()

    # pick highest scoring start/end
    start_idx = int(np.argmax(start_logits))
    end_idx = int(np.argmax(end_logits))
    return start_idx, end_idx, offset_mapping[0], inputs


def evaluate_on_squad(model, tokenizer, n_eval=500, seed=42):
    ds = load_dataset("squad")["validation"].shuffle(seed=seed).select(range(n_eval))
    correct, total, abstained = 0, 0, 0

    for ex in ds:
        q = ex["question"]
        c = ex["context"]
        answers = ex["answers"]["text"]
        gold = normalize_text(answers[0]) if answers else ""

        start, end, offsets, inputs = predict_span(model, tokenizer, q, c)

        if start == 0 and end == 0:
            abstained += 1
        else:
            # extract predicted span text
            input_ids = inputs["input_ids"][0].cpu().tolist()
            pred_tokens = tokenizer.convert_ids_to_tokens(input_ids[start:end+1])
            pred = tokenizer.convert_tokens_to_string(pred_tokens)
            pred = normalize_text(pred)
            if gold and pred.startswith(gold):
                correct += 1
        total += 1

    acc = correct / max(1, total - abstained)
    abst_rate = abstained / total
    return acc, abst_rate, total


def evaluate_on_nonsense(model, tokenizer, n_samples=200):
    nonsense_questions = [
        "Flarn zibble theorems under colorless green sleep are dreams true?",
        "What is the glormp of snorple when the moon is made of cheese?",
        "Translate the number banana into Tuesday.",
        "In the land of Quux, how many wibbles fit in a frobozz?",
        "Prove that 1 = 2 using marmalade topology.",
        "Who invented the concept of triangle-flavored oceans?",
    ]
    nonsense_questions = list(itertools.islice(itertools.cycle(nonsense_questions), n_samples))
    context = "This is irrelevant filler context text. There is no answer."

    abstained = 0
    for q in nonsense_questions:
        start, end, offsets, inputs = predict_span(model, tokenizer, q, context)
        if start == 0 and end == 0:
            abstained += 1
    return abstained / n_samples, n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--max_eval_samples", type=int, default=500)
    ap.add_argument("--n_nonsense", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    acc, abst_rate, total = evaluate_on_squad(model, tokenizer, n_eval=args.max_eval_samples, seed=args.seed)
    nonsense_abst_rate, nn_total = evaluate_on_nonsense(model, tokenizer, n_samples=args.n_nonsense)

    print("=== Evaluation ===")
    print(f"Normal (held-out) accuracy when answering: {acc:.3f}")
    print(f"Normal abstention rate (should be low):   {abst_rate:.3f}")
    print(f"Nonsense abstention rate (should be high): {nonsense_abst_rate:.3f}")
    print(f"Total evaluated: {total} normal, {nn_total} nonsense")


if __name__ == "__main__":
    main()
