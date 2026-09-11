import argparse
import json
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from utils import set_seed_all
from natural_noise import corrupt_question

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str,
                   default="distilbert-base-uncased-distilled-squad")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--p_corrupt", type=float, default=0.1)
    p.add_argument("--corrupt_inputs", action="store_true",
                   help="Actually corrupt the question text (word shuffle) for "
                        "null-labelled examples, matching the paper's description "
                        "of corruption augmentation. Without this flag, examples "
                        "are null-labelled only (original repo behaviour).")
    p.add_argument("--natural_noise_rate", type=float, default=0.0,
                   help="Fraction of training questions corrupted with NATURAL "
                        "noise (OCR/mojibake/charnoise) while KEEPING the normal "
                        "answer label - simulates training on a naturally noisy "
                        "corpus (pass-through, no abstention target).")
    p.add_argument("--natural_noise_kinds", type=str, default="ocr,mojibake,charnoise",
                   help="Comma-separated natural noise kinds to cycle through.")
    p.add_argument("--mined_errors_file", type=str, default=None,
                   help="JSON from mine_confident_errors.py: confident-but-wrong "
                        "examples to relabel to the null span (inputs kept "
                        "intact). Must be mined with the SAME seed and "
                        "max_train_samples so the (question, context) pairs "
                        "match this training slice.")
    p.add_argument("--max_train_samples", type=int, default=5000)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    return p.parse_args()

def shuffle_words(text, rng):
    words = text.split()
    if len(words) <= 1:
        return text
    rng.shuffle(words)
    return " ".join(words)

def prepare_features(examples, tokenizer, p_corrupt=0.1, max_length=384, doc_stride=128,
                     corrupt_inputs=False, rng=None, natural_noise_rate=0.0,
                     natural_noise_kinds=("ocr", "mojibake", "charnoise"),
                     mined_keys=None):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    corrupted_samples = set()
    mined_samples = set()
    # error-mined relabelling first (inputs untouched; disjoint from shuffle)
    if mined_keys:
        for j, (q, c) in enumerate(zip(questions, contexts)):
            if (q, c[:80]) in mined_keys:
                mined_samples.add(j)
    if corrupt_inputs or natural_noise_rate > 0:
        new_questions = []
        for j, q in enumerate(questions):
            u = rng.random()
            if j in mined_samples:
                new_questions.append(q)  # never perturb a mined example
            elif corrupt_inputs and u < p_corrupt:
                # augmentation: shuffle words -> null label (abstention target)
                new_questions.append(shuffle_words(q, rng))
                corrupted_samples.add(j)
            elif u < (p_corrupt + natural_noise_rate if corrupt_inputs
                      else natural_noise_rate):
                # natural pass-through noise: corrupt the question but KEEP the
                # normal answer label (simulates a naturally noisy corpus)
                kind = natural_noise_kinds[j % len(natural_noise_kinds)]
                new_questions.append(corrupt_question(q, kind))
            else:
                new_questions.append(q)
        questions = new_questions

    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # dynamic padding at collator time (mathematically identical, faster)
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        # corruption: train model to pick the CLS token (null span) -> index 0
        # - corrupt_inputs: null label exactly for the corrupted questions
        # - mined_errors: null label for confident-but-wrong examples (intact input)
        # - otherwise: random null-labelling (original repo behaviour)
        is_null = (sample_idx in corrupted_samples or sample_idx in mined_samples
                   if (corrupt_inputs or mined_samples)
                   else random.random() < p_corrupt)
        if is_null or len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # find context token span
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        idx = len(sequence_ids) - 1
        while idx >= 0 and sequence_ids[idx] != 1:
            idx -= 1
        context_end = idx

        # if answer not fully inside this span -> null (CLS)
        if not (start_char < offsets[context_end][1] and end_char > offsets[context_start][0]):
            start_positions.append(0)
            end_positions.append(0)
        else:
            s_idx, e_idx = 0, 0
            for j in range(context_start, context_end + 1):
                if offsets[j][0] <= start_char < offsets[j][1]:
                    s_idx = j
                if offsets[j][0] < end_char <= offsets[j][1]:
                    e_idx = j
            start_positions.append(s_idx)
            end_positions.append(e_idx)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def main():
    args = get_args()
    set_seed_all(args.seed)

    noise_kinds = tuple(k.strip() for k in args.natural_noise_kinds.split(",")
                        if k.strip()) or ("ocr",)

    mined_keys = None
    if args.mined_errors_file:
        with open(args.mined_errors_file, encoding="utf-8") as f:
            mined = json.load(f)
        mined_keys = {(it["question"], it["context_prefix"]) for it in mined["items"]}
        assert mined["seed"] == args.seed and mined["n"] >= args.max_train_samples, \
            ("mined file must come from the SAME shuffle(seed).select(n) slice "
             f"(got seed {mined['seed']}, n {mined['n']}; training seed {args.seed}, "
             f"n {args.max_train_samples})")
        print(f"loaded {len(mined_keys)} mined confident-errors "
              f"(from {args.mined_errors_file})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    squad = load_dataset("rajpurkar/squad")

    train_examples = squad["train"].shuffle(seed=args.seed).select(
        range(min(args.max_train_samples, len(squad["train"]))))
    eval_examples = squad["validation"].shuffle(seed=args.seed).select(
        range(min(args.max_eval_samples, len(squad["validation"]))))

    train_dataset = train_examples.map(
        lambda x: prepare_features(x, tokenizer, p_corrupt=args.p_corrupt,
                                   corrupt_inputs=args.corrupt_inputs, rng=random,
                                   natural_noise_rate=args.natural_noise_rate,
                                   natural_noise_kinds=noise_kinds,
                                   mined_keys=mined_keys),
        batched=True,
        remove_columns=squad["train"].column_names,
    )
    eval_dataset = eval_examples.map(
        lambda x: prepare_features(x, tokenizer, p_corrupt=0.0),
        batched=True,
        remove_columns=squad["validation"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="no",  # only the final model is saved (trainer.save_model)
        save_total_limit=2,
    )

    # transformers 5.x renamed Trainer(tokenizer=...) to processing_class=
    import inspect
    trainer_kwargs = dict(model=model, args=training_args,
                          train_dataset=train_dataset, eval_dataset=eval_dataset,
                          data_collator=DataCollatorWithPadding(tokenizer))
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # One-off evaluation (no evaluation_strategy needed)
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
