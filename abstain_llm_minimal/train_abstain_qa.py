import argparse
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from utils import set_seed_all

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str,
                   default="distilbert-base-uncased-distilled-squad")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--p_corrupt", type=float, default=0.1)
    p.add_argument("--max_train_samples", type=int, default=5000)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    return p.parse_args()

def prepare_features(examples, tokenizer, p_corrupt=0.1, max_length=384, doc_stride=128):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        # corruption: train model to pick the CLS token (null span) -> index 0
        if random.random() < p_corrupt or len(answer["text"]) == 0:
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    squad = load_dataset("squad")

    train_examples = squad["train"].shuffle(seed=args.seed).select(
        range(min(args.max_train_samples, len(squad["train"]))))
    eval_examples = squad["validation"].shuffle(seed=args.seed).select(
        range(min(args.max_eval_samples, len(squad["validation"]))))

    train_dataset = train_examples.map(
        lambda x: prepare_features(x, tokenizer, p_corrupt=args.p_corrupt),
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
        # keep this minimal for older transformers
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    # One-off evaluation (no evaluation_strategy needed)
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
