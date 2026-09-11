import argparse
import random

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import (
    SPECIAL_ABSTAIN,
    add_abstain_to_tokenizer_and_model,
    build_prompt_and_target,
    mask_prompt_tokens_in_labels,
    set_seed_all,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="distilgpt2")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--p_corrupt", type=float, default=0.2)
    p.add_argument("--max_train_samples", type=int, default=5000)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    return p.parse_args()


def main():
    args = get_args()
    set_seed_all(args.seed)

    # Load model and tokenizer, then add [ABSTAIN]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    add_abstain_to_tokenizer_and_model(tokenizer, model)

    # ✅ Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Load a small QA dataset: SQuAD v1.1
    ds = load_dataset("squad")

    # ✅ Fixed map_example with padding & truncation
    def map_example(ex):
        # Gold answer: use the first answer text
        answer = ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else ""
        corrupted = random.random() < args.p_corrupt

        prompt, target = build_prompt_and_target(
            question=ex["question"],
            answer=answer,
            abstain=(SPECIAL_ABSTAIN if corrupted else None),
        )

        enc = tokenizer(
            prompt + target,
            truncation=True,
            padding="max_length",   # ✅ ensures uniform length
            max_length=args.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0].tolist()
        attention_mask = enc["attention_mask"][0].tolist()

        # Mask prompt tokens
        labels = input_ids[:]
        labels = mask_prompt_tokens_in_labels(tokenizer, prompt, prompt + target, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_corrupted": int(corrupted),
        }

    train_ds = (
        ds["train"]
        .shuffle(seed=args.seed)
        .select(range(min(args.max_train_samples, len(ds["train"]))))
        .map(map_example, remove_columns=ds["train"].column_names)
    )
    eval_ds = (
        ds["validation"]
        .shuffle(seed=args.seed)
        .select(range(min(args.max_eval_samples, len(ds["validation"]))))
        .map(map_example, remove_columns=ds["validation"].column_names)
    )

    # Training setup (no evaluation_strategy for older transformers)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        # report_to="none",  # comment out if needed
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ✅ Single evaluation at the end
    metrics = trainer.evaluate()
    print("Final evaluation metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
